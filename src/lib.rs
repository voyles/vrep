//! `vrep-core` is a stateless "semantic grep" engine for binary vector search.
//!
//! The design keeps the hot path narrow:
//! - input is streamed through an `mmap`, so file IO is delegated to the kernel page cache
//!   rather than copied through user-space buffers;
//! - every line is treated as an independent record, which preserves Unix-style
//!   statelessness and makes the scanner easy to compose with existing tools;
//! - vector comparison is reduced to XOR plus population count over 512 bits,
//!   which maps well to modern CPUs and leaves room for future SIMD-friendly encoders.
//!
//! The current encoder is intentionally mocked. Future implementations are expected to
//! produce the same 512-bit shape through sign-based quantization, where floating-point
//! activations are converted into packed bits before entering the Hamming scan.

use std::cmp::min;
use std::collections::HashMap;
use std::mem::size_of;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use memchr::memchr_iter;
use memmap2::{Mmap, MmapOptions};
use ort::{inputs, session::Session, value::TensorRef};
use rayon::prelude::*;
use regex::Regex;
use tokenizers::{Encoding, Tokenizer};

/// Number of `u64` lanes in a binary embedding.
///
/// Eight lanes yield a 512-bit vector, which is large enough to mimic a compact
/// semantic fingerprint while still fitting naturally into a tight XOR plus POPCNT loop.
pub const VECTOR_WORDS: usize = 8;
pub const ONNX_BATCH_SIZE: usize = 32;
const SEARCH_CHUNK_SIZE: usize = 5000;

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;
const DUMMY_LINE: &[u8] = b"vector-placeholder payload for binary scan benchmark\n";
const EMBEDDING_DIMS: usize = 384;
const INDEX_MAGIC: &[u8; 8] = b"VRPXIDX1";
const INDEX_HEADER_BYTES: usize = 8 + 8 + 16 + 8;
const CHEAP_BYPASS_MIN_LEN: usize = 12;

/// Generates a packed 512-bit embedding for a line of text.
///
/// Implementers are expected to be deterministic and thread-safe because the scan engine
/// may call them from parallel contexts. A future model-backed encoder should perform
/// sign-based quantization here: compute floating-point activations, threshold each value
/// by sign, and pack the resulting bits into the returned `[u64; 8]`.
pub trait EmbeddingGenerator: Sync {
    /// Converts `text` into a fixed-width 512-bit binary embedding.
    ///
    /// The returned array is consumed directly by the Hamming distance loop, so avoiding
    /// heap allocation and preserving stable bit layout matters for throughput.
    fn generate(&self, text: &str) -> [u64; VECTOR_WORDS];
}

/// Placeholder encoder used to establish a stable performance baseline.
///
/// This encoder uses a lightweight hash mixer instead of a real ML model so the current
/// benchmarks isolate chunking, memory access, and Hamming scan costs.
#[derive(Clone, Copy, Debug, Default)]
pub struct MockEncoder;

impl EmbeddingGenerator for MockEncoder {
    fn generate(&self, text: &str) -> [u64; VECTOR_WORDS] {
        mock_binary_embedding(text.as_bytes())
    }
}

/// ONNX-backed encoder using all-MiniLM-L6-v2 artifacts in `./model`.
///
/// The encoder loads `tokenizer.json` and `model.onnx` once, validates required input bindings,
/// then serves inference from per-thread ONNX sessions for high concurrency under Rayon.
/// It applies mean pooling to token embeddings and then
/// quantizes the pooled 384-dim float vector into the 512-bit `[u64; 8]` signature.
pub struct OnnxEncoder {
    tokenizer: Tokenizer,
    model_path: Arc<PathBuf>,
    input_bindings: InputBindings,
    cache: Mutex<HashMap<u64, [u64; VECTOR_WORDS]>>,
    cache_capacity: usize,
    warned_once: AtomicBool,
}

#[derive(Clone, Debug)]
struct InputBindings {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
}

thread_local! {
    static TLS_MODEL_SESSION: OnceLock<io::Result<(String, UnsafeCell<Session>)>> = const { OnceLock::new() };
}

impl OnnxEncoder {
    /// Loads ONNX model assets from `model_dir`.
    ///
    /// Expected files:
    /// - `model.onnx`
    /// - `tokenizer.json`
    pub fn from_model_dir(model_dir: &Path) -> io::Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_path = model_dir.join("model.onnx");

        // eprintln!("Initializing ONNX session...");

        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("failed to load tokenizer.json: {error}"),
            )
        })?;

        let mut session = Self::build_session(&model_path)?;
        let input_bindings = Self::discover_input_bindings(&session)?;

        // Force one synchronous inference during startup so model-level stalls surface
        // during initialization rather than inside the parallel scan loop.
        let warmup = tokenizer.encode("warmup probe", true).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("tokenization failed during warm-up: {error}"),
            )
        })?;
        let _ = Self::run_inference(&mut session, &input_bindings, &warmup)?;

        // eprintln!("ONNX Ready.");

        Ok(Self {
            tokenizer,
            model_path: Arc::new(model_path),
            input_bindings,
            cache: Mutex::new(HashMap::new()),
            cache_capacity: 16_384,
            warned_once: AtomicBool::new(false),
        })
    }

    fn build_session(model_path: &std::path::Path) -> std::io::Result<ort::session::Session> {
        // 1. We create the builder.
        let mut builder = ort::session::Session::builder()
            .map_err(|e| std::io::Error::other(format!("ORT builder fail: {e}")))?;

        // 2. We set the thread limits.
        // In rc.12, these return a Result<&mut SessionBuilder, Error>.
        builder = builder
            .with_intra_threads(1)
            .map_err(|e| std::io::Error::other(format!("ORT intra fail: {e}")))?;

        builder = builder
            .with_inter_threads(1)
            .map_err(|e| std::io::Error::other(format!("ORT inter fail: {e}")))?;

        // 3. We load the model file.
        builder
            .commit_from_file(model_path)
            .map_err(|e| std::io::Error::other(format!("failed to load model.onnx: {e}")))
    }

    fn discover_input_bindings(session: &Session) -> io::Result<InputBindings> {
        let names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();

        let input_ids = names
            .iter()
            .find(|name| name.contains("input_ids"))
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("model inputs missing input_ids; got {:?}", names),
                )
            })?;

        let attention_mask = names
            .iter()
            .find(|name| name.contains("attention_mask"))
            .cloned()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("model inputs missing attention_mask; got {:?}", names),
                )
            })?;

        let token_type_ids = names
            .iter()
            .find(|name| name.contains("token_type_ids"))
            .cloned();

        Ok(InputBindings {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    fn with_thread_session<F, T>(&self, op: F) -> io::Result<T>
    where
        F: FnOnce(&mut Session) -> io::Result<T>,
    {
        let key = self.model_path.to_string_lossy().to_string();
        let path = self.model_path.clone();

        TLS_MODEL_SESSION.with(|slot| {
            let (session_key, session_cell) = match slot.get_or_init(|| {
                Self::build_session(path.as_path())
                    .map(|session| (key.clone(), UnsafeCell::new(session)))
            }) {
                Ok(entry) => entry,
                Err(error) => {
                    return Err(io::Error::other(format!(
                        "failed to initialize thread-local ONNX session: {error}"
                    )))
                }
            };

            if session_key != &key {
                return Err(io::Error::other(
                    "thread-local ONNX session is bound to a different model path",
                ));
            }

            // SAFETY: `TLS_MODEL_SESSION` is thread-local, so each worker thread owns its
            // own `Session` instance. We only hand out a temporary mutable borrow during this
            // closure invocation on the current thread.
            let session = unsafe { &mut *session_cell.get() };
            op(session)
        })
    }

    pub fn batch_encode<'a>(&self, texts: &[&'a str]) -> Vec<[u64; VECTOR_WORDS]> {
        if texts.is_empty() {
            return Vec::new();
        }

        let mut results = vec![[0_u64; VECTOR_WORDS]; texts.len()];
        let mut missing_meta = Vec::with_capacity(texts.len());
        let mut missing_texts = Vec::with_capacity(texts.len());

        if let Ok(cache) = self.cache.lock() {
            for (index, &text) in texts.iter().enumerate() {
                if let Some(bits) = cheap_bypass_bits(text) {
                    results[index] = bits;
                    continue;
                }

                let cache_key = simple_hash(text);
                if let Some(bits) = cache.get(&cache_key) {
                    results[index] = *bits;
                } else {
                    missing_meta.push((index, cache_key));
                    missing_texts.push(text);
                }
            }
        } else {
            for (index, &text) in texts.iter().enumerate() {
                if let Some(bits) = cheap_bypass_bits(text) {
                    results[index] = bits;
                    continue;
                }

                missing_meta.push((index, simple_hash(text)));
                missing_texts.push(text);
            }
        }

        if missing_meta.is_empty() {
            return results;
        }

        let encoded: Vec<[u64; VECTOR_WORDS]> = match self.infer_embeddings_batch(&missing_texts) {
            Ok(bits_batch) => bits_batch,
            Err(error) => {
                if !self.warned_once.swap(true, Ordering::Relaxed) {
                    eprintln!("warning: ONNX inference unavailable, falling back to mock encoder: {error}");
                }
                missing_texts
                    .iter()
                    .map(|text| mock_binary_embedding(text.as_bytes()))
                    .collect()
            }
        };

        if let Ok(mut cache) = self.cache.lock() {
            for ((index, cache_key), bits) in missing_meta.into_iter().zip(encoded.into_iter()) {
                if cache.len() >= self.cache_capacity {
                    cache.clear();
                }
                cache.insert(cache_key, bits);
                results[index] = bits;
            }
        } else {
            for ((index, _), bits) in missing_meta.into_iter().zip(encoded.into_iter()) {
                results[index] = bits;
            }
        }

        results
    }

    pub fn batch_encode_owned(&self, texts: Vec<String>) -> Vec<Option<[u64; VECTOR_WORDS]>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let mut results = vec![None; texts.len()];
        let mut missing_meta = Vec::with_capacity(texts.len());
        let mut missing_indices = Vec::with_capacity(texts.len());
        let mut missing_refs = Vec::with_capacity(texts.len());

        if let Ok(cache) = self.cache.lock() {
            for (index, text) in texts.iter().enumerate() {
                if let Some(bits) = cheap_bypass_bits(text) {
                    results[index] = Some(bits);
                    continue;
                }

                let cache_key = simple_hash(text);
                if let Some(bits) = cache.get(&cache_key) {
                    results[index] = Some(*bits);
                } else {
                    missing_meta.push((index, cache_key));
                    missing_indices.push(index);
                    missing_refs.push(text.as_str());
                }
            }
        } else {
            for (index, text) in texts.iter().enumerate() {
                if let Some(bits) = cheap_bypass_bits(text) {
                    results[index] = Some(bits);
                    continue;
                }

                missing_meta.push((index, simple_hash(text)));
                missing_indices.push(index);
                missing_refs.push(text.as_str());
            }
        }

        if missing_meta.is_empty() {
            return results;
        }

        match self.infer_embeddings_batch(&missing_refs) {
            Ok(encoded) => {
                if let Ok(mut cache) = self.cache.lock() {
                    for ((index, cache_key), bits) in missing_meta.into_iter().zip(encoded.into_iter()) {
                        if cache.len() >= self.cache_capacity {
                            cache.clear();
                        }
                        cache.insert(cache_key, bits);
                        results[index] = Some(bits);
                    }
                } else {
                    for ((index, _), bits) in missing_meta.into_iter().zip(encoded.into_iter()) {
                        results[index] = Some(bits);
                    }
                }
            }
            Err(error) => {
                if !self.warned_once.swap(true, Ordering::Relaxed) {
                    eprintln!("warning: ONNX batch inference unavailable, retrying per record: {error}");
                }
                for index in missing_indices {
                    let text = &texts[index];
                    let bits = match catch_unwind(AssertUnwindSafe(|| self.try_generate_strict(text))) {
                        Ok(Ok(bits)) => Some(bits),
                        _ => None,
                    };
                    results[index] = bits;
                }
            }
        }

        results
    }

    fn try_generate_strict(&self, text: &str) -> io::Result<[u64; VECTOR_WORDS]> {
        if let Some(bits) = cheap_bypass_bits(text) {
            return Ok(bits);
        }

        let cache_key = simple_hash(text);
        if let Ok(cache) = self.cache.lock() {
            if let Some(bits) = cache.get(&cache_key) {
                return Ok(*bits);
            }
        }

        let encoding = self.tokenizer.encode(text, true).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("tokenization failed for record: {error}"),
            )
        })?;

        let embedding = self.with_thread_session(|session| {
            Self::run_inference(session, &self.input_bindings, &encoding)
        })?;
        let bits = quantize_sign_bits(&embedding);

        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= self.cache_capacity {
                cache.clear();
            }
            cache.insert(cache_key, bits);
        }

        Ok(bits)
    }

    fn infer_embeddings_batch<'a>(&self, texts: &[&'a str]) -> io::Result<Vec<[u64; VECTOR_WORDS]>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("batch tokenization failed: {error}"),
                )
            })?;

        self.with_thread_session(|session| {
            Self::run_batched_inference(session, &self.input_bindings, &encodings)
        })
    }

    fn run_inference(
        session: &mut Session,
        bindings: &InputBindings,
        encoding: &Encoding,
    ) -> io::Result<Vec<f32>> {
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| i64::from(m))
            .collect();

        Self::run_inference_from_token_ids(session, bindings, &ids, &mask)
    }

    fn run_inference_from_token_ids(
        session: &mut Session,
        bindings: &InputBindings,
        ids: &[i64],
        mask: &[i64],
    ) -> io::Result<Vec<f32>> {
        if ids.is_empty() {
            return Ok(vec![0.0; EMBEDDING_DIMS]);
        }

        let seq_len = ids.len();
        let input_ids =
            ndarray::Array2::from_shape_vec((1, seq_len), ids.to_vec()).map_err(|error| {
                io::Error::other(format!("invalid input_ids shape [1,{seq_len}]: {error}"))
            })?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((1, seq_len), mask.to_vec()).map_err(|error| {
                io::Error::other(format!(
                    "invalid attention_mask shape [1,{seq_len}]: {error}"
                ))
            })?;

        let input_ids_tensor = TensorRef::from_array_view(input_ids.view()).map_err(|error| {
            io::Error::other(format!("failed to build input_ids tensor: {error}"))
        })?;
        let attention_mask_tensor =
            TensorRef::from_array_view(attention_mask.view()).map_err(|error| {
                io::Error::other(format!("failed to build attention_mask tensor: {error}"))
            })?;

        let outputs = if let Some(token_type_name) = &bindings.token_type_ids {
            let token_type_ids = ndarray::Array2::<i64>::zeros((1, seq_len));
            let token_type_tensor =
                TensorRef::from_array_view(token_type_ids.view()).map_err(|error| {
                    io::Error::other(format!("failed to build token_type_ids tensor: {error}"))
                })?;
            session
                .run(inputs![
                    bindings.input_ids.as_str() => input_ids_tensor,
                    bindings.attention_mask.as_str() => attention_mask_tensor,
                    token_type_name.as_str() => token_type_tensor,
                ])
                .map_err(|error| io::Error::other(format!("onnx inference failed: {error}")))?
        } else {
            session
                .run(inputs![
                    bindings.input_ids.as_str() => input_ids_tensor,
                    bindings.attention_mask.as_str() => attention_mask_tensor,
                ])
                .map_err(|error| io::Error::other(format!("onnx inference failed: {error}")))?
        };

        let first_output = outputs
            .values()
            .next()
            .ok_or_else(|| io::Error::other("onnx model produced no outputs"))?;
        let (shape, data): (Vec<i64>, Vec<f32>) =
            if let Ok((shape, data)) = first_output.try_extract_tensor::<f32>() {
                (shape.to_vec(), data.to_vec())
            } else if let Ok((shape, data)) = first_output.try_extract_tensor::<i8>() {
                (
                    shape.to_vec(),
                    data.iter().map(|value| f32::from(*value)).collect(),
                )
            } else if let Ok((shape, data)) = first_output.try_extract_tensor::<u8>() {
                (
                    shape.to_vec(),
                    data.iter().map(|value| f32::from(*value)).collect(),
                )
            } else {
                return Err(io::Error::other(
                    "failed to extract output tensor as f32/i8/u8",
                ));
            };

        if shape.len() == 2 {
            let batch = shape[0] as usize;
            let hidden_size = shape[1] as usize;
            if batch == 0 || hidden_size == 0 {
                return Ok(vec![0.0; EMBEDDING_DIMS]);
            }
            return Ok(data[..hidden_size].to_vec());
        }

        if shape.len() != 3 {
            return Err(io::Error::other(format!(
                "expected [batch, hidden] or [batch, seq, hidden] tensor, got shape {:?}",
                shape
            )));
        }

        let batch = shape[0] as usize;
        let seq = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        if batch == 0 || seq == 0 || hidden_size == 0 {
            return Ok(vec![0.0; EMBEDDING_DIMS]);
        }

        let mut pooled = vec![0.0_f32; hidden_size];
        let mut token_count = 0_f32;

        for (token_index, &mask_value) in mask.iter().enumerate().take(seq) {
            if mask_value == 0 {
                continue;
            }

            token_count += 1.0;
            let token_offset = token_index * hidden_size;
            for (feature, pooled_value) in pooled.iter_mut().enumerate().take(hidden_size) {
                *pooled_value += data[token_offset + feature];
            }
        }

        if token_count > 0.0 {
            for value in &mut pooled {
                *value /= token_count;
            }
        }

        Ok(pooled)
    }

    fn run_batched_inference(
        session: &mut Session,
        bindings: &InputBindings,
        encodings: &[Encoding],
    ) -> io::Result<Vec<[u64; VECTOR_WORDS]>> {
        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(0);

        if seq_len == 0 {
            return Ok(vec![[0_u64; VECTOR_WORDS]; batch_size]);
        }

        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| {
                let row = encoding.get_ids();
                row.iter().map(|&token| i64::from(token)).chain(
                    std::iter::repeat(0_i64).take(seq_len.saturating_sub(row.len())),
                )
            })
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| {
                let row = encoding.get_attention_mask();
                row.iter().map(|&token_mask| i64::from(token_mask)).chain(
                    std::iter::repeat(0_i64).take(seq_len.saturating_sub(row.len())),
                )
            })
            .collect();

        let input_ids =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), ids).map_err(|error| {
                io::Error::other(format!(
                    "invalid input_ids shape [{batch_size},{seq_len}]: {error}"
                ))
            })?;
        let attention_mask =
            ndarray::Array2::from_shape_vec((batch_size, seq_len), mask).map_err(|error| {
                io::Error::other(format!(
                    "invalid attention_mask shape [{batch_size},{seq_len}]: {error}"
                ))
            })?;
        let mask_values = attention_mask
            .as_slice()
            .ok_or_else(|| io::Error::other("attention_mask backing storage not contiguous"))?;

        let input_ids_tensor = TensorRef::from_array_view(input_ids.view()).map_err(|error| {
            io::Error::other(format!("failed to build input_ids tensor: {error}"))
        })?;
        let attention_mask_tensor =
            TensorRef::from_array_view(attention_mask.view()).map_err(|error| {
                io::Error::other(format!("failed to build attention_mask tensor: {error}"))
            })?;

        let outputs = if let Some(token_type_name) = &bindings.token_type_ids {
            let token_type_ids = ndarray::Array2::<i64>::zeros((batch_size, seq_len));
            let token_type_tensor =
                TensorRef::from_array_view(token_type_ids.view()).map_err(|error| {
                    io::Error::other(format!("failed to build token_type_ids tensor: {error}"))
                })?;
            session
                .run(inputs![
                    bindings.input_ids.as_str() => input_ids_tensor,
                    bindings.attention_mask.as_str() => attention_mask_tensor,
                    token_type_name.as_str() => token_type_tensor,
                ])
                .map_err(|error| io::Error::other(format!("onnx inference failed: {error}")))?
        } else {
            session
                .run(inputs![
                    bindings.input_ids.as_str() => input_ids_tensor,
                    bindings.attention_mask.as_str() => attention_mask_tensor,
                ])
                .map_err(|error| io::Error::other(format!("onnx inference failed: {error}")))?
        };

        let first_output = outputs
            .values()
            .next()
            .ok_or_else(|| io::Error::other("onnx model produced no outputs"))?;
        let (shape, data): (Vec<i64>, Vec<f32>) =
            if let Ok((shape, data)) = first_output.try_extract_tensor::<f32>() {
                (shape.to_vec(), data.to_vec())
            } else if let Ok((shape, data)) = first_output.try_extract_tensor::<i8>() {
                (
                    shape.to_vec(),
                    data.iter().map(|value| f32::from(*value)).collect(),
                )
            } else if let Ok((shape, data)) = first_output.try_extract_tensor::<u8>() {
                (
                    shape.to_vec(),
                    data.iter().map(|value| f32::from(*value)).collect(),
                )
            } else {
                return Err(io::Error::other(
                    "failed to extract output tensor as f32/i8/u8",
                ));
            };

        if shape.is_empty() {
            return Err(io::Error::other(
                "onnx model produced an empty output shape",
            ));
        }

        let output_batch = shape[0] as usize;
        if output_batch != batch_size {
            return Err(io::Error::other(format!(
                "expected output batch {batch_size}, got {output_batch}"
            )));
        }

        if shape.len() == 2 {
            let hidden_size = shape[1] as usize;
            if hidden_size == 0 {
                return Ok(vec![[0_u64; VECTOR_WORDS]; batch_size]);
            }

            let mut bits_batch = Vec::with_capacity(batch_size);
            for row in 0..batch_size {
                let start = row * hidden_size;
                bits_batch.push(quantize_sign_bits_row(&data, start, hidden_size));
            }
            return Ok(bits_batch);
        }

        if shape.len() != 3 {
            return Err(io::Error::other(format!(
                "expected [batch, hidden] or [batch, seq, hidden] tensor, got shape {:?}",
                shape
            )));
        }

        let output_seq = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        if output_seq != seq_len {
            return Err(io::Error::other(format!(
                "expected output sequence length {seq_len}, got {output_seq}"
            )));
        }
        if hidden_size == 0 {
            return Ok(vec![[0_u64; VECTOR_WORDS]; batch_size]);
        }

        let mut bits_batch = Vec::with_capacity(batch_size);
        let mut pooled_sums = vec![0.0_f32; hidden_size];

        for row in 0..batch_size {
            pooled_sums.fill(0.0);
            let mut has_token = false;
            let row_offset = row * seq_len;

            for token_index in 0..seq_len {
                if mask_values[row_offset + token_index] == 0 {
                    continue;
                }

                has_token = true;
                let embedding_offset = (row_offset + token_index) * hidden_size;
                for (feature, pooled_value) in pooled_sums.iter_mut().enumerate() {
                    *pooled_value += data[embedding_offset + feature];
                }
            }

            if has_token {
                bits_batch.push(quantize_sign_bits(&pooled_sums));
            } else {
                bits_batch.push([0_u64; VECTOR_WORDS]);
            }
        }

        Ok(bits_batch)
    }
}

impl EmbeddingGenerator for OnnxEncoder {
    fn generate(&self, text: &str) -> [u64; VECTOR_WORDS] {
        let cache_key = simple_hash(text);
        if let Ok(cache) = self.cache.lock() {
            if let Some(bits) = cache.get(&cache_key) {
                return *bits;
            }
        }

        let bits = match self.infer_embeddings_batch(&[text]) {
            Ok(mut batch_bits) => batch_bits.pop().unwrap_or([0_u64; VECTOR_WORDS]),
            Err(error) => {
                if !self.warned_once.swap(true, Ordering::Relaxed) {
                    eprintln!("warning: ONNX inference unavailable, falling back to mock encoder: {error}");
                }
                mock_binary_embedding(text.as_bytes())
            }
        };

        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= self.cache_capacity {
                cache.clear();
            }
            cache.insert(cache_key, bits);
        }

        bits
    }
}

/// Location metadata for one indexed line in the source file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceRef {
    /// One-based line number in the mapped file.
    pub line_number: u64,
    /// Byte offset from the start of the file.
    pub byte_offset: usize,
    /// Length of the line payload after trimming a trailing carriage return.
    pub byte_len: usize,
}

/// A 512-bit embedding together with the line it was derived from.
#[derive(Clone, Copy, Debug)]
pub struct VectorChunk {
    /// Packed binary embedding stored as eight machine words.
    pub bits: [u64; VECTOR_WORDS],
    /// Original source location for result reporting.
    pub source: SourceRef,
}

/// Result entry returned by the line-oriented top-k scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchHit {
    /// Hamming distance between the query vector and this chunk.
    pub distance: u32,
    /// Source line associated with the hit.
    pub source: SourceRef,
}

/// End-to-end timing and throughput counters for a full file scan.
#[derive(Debug)]
pub struct ScanStats {
    /// Number of bytes read through the memory map.
    pub bytes_scanned: usize,
    /// Number of newline-delimited vectors materialized from the file.
    pub vector_count: usize,
    /// Number of vectors that reached distance evaluation after pre-filtering.
    pub candidate_count: usize,
    /// Wall-clock duration for mmap, chunking, and search combined.
    pub total_latency: Duration,
    /// Time spent splitting lines and generating embeddings.
    pub chunk_latency: Duration,
    /// Time spent in the Hamming scan and top-k reduction.
    pub search_latency: Duration,
}

/// Complete output of an end-to-end scan.
#[derive(Debug)]
pub struct ScanResult {
    /// Top-k hits sorted by ascending Hamming distance.
    pub hits: Vec<SearchHit>,
    /// Timing counters for the scan that produced `hits`.
    pub stats: ScanStats,
}

/// Top-k result for a raw vector scan where only the embedding index is known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedSearchHit {
    /// Index into the input slice passed to [`search_top_k_bits`].
    pub index: usize,
    /// Hamming distance between the query and the indexed vector.
    pub distance: u32,
}

/// Memory-maps `path`, chunks it into newline-delimited records, and scans the resulting
/// embeddings against `query`.
///
/// `top_k` controls how many nearest neighbors are retained per scan. The function is
/// generic over `EmbeddingGenerator` so model-backed encoders can be introduced without
/// disturbing the parallel scan architecture.
pub fn scan_file_with_encoder<E: EmbeddingGenerator>(
    path: &Path,
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &E,
) -> io::Result<ScanResult> {
    let total_start = Instant::now();
    let mmap = map_file_read_only(path)?;

    Ok(scan_bytes_with_encoder_from(
        &mmap,
        query,
        top_k,
        max_distance,
        encoder,
        total_start,
    ))
}

/// Scans newline-delimited `bytes` directly without going through the filesystem.
///
/// This path exists for stdin and other streaming sources that cannot be memory-mapped.
/// The caller owns the backing buffer and can decide whether it came from a file, a pipe,
/// or any other buffered reader.
pub fn scan_bytes_with_encoder<E: EmbeddingGenerator>(
    bytes: &[u8],
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &E,
) -> ScanResult {
    let total_start = Instant::now();
    scan_bytes_with_encoder_from(bytes, query, top_k, max_distance, encoder, total_start)
}

pub fn scan_file_with_onnx_batches(
    path: &Path,
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &OnnxEncoder,
    batch_size: usize,
    filter: Option<&str>,
) -> io::Result<ScanResult> {
    let total_start = Instant::now();
    let mmap = map_file_read_only(path)?;

    if let Some(index) = load_vector_index(path)? {
        return Ok(scan_cached_vectors_from(
            &mmap,
            query,
            top_k,
            max_distance,
            encoder,
            filter,
            &index,
            total_start,
        ));
    }

    if let Ok(mut index) = build_vector_index_from_bytes(&mmap, encoder, batch_size) {
        index.file_size = mmap.len() as u64;
        index.last_modified = path_modified_nanos(path)?;
        let _ = save_vector_index(path, &index);
        return Ok(scan_cached_vectors_from(
            &mmap,
            query,
            top_k,
            max_distance,
            encoder,
            filter,
            &index,
            total_start,
        ));
    }

    Ok(scan_bytes_with_onnx_batches_from(
        &mmap,
        query,
        top_k,
        max_distance,
        encoder,
        batch_size,
        filter,
        total_start,
    ))
}

pub fn scan_bytes_with_onnx_batches(
    bytes: &[u8],
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &OnnxEncoder,
    batch_size: usize,
    filter: Option<&str>,
) -> ScanResult {
    let total_start = Instant::now();
    scan_bytes_with_onnx_batches_from(
        bytes,
        query,
        top_k,
        max_distance,
        encoder,
        batch_size,
        filter,
        total_start,
    )
}

/// Applies a semantic distance threshold to `result.hits`.
///
/// Lower Hamming distance means a tighter semantic match. When `max_distance` is set,
/// any hit with `distance > max_distance` is removed from the result set.
pub fn apply_max_distance_filter(result: &mut ScanResult, max_distance: Option<u32>) {
    if let Some(limit) = max_distance {
        result.hits.retain(|hit| hit.distance <= limit);
    }
}

fn scan_bytes_with_encoder_from<E: EmbeddingGenerator>(
    bytes: &[u8],
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &E,
    total_start: Instant,
) -> ScanResult {
    if top_k == 0 {
        return ScanResult {
            hits: Vec::new(),
            stats: ScanStats {
                bytes_scanned: bytes.len(),
                vector_count: 0,
                candidate_count: 0,
                total_latency: total_start.elapsed(),
                chunk_latency: Duration::ZERO,
                search_latency: Duration::ZERO,
            },
        };
    }

    let query_bits = encoder.generate(query);
    let line_sources = collect_line_sources(bytes);
    let search_start = Instant::now();
    let vector_count = line_sources.len();
    let (hits, candidate_count) = line_sources
        .par_chunks(SEARCH_CHUNK_SIZE)
        .fold(
            || (Vec::with_capacity(top_k), 0usize),
            |(mut local_hits, mut local_candidates), chunk| {
                // Precompute candidate embeddings in a separate pass so the hottest
                // comparison loop only executes Hamming distance plus thresholding.
                let texts: Vec<String> = chunk
                    .iter()
                    .map(|source| line_as_str_lossy(line_bytes(bytes, *source)))
                    .collect();
                let candidate_bits: Vec<[u64; VECTOR_WORDS]> =
                    texts.iter().map(|text| encoder.generate(text)).collect();

                local_candidates += chunk.len();
                for (source, bits) in chunk.iter().copied().zip(candidate_bits.into_iter()) {
                    let distance = hamming_distance(&bits, &query_bits);
                    if max_distance.is_none_or(|limit| distance <= limit) {
                        insert_top_hit(&mut local_hits, SearchHit { distance, source }, top_k);
                    }
                }

                (local_hits, local_candidates)
            },
        )
        .reduce(
            || (Vec::with_capacity(top_k), 0usize),
            |(mut left_hits, left_count), (right_hits, right_count)| {
                for hit in right_hits {
                    insert_top_hit(&mut left_hits, hit, top_k);
                }
                (left_hits, left_count + right_count)
            },
        );

    let search_latency = search_start.elapsed();
    let total_latency = total_start.elapsed();

    ScanResult {
        hits,
        stats: ScanStats {
            bytes_scanned: bytes.len(),
            vector_count,
            candidate_count,
            total_latency,
            chunk_latency: Duration::ZERO,
            search_latency,
        },
    }
}

fn scan_bytes_with_onnx_batches_from(
    bytes: &[u8],
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &OnnxEncoder,
    batch_size: usize,
    filter: Option<&str>,
    total_start: Instant,
) -> ScanResult {
    if top_k == 0 {
        return ScanResult {
            hits: Vec::new(),
            stats: ScanStats {
                bytes_scanned: bytes.len(),
                vector_count: 0,
                candidate_count: 0,
                total_latency: total_start.elapsed(),
                chunk_latency: Duration::ZERO,
                search_latency: Duration::ZERO,
            },
        };
    }

    if let Ok(index) = build_vector_index_from_bytes(bytes, encoder, batch_size) {
        return scan_cached_vectors_from(
            bytes,
            query,
            top_k,
            max_distance,
            encoder,
            filter,
            &index,
            total_start,
        );
    }

    let query_bits = encoder.generate(query);
    let batch_size = batch_size.max(1);
    let line_sources = collect_line_sources(bytes);
    let search_start = Instant::now();
    let matcher = filter.map(|pattern| {
        aho_corasick::AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build([pattern])
            .expect("Failed to build AhoCorasick matcher")
    });

    let vector_count = line_sources.len();

    let (hits, candidate_count) = line_sources
        .chunks(batch_size)
        .fold(
            (Vec::with_capacity(top_k), 0usize),
            |(mut local_hits, mut local_candidates), batch| {
                // Pass 1: filter on bytes to avoid UTF-8 conversion for skipped lines.
                let mut candidates: Vec<SourceRef> = Vec::with_capacity(batch.len());
                if let Some(m) = matcher.as_ref() {
                    for source in batch {
                        let line = line_bytes(bytes, *source);
                        if m.is_match(line) {
                            candidates.push(*source);
                        }
                    }
                } else {
                    candidates.extend_from_slice(batch);
                }

                if candidates.is_empty() {
                    return (local_hits, local_candidates);
                }

                // Pass 2: encode only filtered candidates.
                local_candidates += candidates.len();
                let texts: Vec<String> = candidates
                    .iter()
                    .map(|source| line_as_str_lossy(line_bytes(bytes, *source)))
                    .collect();
                let text_refs: Vec<&str> = texts.iter().map(|text| text.as_str()).collect();
                let bits_batch = encoder.batch_encode(&text_refs);

                for (source, bits) in candidates.into_iter().zip(bits_batch.into_iter()) {
                    let distance = hamming_distance(&bits, &query_bits);
                    if max_distance.is_none_or(|limit| distance <= limit) {
                        insert_top_hit(
                            &mut local_hits,
                            SearchHit { distance, source },
                            top_k,
                        );
                    }
                }
                (local_hits, local_candidates)
            },
        );

    ScanResult {
        hits,
        stats: ScanStats {
            bytes_scanned: bytes.len(),
            vector_count,
            candidate_count,
            total_latency: total_start.elapsed(),
            chunk_latency: Duration::ZERO,
            search_latency: search_start.elapsed(),
        },
    }
}

/// Converts a mapped byte slice into `VectorChunk` records using newline boundaries.
///
/// The chunker uses `memchr` so boundary detection stays close to libc-grade byte search
/// performance. Each line is encoded independently, which keeps the index stateless and
/// avoids cross-record coupling.
pub fn chunk_vectors_with_encoder<E: EmbeddingGenerator>(
    bytes: &[u8],
    encoder: &E,
) -> Vec<VectorChunk> {
    let line_count = memchr_iter(b'\n', bytes).count();
    let has_trailing_line = matches!(bytes.last(), Some(last) if *last != b'\n');
    let mut chunks = Vec::with_capacity(line_count + usize::from(has_trailing_line));
    let mut line_start = 0usize;
    let mut line_number = 0u64;

    for line_end in memchr_iter(b'\n', bytes) {
        line_number += 1;
        let line = trim_carriage_return(&bytes[line_start..line_end]);
        if is_corrupted_line(line) {
            line_start = line_end + 1;
            continue;
        }
        chunks.push(VectorChunk {
            bits: encoder.generate(&line_as_str_lossy(line)),
            source: SourceRef {
                line_number,
                byte_offset: line_start,
                byte_len: line.len(),
            },
        });
        line_start = line_end + 1;
    }

    if line_start < bytes.len() {
        line_number += 1;
        let line = trim_carriage_return(&bytes[line_start..]);
        if !is_corrupted_line(line) {
            chunks.push(VectorChunk {
                bits: encoder.generate(&line_as_str_lossy(line)),
                source: SourceRef {
                    line_number,
                    byte_offset: line_start,
                    byte_len: line.len(),
                },
            });
        }
    }

    chunks
}

/// Returns the closest `top_k` line-backed vectors to `query_bits`.
///
/// This variant retains source metadata for CLI reporting. It parallelizes across chunks
/// and performs a local top-k reduction per worker before merging, which limits contention
/// and keeps memory traffic predictable.
pub fn search_top_k(
    chunks: &[VectorChunk],
    query_bits: &[u64; VECTOR_WORDS],
    top_k: usize,
) -> Vec<SearchHit> {
    if chunks.is_empty() || top_k == 0 {
        return Vec::new();
    }

    chunks
        .par_iter()
        .fold(
            || Vec::with_capacity(top_k),
            |mut local_hits, chunk| {
                // XOR marks every differing bit between the query and candidate.
                // `count_ones()` then counts those set bits per 64-bit lane, which LLVM
                // lowers to the CPU's POPCNT instruction on supported targets. Summed over
                // eight lanes, this yields the 512-bit Hamming distance with no branching.
                let hit = SearchHit {
                    distance: hamming_distance(&chunk.bits, query_bits),
                    source: chunk.source,
                };
                insert_top_hit(&mut local_hits, hit, top_k);
                local_hits
            },
        )
        .reduce(
            || Vec::with_capacity(top_k),
            |mut left, right| {
                for hit in right {
                    insert_top_hit(&mut left, hit, top_k);
                }
                left
            },
        )
}

/// Returns the closest `top_k` vectors when the caller already owns a flat slice of packed
/// embeddings.
///
/// This is the core micro-benchmark target because it isolates the hot Hamming loop from
/// mmap setup and line chunking overhead.
pub fn search_top_k_bits(
    vectors: &[[u64; VECTOR_WORDS]],
    query_bits: &[u64; VECTOR_WORDS],
    top_k: usize,
) -> Vec<IndexedSearchHit> {
    if vectors.is_empty() || top_k == 0 {
        return Vec::new();
    }

    vectors
        .par_iter()
        .enumerate()
        .fold(
            || Vec::with_capacity(top_k),
            |mut local_hits, (index, bits)| {
                // This is the same scan kernel as `search_top_k`, minus source metadata.
                // The work remains compute-dense: XOR to surface differing bits, POPCNT to
                // count them, then a tiny ordered top-k buffer per worker thread.
                let hit = IndexedSearchHit {
                    index,
                    distance: hamming_distance(bits, query_bits),
                };
                insert_indexed_top_hit(&mut local_hits, hit, top_k);
                local_hits
            },
        )
        .reduce(
            || Vec::with_capacity(top_k),
            |mut left, right| {
                for hit in right {
                    insert_indexed_top_hit(&mut left, hit, top_k);
                }
                left
            },
        )
}

/// Computes the Hamming distance between two packed 512-bit vectors.
///
/// The implementation is intentionally minimal so the compiler can inline it into the scan
/// loop and emit efficient XOR plus population-count instructions.
pub fn hamming_distance(lhs: &[u64; VECTOR_WORDS], rhs: &[u64; VECTOR_WORDS]) -> u32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

/// Writes hits in a grep-like `line_number:content` format.
///
/// `bytes` must be the same corpus used to produce `hits`, because each hit stores only
/// offsets into the original input. Output is written to `writer` so callers can direct
/// match streams to stdout while reserving stderr for diagnostics.
pub fn write_hits<W: Write>(writer: &mut W, bytes: &[u8], hits: &[SearchHit]) -> io::Result<()> {
    if hits.is_empty() {
        return Ok(());
    }

    for hit in hits {
        let line = line_bytes(bytes, hit.source);
        writeln!(
            writer,
            "{}:{}",
            hit.source.line_number,
            String::from_utf8_lossy(line),
        )?;
    }

    Ok(())
}

/// Writes end-to-end scan metrics for the CLI `--bench` mode.
///
/// This reports wall-clock timings that include file mapping and chunk construction. For
/// isolated regression tracking of the Hamming loop itself, use the Criterion benchmark.
pub fn write_benchmark<W: Write>(writer: &mut W, stats: &ScanStats) -> io::Result<()> {
    let total_seconds = stats.total_latency.as_secs_f64().max(f64::MIN_POSITIVE);
    let throughput_mb_s = (stats.bytes_scanned as f64 / 1_000_000.0) / total_seconds;
    let vectors_per_second = stats.vector_count as f64 / total_seconds;
    let selectivity = if stats.vector_count == 0 {
        0.0
    } else {
        stats.candidate_count as f64 / stats.vector_count as f64
    };

    writeln!(writer, "benchmark:")?;
    writeln!(
        writer,
        "  bytes_scanned={} ({:.2} MiB)",
        stats.bytes_scanned,
        bytes_to_mib(stats.bytes_scanned)
    )?;
    writeln!(writer, "  vectors_scanned={}", stats.vector_count)?;
    writeln!(
        writer,
        "  total_latency_ms={:.3}",
        stats.total_latency.as_secs_f64() * 1_000.0
    )?;
    writeln!(
        writer,
        "  chunk_latency_ms={:.3}",
        stats.chunk_latency.as_secs_f64() * 1_000.0
    )?;
    writeln!(
        writer,
        "  search_latency_ms={:.3}",
        stats.search_latency.as_secs_f64() * 1_000.0
    )?;
    writeln!(writer, "  vectors_after_filter={}", stats.candidate_count)?;
    writeln!(writer, "  filter_selectivity_pct={:.2}", selectivity * 100.0)?;
    writeln!(writer, "  throughput_mb_s={throughput_mb_s:.2}")?;
    writeln!(writer, "  vectors_per_second={vectors_per_second:.0}")?;
    if stats.vector_count > 0 && selectivity > 0.30 {
        writeln!(
            writer,
            "  warning=high filter selectivity ({:.2}%); ONNX bypass impact is limited",
            selectivity * 100.0
        )?;
    }
    Ok(())
}

struct VectorIndex {
    file_size: u64,
    last_modified: u128,
    storage: VectorIndexStorage,
}

enum VectorIndexStorage {
    Owned {
        vectors: Vec<[u64; VECTOR_WORDS]>,
        sources: Vec<SourceRef>,
    },
    Mapped {
        mmap: Mmap,
        entry_count: usize,
    },
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VectorIndexEntryDisk {
    line_number: u64,
    byte_offset: u64,
    byte_len: u64,
    bits: [u64; VECTOR_WORDS],
}

impl VectorIndexEntryDisk {
    #[inline]
    fn source(self) -> SourceRef {
        SourceRef {
            line_number: u64::from_le(self.line_number),
            byte_offset: u64::from_le(self.byte_offset) as usize,
            byte_len: u64::from_le(self.byte_len) as usize,
        }
    }

    #[cfg(not(target_endian = "little"))]
    #[inline]
    fn bits(self) -> [u64; VECTOR_WORDS] {
        self.bits.map(u64::from_le)
    }
}

impl VectorIndex {
    #[inline]
    fn len(&self) -> usize {
        match &self.storage {
            VectorIndexStorage::Owned { vectors, .. } => vectors.len(),
            VectorIndexStorage::Mapped { entry_count, .. } => *entry_count,
        }
    }

    fn mapped_entries(&self) -> Option<&[VectorIndexEntryDisk]> {
        match &self.storage {
            VectorIndexStorage::Owned { .. } => None,
            VectorIndexStorage::Mapped { mmap, entry_count } => {
                let ptr = unsafe { mmap.as_ptr().add(INDEX_HEADER_BYTES).cast::<VectorIndexEntryDisk>() };
                Some(unsafe { std::slice::from_raw_parts(ptr, *entry_count) })
            }
        }
    }
}

fn index_path_for_input(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".vrepidx");
    PathBuf::from(os)
}

fn path_modified_nanos(path: &Path) -> io::Result<u128> {
    let modified = std::fs::metadata(path)?.modified()?;
    match modified.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => Ok(duration.as_nanos()),
        Err(_) => Ok(0),
    }
}

fn build_vector_index_from_bytes(
    bytes: &[u8],
    encoder: &OnnxEncoder,
    batch_size: usize,
) -> io::Result<VectorIndex> {
    let sources = collect_line_sources(bytes);
    let batch_size = batch_size.max(1);

    let parts: Vec<Vec<(SourceRef, [u64; VECTOR_WORDS])>> = sources
        .par_chunks(batch_size)
        .map(|batch| {
            let texts: Vec<String> = batch
                .iter()
                .map(|source| line_as_str_lossy(line_bytes(bytes, *source)))
                .collect();
            let bits_batch = encoder.batch_encode_owned(texts);
            let mut encoded = Vec::with_capacity(batch.len());
            for (source, maybe_bits) in batch.iter().copied().zip(bits_batch.into_iter()) {
                if let Some(bits) = maybe_bits {
                    encoded.push((source, bits));
                }
            }
            encoded
        })
        .collect();

    let mut flattened = Vec::with_capacity(sources.len());
    for mut part in parts {
        flattened.append(&mut part);
    }
    flattened.sort_by_key(|(source, _)| source.byte_offset);
    let (sources, vectors): (Vec<SourceRef>, Vec<[u64; VECTOR_WORDS]>) =
        flattened.into_iter().unzip();

    Ok(VectorIndex {
        file_size: bytes.len() as u64,
        last_modified: 0,
        storage: VectorIndexStorage::Owned { vectors, sources },
    })
}

fn save_vector_index(path: &Path, index: &VectorIndex) -> io::Result<()> {
    let index_path = index_path_for_input(path);
    let tmp_path = PathBuf::from(format!("{}.tmp", index_path.display()));
    let file = File::create(&tmp_path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(INDEX_MAGIC)?;
    writer.write_all(&index.file_size.to_le_bytes())?;
    writer.write_all(&index.last_modified.to_le_bytes())?;
    writer.write_all(&(index.len() as u64).to_le_bytes())?;

    match &index.storage {
        VectorIndexStorage::Owned { vectors, sources } => {
            for (source, bits) in sources.iter().zip(vectors.iter()) {
                writer.write_all(&source.line_number.to_le_bytes())?;
                writer.write_all(&(source.byte_offset as u64).to_le_bytes())?;
                writer.write_all(&(source.byte_len as u64).to_le_bytes())?;
                for lane in bits {
                    writer.write_all(&lane.to_le_bytes())?;
                }
            }
        }
        VectorIndexStorage::Mapped { mmap, .. } => {
            writer.write_all(&mmap[INDEX_HEADER_BYTES..])?;
        }
    }

    writer.flush()?;
    writer.get_ref().sync_all()?;
    std::fs::rename(&tmp_path, &index_path)?;
    Ok(())
}

fn load_vector_index(path: &Path) -> io::Result<Option<VectorIndex>> {
    let index_path = index_path_for_input(path);
    if !index_path.exists() {
        return Ok(None);
    }

    let file_size = std::fs::metadata(path)?.len();
    let last_modified = path_modified_nanos(path)?;
    let index_len = std::fs::metadata(&index_path)?.len() as usize;
    if index_len < INDEX_HEADER_BYTES {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }

    let mut header = [0_u8; INDEX_HEADER_BYTES];
    let mut reader = File::open(&index_path)?;
    if reader.read_exact(&mut header).is_err() {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }
    if &header[..INDEX_MAGIC.len()] != INDEX_MAGIC {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }

    let entry_count = u64::from_le_bytes(header[32..40].try_into().unwrap()) as usize;
    let entry_bytes = size_of::<VectorIndexEntryDisk>();
    let expected_len = INDEX_HEADER_BYTES
        .checked_add(entry_count.checked_mul(entry_bytes).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "vector index entry count overflow")
        })?)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "vector index length overflow"))?;

    // Reject truncated or partially written index files before memory mapping.
    if index_len != expected_len {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }

    let mmap = map_file_read_only(&index_path)?;
    let bytes = mmap.as_ref();
    if bytes.len() != expected_len || &bytes[..INDEX_MAGIC.len()] != INDEX_MAGIC {
        return Ok(None);
    }

    let indexed_size = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let indexed_last_modified = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
    if indexed_size != file_size || indexed_last_modified != last_modified {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }

    let entry_region = &bytes[INDEX_HEADER_BYTES..];
    let (prefix, entries, suffix) = unsafe { entry_region.align_to::<VectorIndexEntryDisk>() };
    if !prefix.is_empty() || !suffix.is_empty() || entries.len() != entry_count {
        let _ = std::fs::remove_file(&index_path);
        return Ok(None);
    }

    Ok(Some(VectorIndex {
        file_size,
        last_modified,
        storage: VectorIndexStorage::Mapped { mmap, entry_count },
    }))
}

fn scan_cached_vectors_from(
    bytes: &[u8],
    query: &str,
    top_k: usize,
    max_distance: Option<u32>,
    encoder: &OnnxEncoder,
    filter: Option<&str>,
    index: &VectorIndex,
    total_start: Instant,
) -> ScanResult {
    if top_k == 0 {
        return ScanResult {
            hits: Vec::new(),
            stats: ScanStats {
                bytes_scanned: bytes.len(),
                vector_count: 0,
                candidate_count: 0,
                total_latency: total_start.elapsed(),
                chunk_latency: Duration::ZERO,
                search_latency: Duration::ZERO,
            },
        };
    }

    let query_bits = encoder.generate(query);
    let matcher = filter.map(|pattern| {
        aho_corasick::AhoCorasickBuilder::new()
            .ascii_case_insensitive(true)
            .build([pattern])
            .expect("Failed to build AhoCorasick matcher")
    });

    let search_start = Instant::now();
    let (hits, candidate_count) = match &index.storage {
        VectorIndexStorage::Owned { vectors, sources } => vectors
            .par_chunks(SEARCH_CHUNK_SIZE)
            .zip(sources.par_chunks(SEARCH_CHUNK_SIZE))
            .fold(
                || (Vec::with_capacity(top_k), 0usize),
                |(mut local_hits, mut local_candidates), (bits_chunk, sources_chunk)| {
                    for (bits, source) in bits_chunk.iter().zip(sources_chunk.iter()) {
                        if !source_is_in_bounds(bytes.len(), *source) {
                            continue;
                        }
                        if let Some(m) = matcher.as_ref() {
                            let line = line_bytes(bytes, *source);
                            if !m.is_match(line) {
                                continue;
                            }
                        }

                        local_candidates += 1;
                        let distance = hamming_distance(bits, &query_bits);
                        if max_distance.is_none_or(|limit| distance <= limit) {
                            insert_top_hit(
                                &mut local_hits,
                                SearchHit {
                                    distance,
                                    source: *source,
                                },
                                top_k,
                            );
                        }
                    }
                    (local_hits, local_candidates)
                },
            )
            .reduce(
                || (Vec::with_capacity(top_k), 0usize),
                |(mut left_hits, left_candidates), (right_hits, right_candidates)| {
                    for hit in right_hits {
                        insert_top_hit(&mut left_hits, hit, top_k);
                    }
                    (left_hits, left_candidates + right_candidates)
                },
            ),
        VectorIndexStorage::Mapped { .. } => index
            .mapped_entries()
            .unwrap_or(&[])
            .par_chunks(SEARCH_CHUNK_SIZE)
            .fold(
                || (Vec::with_capacity(top_k), 0usize),
                |(mut local_hits, mut local_candidates), entries| {
                    for entry in entries {
                        let source = entry.source();
                        if !source_is_in_bounds(bytes.len(), source) {
                            continue;
                        }
                        if let Some(m) = matcher.as_ref() {
                            let line = line_bytes(bytes, source);
                            if !m.is_match(line) {
                                continue;
                            }
                        }

                        local_candidates += 1;
                        #[cfg(target_endian = "little")]
                        let distance = hamming_distance(&entry.bits, &query_bits);
                        #[cfg(not(target_endian = "little"))]
                        let distance = {
                            let bits = entry.bits();
                            hamming_distance(&bits, &query_bits)
                        };
                        if max_distance.is_none_or(|limit| distance <= limit) {
                            insert_top_hit(&mut local_hits, SearchHit { distance, source }, top_k);
                        }
                    }
                    (local_hits, local_candidates)
                },
            )
            .reduce(
                || (Vec::with_capacity(top_k), 0usize),
                |(mut left_hits, left_candidates), (right_hits, right_candidates)| {
                    for hit in right_hits {
                        insert_top_hit(&mut left_hits, hit, top_k);
                    }
                    (left_hits, left_candidates + right_candidates)
                },
            ),
    };

    ScanResult {
        hits,
        stats: ScanStats {
            bytes_scanned: bytes.len(),
            vector_count: index.len(),
            candidate_count,
            total_latency: total_start.elapsed(),
            chunk_latency: Duration::ZERO,
            search_latency: search_start.elapsed(),
        },
    }
}

/// Creates a newline-delimited dummy corpus of approximately `size_gib` gibibytes.
///
/// The file is overwritten if it already exists. This helper exists to make end-to-end
/// throughput testing repeatable without requiring a real embedding corpus.
pub fn create_dummy_file(path: &Path, size_gib: u64) -> io::Result<()> {
    let target_size = size_gib.saturating_mul(BYTES_PER_GIB);
    let file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut written = 0u64;

    while written < target_size {
        let remaining = (target_size - written) as usize;
        if remaining >= DUMMY_LINE.len() {
            writer.write_all(DUMMY_LINE)?;
            written += DUMMY_LINE.len() as u64;
            continue;
        }

        let slice_len = min(
            remaining.saturating_sub(1),
            DUMMY_LINE.len().saturating_sub(1),
        );
        if slice_len > 0 {
            writer.write_all(&DUMMY_LINE[..slice_len])?;
            written += slice_len as u64;
        }
        writer.write_all(b"\n")?;
        written += 1;
    }

    writer.flush()?;
    Ok(())
}

/// Creates a read-only memory map for `path`.
///
/// This is the preferred file-input path for the CLI because it avoids copying the corpus
/// into a separate heap buffer and lets the operating system page data in on demand.
pub fn map_file_read_only(path: &Path) -> io::Result<Mmap> {
    let file = File::open(path)?;
    // SAFETY: The mapping is created read-only from an open file descriptor and returned as
    // `Mmap`, which owns the mapping lifetime independently of `file`. We never mutate the
    // mapped bytes, and all later access stays within the slice bounds reported by memmap2.
    unsafe { MmapOptions::new().map(&file) }
}

fn line_bytes(bytes: &[u8], source: SourceRef) -> &[u8] {
    let start = source.byte_offset.min(bytes.len());
    let end = source
        .byte_offset
        .saturating_add(source.byte_len)
        .min(bytes.len());
    &bytes[start..end]
}

fn collect_line_sources(bytes: &[u8]) -> Vec<SourceRef> {
    let line_count = memchr_iter(b'\n', bytes).count();
    let has_trailing_line = matches!(bytes.last(), Some(last) if *last != b'\n');
    let mut sources = Vec::with_capacity(line_count + usize::from(has_trailing_line));
    let mut line_start = 0usize;
    let mut line_number = 0u64;

    for line_end in memchr_iter(b'\n', bytes) {
        line_number += 1;
        let line = trim_carriage_return(&bytes[line_start..line_end]);
        if is_corrupted_line(line) {
            line_start = line_end + 1;
            continue;
        }
        sources.push(SourceRef {
            line_number,
            byte_offset: line_start,
            byte_len: line.len(),
        });
        line_start = line_end + 1;
    }

    if line_start < bytes.len() {
        line_number += 1;
        let line = trim_carriage_return(&bytes[line_start..]);
        if !is_corrupted_line(line) {
            sources.push(SourceRef {
                line_number,
                byte_offset: line_start,
                byte_len: line.len(),
            });
        }
    }

    sources
}

fn line_as_str_lossy(bytes: &[u8]) -> String {
    let filtered = sanitize_for_embedding(bytes);
    String::from_utf8_lossy(filtered.as_ref()).into_owned()
}

fn cheap_bypass_bits(text: &str) -> Option<[u64; VECTOR_WORDS]> {
    if text.is_empty() {
        return Some([0_u64; VECTOR_WORDS]);
    }

    if text.len() < CHEAP_BYPASS_MIN_LEN {
        return Some(mock_binary_embedding(text.as_bytes()));
    }

    if !semantic_content_regex().is_match(text) {
        return Some(mock_binary_embedding(text.as_bytes()));
    }

    None
}

fn semantic_content_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?i)[a-z]{3,}").expect("semantic regex must compile"))
}

fn sanitize_for_embedding(bytes: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if bytes.iter().all(|&byte| is_allowed_embedding_byte(byte)) {
        return std::borrow::Cow::Borrowed(bytes);
    }

    let filtered: Vec<u8> = bytes
        .iter()
        .copied()
        .filter(|&byte| is_allowed_embedding_byte(byte))
        .collect();
    std::borrow::Cow::Owned(filtered)
}

fn is_allowed_embedding_byte(byte: u8) -> bool {
    matches!(byte, b'\t' | b'\r' | b' '..=b'~')
}

fn is_corrupted_line(line: &[u8]) -> bool {
    line.contains(&0)
}

fn source_is_in_bounds(bytes_len: usize, source: SourceRef) -> bool {
    source.byte_offset <= bytes_len && source.byte_len <= bytes_len.saturating_sub(source.byte_offset)
}

fn quantize_sign_bits(embedding: &[f32]) -> [u64; VECTOR_WORDS] {
    quantize_sign_bits_row(embedding, 0, embedding.len())
}

fn quantize_sign_bits_row(values: &[f32], start: usize, hidden_size: usize) -> [u64; VECTOR_WORDS] {
    let mut bits = [0_u64; VECTOR_WORDS];
    let dims = hidden_size.min(EMBEDDING_DIMS);

    for index in 0..dims {
        if values[start + index] > 0.0 {
            let lane = index / 64;
            let bit = index % 64;
            bits[lane] |= 1_u64 << bit;
        }
    }

    bits
}

fn simple_hash(text: &str) -> u64 {
    let mut hash = 0x9E37_79B9_7F4A_7C15_u64;
    for &byte in text.as_bytes() {
        hash ^= u64::from(byte);
        hash = hash.rotate_left(5).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    }
    hash ^ (text.len() as u64).rotate_left(17)
}

fn mock_binary_embedding(bytes: &[u8]) -> [u64; VECTOR_WORDS] {
    let mut lanes = [
        0x243F_6A88_85A3_08D3_u64,
        0x1319_8A2E_0370_7344_u64,
        0xA409_3822_299F_31D0_u64,
        0x082E_FA98_EC4E_6C89_u64,
        0x4528_21E6_38D0_1377_u64,
        0xBE54_66CF_34E9_0C6C_u64,
        0xC0AC_29B7_C97C_50DD_u64,
        0x3F84_D5B5_B547_0917_u64,
    ];

    for (index, &byte) in bytes.iter().enumerate() {
        let lane = index & (VECTOR_WORDS - 1);
        let shift = ((index >> 3) & 7) * 8;
        let mixed = (u64::from(byte) << shift)
            ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (bytes.len() as u64).rotate_left((lane as u32) + 7);
        lanes[lane] = lanes[lane].rotate_left(5) ^ mixed;
        lanes[(lane + 3) & (VECTOR_WORDS - 1)] = lanes[(lane + 3) & (VECTOR_WORDS - 1)]
            .wrapping_add(mixed.rotate_left(17))
            .rotate_left(11);
    }

    for (index, lane) in lanes.iter_mut().enumerate() {
        *lane ^= (bytes.len() as u64).wrapping_mul(0xA24B_AED4_963E_E407 ^ index as u64);
        *lane = avalanche(*lane);
    }

    lanes
}

fn trim_carriage_return(line: &[u8]) -> &[u8] {
    if matches!(line.last(), Some(b'\r')) {
        &line[..line.len() - 1]
    } else {
        line
    }
}

fn avalanche(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 33;
    value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    value ^ (value >> 33)
}

fn insert_top_hit(hits: &mut Vec<SearchHit>, candidate: SearchHit, top_k: usize) {
    let insert_at = hits
        .binary_search_by_key(&candidate.distance, |hit| hit.distance)
        .unwrap_or_else(|position| position);

    if insert_at >= top_k {
        return;
    }

    hits.insert(insert_at, candidate);
    if hits.len() > top_k {
        hits.pop();
    }
}

fn insert_indexed_top_hit(
    hits: &mut Vec<IndexedSearchHit>,
    candidate: IndexedSearchHit,
    top_k: usize,
) {
    let insert_at = hits
        .binary_search_by_key(&candidate.distance, |hit| hit.distance)
        .unwrap_or_else(|position| position);

    if insert_at >= top_k {
        return;
    }

    hits.insert(insert_at, candidate);
    if hits.len() > top_k {
        hits.pop();
    }
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
