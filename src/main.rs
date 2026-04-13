//! # vrep-core CLI
//!
//! This binary acts as the frontend for the vrep semantic search engine.
//! It prioritizes high-throughput IO by switching between two modes:
//!
//! 1. **Zero-Copy Mode:** When a file path is provided, it uses `mmap` to map
//!    the file directly into the process address space.
//! 2. **Streaming Mode:** When reading from `stdin`, it buffers the input into
//!    a heap-allocated vector for processing.
//!
//! Output is directed to `stdout` for matches and `stderr` for operational
//! telemetry, following the Unix tool standard.
//! The CLI also includes a built-in benchmark mode that can be enabled with the `--bench` flag,
//! which prints throughput and latency metrics to `stderr` after the scan completes.
//! The `--prepare-dummy-gib` option allows users to create a large dummy dataset for testing
//! without needing to generate real embeddings.
//! The CLI is designed to be flexible and efficient, making it suitable for both real-world usage and performance testing.
//! The `--quiet` flag can be used to suppress match output, allowing the CLI to be used in scripts where only the exit code is relevant.
//! Exit code `0` indicates that at least one match was found, while `1` indicates no matches were found.
//! This design allows the CLI to be easily integrated into larger workflows and pipelines, while still providing detailed performance insights when needed.
//! The CLI uses the `MockEncoder` from `vrep_core` to generate embeddings for the query and input data, making it a self-contained tool for testing and benchmarking the core scanning functionality without relying on external models or encoders.
//! The CLI is implemented in Rust and uses the `clap` crate for argument parsing, the `memmap2` crate for memory mapping, and standard Rust libraries for IO operations. It is designed to be efficient and easy to use, making it a valuable tool for developers working with the vrep semantic search engine.

use std::io::{self, BufReader, Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;

use clap::Parser;
use memmap2::Mmap;

use vrep_core::{
    create_dummy_file, scan_bytes_with_encoder, scan_bytes_with_onnx_batches,
    scan_file_with_onnx_batches,
    write_benchmark, write_hits, MockEncoder, OnnxEncoder, ONNX_BATCH_SIZE,
};
use tokenizers::Tokenizer;

const DEFAULT_TOP_K: usize = 5;
const DEFAULT_MAX_DISTANCE: u32 = 512;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "High-performance semantic log scanner using ONNX embeddings"
)]
struct Args {
    #[arg(value_name = "QUERY")]
    query: Option<String>,
    #[arg(value_name = "INPUT")]
    input: Option<PathBuf>,
    #[arg(long, help = "Print throughput and latency metrics for the scan")]
    bench: bool,
    #[arg(
        short = 'q',
        long,
        help = "Suppress match output and return only the exit code"
    )]
    quiet: bool,
    #[arg(
        long,
        value_name = "GIB",
        help = "Create or overwrite INPUT with a newline-delimited dummy dataset of the requested size before scanning"
    )]
    prepare_dummy_gib: Option<u64>,
    #[arg(long, default_value_t = DEFAULT_TOP_K, help = "Number of nearest matches to keep")]
    top_k: usize,
    #[arg(
        short = 'd',
        long,
        value_name = "DISTANCE",
        default_value_t = DEFAULT_MAX_DISTANCE,
        value_parser = clap::value_parser!(u32).range(0..=512),
        help = "Maximum Hamming distance threshold (0 to 512); lower is a tighter match."
    )]
    max_distance: u32,
    #[arg(
        long,
        help = "Validate ONNX model assets in ./model and exit without scanning"
    )]
    check_model: bool,
    #[arg(
        short = 'f',
        long,
        value_name = "PATTERN",
        help = "Pre-filter lines before semantic search using exact substring matching (fast path)"
    )]
    filter: Option<String>,
}

fn main() {
    let args = Args::parse();

    match run(args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

/// Executes the CLI request using the mock encoder and shared library scan path.
fn run(args: Args) -> io::Result<i32> {
    if args.check_model {
        return run_model_check();
    }

    let query = args.query.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "QUERY is required unless --check-model is used",
        )
    })?;

    if let Some(size_gib) = args.prepare_dummy_gib {
        let input_path = args.input.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "--prepare-dummy-gib requires an INPUT path",
            )
        })?;
        create_dummy_file(input_path, size_gib)?;
    }

    let model_dir = PathBuf::from("model");
    let encoder = match OnnxEncoder::from_model_dir(&model_dir) {
        Ok(onnx) => CliEncoder::Onnx(Box::new(onnx)),
        Err(error) => {
            eprintln!("warning: failed to initialize ONNX encoder from ./model: {error}");
            eprintln!("warning: falling back to mock encoder");
            CliEncoder::Mock(MockEncoder)
        }
    };

    let fenced = catch_unwind(AssertUnwindSafe(|| -> io::Result<InputData> {
        let input = match args.input.as_ref() {
            Some(input_path) => {
                let bytes = vrep_core::map_file_read_only(input_path)?;
                let result = match &encoder {
                        CliEncoder::Onnx(onnx) => scan_file_with_onnx_batches(
                        input_path,
                        query,
                        args.top_k,
                        Some(args.max_distance),
                        onnx,
                        ONNX_BATCH_SIZE,
                            args.filter.as_deref(),
                    )?,
                    CliEncoder::Mock(mock) => scan_bytes_with_encoder(
                        &bytes,
                        query,
                        args.top_k,
                        Some(args.max_distance),
                        mock,
                    ),
                };
                InputData::Mapped { bytes, result }
            }
            None => {
                let stdin = io::stdin();
                let mut reader = BufReader::new(stdin.lock());
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes)?;
                let result = match &encoder {
                    CliEncoder::Onnx(onnx) => scan_bytes_with_onnx_batches(
                        &bytes,
                        query,
                        args.top_k,
                        Some(args.max_distance),
                        onnx,
                        ONNX_BATCH_SIZE,
                        args.filter.as_deref(),
                    ),
                    CliEncoder::Mock(mock) => scan_bytes_with_encoder(
                        &bytes,
                        query,
                        args.top_k,
                        Some(args.max_distance),
                        mock,
                    ),
                };
                InputData::Buffered { bytes, result }
            }
        };

        Ok(input)
    }));

    let input = match fenced {
        Ok(result) => result?,
        Err(_) => {
            return Err(io::Error::other(
                "semantic query panicked inside execution fence",
            ));
        }
    };

    if !args.quiet {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        write_hits(&mut handle, input.bytes(), &input.result().hits)?;
        handle.flush()?;
    }

    if args.bench {
        let stderr = io::stderr();
        let mut handle = stderr.lock();
        write_benchmark(&mut handle, &input.result().stats)?;
        handle.flush()?;
    }

    Ok(if input.result().hits.is_empty() { 1 } else { 0 })
}

fn run_model_check() -> io::Result<i32> {
    let model_dir = PathBuf::from("model");
    eprintln!("model check: validating assets in {:?}", model_dir);

    let tokenizer_path = model_dir.join("tokenizer.json");
    let model_path = model_dir.join("model.onnx");

    if !model_path.exists() {
        eprintln!("model check: FAILED - missing {:?}", model_path);
        return Ok(1);
    }
    if !tokenizer_path.exists() {
        eprintln!("model check: FAILED - missing {:?}", tokenizer_path);
        return Ok(1);
    }

    if let Err(error) = Tokenizer::from_file(&tokenizer_path) {
        eprintln!("model check: FAILED - tokenizer parse error: {error}");
        return Ok(1);
    }

    eprintln!("model check: OK");
    eprintln!("model check: expected inputs: input_ids, attention_mask");
    eprintln!("model check: tokenizer and model assets are present");
    eprintln!("model check: deep ORT validation occurs during first real query");
    Ok(0)
}

enum CliEncoder {
    Onnx(Box<OnnxEncoder>),
    Mock(MockEncoder),
}

enum InputData {
    Mapped {
        bytes: Mmap,
        result: vrep_core::ScanResult,
    },
    Buffered {
        bytes: Vec<u8>,
        result: vrep_core::ScanResult,
    },
}

impl InputData {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped { bytes, .. } => bytes,
            Self::Buffered { bytes, .. } => bytes,
        }
    }

    fn result(&self) -> &vrep_core::ScanResult {
        match self {
            Self::Mapped { result, .. } => result,
            Self::Buffered { result, .. } => result,
        }
    }
}
