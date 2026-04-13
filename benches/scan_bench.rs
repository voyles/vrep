//! Criterion regression harness for the pure Hamming scan loop.
//!
//! This benchmark intentionally excludes mmap and chunking so it can measure the "inference
//! tax" later: swap in a model-backed encoder elsewhere, keep this loop fixed, and compare.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use vrep_core::{search_top_k_bits, EmbeddingGenerator, MockEncoder, VECTOR_WORDS};

const VECTOR_COUNT: usize = 1_000_000;
const TOP_K: usize = 10;

fn scan_benchmark(c: &mut Criterion) {
    let encoder = MockEncoder::default();
    let query = encoder.generate("benchmark-query");
    let vectors = build_vectors(&encoder, VECTOR_COUNT);

    let mut group = c.benchmark_group("hamming_scan");
    group.throughput(Throughput::Elements(VECTOR_COUNT as u64));
    group.bench_with_input(
        BenchmarkId::new("top_k", VECTOR_COUNT),
        &vectors,
        |bench, embeddings| {
            bench.iter(|| {
                // `black_box` prevents the optimizer from proving that the scan result is
                // unused. That keeps the benchmark focused on real XOR plus POPCNT work
                // rather than measuring a partially eliminated code path.
                let hits = search_top_k_bits(black_box(embeddings), black_box(&query), TOP_K);
                black_box(hits);
            });
        },
    );
    group.finish();
}

fn build_vectors(encoder: &MockEncoder, vector_count: usize) -> Vec<[u64; VECTOR_WORDS]> {
    (0..vector_count)
        .map(|index| encoder.generate(&format!("mock-vector-{index:08}")))
        .collect()
}

criterion_group!(benches, scan_benchmark);
criterion_main!(benches);
