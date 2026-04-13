use std::fs;
use std::io::{Cursor, Read};
use std::path::Path;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use vrep_core::{
    scan_bytes_with_encoder, scan_file_with_encoder, write_hits, EmbeddingGenerator, MockEncoder,
    OnnxEncoder,
};

enum TestEncoder {
    Onnx(OnnxEncoder),
    Mock(MockEncoder),
}

impl EmbeddingGenerator for TestEncoder {
    fn generate(&self, text: &str) -> [u64; vrep_core::VECTOR_WORDS] {
        match self {
            Self::Onnx(encoder) => encoder.generate(text),
            Self::Mock(encoder) => encoder.generate(text),
        }
    }
}

fn select_test_encoder() -> TestEncoder {
    match OnnxEncoder::from_model_dir(Path::new("model")) {
        Ok(onnx) => TestEncoder::Onnx(onnx),
        Err(_) => TestEncoder::Mock(MockEncoder::default()),
    }
}

fn unique_temp_file(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("vrep-{name}-{}-{nanos}.txt", std::process::id()))
}

fn write_temp_file(name: &str, content: &str) -> PathBuf {
    let path = unique_temp_file(name);
    fs::write(&path, content).expect("failed to write temp fixture");
    path
}

fn binary_path() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_vrep-core")
        .map(PathBuf::from)
        .expect("CARGO_BIN_EXE_vrep-core was not set by cargo test")
}

#[test]
fn file_path_scan_returns_hits() {
    let path = write_temp_file("path-scan", "alpha\nbeta\ngamma\n");
    let encoder = MockEncoder::default();

    let result =
        scan_file_with_encoder(&path, "beta", 5, None, &encoder).expect("scan should succeed");

    assert!(!result.hits.is_empty(), "expected at least one hit");
    assert_eq!(result.hits[0].source.line_number, 2);
    assert_eq!(result.hits[0].distance, 0);

    let _ = fs::remove_file(path);
}

#[test]
fn stdin_scan_via_mock_reader_returns_hits() {
    let encoder = MockEncoder::default();
    let mut reader = Cursor::new("alpha\nbeta\ngamma\n");
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .expect("cursor should be readable");

    let result = scan_bytes_with_encoder(&bytes, "beta", 5, None, &encoder);

    assert!(
        !result.hits.is_empty(),
        "expected at least one hit from stdin bytes"
    );
    assert_eq!(result.hits[0].source.line_number, 2);
}

#[test]
fn threshold_filter_drops_distant_hits_and_empty_write_is_silent() {
    let encoder = MockEncoder::default();
    let bytes = b"alpha\nbeta\ngamma\n";
    let result = scan_bytes_with_encoder(bytes, "beta", 5, Some(0), &encoder);

    assert!(
        !result.hits.is_empty(),
        "exact line should survive distance=0"
    );
    assert!(result.hits.iter().all(|hit| hit.distance == 0));

    let mut out = Vec::new();
    write_hits(&mut out, bytes, &[]).expect("write_hits should not fail on empty hits");
    assert!(out.is_empty(), "empty hits should produce no output");
}

fn query_with_positive_min_distance<E: EmbeddingGenerator>(bytes: &[u8], encoder: &E) -> (String, u32) {
    let candidates = [
        "zeta-not-present",
        "theta-not-present",
        "omega-not-present",
        "kappa-not-present",
    ];

    for candidate in candidates {
        let result = scan_bytes_with_encoder(bytes, candidate, 5, None, encoder);
        let min_distance = result
            .hits
            .iter()
            .map(|hit| hit.distance)
            .min()
            .expect("non-empty corpus should produce at least one nearest hit");
        if min_distance > 0 {
            return (candidate.to_string(), min_distance);
        }
    }

    panic!("unable to find a query with strictly positive minimum distance");
}

#[test]
fn quiet_mode_preserves_exit_codes() {
    let path = write_temp_file("quiet-exit", "alpha\nbeta\ngamma\n");
    let bin = binary_path();
    let bytes = b"alpha\nbeta\ngamma\n";
    let encoder = select_test_encoder();
    let (no_match_query, min_distance) = query_with_positive_min_distance(bytes, &encoder);
    let strict_threshold = min_distance - 1;

    let success = Command::new(&bin)
        .arg("-q")
        .arg("-d")
        .arg("0")
        .arg("beta")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("quiet command should run");

    assert_eq!(success.status.code(), Some(0));
    assert!(
        success.stdout.is_empty(),
        "quiet mode should suppress stdout"
    );

    let failure = Command::new(&bin)
        .arg("-q")
        .arg("-d")
        .arg(strict_threshold.to_string())
        .arg(no_match_query)
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("quiet command should run");

    assert_eq!(failure.status.code(), Some(1));
    assert!(
        failure.stdout.is_empty(),
        "quiet mode should suppress stdout"
    );

    let _ = fs::remove_file(path);
}

#[test]
fn help_lists_max_distance_and_cli_parses_long_and_short_flags() {
    let bin = binary_path();
    let help = Command::new(&bin)
        .arg("--help")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("--help should run");

    assert_eq!(help.status.code(), Some(0));
    let help_stdout = String::from_utf8_lossy(&help.stdout);
    assert!(
        help_stdout.contains("--max-distance"),
        "help output should include --max-distance"
    );
    assert!(
        help_stdout.contains("-d"),
        "help output should include short -d flag"
    );

    let path = write_temp_file("max-distance-parse", "alpha\nbeta\ngamma\n");

    let long_form = Command::new(&bin)
        .arg("-q")
        .arg("--max-distance")
        .arg("0")
        .arg("beta")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("long-form max-distance command should run");

    let short_form = Command::new(&bin)
        .arg("-q")
        .arg("-d")
        .arg("0")
        .arg("beta")
        .arg(&path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("short-form max-distance command should run");

    assert_eq!(long_form.status.code(), Some(0));
    assert_eq!(short_form.status.code(), Some(0));
    assert!(long_form.stdout.is_empty());
    assert!(short_form.stdout.is_empty());

    let _ = fs::remove_file(path);
}
