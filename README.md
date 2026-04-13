# vrep-core (Vector Grep) v1.0.0

A high-performance semantic log scanner that bypasses the "Neural Wall" using a two-pass architecture and sidecar indexing.

## Install as a command

Build and install from the repository:

```bash
cargo install --path .
```

Or run the bootstrap installer (recommended):

```bash
chmod +x scripts/install-linux.sh
./scripts/install-linux.sh
```

This installs `vrep-core` into `~/.cargo/bin`.

Make sure it is on your `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Model layout

`OnnxEncoder` expects:

- `model/model.onnx`
- `model/tokenizer.json`

from the current working directory.

## Quick checks

Safe model check (with fail-stop timeout):

```bash
chmod +x scripts/check-model-safe.sh
./scripts/check-model-safe.sh --timeout 20 --profile release
```

Safe semantic query run (with fail-stop timeout):

```bash
chmod +x scripts/run-semantic-safe.sh
./scripts/run-semantic-safe.sh "bad internet" semantic_test.txt --top-k 3 --bench --timeout 20 --profile release
```

## Direct CLI examples

```bash
vrep-core --check-model
vrep-core "bad internet" semantic_test.txt --top-k 3 --bench
cat semantic_test.txt | vrep-core "bad internet" --top-k 3
```

## Exit codes

- `0`: at least one match
- `1`: no matches
- `124`: timed out by safe wrapper scripts
