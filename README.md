# vrep-core (Vector Grep) v1.0.0

## 🚀 Performance & Data Resilience

`vrep-core` uses a "Compute Once, Search Forever" architecture. It distinguishes between the intensive **Cold Pass** (initial indexing) and the near-instant **Hot Pass** (semantic searching).

### The Two-Pass Architecture

| Pass | Action | Performance (Nog-class CPU) |
| :--- | :--- | :--- |
| **Cold Pass** | Neural Encoding & Indexing | ~800 - 2,000 vectors/sec |
| **Hot Pass** | Semantic Bit-Vector Search | **30M+ vectors/sec** |

### Data Robustness

The engine is designed to handle "dirty" real-world datasets (e.g., `rockyou.txt`) without manual pre-processing:

* **Encoding Agnostic:** Automatically handles non-UTF8 bytes via lossy conversion—no more hanging on legacy wordlist "garbage."
* **Zero-Copy Search:** Uses `mmap` to treat the `.vrepidx` as local memory, bypassing heap allocation bottlenecks.
* **Semantic Intelligence:** Finds matches based on meaning, not just characters. Searching for **"sport"** successfully returns `football`, `baseball`, and `soccer`.

## 🔍 Operational Notes

### Indexing & Permissions

By default, `vrep-core` attempts to save a sidecar `.vrepidx` file next to the source file.

1. **Read-Only Directories:** If scanning system wordlists (e.g., `/usr/share/wordlists/`), copy the file to a writable directory or run with `sudo` to save the index.
2. **The First Run:** Expect high CPU/Memory usage during the "Cold Pass." The engine is performing heavy neural inference to map the file.
3. **Subsequent Runs:** Once the index exists, results are returned in milliseconds.

*Note: A future update will redirect indices to `~/.cache/vrep` automatically to avoid permission conflicts.*

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

* `model/model.onnx`
* `model/tokenizer.json`

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

## Data Resilience

`vrep-core` is designed to handle "dirty" real-world datasets (e.g., `rockyou.txt`):

* **Encoding:** Automatically handles non-UTF8 bytes via lossy conversion.
* **Zero-Copy:** Uses `mmap` for instant access to large files without heap-allocating every line.
* **Sanitization:** If a file contains null bytes or inconsistent line endings, the indexer skips corrupted segments rather than stalling.

### ⚠️ A Note on Benchmarking

The provided `stress.txt` is a **throughput benchmark only**.

* The index contains **vector-placeholders** to test binary scanning speed (~18M vps).
* It will **not** return semantically accurate results.
* For real-world semantic accuracy, use a "Cold Pass" on a real dataset like `rockyou.txt`.

## Exit codes

* `0`: at least one match
* `1`: no matches
* `124`: timed out by safe wrapper scripts
