#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run as root. Use your normal user account." >&2
  exit 1
fi

echo "[1/6] Checking required commands..."
missing=()
for cmd in bash curl grep sed awk chmod timeout pkill; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    missing+=("$cmd")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing required commands: ${missing[*]}" >&2
  echo "Install core utilities (for example: coreutils, procps) and retry." >&2
  exit 1
fi

echo "[2/6] Checking Rust toolchain..."
if ! command -v cargo >/dev/null 2>&1; then
  echo "Rust is not installed. Installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  export PATH="$HOME/.cargo/bin:$PATH"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo was not found after rustup install." >&2
  exit 1
fi

echo "[3/6] Ensuring script executability..."
chmod +x scripts/check-model-safe.sh scripts/run-semantic-safe.sh

echo "[4/6] Building and installing vrep-core..."
cargo install --path .

echo "[5/6] Verifying PATH includes ~/.cargo/bin..."
if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
  shell_rc=""
  if [[ -n "${ZSH_VERSION:-}" ]]; then
    shell_rc="$HOME/.zshrc"
  else
    shell_rc="$HOME/.bashrc"
  fi

  if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$shell_rc" 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$shell_rc"
    echo "Added ~/.cargo/bin PATH export to $shell_rc"
  fi

  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[6/6] Quick sanity check..."
if ! command -v vrep-core >/dev/null 2>&1; then
  echo "vrep-core command not found on PATH after install." >&2
  echo "Open a new shell or source your rc file, then run: vrep-core --help" >&2
  exit 1
fi

echo "Installation complete."
echo "Next steps:"
echo "  1) ./scripts/check-model-safe.sh --timeout 20 --profile release"
echo "  2) ./scripts/run-semantic-safe.sh \"bad internet\" semantic_test.txt --top-k 3 --bench --timeout 20 --profile release"
