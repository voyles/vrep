#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <query> <input_path> [--top-k N] [--max-distance D] [--bench] [--timeout S] [--profile release|debug]" >&2
  exit 2
fi

QUERY="$1"
INPUT_PATH="$2"
shift 2

TOP_K=5
TIMEOUT_SECONDS=30
PROFILE="release"
BENCH=0
MAX_DISTANCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --max-distance)
      MAX_DISTANCE="$2"
      shift 2
      ;;
    --bench)
      BENCH=1
      shift
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

pkill -f "vrep-core" >/dev/null 2>&1 || true

cargo_args=(run)
if [[ "$PROFILE" == "release" ]]; then
  cargo_args+=(--release)
fi
cargo_args+=(-- "$QUERY" "$INPUT_PATH" --top-k "$TOP_K")
if [[ $BENCH -eq 1 ]]; then
  cargo_args+=(--bench)
fi
if [[ -n "$MAX_DISTANCE" ]]; then
  cargo_args+=(--max-distance "$MAX_DISTANCE")
fi

if ! timeout --signal=KILL "${TIMEOUT_SECONDS}s" cargo "${cargo_args[@]}"; then
  status=$?
  if [[ $status -eq 124 || $status -eq 137 ]]; then
    pkill -f "vrep-core" >/dev/null 2>&1 || true
    echo "semantic query timed out after ${TIMEOUT_SECONDS}s and was terminated." >&2
    exit 124
  fi
  exit $status
fi
