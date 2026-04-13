#!/usr/bin/env bash
set -euo pipefail

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-15}"
PROFILE="${PROFILE:-release}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
cargo_args+=(-- --check-model)

if ! timeout --signal=KILL "${TIMEOUT_SECONDS}s" cargo "${cargo_args[@]}"; then
  status=$?
  if [[ $status -eq 124 || $status -eq 137 ]]; then
    pkill -f "vrep-core" >/dev/null 2>&1 || true
    echo "check-model timed out after ${TIMEOUT_SECONDS}s and was terminated." >&2
    exit 124
  fi
  exit $status
fi
