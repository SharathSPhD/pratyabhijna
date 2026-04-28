#!/bin/bash
# PCE ralph-loop promise gate.
#
# This script is invoked by the ralph-loop Stop hook (via a small wrapper that
# only fires when the active <promise> matches one of the PCE_PHASE_*_COMPLETE
# strings). It runs the four honesty gates in order and writes the combined
# result to audit/phase<N>/promise.json. Any non-zero exit code in any gate
# causes this script to exit 1, which the wrapper translates back into a Stop
# hook decision of "block" so ralph-loop re-injects the same prompt.
#
# Usage:
#   ralph_promise_gate.sh <phase_number> [--allow-dirty]
#
# Exit codes:
#   0 - all green, ralph-loop may accept the promise
#   1 - at least one gate failed, ralph-loop must continue iterating
#   2 - infrastructure error (missing tool, etc.)

set -euo pipefail

PHASE="${1:-}"
shift || true
EXTRA_ARGS=("$@")
if [[ -z "$PHASE" ]]; then
  echo "ralph_promise_gate: usage: $0 <phase_number> [extra args...]" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_DIR="$REPO_ROOT/audit/phase$PHASE"
mkdir -p "$AUDIT_DIR"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ralph_promise_gate: python3 not found on PATH" >&2
  exit 2
fi

# Use the project's uv-managed venv if present, falling back to system python3.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi

run_gate() {
  local name="$1"; shift
  local out_file="$AUDIT_DIR/${name}.json"
  set +e
  "$PYTHON_BIN" "$@" > "$out_file" 2> "$AUDIT_DIR/${name}.stderr"
  local code=$?
  set -e
  echo "$name $code"
  return $code
}

ALL_OK=1
declare -a RESULTS=()

set +e
run_gate anti_stub "$REPO_ROOT/scripts/anti_stub_check.py" --json --phase "$PHASE"
RESULTS+=("anti_stub:$?")
[[ "${RESULTS[-1]}" == "anti_stub:0" ]] || ALL_OK=0

run_gate verify_real_model "$REPO_ROOT/scripts/verify_real_model.py" --phase "$PHASE"
RESULTS+=("verify_real_model:$?")
[[ "${RESULTS[-1]}" == "verify_real_model:0" ]] || ALL_OK=0

run_gate verify_artifact "$REPO_ROOT/scripts/verify_artifact.py" --phase "$PHASE"
RESULTS+=("verify_artifact:$?")
[[ "${RESULTS[-1]}" == "verify_artifact:0" ]] || ALL_OK=0

run_gate verify_remote_pushed "$REPO_ROOT/scripts/verify_remote_pushed.py" "${EXTRA_ARGS[@]}"
RESULTS+=("verify_remote_pushed:$?")
[[ "${RESULTS[-1]}" == "verify_remote_pushed:0" ]] || ALL_OK=0
set -e

# Build the combined promise.json
{
  printf '{\n  "phase": %s,\n  "results": [' "$PHASE"
  first=1
  for r in "${RESULTS[@]}"; do
    name="${r%%:*}"
    code="${r##*:}"
    if [[ $first -eq 1 ]]; then
      first=0
    else
      printf ','
    fi
    printf '\n    {"gate": "%s", "exit_code": %s}' "$name" "$code"
  done
  printf '\n  ],\n  "ok": %s\n}\n' "$([[ $ALL_OK -eq 1 ]] && echo true || echo false)"
} > "$AUDIT_DIR/promise.json"

if [[ $ALL_OK -eq 1 ]]; then
  echo "[ralph_promise_gate] phase $PHASE: ALL GATES GREEN"
  exit 0
fi
echo "[ralph_promise_gate] phase $PHASE: gates red - ${RESULTS[*]}" >&2
exit 1
