#!/bin/bash
# PCE ralph-loop promise gate.
#
# Runs the four honesty gates in order and writes the combined result to
# audit/phase<N>/promise.json. Exit code:
#   0 - all green, ralph-loop may accept the promise
#   1 - at least one gate failed, ralph-loop must continue iterating
#   2 - infrastructure error (missing tool, etc.)
#
# Portable across macOS bash 3.2 and Linux bash 4+.

set -o pipefail

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "ralph_promise_gate: usage: $0 <phase_number> [extra args...]" >&2
  exit 2
fi
shift || true
# Capture remaining args (forwarded to verify_remote_pushed); avoid empty-array
# expansion issues on bash 3.2.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_DIR="$REPO_ROOT/audit/phase$PHASE"
mkdir -p "$AUDIT_DIR"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ralph_promise_gate: python3 not found on PATH" >&2
  exit 2
fi

run_one() {
  local name="$1"; shift
  local out_file="$AUDIT_DIR/${name}.json"
  local err_file="$AUDIT_DIR/${name}.stderr"
  "$PYTHON_BIN" "$@" > "$out_file" 2> "$err_file"
  local code=$?
  echo "$code"
}

ANTI_STUB_CODE=$(run_one anti_stub "$REPO_ROOT/scripts/anti_stub_check.py" --json --phase "$PHASE")
VRM_CODE=$(run_one verify_real_model "$REPO_ROOT/scripts/verify_real_model.py" --phase "$PHASE")
VA_CODE=$(run_one verify_artifact "$REPO_ROOT/scripts/verify_artifact.py" --phase "$PHASE")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  VRP_CODE=$(run_one verify_remote_pushed "$REPO_ROOT/scripts/verify_remote_pushed.py")
else
  VRP_CODE=$(run_one verify_remote_pushed "$REPO_ROOT/scripts/verify_remote_pushed.py" "${EXTRA_ARGS[@]}")
fi

ALL_OK=1
if [[ "$ANTI_STUB_CODE" -ne 0 ]] || [[ "$VRM_CODE" -ne 0 ]] || [[ "$VA_CODE" -ne 0 ]] || [[ "$VRP_CODE" -ne 0 ]]; then
  ALL_OK=0
fi

OK_STR=$([[ "$ALL_OK" -eq 1 ]] && echo true || echo false)
cat > "$AUDIT_DIR/promise.json" <<EOF
{
  "phase": $PHASE,
  "results": [
    {"gate": "anti_stub", "exit_code": $ANTI_STUB_CODE},
    {"gate": "verify_real_model", "exit_code": $VRM_CODE},
    {"gate": "verify_artifact", "exit_code": $VA_CODE},
    {"gate": "verify_remote_pushed", "exit_code": $VRP_CODE}
  ],
  "ok": $OK_STR
}
EOF

if [[ "$ALL_OK" -eq 1 ]]; then
  echo "[ralph_promise_gate] phase $PHASE: ALL GATES GREEN"
  exit 0
fi
echo "[ralph_promise_gate] phase $PHASE: gates RED  anti_stub=$ANTI_STUB_CODE  verify_real_model=$VRM_CODE  verify_artifact=$VA_CODE  verify_remote_pushed=$VRP_CODE" >&2
exit 1
