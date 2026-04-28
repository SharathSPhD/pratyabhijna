#!/bin/bash
# Pratyabhijna Creative Engine - PreToolUse hook for cascade/vimarsa.
# Stamps a pre-call event so the audit trail can reconstruct call latency.
set -euo pipefail
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd -- "$PLUGIN_ROOT/.." && pwd)"
AUDIT_DIR="$REPO_ROOT/audit/phase8"
mkdir -p "$AUDIT_DIR"
TS=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")
echo "{\"event\": \"pre_tool\", \"ts\": \"$TS\", \"tool\": \"${TOOL_NAME:-unknown}\"}" >> "$AUDIT_DIR/hook_events.jsonl"
