#!/bin/bash
# Pratyabhijna Creative Engine - PostToolUse hook for cascade.
# After every cascade call, append the tail of mcp_calls.jsonl to a hot
# consolidation queue. (The actual SWS/REM cycle is invoked manually via the
# `consolidate_cycle` tool to keep the per-call latency low.)
set -euo pipefail
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd -- "$PLUGIN_ROOT/.." && pwd)"
AUDIT_DIR="$REPO_ROOT/audit/phase8"
mkdir -p "$AUDIT_DIR"
TS=$(date -u +"%Y-%m-%dT%H:%M:%S.%NZ")
QUEUE="$AUDIT_DIR/consolidation_queue.jsonl"
LOG="$AUDIT_DIR/mcp_calls.jsonl"
echo "{\"event\": \"post_tool\", \"ts\": \"$TS\", \"tool\": \"${TOOL_NAME:-unknown}\"}" >> "$AUDIT_DIR/hook_events.jsonl"
if [[ -f "$LOG" ]]; then
  tail -n 1 "$LOG" >> "$QUEUE"
fi
