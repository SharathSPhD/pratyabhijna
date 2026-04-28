#!/bin/bash
# Pratyabhijna Creative Engine - SessionStart hook.
# Records the session-start timestamp and emits a one-line status banner so the
# user knows the cascade is loaded.
set -euo pipefail
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd -- "$PLUGIN_ROOT/.." && pwd)"
AUDIT_DIR="$REPO_ROOT/audit/phase8"
mkdir -p "$AUDIT_DIR"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"event\": \"session_start\", \"ts\": \"$TS\"}" >> "$AUDIT_DIR/hook_events.jsonl"
echo "[pratyabhijna] cascade loaded; 15 MCP tools, 5 skills, 5 agents available."
