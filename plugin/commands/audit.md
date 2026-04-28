---
name: pce-audit
description: Read the cascade audit log and explain the most recent decision. Args: optional `--n <int>` (default 1, the most recent cascade call).
---

Delegate to the `pratyabhijna-auditor` agent. Read the last N entries of `audit/phase8/mcp_calls.jsonl` and produce the auditor's standard report (substrate state, candidate table, vimarsa diagnostics, bottleneck gate, suggested next call).
