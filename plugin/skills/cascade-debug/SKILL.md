---
name: cascade-debug
description: Use when the user asks "why did the cascade pick this candidate?", "trace the BMR for this prompt", or wants to inspect the cascade's audit trail. Reads `audit/phase8/mcp_calls.jsonl` and produces a per-step summary.
---

# Cascade debug / audit-trail inspector

## When to use

The user has run a cascade and wants to understand *why* it produced what it produced. Or they want to see whether `vimarsa_event` fired and which gate held it back. Or a benchmark result looks suspect.

## Workflow

1. Call MCP tool `pratyabhijna_mcp__report` to confirm substrate state.
2. Read `audit/phase8/mcp_calls.jsonl` (newest entries first).
3. For the most recent `cascade` call:
   - List all 7 candidates (from the ananda + apoha arrays in `audit`).
   - Show the BMR posterior, the ΔF, and the selected_idx.
   - Show the vimarsa diagnostics: novelty, aspect_count, switching, ananda.
4. Identify which axis is the bottleneck: was novelty too low? Were < 2 aspects detected? Was ananda below the floor?
5. Suggest a fix:
   - Low novelty → broaden retrieval_set or raise cit_temperature.
   - Low aspect_count → enrich aspects list or lower aspect_cosine_hit.
   - Low ananda → reword constraint or add reward-model term.
   - Low ΔF → uniform candidate pool, consider raising K or λ_a.

## Output format

A bulleted diagnostic report. Always include a single suggested next call (full tool name and args) that the user can run to test the fix.
