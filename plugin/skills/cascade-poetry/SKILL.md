---
name: cascade-poetry
description: Use when the user asks for a poem, haiku, sonnet, ghazal, or any short literary form, or when they want a constraint-anchored creative composition. Routes through the Pratyabhijna cascade so the result reflects vimarsa aspect-shifts and BMR-driven candidate selection rather than raw LM sampling.
---

# Cascade poetry generation

## When to use

Poetry, song lyrics, haiku, micro-fiction, or any short literary form where the user supplies a topic + form constraint. Also use when the user asks "compose with the cascade" or "use PCE for this poem".

## Workflow

1. Parse the user's topic into a plain `prompt` string (the brief / image / event).
2. Construct a one-line `constraint_text` that names the form and tone (e.g. `"a 5-7-5 haiku about autumn rain, restrained voice"`).
3. Build optional `must_avoid` from concrete clichés the user wants to dodge.
4. Build optional `aspects` (2-4 short phrases naming axes of meaning the user wants present).
5. Call MCP tool `pratyabhijna_mcp__cascade` with `K=6..8`, `max_tokens=64..96`, `render_mode="verbatim"`.
6. Read back `surface`, `vimarsa_event`, `selected_idx`, `audit.delta_F`, `audit.ananda_scores`.
7. If `vimarsa_event = false` and the user asked for an aspect-shift poem, retry with a slightly higher `cit_temperature` (`1.1` → `1.3`) and a larger `K`.

## Output format

Return the `surface` text. If `vimarsa_event = true`, add a one-line italicised note saying which aspects were detected. Never return raw JSON to the user.

## Anti-patterns

- Do **not** call the LM directly when this skill is active. Always go through `cascade`.
- Do **not** pass a long prompt unmodified - distill to ≤ 30 words.
- Do **not** invent `aspects` the user did not imply.
