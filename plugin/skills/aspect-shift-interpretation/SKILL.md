---
name: aspect-shift-interpretation
description: Use when the user asks to interpret a poem, image, ambiguous figure, or text that admits multiple readings - especially when they want both readings surfaced. Routes through vimarsa to detect duck-rabbit aspect-shifts.
---

# Aspect-shift interpretation

## When to use

The user supplies a poem, koan, ambiguous figure description, or short text and asks "what does this mean?" or "are there multiple readings here?". Also: any explicit duck-rabbit / Wittgenstein / Necker-cube style probe.

## Workflow

1. Extract the `surface` (the user-supplied text).
2. Construct a `prompt` that names the interpretive task (e.g. `"What does this poem mean?"`).
3. Build a `retrieval_set` from the user's contextual material: lines from the poem, prior interpretations, etc.
4. Build an `aspects` list of 2-4 candidate readings (each a short phrase). If the user names them, use those.
5. Estimate `ananda_score` from prior context (`0.7` is a reasonable default for coherent texts).
6. Call MCP tool `pratyabhijna_mcp__vimarsa`.
7. If `event = true`, the surface carries multiple aspects; report each with its cosine match. If `event = false`, report which gates failed (novelty, aspect_count, aesthetic) so the user can refine.

## Tuning

- For LM-generated surfaces, default `aspect_cosine_hit = 0.40`.
- For human-written poetry, try `aspect_cosine_hit = 0.50` (humans embed more sharply).
- If the surface is borrowed (high overlap with retrieval_set), novelty will be low - that is correct, not a bug.

## Output format

A short paragraph: "This text fires a vimarsa event (novelty = X.XX, aspects = N). Reading A: ... Reading B: ...". Or: "No aspect-shift detected; the dominant reading is ...".
