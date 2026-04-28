---
name: scientific-creativity
description: Use for scientific or mathematical creativity tasks - hypothesis generation, analogy-finding across domains, BIG-Bench-Hard "Causal Judgement"-style problems, or any "give me a non-obvious explanation/analogy" prompt. The cascade is configured for higher cit-temperature and a wider sampler grid to favor exploration.
---

# Scientific / mathematical creativity

## When to use

Hypothesis brainstorming, cross-domain analogy hunting, "non-obvious explanation" tasks, BIG-Bench-Hard creative reasoning, or whenever the user wants "the kind of leap that wins a paper, not the standard answer".

## Workflow

1. Frame the user's question as a constraint: `constraint_text = "non-obvious explanation/analogy for: {question}"`.
2. Set `must_avoid` to standard textbook explanations the user has already heard.
3. Build `aspects` from 2-3 candidate framings (e.g. ["thermodynamic", "graph-theoretic", "biological"]).
4. Call MCP tool `pratyabhijna_mcp__cascade` with:
   - `K=8`
   - `max_tokens=128`
   - `render_mode="verbatim"`
5. If `vimarsa_event = true`, the cascade found a cross-frame analogy; surface it with the diagnostics.
6. Otherwise, report the BMR-winning candidate with `delta_F` and the runner-up.

## Output format

```
Hypothesis: <the BMR winner>
Cross-frame analogy: <if vimarsa fired, name which frames>
Why this is non-obvious: <novelty score, contrast with retrieval set>
Runner-up: <next-best>
```

## Tuning

- Phase-9 calibrated `λ_a = 2.0`, `λ_p = 2.0`. Increase `λ_p` if must_avoid is doing the heavy lifting.
- For mathematical problems, set `top_p = 0.90` and `temperature ≤ 1.0` (the LM shouldn't hallucinate axioms).
