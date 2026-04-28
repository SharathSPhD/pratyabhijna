# PCE v0.1 — As Shipped (post-mortem)

This document captures what `pratyabhijna-creative-engine` v0.1 actually shipped, as opposed to what `docs/SPEC.md` and `docs/PRD.md` v0.1 said it would ship. It is the single source of truth for the *gap* the v0.2 cycle is closing. Source: [docs/reviews/2026-04-28-adversarial-plugin-review.md](reviews/2026-04-28-adversarial-plugin-review.md).

## What shipped

- 7-operator typed cascade (`cit -> iccha -> apohana -> ananda -> jnana -> kriya -> vimarsa`).
- 15-tool MCP server, 5 skills, 5 agents, 5 commands, 3 hooks; all reachable from `claude -p --plugin-dir ./plugin`.
- Hopfield ālayavijñāna store with SWS k-means and REM Metropolis consolidation, exposed as MCP tools but **not part of the benchmark causal path**.
- Local LM substrate: `Qwen/Qwen2-1.5B-Instruct` via `transformers`, with auto-detected device (`mps`/`cuda`/`cpu`) + dtype (`float16`/`float32`).
- Sentence-transformers embedder (`all-MiniLM-L6-v2`).
- Three-arm benchmark driver: `claude_haiku`, `local_bare`, `local_cascade`.
- Per-domain JSON results (`benchmarks/results/*.json`), `stats.json`, regenerable figures, autoreport-substituted LaTeX paper, single-page HTML presentation.
- Pipeline tests (`pytest tests/`), strict mypy + ruff, plugin smoke + verify scripts.

## What did not ship as the SPEC promised

| SPEC claim | Reality | Severity |
|---|---|---|
| H1-H4 framed as "PCE-Haiku vs no-PCE Haiku" | Driver runs `local_cascade` (Qwen2-1.5B + PCE) vs `claude_haiku` and vs `local_bare`. Apples-to-oranges substrate confound. | P0 |
| `vimarsa_event=True` on >= 9/10 duck-rabbit textual probes | `vimarsa_event` was 0/30 across the committed benchmark rows. | P0 |
| `vimarsa` is a recursive self-reflexivity layer | `vimarsa` only writes telemetry; it does not change the surface text. Even if it fired, it would not affect the score. | P0 |
| Sonnet/Opus judging | Local-only proxy scoring (lexical diversity, embedding cosines, keyword overlap, length). | P1 |
| `cascade_run`, `op_*`, `bench_score`, `audit_log` MCP tool names | Server exposes `cit`, `iccha`, `apohana`, `ananda`, `jnana`, `kriya`, `vimarsa`, `cascade`, `hopfield_*`, `consolidate_*`, `report`, `reset_state`. | P1 |
| `bypass_vimarsa=True` flag for the no-PCE control | No flag in `run_cascade`; the bare arm uses `lm.generate` directly. | P1 |
| Plugin runtime matches benchmark runtime | `plugin/.mcp.json` hard-pins `PCE_LM_DEVICE=cpu` and `PCE_LM_DTYPE=float32`; benchmark autodetects `mps`/`float16`. | P1 |
| H5 over n=70 paired observations | n=38 paired observations actually shipped (12+10+8+8). | P1 |
| Holm-Bonferroni success on H1+H2+H5 | All hypotheses unsupported (`holm_p=1.0` for the four primary hypotheses; H5 `permutation_p_one_sided≈0.50`). Honest negative result. | P0 |
| `cit_temperature` plumbed into the cascade | Public parameter accepted by `run_cascade` but never passed into `iccha`/`cit`. | P2 |

## Root causes identified by adversarial review

1. **vimarsa structurally blocked**: cascade passes a one-point trajectory `[(e_iccha, e_apoha)]` to `vimarsa`, which requires `switching >= 2`. One point has zero transitions.
2. **vimarsa non-causal**: even when it would fire, surface text is already finalised; `state.surface` is not revised.
3. **Aspect-empty domains (poetry_gen, aut)**: `aspects=[]` makes `aspect_count=0`, triggering `aspect_ok=False` regardless of surface quality.
4. **jnana clips negative apoha**: `np.clip(apoha, 0.0, None)` discards the contrastive penalty for must-avoid neighbourhoods.
5. **Prompt + sampler asymmetry**: `iccha` appends `"Write a response that is <constraint>."` and starts at `tau=0.40`, while `local_bare` uses raw prompt + `tau=0.9`. Not prompt-matched.
6. **Token truncation**: `max_tokens=120` is too tight for paragraphs/8-item lists; both local arms truncate often.
7. **Substrate gap dominates Haiku contrast**: Qwen2-1.5B vs Haiku is unequal hardware, not architecture.
8. **Live runtime config mismatch**: `.mcp.json` forces CPU/float32; benchmark used MPS/float16. Real Claude Code users get the slower runtime.

## What v0.2 must do (frozen scope from planning rounds 1-3)

1. **Substrate (round 1)**: hybrid four-arm benchmark — `local_bare`, `local_cascade`, `haiku_bare`, `haiku_cascade`. Apples-to-apples primary contrast = `haiku_cascade` vs `haiku_bare`.
2. **vimarsa (round 1)**: two-pass-always. Every cascade item runs draft -> vimarsa -> revision. Hard cap = 1 revision. `state.surface = revision`.
3. **Validation cases (round 2)**: duck-rabbit textual probe + AUT "brick" must pass a strict prove-gate before any benchmark.
4. **Pilot budget (round 2)**: ~$15. n=20-30 paired (5-8 per domain), K=4, embedding-proxy judging only.
5. **Prepared offline script (round 2)**: `scripts/run_judge_bridge.py` for ~$100 / n=40-50 / 30-pair Sonnet judge bridge. Built and dry-run-tested in this session, NOT invoked against real Sonnet.
6. **TRIZ five-pack (round 3)**: cost-vs-quality of K Haiku calls, coverage-vs-novelty in apohana, reflection-vs-speed of two-pass vimarsa, substrate-strength-vs-cascade-overhead, determinism-vs-creativity sampler asymmetry.
7. **Branch (round 3)**: `v0.2` branch in-place, plugin `0.1.0 -> 0.2.0`, `paper/v0.1/` archived before edits, main ships v0.2 only after final gate.
8. **Judge model in offline script (round 3)**: Sonnet only.

## What stays out of scope in v0.2

- No new domains beyond the four already in v0.1 (`poetry_gen`, `poetry_interp`, `aut`, `sci_creativity`).
- No retraining or fine-tuning of any LM; Hopfield ālayavijñāna stays as-is.
- No live Sonnet judge run in this session (only the prepared script + dry-run).
- No new plugin marketplace re-submission beyond version bump.
- The v0.1 negative-result narrative is preserved as motivation; not rewritten.
