# PCE v0.2.0 — Haiku substrate, causal vimarśa, TRIZ-resolved cascade

## Headline

The v0.2 pilot reverses the v0.1 sign on every primary hypothesis on the
apples-to-apples `haiku_cascade` vs `haiku_bare` contrast: H1, H2, H3, and
H4 are all directionally positive (Hedges' g ∈ [+0.30, +2.12]) and three
of four BCa CIs are strictly above zero (H1, H2, H4). No hypothesis
crosses the strict pre-registered Holm-adjusted p<0.05 threshold — this
is a power constraint of the pilot (n=5/domain; the exact sign-flip
permutation floor is p=0.0312 and Holm-Bonferroni with m=4 floors the
smallest possible adjusted p at 0.125). A properly-powered run at n≈20
per domain is the v0.3 next step. `vimarśa` now fires on 9/14
`haiku_cascade` trials (vs 0/30 in v0.1) — the operator is causally
active, not just post-hoc telemetry.

## What's new

- **Two-pass-always cascade.** `cascade.py` now performs draft → vimarsa
  brief → revision on every cascade item. The output text in cascade arms
  is the revision, not the draft. `bypass_vimarsa` is preserved for the
  ablation arm. (ADR-003.)
- **Pluggable LM substrate.** New `LMProtocol` abstraction
  (`src/pce/substrate/lm_protocol.py`); `LocalLM` refactored to it and a
  new `HaikuLM` (`src/pce/substrate/haiku_lm.py`) wraps the `claude` CLI
  with cost telemetry, audit logs, retry on empty CLI responses, budget
  enforcement, and an optional Anthropic SDK path. (ADR-001 / ADR-004.)
- **Signed apohana.** Removed the `np.clip(apoha, 0.0, None)` in
  `jnana`; introduced `_shift_apoha` so negative apohana actively
  penalises must-avoid neighborhoods rather than just discounting them.
  (ADR-002.)
- **Prompt and sampler parity.** `iccha` now defaults to
  `prompt_mode="verbatim"` and `sampler_grid_mode="parity"` inside the
  cascade, so the bare and cascade arms cannot be told apart on prompt
  or sampler. (ADR-005.)
- **vimarśa firing fix.** `min_evidence_points=1`,
  `aspect_threshold=1`, and a switching gate that returns N/A on
  one-shot trajectories — the v0.1 thresholds gated out every cascade
  item by construction. (ADR-003.)
- **MCP surface.** Two new tools: `pce.pce_cascade(arm="local"|"haiku")`
  for substrate-agnostic cascade calls and `pce.haiku_bare` for direct
  Haiku calls. The `cascade` tool keeps backward-compatible defaults.
  Total: 17 MCP tools (15 v0.1 + 2 v0.2).
- **Four-arm benchmark matrix.** `benchmarks/driver.py` now supports
  `local_bare`, `local_cascade`, `haiku_bare`, `haiku_cascade` arms with
  a shared cost ledger and parity-mode samplers. New `--pilot` preset
  and `make benchmark.pilot` / `make stats.pilot` targets.
- **Stats payload v0.2.** `benchmarks/stats.py` reports three named
  contrasts side-by-side (`primary`, `local_ablation`,
  `substrate_baseline`) and per-cascade-arm H6 (`H6_haiku_cascade`,
  `H6_local_cascade`).
- **Sonnet judge bridge (prepared).** `scripts/run_judge_bridge.py`
  ships dry-run-tested with a no-key clean exit and full Cohen's-kappa
  pipeline, ready for the offline ~$100 Sonnet bridge run. See
  [`docs/HOWTO_JUDGE.md`](HOWTO_JUDGE.md).

## Documentation

- New: [`docs/SPEC_v0.2.md`](SPEC_v0.2.md), [`docs/PRD_v0.2.md`](PRD_v0.2.md),
  [`docs/COMPLETION_PROMISES_v0.2.md`](COMPLETION_PROMISES_v0.2.md),
  [`docs/AS_SHIPPED_v0.1.md`](AS_SHIPPED_v0.1.md),
  [`docs/HOWTO_JUDGE.md`](HOWTO_JUDGE.md).
- New: five TRIZ contradiction cards under [`docs/triz/`](triz/) and
  five ADRs under [`docs/adr/v0.2/`](adr/v0.2/). Each ADR is end-to-end
  traceable: TRIZ card → ADR → operator code → unit test → pilot result.
- New: prove-gate report at
  [`docs/reviews/2026-04-28-prove-gate.md`](reviews/2026-04-28-prove-gate.md).
- v0.1 paper preserved at [`paper/v0.1/`](../paper/v0.1/) before any
  v0.2 paper edits.

## Pilot summary

- Run: 49 minutes on a single Apple-Silicon laptop.
- Cost: $3.60 over 136 Haiku calls (cap $18, well under).
- Arms run: `local_bare`, `haiku_bare`, `haiku_cascade`. The
  `local_cascade` ablation was deferred to v0.3 due to throughput on
  the pilot host at the chosen K and max_tokens.
- N: 5 items per domain (4 in sci_creativity for `haiku_cascade` after
  one item failed). N_paired = 19.

## Verification

- pytest: 82/82 pass.
- mypy --strict (src/pce + scripts): 32 source files, 0 errors.
- ruff: clean.
- validate_paper: clean.
- verify_plugin: 17/17 tools, 5/5 skills, 5/5 agents, 5/5 commands,
  3/3 hooks.
- smoke_plugin: 16/16 pass (Haiku tests skipped by default; --with-haiku
  available for an explicit live run).
- prove-gate replay: passed=True on both `duck_rabbit_textual` and
  `aut_brick`, $3.97 over 150 Haiku calls.

## TRIZ → ADR → code → result trace

| Contradiction | TRIZ params | ADR | Code change | v0.2 result |
|---|---|---|---|---|
| C1 cost-vs-quality of K Haiku calls | 39 vs 19 | ADR-001 | parallel batching of K candidates | $3.60 / 136 calls in pilot |
| C2 coverage-vs-novelty in apohana  | 27 vs 35 | ADR-002 | signed apohana via `_shift_apoha`, `normalize` flag | jnana penalises must-avoid neighborhoods |
| C3 reflection-vs-speed of two-pass vimarsa | 27 vs 9 | ADR-003 | two-pass-always cascade, revision cap=1, min_evidence=1 | vimarsa fires 9/14 (vs 0/30 in v0.1) |
| C4 substrate-strength-vs-cascade-overhead | 21 vs 36 | ADR-004 | pluggable LMProtocol, HaikuLM | apples-to-apples haiku_cascade vs haiku_bare contrast |
| C5 determinism-vs-creativity sampler asymmetry | 28 vs 35 | ADR-005 | iccha verbatim+parity defaults | bare and cascade arms see identical prompts/samplers |

## Known v0.3 work

- Run the full `local_cascade` ablation (deferred from pilot due to
  local-LM wall-clock).
- Run the prepared Sonnet judge bridge against the live endpoint
  (~$100, n=30 paired).
- Run a properly-powered n≈20/domain pilot to clear the strict Holm
  p<0.05 bar.
- Add per-domain K (early-exit confidence threshold) per ADR-001's
  deferred follow-up.

## Citation

If you use PCE v0.2, please cite the v0.2 abstract and the v0.1 paper
together — the v0.2 pilot supersedes the v0.1 negative result on the
apples-to-apples contrast but the v0.1 paper preserves the original
adversarial-review motivation.
