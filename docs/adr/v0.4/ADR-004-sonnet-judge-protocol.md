# ADR-004 (v0.4) — Sonnet LLM-judge stratified subset protocol

Status: Accepted (frozen at end of Phase 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.4/C4-judge-cost-vs-cap.md](../../triz/v0.4/C4-judge-cost-vs-cap.md).

## Context

The v0.3 review's strongest construct-validity criticism is that local proxy composites may not capture creative nuance the way humans or a frontier judge would. The natural fix is an LLM judge — but the v0.4 cost cap is $30, and Sonnet costs roughly 5× per call vs Haiku. A naive judge stage (every item × every arm × every commit policy) would double the spend.

The C4 TRIZ card resolves the contradiction via principles **26 (Copying)** and **32 (Observability)**: judge a stratified sample, not the full population.

## Decision

Re-enable the existing `scripts/run_judge_bridge.py` (present but disabled in v0.3) under the following protocol:

- **Model**: Anthropic Sonnet via `claude --print --model sonnet` (OAuth-only, no API key — same substrate constraint as Haiku).
- **Frozen prompt**: a single judge prompt template with sha256 versioning. The prompt asks for a single A/B preference plus a one-line rationale. The sha256 is recorded on every judge row.
- **Pairwise A/B with random position swap**: for each judge row, the bridge randomly assigns "A" and "B" to the two compared surfaces; ties are allowed (the judge may answer "tie"). The position assignment is recorded so the analysis can correct for any residual position bias.
- **Stratified subset**: 8 items / domain × 4 domains = **32 items**, drawn from quartiles of the proxy delta (`score(haiku_cascade) - score(haiku_bare)`) on the v0.4 pilot. Stratification ensures the subset spans the proxy delta range, including items the proxy thought cascade lost.
- **Single contrast per item**: judge compares `haiku_cascade.event_gated` vs `haiku_bare`. (Future versions can judge multiple contrasts; v0.4 keeps the contrast minimal to stay under cost cap.)
- **Outputs**: `benchmarks/results_v0.4/judge.jsonl` (one row per judge call) and `benchmarks/results_v0.4/judge_agreement.json` (aggregate agreement metrics).
- **Cost discipline**: a 4-item dry-run is required before the full subset. If the dry-run projects > $5 for the full subset, the bridge aborts and the subset is shrunk.
- **No human raters**: out of scope for v0.4; documented as v0.5 follow-up.

## Pre-registered analysis (H9.v4)

For each of the 32 judge rows:

- `proxy_sign` ∈ {-1, 0, +1} — sign of `score(cascade) - score(bare)` (with `0` only on exact ties).
- `judge_sign` ∈ {-1, 0, +1} — sign of judge preference (cascade > bare = +1, tie = 0, bare > cascade = -1).
- `agree` = 1 if `proxy_sign == judge_sign` else 0.

Aggregate metrics:

- **Sign agreement rate**: `mean(agree)`. Binomial test against 1/2 (excluding ties).
- **Spearman ρ**: between proxy delta magnitude and judge confidence (judge confidence proxied by `judge_sign * (1 - 0.5 * tie_indicator)`).
- **Position bias check**: `mean(judge_sign | position_swap)` should be statistically indistinguishable across swap conditions.

H9.v4 success: sign agreement > 0.5 with binomial p < 0.05 (two-sided). Reported either way.

## Consequences

Positive:

- Adds a second construct signal to the v0.4 mechanism study without exceeding the cost cap.
- The frozen prompt sha256 lets the v0.4 pilot be exactly reproduced post-hoc.
- Sonnet via `claude --print --model sonnet` keeps the OAuth-only substrate honest.

Negative:

- LLM judges are not human raters. The v0.4 paper must say this explicitly.
- 32 items is a small sample. The H9.v4 result will be honest but underpowered for fine-grained correlation claims.
- Sonnet quota exhaustion can interrupt the bridge mid-run. Mitigated by the dry-run + 4-item segment ordering: cascade-vs-bare on each domain first, so partial loss is recoverable.

## Implementation files

- `scripts/judge_subset.py` — v0.4 OAuth-only bridge; reads `benchmarks/results_v0.4/<domain>.json`, draws a quartile-stratified subset, calls Sonnet via `claude --print --model sonnet`, and writes `judge.jsonl` and `judge_agreement.json`. (The legacy v0.2 bridge `scripts/run_judge_bridge.py` is kept for `paper/v0.3/` reproducibility but is *not* used by v0.4.)
- `scripts/judge_prompt_v0_4.txt` — frozen judge prompt; sha256 = `5b39ee653b4aa4fe4d3c007f2f0237b9839975c3347679d8a73a56e16e4ac0d9`. Recorded on every JSONL row and in `judge_agreement.json`.
- `tests/test_judge_bridge_dryrun.py` — 11 tests covering the frozen-prompt sha, position-swap inversion, deterministic dry-run responder, quartile stratification, projected-cost guard, and an end-to-end CLI dry-run that writes a valid 4-row `judge.jsonl` + `judge_agreement.json`.
- `benchmarks/results_v0.4/judge.jsonl`, `benchmarks/results_v0.4/judge_agreement.json` — Phase 7 outputs.

## Acceptance gate (Phase 5)

- `tests/test_judge_bridge_dryrun.py` passes (synthetic 4-item dry-run produces a valid `judge_agreement.json`).
- `scripts/judge_prompt_v0_4.txt` exists, sha256 documented in `benchmarks/results_v0.4/judge_agreement.json`.
- Real-Sonnet dry-run on 4 items completes; projected cost on full 32-item subset ≤ $5.
- Position-bias check returns indistinguishable means on the dry-run subset (sanity check).
