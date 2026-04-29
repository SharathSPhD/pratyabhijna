# C4 — Construct-validity (judge cost) vs cost cap

## Contradiction

The v0.3 review's strongest construct-validity criticism is that local proxy composites may not capture "creative nuance" the way humans or a frontier judge would. The natural fix is a Sonnet LLM-judge stage — but Sonnet costs roughly 5× per call vs Haiku, and v0.4 already operates under a $30 cost cap. A naive judge stage could double the spend.

- **If we run no judge stage** (status quo from v0.3), the construct-validity criticism stands.
- **If we run a full judge stage** (judge every item × every arm × every commit policy), we exceed the cap.

## Improving / worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 28 — Measurement accuracy | Construct-validity: do humans/judges agree with the proxy delta? |
| Worsening | 25 — Loss of time | Total wallclock + total cost (the cap). |

## Matrix lookup

`lookup_matrix(28, 25) -> {26, 32, 28, 18}`.

- **26 — Copying**: judge a sample, not the full population; copying produces representative information cheaper.
- **32 — Color changes / observability**: stratify the sample so a small subset spans the proxy delta range.
- **28 — Replacement of mechanical system**: replace human raters with a frozen LLM judge for the v0.4 mechanism study (humans are v0.5 scope).
- **18 — Mechanical vibration**: alternate judge-A/judge-B positions to remove order bias.

## Ideal Final Result (IFR)

> The judge stage gives a credible second construct signal, agrees with the local proxy where the proxy is right, and disagrees informatively where the proxy is wrong, all at a marginal cost ≤ $5 inside the $30 cap.

## Attractor-flow divergent ideation

1. **Judge every cascade item across all four arms** — exceeds cost cap; *rejected*.
2. **Judge only the cascade arm** — no comparison, no construct signal; *rejected*.
3. **Stratified sample, n = 8 / domain × 4 domains = 32 items**, drawn from quartiles of proxy delta — *kept (primary resolution)*.
4. **Frozen judge prompt with sha256 versioning** — *kept (reproducibility)*.
5. **Pairwise A/B with random position swap, ties allowed** — *kept (anti-position-bias)*.
6. **Bridge dry-run on 4 items required before full subset** — *kept (cost discipline)*.
7. **Sign-agreement + Spearman correlation on the 32-item subset** — *kept (statistic)*.
8. **Human raters** — out of scope for v0.4; documented as v0.5 follow-up.

## Selected resolution

Apply principles **26 (Copying)**, **32 (Observability)** and **28 (Replacement)**:

- Sonnet LLM-judge bridge re-enabled as `scripts/run_judge_bridge.py`.
- Frozen judge prompt versioned by sha256; the prompt asks for a single A/B preference plus a one-line rationale.
- 8 items / domain × 4 domains = 32 items, drawn from quartiles of proxy delta on the cascade vs bare contrast.
- Pairwise A/B with random position swap; ties allowed; the bridge writes `benchmarks/results_v0.4/judge.jsonl`.
- Dry-run of 4 items required before the full subset; if the dry-run projects > $5 for the full subset, abort and shrink the subset.
- H9.v4: Sonnet sign-agreement vs proxy-delta sign on the 32-item subset (binomial test, two-sided), plus Spearman ρ.

Implementation contract: see [ADR-004 — Sonnet judge protocol](../../adr/v0.4/ADR-004-sonnet-judge-protocol.md).
