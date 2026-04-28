# TRIZ Card C1 — Cost vs. quality of K Haiku candidate calls

## Contradiction

Drawing K candidates from a high-quality LM (Haiku) gives `iccha` the breadth needed for a meaningful `jnana` posterior, but every extra candidate is a fixed dollar charge plus per-call latency. Empirically, the marginal lift per extra candidate falls off after K=3-4 while cost stays linear. We want fan-out without paying for it on every item.

- Improving parameter: **39 — Productivity** (useful creative output per dollar / per second).
- Worsening parameter: **19 — Use of energy by moving object** (runtime $ + latency per cascade run).

## Matrix lookup

`lookup_matrix(39, 19)` -> recommended principles `[35, 10, 38, 19]`.

## Selected principles

### Principle 10 — Preliminary Action (primary)

> Perform all or part of the required change before it is needed so critical paths stay short when demand arrives. Software pattern: pre-fetching, lazy-loading inversion.

PCE mapping: precompute a *first* draft cheaply (single Haiku call at K=1, low-temperature) and only fan out to K=4 candidates if a confidence threshold (`logp` of the draft, or `ananda(draft) < tau_anan`) is missed. The draft becomes both the first candidate and the early-exit signal.

### Principle 35 — Parameter Changes (supporting)

> Throttling and scaling are parameter changes for digital loads.

PCE mapping: per-domain K (`K_poetry_gen=2`, `K_poetry_interp=3`, `K_aut=4`, `K_sci=4`) instead of a global K=4. Domains with high inter-candidate variance get more samples; low-variance domains stay cheap.

### Principle 19 — Periodic Action (supporting)

> Replace continuous operation with pulsed or batched cycles.

PCE mapping: batch the K Haiku calls in parallel via Python `asyncio.gather` rather than sequentially, so wallclock = max(per-call) instead of sum.

## Adopted resolution

- Pilot: K=4 with the simple parallel-batch (Principle 19); the early-exit (Principle 10) and per-domain K (Principle 35) are documented but deferred to v0.3 to keep this pilot's design surface small. The cost ledger captures the actual marginal lift per K, so v0.3 has data to tune from.
- ADR: [docs/adr/v0.2/ADR-001-haiku-substrate.md](../adr/v0.2/ADR-001-haiku-substrate.md).
