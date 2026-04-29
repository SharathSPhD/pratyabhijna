# PCE v0.4 — product requirements

## Problem (delta from v0.3)

PCE v0.3 made the substrate clean and the four-arm Haiku benchmark fair, but the pilot was directional-negative on every primary contrast. The [v0.3 adversarial review](reviews/2026-04-29-adversarial-v0.3-review.md) reframed the failure: the cascade's **shadow revisions beat their drafts on 15/20 items (mean Δ = +0.0458)**, but the event gate committed only 3 of them; three "active-inference" mechanisms (FE budget, `cit_temperature`, `vimarsa.switching_ok`) were also audit-only on the Haiku CLI path. v0.4 closes those gaps and runs a focused mechanism study (Experiments A + B + C from the review) so the project can answer two questions cleanly:

1. Does the PCE shadow revision generator have value independent of the gate? (H8a.v4)
2. Can a calibrated commit policy convert that latent value into observed gain? (H8b.v4 / H8c.v4)

**Hard constraint (user-imposed):** OAuth via `claude` CLI remains the only auth path. v0.4 makes `cit_temperature` causal under that constraint via best-of-K candidate width inside `iccha`.

## Users (unchanged from v0.1/v0.2/v0.3)

Same three personas: research, creative practitioner, plugin author.

## Goals (v0.4-specific)

- **G1.v4** — Substrate hardening: `FreeEnergyBudget.should_continue_revision()` is causally active in `run_cascade`; `cit_temperature` is causally active for Haiku CLI via best-of-K width; `HaikuRateLimitError` is surfaced cleanly.
- **G2.v4** — Commit-policy infrastructure: pluggable `CommitPolicy` (`AlwaysDraft`, `AlwaysRevise`, `EventGated`, `LearnedGate`, `OracleCommit`) with leakage-controlled `LearnedGate` training.
- **G3.v4** — Statistics rebuild: H8 split into H8a (shadow revision value), H8b (gate calibration), H8c (commit-policy comparison); H1.v4–H7.v4 retained for backward comparability with v0.3; H9.v4 added for proxy–judge agreement.
- **G4.v4** — Sonnet LLM-judge bridge re-enabled on a stratified 32-item subset.
- **G5.v4** — Powered pilot with **n = 20 / domain** under a **$30 cost cap**; cascade arm multiplexes commit policies on the same artifacts (no extra spend).
- **G6.v4** — All four v0.4 TRIZ contradictions resolved with ADRs in `docs/adr/v0.4/`.
- **G7.v4** — Strict JSON output (`allow_nan=False`) for all benchmark and stats artifacts.
- **G8.v4** — Paper, HTML, README updated with honest active-inference language and the H8a/H8b/H8c result panel.

## Non-goals (v0.4)

- **NG1.v4** — No new domains beyond v0.3's four.
- **NG2.v4** — No human rater study. Construct validity is local proxy + frozen Sonnet LLM-judge only.
- **NG3.v4** — No SDK / API-key path. Substrate stays OAuth-only.
- **NG4.v4** — No Mechanism Ablation pack (Experiment D from the review). Documented as v0.5 scope.
- **NG5.v4** — No powered narrow-domain replication (Experiment F). Documented as v0.5 scope.
- **NG6.v4** — No new local model substrate, no fine-tuning.
- **NG7.v4** — No expansion of scoring composites. `benchmarks/scoring.py` is unchanged.
- **NG8.v4** — No plugin marketplace re-submission beyond the version bump.

## Functional requirements (v0.4)

- **FR-1.v4** — `FreeEnergyBudget.should_continue_revision()` is consulted by `run_cascade` before the shadow-revision pass; when underwater, the cascade emits `revision_skipped_reason="fe_budget_underwater"` and commits the draft.
- **FR-2.v4** — `iccha.generate_candidates` sets `K_runtime = clip(round(K_eff * (0.5 + 1.5 * cit_temperature)), K_min, K_max)`; each candidate uses a deterministic prompt-level perturbation drawn from the seed; `Candidate.sampler` records `cit_temperature`, `K_eff`, `K_runtime`, and the perturbation index.
- **FR-3.v4** — `HaikuLM._call_cli_once` parses `claude --print` stdout JSON even when `rc != 0`; raises `HaikuRateLimitError` when `api_error_status == 429`; surfaces `api_error_status` and `result` into the audit trace.
- **FR-4.v4** — `vimarsa.commit_decision(...)` receives a real `iccha_apoha_trajectory` from `run_cascade`; items without aspect dictionaries still pass via the existing fallback.
- **FR-5.v4** — `src/pce/policies/commit.py` defines `CommitPolicy` ABC with five implementations: `AlwaysDraft`, `AlwaysRevise`, `EventGated`, `LearnedGate`, `OracleCommit`.
- **FR-6.v4** — `LearnedGate` is a scikit-learn `LogisticRegression` over features `[delta_F, novelty, aspect_count, ananda, budget_balance]`; trained by `scripts/train_learned_gate.py` on `audit/v0_3_traces/*.jsonl` with leave-one-domain-out CV; serialized to `artifacts/learned_gate_v0_4.joblib`.
- **FR-7.v4** — `run_cascade(commit_policy: Literal["event_gated","always_draft","always_revise","learned_gate"]="event_gated")` selects the policy at runtime.
- **FR-8.v4** — `benchmarks/driver.py` runs the cascade arm once and rescores with all four commit policies on the same artifacts; per-policy results recorded in `benchmarks/results_v0.4/poetry_gen.json` etc.
- **FR-9.v4** — `benchmarks/stats.py` emits H1.v4..H9.v4 with strict-JSON output; H5.v4 uses fixed-effects pooling (SPEC and code agree).
- **FR-10.v4** — `scripts/run_judge_bridge.py` is re-enabled with a frozen Sonnet prompt (sha256 versioned), pairwise A/B with random position swap, ties allowed; outputs `benchmarks/results_v0.4/judge.jsonl` and `benchmarks/results_v0.4/judge_agreement.json`.
- **FR-11.v4** — Plugin manifest version bumped `0.3.0 → 0.4.0` in `plugin/.claude-plugin/plugin.json` and `pyproject.toml`; `pce_cascade` MCP tool exposes the `commit_policy` enum; new MCP tool `haiku_judge(...)`.
- **FR-12.v4** — `scripts/smoke_plugin.py --with-haiku` includes a `learned_gate` divergence assertion and a one-call judge round-trip.
- **FR-13.v4** — `scripts/prove_gate.py` extended with v0.4 assertions: budget-starved fixture observes ≥ 1 `fe_budget_underwater` event; `learned_gate` differs from `event_gated` on at least one fixture item; n-gram entropy probe at `cit_temperature=0.9` > 0.2.
- **FR-14.v4** — `paper/sections/10_discussion.tex` and `paper/sections/09_results.tex` rewritten with honest active-inference language; H8a/H8b/H8c panel added to the results section.

## Non-functional requirements (v0.4)

- **NF-1.v4** — Pilot wallclock under 90 minutes on Apple Silicon at K=4, max_tokens=200, n=20/domain, four base arms (cascade multiplex adds no Haiku calls).
- **NF-2.v4** — Pilot Haiku + Sonnet spend ≤ $30 (cost cap).
- **NF-3.v4** — All gates (mypy --strict, ruff, pytest, smoke, validate_paper, prove_gate) green at end of session.
- **NF-4.v4** — No mocks, stubs, or canned data in `src/pce/`, `plugin/`, `benchmarks/`, or `scripts/`.
- **NF-5.v4** — Paper figures and HTML data bind to live `benchmarks/results_v0.4/stats.json` and per-domain JSONs (cache-busting already in place).
- **NF-6.v4** — Strict JSON (`allow_nan=False`) for every artifact under `benchmarks/results_v0.4/`, `audit/`, and `paper/figures/`.
- **NF-7.v4** — Outer-host PCE plugin loading is preserved; `scripts/verify_outer_host_loads_pce.py` is a hard gate.
- **NF-8.v4** — `LearnedGate` artifact `artifacts/learned_gate_v0_4.joblib` ≤ 1 MB and committed to the repo.

## Constraints (v0.4)

- **C-1.v4** — OAuth login on host (no `ANTHROPIC_API_KEY`). `claude` CLI is the only path to Haiku.
- **C-2.v4** — macOS Apple Silicon dev platform (CPU/MPS).
- **C-3.v4** — Python 3.11+; uv as the package manager.
- **C-4.v4** — `v0.4-mechanism-study` branch off `main` (post-v0.3 release tag); `paper/v0.3/` archive exists before `paper/main.tex` is edited for v0.4.
- **C-5.v4** — Sonnet judge bridge runs on a 32-item stratified subset only; total Sonnet cost ≤ $5.
- **C-6.v4** — Same v0.2/v0.3 frozen item bank for the first 5/domain; the remaining 15/domain are deterministically generated from `benchmarks/items.py` with seed=4242.
- **C-7.v4** — Inner-subprocess isolation only — outer host PCE plugin loading is preserved by design.

## Key user journeys (v0.4)

### UJ-1.v4 — Practitioner picks a commit policy

```
$ claude --plugin-dir ./plugin
> /pce_run --arm haiku --commit-policy learned_gate "Compose a haiku about a duck that becomes a rabbit"
[probe] integrity: 0 plugins, 0 skills loaded; leakage: clean
[draft -> vimarsa: event=False, learned_gate=True (p=0.71) -> commit=revision]
HAIKU: <revision text> [committed=revision, learned_gate p=0.71, fe_budget=+0.42]
```

### UJ-2.v4 — Researcher runs the v0.4 powered pilot

```
$ make benchmark.pilot.v4
[probe] integrity: 16 probes, 16 clean
[bench] 4 arms x 80 items -> audit/cost_ledger_v0_4.json shows $24.10
[bench] cascade multiplexed across {event_gated, always_draft, always_revise, learned_gate}
[judge] sonnet bridge on 32 stratified items -> judge_agreement.json
[stats] H1.v4..H9.v4 written to benchmarks/results_v0.4/stats.json
[stats] H8a.v4 shadow-revision Hedges' g = +0.31, p = 0.018 *
[stats] H8c.v4 learned_gate vs event_gated g = +0.22, p = 0.041 *
```

### UJ-3.v4 — Plugin author exercises the new commit_policy enum

```
$ python scripts/smoke_plugin.py --with-haiku
[ok] pce_cascade(commit_policy=event_gated)   ...
[ok] pce_cascade(commit_policy=learned_gate)  ... (differs on 1/3 items)
[ok] haiku_judge(arm_a=draft, arm_b=revision) ...
[ok] 23 / 23 tools pass
```

## Success metrics (v0.4)

- **SM-1.v4** — Plugin loads in Claude Code without error: 100% on macOS Apple Silicon (outer host).
- **SM-2.v4** — All operator-level pytests pass after operator changes.
- **SM-3.v4** — Prove-gate v0.4 passes: FE-budget abort observable; `learned_gate` ≠ `event_gated` on ≥ 1 fixture item; n-gram entropy probe monotonic in `cit_temperature`.
- **SM-4.v4** — IntegrityProbe leakage-free rate: 100%.
- **SM-5.v4** — Outer-host PCE plugin loading preserved.
- **SM-6.v4** — At least one of `{H8a.v4, H8b.v4, H8c.v4}` directionally supported, OR a clean and well-instrumented negative result with a clear v0.5 follow-up.
- **SM-7.v4** — vimarsa specificity (no event on bare arms): ≥ 95%.
- **SM-8.v4** — Cost ledger total ≤ $30.
- **SM-9.v4** — BMR `delta_F` distribution non-degenerate (≥ 50% of `haiku_cascade` items have `|delta_F| > 0.01`).
- **SM-10.v4** — `LearnedGate` leave-one-domain-out CV AUROC > 0.55 on v0.3 traces.
- **SM-11.v4** — Sonnet judge sign-agreement with proxy delta > 0.5 on the 32-item subset (binomial p < 0.05 not required for SM but reported).

## Out-of-scope reminders for implementer (v0.4)

- Don't sanitize the outer host. Only the spawned `claude --print` subprocess is sanitized.
- Don't add the Anthropic SDK code path; v0.4 stays OAuth-CLI only.
- Don't substitute Haiku for a different model in the pilot; if `claude` is unavailable, abort with a clear message.
- Don't synthesize benchmark scores; every score from a real Haiku call recorded in `benchmarks/results_v0.4/*.json`.
- Don't run the live human-rater study in this session.
- Don't expand the benchmark sample beyond v0.4's `n=20/domain`.
- Don't break v0.3 backward compatibility unnecessarily — `LMProtocol` alias kept; `LocalLM` kept importable; `commit_policy` enum extended (not replaced).

## Review-finding → phase mapping

The v0.4 plan addresses the v0.3 adversarial review findings as follows:

| Review finding (severity, file) | v0.4 phase | Resolution |
|---|---|---|
| P0 — Event gate discards useful revisions ([cascade.py](../src/pce/cascade.py), [stats.py](../benchmarks/stats.py)) | Phase 3 + 4 | `LearnedGate` policy + H8a/H8b/H8c split |
| P0 — FE budget not behavior-gating ([budget.py](../src/pce/active_inference/budget.py), [cascade.py](../src/pce/cascade.py)) | Phase 2 | Wire `should_continue_revision()` into `run_cascade` |
| P0 — `cit_temperature` not causal for Haiku CLI ([iccha.py](../src/pce/operators/iccha.py), [haiku_lm.py](../src/pce/substrate/haiku_lm.py)) | Phase 2 | Best-of-K width mechanism inside `iccha` |
| P1 — H8.v3 misses the diagnostic signal ([stats.py](../benchmarks/stats.py)) | Phase 4 | Split into H8a/H8b/H8c |
| P1 — Paper overstates AI completion ([paper/sections/10_discussion.tex](../paper/sections/10_discussion.tex)) | Phase 8 | Honest rewrite |
| P1 — H5 SPEC vs code drift (fixed vs random) ([SPEC_v0.3.md](SPEC_v0.3.md), [stats.py](../benchmarks/stats.py)) | Phase 4 | Lock fixed-effects in SPEC and code |
| P1 — Quota error not surfaced cleanly ([haiku_lm.py](../src/pce/substrate/haiku_lm.py)) | Phase 2 | `HaikuRateLimitError` typed exception |
| P2 — Hopfield mostly cold-start ([apohana.py](../src/pce/operators/apohana.py)) | (Out of scope, v0.5) | Documented as NG; revisit with burn-in design |
| P2 — Discrete MCP tools ≠ full cascade ([plugin/mcp/server.py](../plugin/mcp/server.py)) | Phase 6 | Document `pce_cascade` as research-grade path |
