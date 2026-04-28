# PCE — implementation plan (snapshot)

This is the in-repo working copy of the implementation plan. The upstream Cursor plan-files are at `~/.cursor/plans/pratyabhijna_creative_engine_572f1aca.plan.md` (v0.1) and `~/.cursor/plans/pce_v0.2_haiku_cascade_*.plan.md` (v0.2) and are not edited from inside this repo.

## v0.1 phases (shipped)

| # | Phase | Outputs | Status |
|---|-------|---------|--------|
| 0 | Spine + connectors + gates | scripts/{anti_stub_check,verify_*,ralph_promise_gate}.* + COMPLETION_PROMISES.md + remote repo | DONE |
| 1 | Research grounding | docs/research-extended.md, docs/operator-spec.md, paper/references.bib seed | DONE |
| 2 | Brainstorm + scaffolding | docs/SPEC.md (with H1-H6), docs/PRD.md, docs/plan.md (this), ADR-001..004, CLAUDE.md, AGENTS.md | DONE |
| 3-5 | Engine TDD | src/pce/operators/* + cascade.py + pytest + mypy-strict + hf_downloads.jsonl | DONE |
| 6 | Refinement | audit/phase6/probes.jsonl with vimarsa events on probes | DONE |
| 7 | Plugin wrap | plugin/.claude-plugin/plugin.json + .mcp.json + marketplace.json + 15 MCP + 5 skills + 5 agents + 5 commands + 3 hooks | DONE |
| 8 | Install + smoke | audit/phase8/smoke.json with non-canned outputs from every plugin component | DONE |
| 9 | Benchmarks | benchmarks/results/{poetry_gen,poetry_interp,aut,sci_creativity}.json + stats.json | DONE (negative result) |
| 10-11 | HTML + paper | presentation/index.html + paper/main.tex with stats-bound figures | DONE |
| F | Final verification | global anti-stub sweep, all audit/phaseN/promise.json, mypy strict + ruff clean | DONE |

v0.1 outcome: directional null on H1-H4 + H5; H6 undefined (vimarsa never fired in committed benchmark rows). Adversarial review documented in [docs/reviews/2026-04-28-adversarial-plugin-review.md](reviews/2026-04-28-adversarial-plugin-review.md). v0.1 paper archived at `paper/v0.1/`.

## v0.2 phases (current)

Frozen scope: see [docs/SPEC_v0.2.md](SPEC_v0.2.md), [docs/PRD_v0.2.md](PRD_v0.2.md), [docs/AS_SHIPPED_v0.1.md](AS_SHIPPED_v0.1.md). Each phase carries a ralph-loop completion promise (cap 3 retries, then escalate).

| # | Phase | Outputs | Status |
|---|-------|---------|--------|
| 0 | Worktree + scope freeze | v0.2 branch, paper/v0.1/ archive, manifest 0.2.0 bump, SPEC_v0.2 + PRD_v0.2 + AS_SHIPPED_v0.1 | IN-PROGRESS |
| 1 | TRIZ five-pack + ADRs | docs/triz/ five contradiction cards, docs/adr/v0.2/ five ADRs | pending |
| 2 | Haiku substrate adapter | src/pce/substrate/lm_protocol.py + haiku_lm.py + cost telemetry + audit logs | pending |
| 3 | Causal vimarsa + operator fixes | cascade.py two-pass-always, vimarsa min_evidence, jnana signed apoha, iccha verbatim mode | pending |
| 4 | Two-case prove gate | scripts/prove_gate.py + duck-rabbit textual + AUT brick + calibration | pending |
| 5 | Plugin MCP refresh | .mcp.json device fix, PCE_HAIKU_* env vars, pce_cascade(arm=...) tool | pending |
| 6 | Pilot benchmark (~$15) | driver.py with haiku_bare/haiku_cascade arms, audit/cost_ledger.json, n>=20 paired | pending |
| 7 | Reporting refresh | regenerate stats/figures/autoreport, paper v0.2 sections, HTML v0.2 panel, README headline | pending |
| 8 | Sonnet judge bridge (prepared) | scripts/run_judge_bridge.py with --dry-run + --live, docs/HOWTO_JUDGE.md | pending |
| 9 | Final verification + push | full QA, push v0.2, merge to main, tag v0.2.0 | pending |

## Worktree → branch → agent stack

v0.1 worktrees `wt-{engine,plugin,bench,paper}` are kept on disk for reference. v0.2 is implemented in-place on the `v0.2` branch (single-session sequential work) and merged to `main` at the final phase.

## Worktree → branch → agent stack

| Worktree | Branch | Agent stack |
|----------|--------|-------------|
| `wt-engine` | `engine` | python-pro + ce-correctness-reviewer + ce-performance-reviewer + ce-kieran-python-reviewer |
| `wt-plugin` | `plugin` | plugin-architect + agent-sdk-verifier-py + ce-cli-readiness-reviewer + ce-api-contract-reviewer |
| `wt-bench` | `bench` | data-scientist + ce-data-migration-expert + ce-correctness-reviewer |
| `wt-paper` | `paper` | docs-architect + ce-coherence-reviewer + ce-feasibility-reviewer + ce-product-lens-reviewer |

## Per-phase ralph-loop completion-promise contract

See [docs/COMPLETION_PROMISES.md](COMPLETION_PROMISES.md). Every phase ends with `bash scripts/ralph_promise_gate.sh <N>` returning 0 and `audit/phase<N>/promise.json` showing all-green; only then is the phase marked done.

## Build / test commands

```bash
# Set up dev env (Phase 3)
uv sync --extra dev

# Run all tests
uv run pytest -q

# Type-check
uv run mypy src/pce scripts

# Lint
uv run ruff check src/pce scripts plugin

# Run a phase gate
bash scripts/ralph_promise_gate.sh <phase>

# Verify a real model
uv run python scripts/verify_real_model.py --model microsoft/Phi-3-mini-4k-instruct

# Run benchmarks
uv run python -m benchmarks.run --domain all --seed 42
```

## Open issues (rolling)

* The duck-rabbit probe corpus needs to be hand-curated (~10 short English-language poems with two clear readings); deferred to Phase 6.
* `pymdp` v1 may not run on macOS Apple Silicon under JAX; ADR-003 specifies fall-back to `pymdp.legacy` if so.
* Sonnet/Opus judge cost during Phase 9; budget recorded in `audit/phase9/budget.json` (run-time addition).
