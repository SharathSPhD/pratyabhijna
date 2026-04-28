# PCE — implementation plan (snapshot)

This is the in-repo working copy of the implementation plan; the Cursor plan-file at `~/.cursor/plans/pratyabhijna_creative_engine_572f1aca.plan.md` is the upstream and is not edited from inside this repo.

## Phases

| # | Phase | Outputs | Status |
|---|-------|---------|--------|
| 0 | Spine + connectors + gates | scripts/{anti_stub_check,verify_*,ralph_promise_gate}.* + COMPLETION_PROMISES.md + remote repo | DONE (audit/phase0/promise.json) |
| 1 | Research grounding | docs/research-extended.md, docs/operator-spec.md, paper/references.bib seed | DONE (audit/phase1/promise.json) |
| 2 | Brainstorm + scaffolding | docs/SPEC.md (with H1-H6), docs/PRD.md, docs/plan.md (this), ADR-001..004, CLAUDE.md, AGENTS.md | IN-PROGRESS |
| 3 | Project scaffolding | pyproject.toml resolved + dev tools installed + `python -c "import pce"` smoke + plugin/triz-engine + attractor-flow registered as Cursor dev aids | pending |
| 4 | Worktrees | wt-{engine,plugin,bench,paper} on disk; remote branches `engine`, `plugin`, `bench`, `paper` | pending |
| 5 | Engine TDD | src/pce/operators/{cit,ananda,iccha,jnana,kriya,apohana,vimarsa}.py + cascade.py + 100% pytest + mypy-strict + hf_downloads.jsonl | pending |
| 6 | Refinement | audit/phase6/probes.jsonl with ≥1 vimarsa event per non-trivial probe and 0 on bypass-control | pending |
| 7 | Plugin wrap | plugin/.claude-plugin/plugin.json + .mcp.json + marketplace.json + 15 MCP + 5 skills + 5 agents + 5 commands + 3 hooks | pending |
| 8 | Install + smoke | audit/phase8/smoke.json with non-canned outputs from every plugin component | pending |
| 9 | Benchmarks | audit/phase9/calls.jsonl + benchmarks/results/{poetry_gen,poetry_interp,aut,sci_creativity}.json + statistical report | pending |
| 10 | HTML | presentation/index.html with `data-trace` attributes resolving | pending |
| 11 | Paper | paper/main.pdf + citations.checksum | pending |
| F | Final verification | global anti-stub sweep, all audit/phaseN/promise.json present, all pytest pass, plugin smoke pass, paper compiles, HTML renders, plan.md and SPEC.md updated to reflect actual implementation | pending |

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
