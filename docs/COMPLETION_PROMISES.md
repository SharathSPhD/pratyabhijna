<!-- placeholder-policy: allow -->
# Per-phase completion promises

Each phase of the PCE project runs inside `/ralph-loop` with the corresponding completion promise below. The Stop hook (see `scripts/ralph_promise_gate.sh`) rejects the promise unless every gate listed for the phase is green; rejection re-injects the same prompt and the loop iterates.

## Gate stack (in order)

1. `scripts/anti_stub_check.py`   - code honesty (no stubs / mocks / TODO in `src/pce`, `plugin/`, `scripts/`)
2. `scripts/verify_real_model.py` - substrate honesty (HF models actually downloaded + non-degenerate logits) [Phase >= 5]
3. `scripts/verify_artifact.py`   - output honesty (every artifact required by the phase exists, validates schema, has no placeholders)
4. `scripts/verify_remote_pushed.py` - provenance honesty (local HEAD pushed to `SharathSPhD/pratyabhijna` on the active branch)

A red on any step rejects the `<promise>`; ralph-loop's re-injection prompt names the failing step.

## Phase contracts

| Phase | Promise string | Branch | Required artifacts |
|------:|----------------|--------|--------------------|
| 0 | `PCE_PHASE_0_SPINE_COMPLETE` | `main` | `scripts/{anti_stub_check,verify_artifact,verify_real_model,verify_remote_pushed,ralph_promise_gate.sh}.py`, `docs/COMPLETION_PROMISES.md`, `pyproject.toml`, `.gitignore`, `README.md` |
| 1 | `PCE_PHASE_1_RESEARCH_COMPLETE` | `main` | `docs/research-extended.md`, `docs/operator-spec.md`, `paper/references.bib` (seed) |
| 2 | `PCE_PHASE_2_SCAFFOLD_COMPLETE` | `main` | `docs/SPEC.md` (with H1-H6), `docs/PRD.md`, `docs/plan.md`, `docs/ADR-001..004*.md`, `CLAUDE.md`, `AGENTS.md` |
| 3 | `PCE_PHASE_3_TREE_COMPLETE` | `main` | `pyproject.toml` resolved, dev tools installed, smoke `python -c "import pce"` passes |
| 4 | `PCE_PHASE_4_WORKTREES_COMPLETE` | `main` | worktrees `wt-{engine,plugin,bench,paper}` exist on disk; remote branches `engine`/`plugin`/`bench`/`paper` exist |
| 5 | `PCE_PHASE_5_ENGINE_COMPLETE` | `engine` | `src/pce/operators/{cit,ananda,iccha,jnana,kriya,apohana,vimarsa}.py`; `src/pce/cascade.py`; `audit/hf_downloads.jsonl`; pytest 100%; mypy --strict clean |
| 6 | `PCE_PHASE_6_TUNED_COMPLETE` | `engine` | `audit/phase6/probes.jsonl` with >=1 vimarsa event on probes and 0 on bypass-control |
| 7 | `PCE_PHASE_7_PLUGIN_COMPLETE` | `plugin` | `plugin/.claude-plugin/plugin.json`, `plugin/.mcp.json`, `plugin/marketplace.json`, all 15 MCP tools wired to engine; FastMCP boot-smoke green |
| 8 | `PCE_PHASE_8_SMOKE_COMPLETE` | `plugin` | `audit/phase8/smoke.json` with non-canned outputs from every tool/skill/agent/command/hook |
| 9 | `PCE_PHASE_9_BENCH_COMPLETE` | `bench` | `audit/phase9/calls.jsonl` with n>=15/domain, no duplicate hashes; `benchmarks/results/{poetry_gen,poetry_interp,aut,sci_creativity}.json`; statistical report |
| 10 | `PCE_PHASE_10_HTML_COMPLETE` | `paper` | `presentation/index.html` with `data-trace` attributes resolving to `benchmarks/results/*.json` |
| 11 | `PCE_PHASE_11_PAPER_COMPLETE` | `paper` | `paper/main.pdf` compiled; `paper/citations.checksum`; every `\srcfile{...}` resolves |
| Final | `PCE_PROJECT_COMPLETE` | `main` | global anti-stub sweep green, all `audit/phase*/promise.json` present, `v0.1.0` release tagged |

## Anti-substitution policy

If a HuggingFace download is gated, rate-limited, or blocked, ralph-loop **stops and surfaces** the failure to the user. It must not substitute a smaller / different / random-init model, and it must not synthesize benchmark scores. The completion promise stays unsatisfied until the real artifact is in place.
