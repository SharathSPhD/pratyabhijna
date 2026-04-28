<!-- placeholder-policy: allow -->
# Per-phase completion promises (v0.2)

Each v0.2 phase carries a ralph-loop completion promise; the phase only closes when its gate test passes. Cap = 3 retries per phase, then the loop escalates to the user with a structured failure report.

## Gate stack (in order)

1. `scripts/anti_stub_check.py` — code honesty (no stubs / mocks / TODO in `src/pce`, `plugin/`, `scripts/`, `benchmarks/`).
2. `scripts/verify_real_model.py` — substrate honesty (`Qwen/Qwen2-1.5B-Instruct` actually loadable + non-degenerate logits).
3. `scripts/verify_artifact.py` — output honesty (every artifact required by the phase exists, validates schema, has no placeholders).
4. `scripts/verify_remote_pushed.py` — provenance honesty (local HEAD pushed to `SharathSPhD/pratyabhijna` on `v0.2`).

A red on any step rejects the phase promise; ralph-loop's re-injection prompt names the failing gate and the specific failing artifact.

## v0.2 phase contracts

| Phase | Promise string | Branch | Required artifacts |
|------:|----------------|--------|--------------------|
| v0.2-0 | `PCE_V02_PHASE_0_SCAFFOLD_COMPLETE` | `v0.2` | `paper/v0.1/` archive (full tree), `plugin/.claude-plugin/plugin.json` v0.2.0, `pyproject.toml` v0.2.0, `docs/SPEC_v0.2.md`, `docs/PRD_v0.2.md`, `docs/AS_SHIPPED_v0.1.md`, refreshed `docs/plan.md`, `docs/COMPLETION_PROMISES_v0.2.md` |
| v0.2-1 | `PCE_V02_PHASE_1_TRIZ_COMPLETE` | `v0.2` | `docs/triz/` with five contradiction cards (cost-vs-quality, coverage-vs-novelty, reflection-vs-speed, substrate-vs-overhead, determinism-vs-creativity); `docs/adr/v0.2/` with ADR-001..005 each citing its TRIZ card |
| v0.2-2 | `PCE_V02_PHASE_2_HAIKU_LM_COMPLETE` | `v0.2` | `src/pce/substrate/lm_protocol.py`, `src/pce/substrate/haiku_lm.py`, refactored `src/pce/substrate/lm.py`, unit tests in `tests/substrate/`, one duck-rabbit probe through `HaikuLM` returning non-empty text, `audit/cost_ledger.json` schema present |
| v0.2-3 | `PCE_V02_PHASE_3_CAUSAL_VIMARSA_COMPLETE` | `v0.2` | rewritten `src/pce/cascade.py` (two-pass-always), updated `src/pce/operators/{vimarsa,jnana,iccha,apohana}.py`, `tests/cascade_two_pass_test.py`, mypy --strict + ruff + pytest green |
| v0.2-4 | `PCE_V02_PHASE_4_PROVE_GATE_COMPLETE` | `v0.2` | `tests/fixtures/{duck_rabbit_textual,aut_brick}.json`, `scripts/prove_gate.py`, `audit/prove_gate/<case>/<arm>/`, `docs/reviews/2026-04-28-prove-gate.md` |
| v0.2-5 | `PCE_V02_PHASE_5_PLUGIN_REFRESH_COMPLETE` | `v0.2` | updated `plugin/.mcp.json` (no device pin), `plugin/mcp/server.py` with `pce_cascade(arm=...)`, smoke + verify scripts pass with v0.2 manifest |
| v0.2-6 | `PCE_V02_PHASE_6_PILOT_BENCH_COMPLETE` | `v0.2` | `benchmarks/results/{poetry_gen,poetry_interp,aut,sci_creativity}.json` with all four arms, `benchmarks/results/stats.json` with H1.v2-H8.v2, `audit/cost_ledger.json` total < $20 |
| v0.2-7 | `PCE_V02_PHASE_7_REPORT_COMPLETE` | `v0.2` | regenerated `paper/figures/`, `paper/main.tex` v0.2 sections, `presentation/index.html` v0.2 panel, README headline block bound to v0.2 stats |
| v0.2-8 | `PCE_V02_PHASE_8_JUDGE_BRIDGE_COMPLETE` | `v0.2` | `scripts/run_judge_bridge.py` with `--dry-run` mode green on synthetic data, `docs/HOWTO_JUDGE.md`, dry-run output at `audit/judge/sonnet_30pair_DRY.jsonl` |
| v0.2-Final | `PCE_V02_PROJECT_COMPLETE` | `main` | full QA green, `v0.2` merged to `main`, tag `v0.2.0` pushed |

## Anti-substitution policy (v0.2)

- If `claude` CLI is unavailable for a Haiku arm, stop and surface; do not silently fall back to local-only arms.
- If `ANTHROPIC_API_KEY` is unset for the judge bridge, exit cleanly with a usage banner; do not run with a mock judge.
- If `Qwen/Qwen2-1.5B-Instruct` is gated/blocked at HuggingFace, stop and surface; do not substitute a different model.
- The pilot must complete with real Haiku calls; if cost ledger crosses $18, abort gracefully (cap with safety margin under $20).
