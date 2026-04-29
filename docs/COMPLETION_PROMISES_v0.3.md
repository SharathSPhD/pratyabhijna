<!-- placeholder-policy: allow -->
# Per-phase completion promises (v0.3)

Each v0.3 phase carries a ralph-loop completion promise; the phase only closes when its gate test passes. Cap = 3 retries per phase, then the loop escalates to the user with a structured failure report.

## Gate stack (in order)

1. `scripts/anti_stub_check.py` — code honesty (no stubs / mocks / TODO in `src/pce`, `plugin/`, `scripts/`, `benchmarks/`).
2. `scripts/verify_outer_host_loads_pce.py` — outer-host honesty (PCE plugin still discoverable from the parent host).
3. `scripts/verify_artifact.py` — output honesty (every artifact required by the phase exists, validates schema, has no placeholders, JSON is strict (`allow_nan=False`)).
4. `scripts/verify_remote_pushed.py` — provenance honesty (local HEAD pushed to `SharathSPhD/pratyabhijna` on `v0.3`).

Inner-substrate honesty (probe + leakage regex) is enforced inside `HaikuLM` itself and is exercised by phase-2's gate.

## Two-tier substrate isolation reminder

- *Outer host*: parent Python process (or Claude Code session). Keeps PCE plugin loaded so `pce_cascade(...)` works at all. NEVER sanitized.
- *Inner subprocess*: each `claude --print` spawned by `HaikuLM`. Sanitized via flags + `subprocess.run(env=clean_env, cwd=tmp_clean_dir)`. IntegrityProbe runs *inside* this subprocess.

The four gates above interpret this distinction correctly: Gate 2 verifies the outer host is healthy; Gate 3 verifies inner-subprocess outputs are leak-free.

## v0.3 phase contracts

| Phase | Promise string | Branch | Required artifacts |
|------:|----------------|--------|--------------------|
| v0.3-0 | `PCE_V03_PHASE_0_SCAFFOLD_COMPLETE` | `v0.3` | `paper/v0.2/` archive (full tree), `plugin/.claude-plugin/plugin.json` v0.3.0, `pyproject.toml` v0.3.0, `docs/SPEC_v0.3.md`, `docs/PRD_v0.3.md`, `docs/COMPLETION_PROMISES_v0.3.md` |
| v0.3-1 | `PCE_V03_PHASE_1_TRIZ_COMPLETE` | `v0.3` | `docs/triz/v0.3/` with five contradiction cards (C1 fairness vs depth, C2 clean substrate vs OAuth, C3 active inference vs CLI black-box, C4 vimarsa as event vs guarantee, C5 memory in cascade vs purity); `docs/adr/v0.3/` with ADR-001..005 each citing its TRIZ card |
| v0.3-2 | `PCE_V03_PHASE_2_CLEAN_SUBSTRATE_COMPLETE` | `v0.3` | rewritten `src/pce/substrate/haiku_lm.py` (clean inner subprocess), new `src/pce/substrate/integrity.py`, broadened `src/pce/substrate/lm_protocol.py` (`GeneratorProtocol` with capability flags), `scripts/verify_outer_host_loads_pce.py`, 50/50 leak-free Haiku subprocess calls recorded, IntegrityProbe 10/10 from fresh subprocesses, mypy --strict + ruff green |
| v0.3-3 | `PCE_V03_PHASE_3_ACTIVE_INFERENCE_COMPLETE` | `v0.3` | rewritten `src/pce/operators/jnana.py` (aspect-conditioned reductions), updated `src/pce/operators/apohana.py` (Hopfield query), updated `src/pce/operators/iccha.py` (cit_temperature plumbed), new `src/pce/active_inference/budget.py`, four new test files green, ΔF non-degenerate on duck-rabbit |
| v0.3-4 | `PCE_V03_PHASE_4_CAUSAL_VIMARSA_COMPLETE` | `v0.3` | rewritten `src/pce/cascade.py` (event-gated commit + always-shadow revision), updated `src/pce/operators/vimarsa.py` (delta_F evidence + consolidate hook), `tests/cascade_event_gated_test.py`, `bypass_vimarsa` removed in favor of `commit_policy` |
| v0.3-5 | `PCE_V03_PHASE_5_PROVE_GATE_COMPLETE` | `v0.3` | `scripts/prove_gate.py` (haiku_cascade-specific assertions + IntegrityProbe + leakage), extended `tests/fixtures/{duck_rabbit_textual,aut_brick}.json` (delta_F_floor + no_leakage signals), `audit/prove_gate/v0_3/` populated, `docs/HOWTO_JUDGE.md` v0.3 addendum |
| v0.3-6 | `PCE_V03_PHASE_6_PLUGIN_REFRESH_COMPLETE` | `v0.3` | updated `plugin/mcp/server.py` (new `pce_cascade(arm=)` enum + `haiku_clean_substrate_probe` + `hopfield_state`), v0.3 manifest, `scripts/smoke_plugin.py --with-haiku` 18/18, trace at `audit/v0_3_smoke_trace.jsonl` |
| v0.3-7 | `PCE_V03_PHASE_7_PILOT_BENCH_COMPLETE` | `v0.3` | `benchmarks/driver.py` (4-arm matrix), `benchmarks/results_v3/{poetry_gen,poetry_interp,aut,sci_creativity}.json`, `benchmarks/results_v3/stats.json` with H1.v3-H8.v3, strict JSON, `audit/cost_ledger.json` total < $20, ΔF distribution non-degenerate |
| v0.3-8 | `PCE_V03_PHASE_8_REPORT_COMPLETE` | `v0.3` | regenerated `paper/figures/`, `paper/main.tex` v0.3 abstract leading with H6.v3/H7.v3, `paper/sections/*.tex` v0.3 methods + results, `presentation/index.html` v0.3 panel, README v0.3 headline block, `docs/RELEASE_NOTES_v0.3.md` |
| v0.3-Final | `PCE_V03_PROJECT_COMPLETE` | `main` | full QA green, `v0.3` merged to `main`, tag `v0.3.0` pushed, GitHub release published |

## Anti-substitution policy (v0.3)

- If `claude` CLI is unavailable, stop and surface; do not silently fall back to local-only arms or to a synthetic substrate.
- If OAuth credentials are missing (clean subprocess returns `401`), exit cleanly with a usage banner; do not run with a mock.
- If IntegrityProbe fails, stop the run and surface; do not record the contaminated outputs as benchmark data.
- The pilot must complete with real Haiku calls; if cost ledger crosses $18, abort gracefully (cap with safety margin under $20).
- The outer host must keep PCE loaded; if `scripts/verify_outer_host_loads_pce.py` fails, fix the regression in the same commit. Do not ship a v0.3 that breaks the very plugin it intends to defend.
- Inner-substrate sanitization MUST NOT mutate `os.environ` or any parent state; only `subprocess.run(env=clean_env, cwd=tmp_clean_dir)` may carry the scrubbed environment.
