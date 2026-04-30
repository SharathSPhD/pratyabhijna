<!-- placeholder-policy: allow -->
# Per-phase completion promises (v0.4)

Each v0.4 phase carries a ralph-loop completion promise; the phase only closes when its gate test passes. Cap = 3 retries per phase, then the loop escalates to the user with a structured failure report.

## Gate stack (in order, applies to every phase)

1. `scripts/anti_stub_check.py` — code honesty (no stubs / mocks / TODO in `src/pce`, `plugin/`, `scripts/`, `benchmarks/`).
2. `scripts/verify_outer_host_loads_pce.py` — outer-host honesty (PCE plugin still discoverable from the parent host).
3. `scripts/verify_artifact.py` — output honesty (every artifact required by the phase exists, validates schema, has no placeholders, JSON is strict (`allow_nan=False`)).
4. `scripts/verify_remote_pushed.py` — provenance honesty (local HEAD pushed to `SharathSPhD/pratyabhijna` on `v0.4-mechanism-study`).

Inner-substrate honesty (probe + leakage regex) is enforced inside `HaikuLM` itself and is exercised by Phase 2's gate.

## Two-tier substrate isolation reminder

- *Outer host*: parent Python process (or Claude Code session). Keeps PCE plugin loaded so `pce_cascade(...)` works at all. NEVER sanitized.
- *Inner subprocess*: each `claude --print` spawned by `HaikuLM`. Sanitized via flags + `subprocess.run(env=clean_env, cwd=tmp_clean_dir)`. IntegrityProbe runs *inside* this subprocess.

## v0.4 phase contracts

| Phase | Promise string | Branch | Required artifacts |
|------:|----------------|--------|--------------------|
| v0.4-0 | `PCE_V04_PHASE_0_SCAFFOLD_COMPLETE` | `v0.4-mechanism-study` | `paper/v0.3/` archive (full tree), `docs/SPEC_v0.4.md`, `docs/PRD_v0.4.md`, `docs/COMPLETION_PROMISES_v0.4.md`, `docs/triz/v0.4/` and `docs/adr/v0.4/` directories created |
| v0.4-1 | `PCE_V04_PHASE_1_TRIZ_ADR_COMPLETE` | `v0.4-mechanism-study` | `docs/triz/v0.4/` with four contradiction cards (C1 theory-vs-utility, C2 OAuth-vs-cit_temp, C3 budget-as-ledger-vs-authority, C4 judge-cost-vs-cap); `docs/adr/v0.4/` with ADR-001..006 each citing its TRIZ card |
| v0.4-2 | `PCE_V04_PHASE_2_SUBSTRATE_HARDENING_COMPLETE` | `v0.4-mechanism-study` | wired `FreeEnergyBudget.should_continue_revision()` in `src/pce/cascade.py`, `iccha` best-of-K width with `cit_temperature`, `src/pce/substrate/errors.py` with `HaikuRateLimitError`, hardened `_call_cli_once`, `tests/test_fe_budget_gating.py`, `tests/test_cit_temperature_kruntime.py`, `tests/test_haiku_rate_limit_error.py`, mypy --strict + ruff green, prove-gate v0.4-α green |
| v0.4-3 | `PCE_V04_PHASE_3_COMMIT_POLICIES_COMPLETE` | `v0.4-mechanism-study` | `src/pce/policies/commit.py` with five policies, `scripts/train_learned_gate.py`, `artifacts/learned_gate_v0_4.joblib`, `tests/test_commit_policies.py`, `tests/test_learned_gate_training.py`, `tests/test_apoha_trajectory_wiring.py`, `benchmarks/driver.py` cascade-policy multiplex |
| v0.4-4 | `PCE_V04_PHASE_4_STATS_REGISTRATION_COMPLETE` | `v0.4-mechanism-study` | `benchmarks/stats.py` emits H1.v4..H9.v4 on synthetic mock data with `allow_nan=False`; H5.v4 fixed-effects in SPEC and code; pre-registration tag `pce-v0.4-prereg` pushed |
| v0.4-5 | `PCE_V04_PHASE_5_JUDGE_BRIDGE_COMPLETE` | `v0.4-mechanism-study` | re-enabled `scripts/run_judge_bridge.py` with frozen Sonnet prompt (sha256), 4-item dry-run JSON committed, projected cost on full subset ≤ $5 documented |
| v0.4-6 | `PCE_V04_PHASE_6_PLUGIN_REFRESH_COMPLETE` | `v0.4-mechanism-study` | manifest 0.4.0 in `plugin/.claude-plugin/plugin.json` and `pyproject.toml`, `pce_cascade(commit_policy=...)` MCP tool, new `haiku_judge` MCP tool, `scripts/smoke_plugin.py --with-haiku` 23/23 (or fewer with documented 429 surface), `scripts/prove_gate.py` v0.4 assertions green |
| v0.4-7 | `PCE_V04_PHASE_7_PILOT_BENCH_COMPLETE` | `v0.4-mechanism-study` | `benchmarks/results_v0.4/{poetry_gen,poetry_interp,aut,sci_creativity}.json`, `benchmarks/results_v0.4/stats.json` with H1.v4..H9.v4, strict JSON, `audit/cost_ledger_v0_4.json` total ≤ $30, judge agreement on 32-item subset committed |
| v0.4-8 | `PCE_V04_PHASE_8_REPORT_COMPLETE` | `v0.4-mechanism-study` | regenerated `paper/figures/`, `paper/main.tex` v0.4 abstract, `paper/sections/*.tex` v0.4 results + honest AI-claim section, `presentation/index.html` v0.4 panel pointing at `benchmarks/results_v0.4/stats.json`, `README.md` v0.4 headline block, `docs/RELEASE_NOTES_v0.4.md` |
| v0.4-Final | `PCE_V04_PROJECT_COMPLETE` | `main` | full QA green, `v0.4-mechanism-study` merged to `main`, tag `v0.4.0` pushed, GitHub release published |

## Anti-substitution policy (v0.4)

- If `claude` CLI is unavailable, stop and surface; do not silently fall back to local-only arms or to a synthetic substrate.
- If OAuth credentials are missing (clean subprocess returns `401`), exit cleanly with a usage banner; do not run with a mock.
- If IntegrityProbe fails, stop the run and surface; do not record the contaminated outputs as benchmark data.
- If `HaikuRateLimitError` raises mid-pilot, the driver records partial state to `audit/cost_ledger_v0_4.json` and exits 1; do not silently retry past the cost cap.
- The pilot must complete with real Haiku calls; if the cost ledger crosses $28, abort gracefully (cap with safety margin under $30).
- The Sonnet judge bridge dry-run must complete before the full subset; if dry-run projects > $5 for the full subset, abort and reduce subset size.
- The `LearnedGate` model never trains on v0.4 evaluation data; it trains exclusively on `audit/v0_3_traces/`.
- The outer host must keep PCE loaded; if `scripts/verify_outer_host_loads_pce.py` fails, fix the regression in the same commit. Do not ship a v0.4 that breaks the very plugin it intends to defend.
- Inner-substrate sanitization MUST NOT mutate `os.environ` or any parent state; only `subprocess.run(env=clean_env, cwd=tmp_clean_dir)` may carry the scrubbed environment.
