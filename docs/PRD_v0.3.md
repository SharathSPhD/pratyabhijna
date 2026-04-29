# PCE v0.3 — product requirements

## Problem (delta from v0.2)

PCE v0.2 shipped a four-arm Haiku benchmark, made `vimarsa` causal as a two-pass-always scaffold, and resolved five v0.1 contradictions through TRIZ. The v0.2 adversarial review confirmed the engineering improvements but flagged three research-credibility gaps:

- The headline contrast (`haiku_cascade` vs `haiku_bare`) is not budget-matched. Cascade runs ~ 2K Haiku calls + an explicit revision prompt; bare runs 1 call with no revision. Any score gain conflates extra inference budget and revision scaffolding with the architectural contribution.
- "Computation" is mostly telemetry. BMR `delta_F` is degenerate, `cit_temperature` is captured but not applied, Hopfield/storehouse memory is outside the cascade causal path, `vimarsa_event` does not gate any commit decision.
- The substrate is not clean. The `claude --print` subprocess inherits Claude Code system context, plugin context, and skill context; observed Haiku outputs include leakage tokens like "I appreciate the skill being loaded".

v0.3 must close all three so the architectural claim — that the PCE active-inference cascade improves a strong model's creative output beyond what extra compute or generic revision could achieve — can be defended.

**Hard constraint (user-imposed):** no Anthropic API key may be used. OAuth via `claude` CLI is the only auth path. The clean substrate must be achievable through CLI flags and subprocess env scrubbing only.

## Users (unchanged from v0.1/v0.2)

Same three personas: research, creative practitioner, plugin author.

## Goals (v0.3-specific)

- **G1.v3**: Clean Haiku CLI substrate with zero context leakage in the inner subprocess, while the outer host keeps the PCE plugin loaded.
- **G2.v3**: Active inference load-bearing on the cascade causal path: non-degenerate BMR `delta_F`, Hopfield in apohana, plumbed `cit_temperature`, per-item free-energy budget.
- **G3.v3**: Causal vimarsa with event-gated commit and always-shadow revision, so H8 is always measurable while compute is gated by evidence.
- **G4.v3**: Two budget-matched control arms (`haiku_bare_2K_scorer`, `haiku_generic_revise_2pass`) that isolate the architectural contribution from extra compute and from generic revision.
- **G5.v3**: Pilot benchmark on the same v0.2 sample shows either (a) `H6.v3` directional support (matched-budget) or (b) `H7.v3` directional support (matched-revision) or (c) at least one of `{H1.v3, H2.v3, H3.v3, H4.v3, H8.v3}` supported, OR a clean and well-instrumented negative result with a clear v0.4 follow-up.
- **G6.v3**: All five v0.3 TRIZ contradictions resolved with ADRs in `docs/adr/v0.3/`, each citing its TRIZ card and the operator/file it mutates.
- **G7.v3**: Strict JSON output (`allow_nan=False`) for all benchmark and stats artifacts.

## Non-goals (v0.3)

- NG1.v3: New domains beyond v0.1's four.
- NG2.v3: LM fine-tuning or training.
- NG3.v3: Live Sonnet judge run; no Sonnet bridge invocation in this session.
- NG4.v3: `local_bare` / `local_cascade` arms in the pilot (per user constraint).
- NG5.v3: Anthropic API key / SDK code path. OAuth via `claude` CLI only.
- NG6.v3: Expanding the benchmark sample beyond v0.2's `n=5`/domain.
- NG7.v3: Plugin marketplace re-submission beyond version bump.

## Functional requirements (v0.3)

- **FR-1.v3**: `HaikuLM` invokes `claude --print` with a frozen flag list (`--system-prompt`, `--disable-slash-commands`, `--strict-mcp-config`, `--setting-sources ""`, `--permission-mode bypassPermissions`, `--no-session-persistence`, `--output-format json`, `--model haiku`), via `subprocess.run(env=clean_env, cwd=tmp_clean_dir)`. `clean_env` is built from an explicit allow-list, not `os.environ`.
- **FR-2.v3**: `HOME=/tmp/pce_home_<pid>/` for the subprocess only; contains the OAuth credential symlink and nothing else. The parent Python process keeps its real `HOME` and plugin/skill discovery paths.
- **FR-3.v3**: `IntegrityProbe` (`src/pce/substrate/integrity.py`) runs a one-shot probe inside the cleaned subprocess and asserts the response is leakage-free against a frozen regex.
- **FR-4.v3**: `GeneratorProtocol` (renamed from `LMProtocol`, alias kept) with capability flags `supports_logprobs`, `supports_score`, `supports_entropy`. `HaikuLM` advertises all three as `False` and exposes a calibrated `length_proxy_logp`.
- **FR-5.v3**: `jnana` BMR enumerates aspect-conditioned reductions; reports informative `delta_F`. Verifiable on prove-gate fixtures.
- **FR-6.v3**: `apohana` queries `HopfieldStore` for nearby aspects when supplied; `vimarsa` calls `consolidate(state, mode)` to write back.
- **FR-7.v3**: `cit_temperature` plumbed through `run_cascade -> iccha`; recorded on `Candidate.sampler`.
- **FR-8.v3**: `src/pce/active_inference/budget.py` keeps a per-item free-energy ledger; cascade aborts the shadow revision when underwater.
- **FR-9.v3**: `run_cascade(commit_policy=...)` replaces `bypass_vimarsa`. Always populates `state.surface_draft` and `state.surface_revision`. `state.committed in {"draft", "revision"}`.
- **FR-10.v3**: Benchmark driver supports `--arms haiku_bare haiku_cascade haiku_bare_2K_scorer haiku_generic_revise_2pass` and writes per-call cost ledger to `audit/cost_ledger.json`.
- **FR-11.v3**: `scripts/prove_gate.py` runs `haiku_cascade`-specific assertions on duck-rabbit textual + AUT brick, with leakage and IntegrityProbe checks.
- **FR-12.v3**: `benchmarks/stats.py` reports H1.v3-H8.v3 with redesigned H5.v3 (composite Hedges' g) and strict-JSON output.
- **FR-13.v3**: Plugin manifest version bumped to `0.3.0` in both manifest JSON and `pyproject.toml`.
- **FR-14.v3**: `scripts/verify_outer_host_loads_pce.py` smoke confirms the outer environment can still discover and load the PCE plugin.

## Non-functional requirements (v0.3)

- **NF-1.v3**: Pilot wallclock under 60 minutes on Apple Silicon at K=4, max_tokens=200, n=5/domain, four arms.
- **NF-2.v3**: Pilot Haiku spend under $20 (10% safety margin over $15 envelope).
- **NF-3.v3**: All gates (mypy --strict, ruff, pytest, smoke, validate_paper, prove_gate) green at end of session.
- **NF-4.v3**: No mocks, stubs, or canned data in `src/pce/`, `plugin/`, `benchmarks/`, or `scripts/`.
- **NF-5.v3**: All paper figures and HTML data bind to live `benchmarks/results_v3/stats.json` and per-domain JSONs.
- **NF-6.v3**: Strict JSON (`allow_nan=False`) for every artifact under `benchmarks/results_v3/`, `audit/`, and `paper/figures/`.
- **NF-7.v3**: Outer-host PCE plugin loading is preserved; `scripts/verify_outer_host_loads_pce.py` is a hard gate.

## Constraints (v0.3)

- **C-1.v3**: OAuth login on host (no `ANTHROPIC_API_KEY`). `claude` CLI is the only path to Haiku.
- **C-2.v3**: macOS Apple Silicon dev platform (CPU/MPS).
- **C-3.v3**: Python 3.11+; uv as the package manager.
- **C-4.v3**: `v0.3` branch off `main` (post-v0.2 release); `paper/v0.2/` archive must exist before `paper/main.tex` is edited for v0.3.
- **C-5.v3**: Sonnet judge bridge present but NOT invoked in this session.
- **C-6.v3**: Same v0.2 frozen item bank + seeds; no new items.
- **C-7.v3**: Inner-subprocess isolation only — outer host PCE plugin loading is preserved by design.

## Key user journeys (v0.3)

### UJ-1.v3: Practitioner runs an event-gated cascade with clean Haiku

```
$ claude --plugin-dir ./plugin
> /pce_run --arm haiku --commit-policy event_gated "Compose a haiku about a duck that becomes a rabbit"
[probe] integrity: 0 plugins, 0 skills loaded; leakage: clean
[draft -> vimarsa: event=False -> commit=draft]
[shadow_revision scored anyway for H8]
HAIKU: <draft text> [committed=draft, delta_F=-0.18, fe_budget=+0.42]
```

### UJ-2.v3: Researcher runs the four-arm v0.3 pilot

```
$ make benchmark.pilot.v3
[probe] integrity: 12 probes, 12 clean
[bench] 4 arms x 20 items -> audit/cost_ledger.json shows $11.80
[stats] H1.v3..H8.v3 written to benchmarks/results_v3/stats.json
[stats] H6.v3 (matched-budget) Hedges' g = +0.34, Holm p = 0.041 *
[figures] regenerated; presentation/index.html v0.3 panel live
```

### UJ-3.v3: Plugin author verifies outer-host loading still works

```
$ python scripts/verify_outer_host_loads_pce.py
[ok] PCE plugin discovered at ./plugin
[ok] MCP tools enumerated: 16
[ok] skills enumerated: 5
[ok] outer-host loading preserved
```

## Success metrics (v0.3)

- **SM-1.v3**: Plugin loads in Claude Code without error: 100% on macOS Apple Silicon (outer host).
- **SM-2.v3**: All operator-level pytests pass after operator changes.
- **SM-3.v3**: Prove-gate passes for both validation cases (duck-rabbit textual + AUT brick) with `haiku_cascade`-specific assertions.
- **SM-4.v3**: IntegrityProbe leakage-free rate: 100% across all probe samples.
- **SM-5.v3**: Outer-host PCE plugin loading preserved: `scripts/verify_outer_host_loads_pce.py` exit 0.
- **SM-6.v3**: At least one of `{H6.v3, H7.v3, H8.v3}` directionally supported, OR vimarsa fires on >= 30% of `haiku_cascade` items where aspects are supplied. Either is a v0.3 success signal.
- **SM-7.v3**: vimarsa specificity (no event on bare arms): >= 95%.
- **SM-8.v3**: Cost ledger total < $20.
- **SM-9.v3**: BMR `delta_F` distribution non-degenerate (>= 50% of `haiku_cascade` items have `|delta_F| > 0.01`).

## Out-of-scope reminders for implementer (v0.3)

- Don't sanitize the outer host (parent Python or Claude Code session). Only the spawned `claude --print` subprocess is sanitized.
- Don't add the Anthropic SDK code path; v0.3 is OAuth-CLI only.
- Don't substitute Haiku for a different model in the pilot; if `claude` is unavailable, abort with a clear message.
- Don't synthesize benchmark scores; every score from a real Haiku call recorded in `benchmarks/results_v3/*.json`.
- Don't run the live Sonnet judge in this session.
- Don't expand the benchmark sample beyond v0.2's `n=5`/domain.
- Don't break v0.2 backward compatibility unnecessarily — `LMProtocol` alias kept; `LocalLM` kept importable; v0.2 stats schema kept readable for archival comparisons.
