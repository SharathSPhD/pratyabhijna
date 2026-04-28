# PCE v0.2 — product requirements

## Problem (delta from v0.1)

PCE v0.1 shipped a typed cascade and an honest negative result, but the central architectural claim — that a recursive `vimarsa` self-reflexivity layer can lift LLM creativity — was never tested causally:

- The `vimarsa` operator was structurally blocked from firing in the cascade.
- Even when it fired, it did not change the surface text.
- The benchmark compared a 1.5B local LM through PCE to a much stronger Haiku, conflating substrate with architecture.
- Live plugin runtime was hard-pinned to CPU/float32 while the benchmark used MPS/float16.

v0.2 must close these gaps so the architectural question can be answered apples-to-apples.

## Users (unchanged)

Same three personas as v0.1 PRD: research, creative practitioner, plugin author.

## Goals (v0.2-specific)

- **G1.v2**: Pluggable LM substrate (`LMProtocol`) with two implementations (`LocalLM`, `HaikuLM`).
- **G2.v2**: Causal two-pass-always `vimarsa` revision loop. The cascade's surface output equals the revision (or the draft when `bypass_vimarsa=True`).
- **G3.v2**: Four-arm benchmark (`local_bare`, `local_cascade`, `haiku_bare`, `haiku_cascade`) with primary apples-to-apples contrast `haiku_cascade` vs `haiku_bare`.
- **G4.v2**: Pilot benchmark (~$15) showing the v0.2 cascade either lifts Haiku or, if not, reports a clean and well-instrumented negative result with a clear v0.3 follow-up.
- **G5.v2**: Prepared (dry-run-tested) `scripts/run_judge_bridge.py` for offline ~$100 30-pair Sonnet judge bridge.
- **G6.v2**: All five TRIZ contradictions resolved with ADRs in `docs/adr/v0.2/`.

## Non-goals (v0.2)

- NG1.v2: New domains beyond v0.1's four.
- NG2.v2: LM fine-tuning or training.
- NG3.v2: Live Sonnet judge run in this session.
- NG4.v2: New Hopfield consolidation features in the benchmark causal path.
- NG5.v2: Plugin marketplace re-submission beyond version bump.

## Functional requirements (v0.2)

- **FR-1.v2**: `LMProtocol` with two implementations; `Candidate` payload identical across both.
- **FR-2.v2**: `run_cascade` is two-pass-always by default; `bypass_vimarsa=True` disables the revision and returns the draft.
- **FR-3.v2**: Cascade returns `surface_draft`, `surface_revision`, `vimarsa_event_draft`, `vimarsa_event_revision`, plus `revision_delta_score` (filled by the benchmark scorer).
- **FR-4.v2**: `iccha._build_prompt(prompt_mode="verbatim")` does not append a constraint suffix; this is the default in the cascade.
- **FR-5.v2**: `jnana` no longer clips negative apoha; v0.1 tests that depend on the old behavior continue to pass when called with `lambda_p=0`.
- **FR-6.v2**: `vimarsa` no longer requires `switching >= 2` for one-point trajectories; uses domain-driven aspect threshold.
- **FR-7.v2**: Benchmark driver supports `--arms haiku_bare haiku_cascade` and writes a per-call cost ledger to `audit/cost_ledger.json`.
- **FR-8.v2**: `scripts/prove_gate.py` runs all four arms on duck-rabbit textual + AUT brick and asserts the v0.2 contract.
- **FR-9.v2**: `scripts/run_judge_bridge.py` defaults to `--dry-run`; requires `--live` and `ANTHROPIC_API_KEY` for real Sonnet calls.
- **FR-10.v2**: `plugin/.mcp.json` does not hard-pin device or dtype.
- **FR-11.v2**: Plugin manifest version bumped to `0.2.0` in both manifest JSON and `pyproject.toml`.

## Non-functional requirements (v0.2)

- **NF-1.v2**: Pilot wallclock under 60 minutes on Apple Silicon at K=4, max_tokens=200, n>=20 paired.
- **NF-2.v2**: Pilot Haiku spend under $20 (10% safety margin over $15 envelope).
- **NF-3.v2**: All gates (mypy --strict, ruff, pytest, smoke, validate_paper, prove_gate) green at end of session.
- **NF-4.v2**: No mocks, stubs, or canned data in `src/pce/`, `plugin/`, `benchmarks/`, or `scripts/`.
- **NF-5.v2**: All paper figures and HTML data bind to live `benchmarks/results/stats.json` and per-domain JSONs.

## Constraints (v0.2)

- **C-1.v2**: Claude Pro / `ANTHROPIC_API_KEY` for Haiku; rate-limit-respecting retry mandated.
- **C-2.v2**: macOS Apple Silicon dev platform (CPU/MPS).
- **C-3.v2**: Python 3.11+; uv as the package manager.
- **C-4.v2**: Worktree-isolated `v0.2` branch; `paper/v0.1/` archive must exist before `paper/main.tex` is edited.
- **C-5.v2**: Sonnet judge bridge built but not invoked in this session; documented for offline run.

## Key user journeys (v0.2)

### UJ-1.v2: Practitioner runs a Haiku-substrate cascade

```
$ claude --plugin-dir ./plugin
> /pce_run --arm haiku "Compose a haiku about a duck that becomes a rabbit"
[draft -> vimarsa: event=False, missing_aspects=[duck->rabbit perceptual flip]]
[revision_brief: "name the perceptual flip"]
[revision -> vimarsa: event=True, novelty=0.51]
HAIKU: <revision text>
```

### UJ-2.v2: Researcher runs the four-arm pilot

```
$ make benchmark.pilot
[bench] 4 arms x 21 items (12 local, 9 haiku-paid) -> audit/cost_ledger.json shows $13.42
[stats] H1.v2..H8.v2 written to benchmarks/results/stats.json
[figures] regenerated; presentation/index.html v0.2 panel live
```

### UJ-3.v2: Researcher runs the prepared Sonnet judge bridge offline

```
$ ANTHROPIC_API_KEY=sk-... python scripts/run_judge_bridge.py --live --pairs 30
[judge] estimated cost $98.40; proceed? --yes assumed via --live
[done] audit/judge/sonnet_30pair.jsonl + stats_with_judge.json + cohen_kappa.json
```

## Success metrics (v0.2)

- **SM-1.v2**: Plugin loads in Claude Code without error: 100% on macOS Apple Silicon.
- **SM-2.v2**: All operator-level pytests pass after operator changes.
- **SM-3.v2**: Prove-gate passes for both validation cases (duck-rabbit textual + AUT brick).
- **SM-4.v2**: At least one of `{H1.v2, H2.v2, H8.v2}` directionally supported, OR vimarsa fires on >=30% of `haiku_cascade` items where aspects are supplied. Either is a v0.2 success signal.
- **SM-5.v2**: vimarsa specificity (no event on bare arms): >=95%.
- **SM-6.v2**: Cost ledger total < $20.
- **SM-7.v2**: Sonnet judge bridge dry-run succeeds; documented run instructions present.

## Out-of-scope reminders for implementer (v0.2)

- Don't introduce new operators outside the seven specified — extensions go through ADRs in `docs/adr/v0.2/`.
- Don't substitute Haiku for a different model in the pilot; if `claude` is unavailable, abort with a clear message rather than fall back silently.
- Don't synthesise benchmark scores; every score from a real Haiku or local LM call recorded in `benchmarks/results/*.json`.
- Don't run the live Sonnet judge in this session.
- Don't rewrite the v0.1 negative-result narrative; preserve it as motivation in the v0.2 paper.
