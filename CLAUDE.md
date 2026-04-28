# CLAUDE.md — guidance for Claude Code working in this repo

## Project identity

This repository is the Pratyabhijna Creative Engine (PCE): a Claude Code plugin that operationalizes the Pratyabhijna 5-shakti generative cascade as typed Python operators over an active-inference / Bayesian-Model-Reduction substrate, with a recursive *vimarsa* aspect-shift detector. The plugin is benchmarked against a no-plugin Haiku baseline across 4 creativity domains.

## Hard rules (non-negotiable)

* **No stubs, no shortcuts, no mocks, no made-up results.** Every operator computes; every test asserts on real numerical behaviour; every benchmark score traces to a real Claude CLI call. The four gate scripts in `scripts/` enforce this.
* **No file edits to `~/.cursor/plans/pratyabhijna_creative_engine_572f1aca.plan.md`.** That is the upstream plan, not a workspace file.
* **Push to remote after every phase completion.** `bash scripts/ralph_promise_gate.sh <N>` must return 0 with `audit/phase<N>/promise.json` showing all-green; only then is the phase done.
* **Don't substitute models.** If a HuggingFace download is gated/blocked, stop and surface to the user. Do not fall back to a smaller / random-init / different model.
* **Don't synthesize benchmark scores or numbers in the paper / HTML.** Every numeric that appears in `paper/*.tex` or `presentation/index.html` must trace via `\srcfile{}` or `data-trace="path#pointer"` to a JSON file under `benchmarks/results/` or `audit/`.

## Where things live

* `src/pce/` — engine. Operators in `operators/`, substrate adapters in `substrate/`, sleep/consolidation in `consolidation/`, cascade orchestrator at the top level.
* `plugin/` — Claude Code plugin manifest, MCP server, skills, agents, commands, hooks.
* `benchmarks/` — domains, runner, stats, judge prompts, results.
* `paper/` — LaTeX preprint following `~/Library/CloudStorage/OneDrive-Personal/wsl_projects/context/paper/main.tex` layout.
* `presentation/index.html` — single-page HTML with Tailwind CDN + Chart.js.
* `docs/` — SPEC, PRD, plan, ADRs, research-extended, operator-spec, COMPLETION_PROMISES.
* `scripts/` — gate scripts and orchestrator.
* `audit/phase<N>/` — per-phase audit logs (JSONL); `promise.json` is the per-phase done-marker.

## Conventions

* Python 3.11+, `uv` for env management, `mypy --strict`, `ruff` for linting.
* All public APIs in `src/pce/` typed with `from __future__ import annotations` and explicit numpy dtypes.
* All operators are pure functions over typed `dataclass(frozen=True)` state; side-effects (HF downloads, audit writes) live behind explicit `Substrate` Protocols.
* Pytest test files mirror module paths: `src/pce/operators/cit.py` → `tests/operators/test_cit.py`.
* No inline imports inside functions (top-of-file only) — see workspace rule `no-inline-imports.mdc`.
* Exhaustive switch on enum / Literal types per workspace rule `typescript-exhaustive-switch.mdc` (the analogous Python pattern is `match/case` with a final `case _: assert_never(...)`).

## Build / test

```bash
uv sync --extra dev
uv run pytest -q                       # all tests
uv run pytest -q -m "not real_model"   # skip slow real-model tests
uv run mypy src/pce scripts            # type-check
uv run ruff check .                     # lint
bash scripts/ralph_promise_gate.sh 5    # phase-5 promise gate
```

## When implementing a new operator (Phase 5)

1. Read the operator's section in `docs/operator-spec.md`.
2. Write `tests/operators/test_<name>.py` first (red).
3. Implement `src/pce/operators/<name>.py` until tests go green.
4. Run `bash scripts/ralph_promise_gate.sh 5 --allow-dirty` for in-flight diagnostics.
5. When all 7 operators are green, commit, push to `engine` branch, drop `--allow-dirty`, and the gate accepts the promise.

## When the gate refuses a phase

`audit/phase<N>/<gate_name>.json` and `<gate_name>.stderr` show why. Common causes:

* `anti_stub`: a `# TODO`, a stub body, a `NotImplementedError`, a `unittest.mock` import, an operator without a paired test.
* `verify_real_model`: HF cache size mismatch (re-download), or `var(logits) < 1e-6` (model loaded but degenerate — likely loaded weights wrong).
* `verify_artifact`: a phase-required path missing, a placeholder string, a duplicate row in `audit/phase9/calls.jsonl`, a `data-trace` that doesn't resolve.
* `verify_remote_pushed`: HEAD not pushed yet, or working tree dirty in non-`audit/` paths.

Fix the cause; do not paper-over. The gate is the contract.

## When in doubt, prefer

* Real models over substituted ones.
* Real Claude CLI calls over hand-typed example outputs.
* Permutation tests over t-tests.
* `gammaln`-based log-Beta computations over direct Gamma evaluation.
* Frozen dataclasses over mutable dicts.
* Explicit `Substrate` Protocol parameters over hidden module-level singletons.
