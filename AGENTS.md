# AGENTS.md

Guidance for AI agents (Claude Code, Cursor, future maintainers) working in this repo.

This file is read at session start. It complements [CLAUDE.md](CLAUDE.md) with the agent-specific workflows.

## Specialist agent map (per worktree)

| Worktree | Branch | Recommended agents |
|----------|--------|---------------------|
| `wt-engine` | `engine` | `python-pro` for implementation; `ce-correctness-reviewer`, `ce-performance-reviewer`, `ce-kieran-python-reviewer` for review |
| `wt-plugin` | `plugin` | `plugin-architect` for scaffolding; `agent-sdk-verifier-py` for SDK conformance; `ce-cli-readiness-reviewer`, `ce-api-contract-reviewer` for review |
| `wt-bench` | `bench` | `data-scientist` for benchmark logic; `ce-data-migration-expert` for results-schema reviews; `ce-correctness-reviewer` |
| `wt-paper` | `paper` | `docs-architect` for the preprint; `ce-coherence-reviewer`, `ce-feasibility-reviewer`, `ce-product-lens-reviewer` |
| Always-on | — | `ce-project-standards-reviewer` (audits against this CLAUDE.md / AGENTS.md) |

## Skills routinely useful here

* `superpowers/test-driven-development` for Phase 5 operator-by-operator TDD.
* `superpowers/using-git-worktrees` for Phase 4.
* `superpowers/dispatching-parallel-agents` for Phase 1 research and Phase 5 multi-operator parallel TDD.
* `superpowers/verification-before-completion` before claiming any phase done.
* `compound-engineering/ce-code-review` before pushing to `main`.
* `compound-engineering/ce-debug` if the engine misbehaves.
* `huggingface-skills/hf-cli` for HF model and dataset operations.
* `python-development/python-testing-patterns` and `python-design-patterns` for Phase 5.

## Plugins routinely loaded in Claude Code for this project

* `ralph-loop@claude-plugins-official` — the spine that enforces completion promises (already installed).
* `attractor-flow@SharathSPhD` — used as a *dev aid* to monitor engine state during Phase 6 refinement (NOT a runtime dependency of PCE).
* `triz-engine@SharathSPhD` (or wherever it resolves) — used as a *dev aid* for inventive-principle ideation during operator design.
* `huggingface-skills` — already authenticated as `qbz506`; covers HF dataset/model interactions.

## Anti-patterns to refuse

* Adding `unittest.mock` or `MagicMock` to `src/pce/` or `plugin/`.
* Hard-coding benchmark numbers in `paper/*.tex` or `presentation/index.html`.
* Editing `~/.cursor/plans/pratyabhijna_creative_engine_572f1aca.plan.md`.
* Skipping a gate ("just run the next phase, the gate will pass eventually") — do not bypass the gate.
* Substituting a smaller / random-init model when a HF download is blocked.
* Padding `audit/phase9/calls.jsonl` with synthetic rows to clear the n>=15 row-count check.

## Required reading before working in this repo

1. `research1.md` — original research vector.
2. `docs/research-extended.md` — Phase 1 grounding.
3. `docs/SPEC.md` — H1-H6 hypotheses and component contracts.
4. `docs/operator-spec.md` — operator type signatures and invariants.
5. `docs/ADR-001..004*.md` — substrate, vimarsa, BMR, stats decisions.
6. `docs/COMPLETION_PROMISES.md` — per-phase contract.

## Pull-request workflow

* Branch off `main` only via `git worktree add` (Phase 4).
* Each operator gets one commit on `engine` once its tests are green.
* Push after every commit; the gate on the receiving end (`scripts/verify_remote_pushed.py`) is sensitive to unpushed local state.
* PR description: copy the relevant section of `docs/operator-spec.md` and the test results.
