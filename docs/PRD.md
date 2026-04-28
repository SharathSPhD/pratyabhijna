# PCE — product requirements

## Problem

Visible LLMs do not reliably exhibit the *form* of human creativity that is best characterised as Wittgensteinian aspect-shift: the ability to look at the same stimulus and produce qualitatively distinct, coherent readings, especially under constraint, where the *meta-trajectory* of selection between candidate readings matters more than the surface text. Existing chat models can be prompted to enumerate alternatives, but rarely show the recursive self-touching that a human poet, mathematician, or scientist performs when they suddenly reorganize their interpretation of a corpus.

The Pratyabhijñā 5-śakti cascade plus a recursive *vimarśa* layer is a candidate architectural answer that no existing LLM tool exposes as a typed pipeline.

## Users

Three personas:

1. **Researcher (creativity / philosophy of mind)**: wants to probe the operator-grammar of insight, with traceable per-operator audit. Cares about reproducibility, statistical rigor, and the ability to bypass `vimarśa` as a control.
2. **Creative practitioner (poet, designer, scientific writer)**: wants to use the cascade through Claude Code as a brainstorming tool that produces *qualitatively distinct* candidate outputs and an aspect-shift verifier.
3. **Plugin author / future self**: wants the engine to be reusable as a Python library separate from the plugin (`pip install pratyabhijna-creative-engine` import path: `from pce import run_cascade`).

## Goals

* G1: A working, typed Python implementation of the 7-operator cascade with full mypy-strict coverage and pytest invariants.
* G2: A Claude Code plugin that exposes the cascade as 15 MCP tools, 5 skills, 5 agents, 5 commands, 3 hooks.
* G3: A statistically-rigorous A/B comparison of Haiku-with-PCE against Haiku-without-PCE on 4 domains (n_total = 70 paired observations).
* G4: A reproducible, end-to-end audit log: every benchmark response traces to a real Claude CLI call with timestamp + prompt SHA + raw output.
* G5: A self-contained HTML presentation + an arxiv-format preprint, both fully sourced from the real audit log.

## Non-goals

* NG1: Production-grade SaaS deployment, multi-tenant, billing.
* NG2: Native UI beyond Claude Code / Cursor MCP.
* NG3: Languages other than English in this version.
* NG4: Multi-modal (images, audio).
* NG5: Mobile.

## Functional requirements

* FR-1: Engine accepts a `(prompt, constraint)` pair and returns a `CascadeState` containing all per-operator artifacts.
* FR-2: Every operator emits a deterministic-on-seed audit dict.
* FR-3: A single `bypass_vimarsa=True` flag turns the cascade into the no-PCE control (used in benchmarks).
* FR-4: Plugin's `cascade_run` MCP tool is fully wired to the engine; no canned data.
* FR-5: Plugin honours Claude Code's tool-namespacing convention (`pratyabhijna_mcp__<tool>`).
* FR-6: Phase 9 driver runs the n=70 paired A/B and produces 4 per-domain results JSONs + 1 aggregate JSON.
* FR-7: Stats module computes paired permutation, Hedges' g, Wilcoxon, BCa CI, Holm-Bonferroni, retrospective power for each Hi.
* FR-8: HTML presentation renders standalone (no live network deps) and every numeric is traceable via `data-trace="path#pointer"` to a JSON file.
* FR-9: Paper compiles via `pdflatex && bibtex && pdflatex && pdflatex` cleanly with all `\cite{}` resolving.

## Non-functional requirements

* NF-1: Single cascade run wall-clock ≤ 30 s on Apple Silicon CPU at K=8 (no GPU required).
* NF-2: Memory usage during cascade ≤ 8 GB resident (Phi-3-mini-4k fp32 ≈ 7.6 GB; fp16 fallback for low-RAM).
* NF-3: All gates (anti-stub, real-model, artifact, remote-push) green for every phase end.
* NF-4: No mocks / stubs / mock-derived test doubles in `src/pce/` or `plugin/`.
* NF-5: Every external download (HuggingFace) verified via `scripts/verify_real_model.py`.
* NF-6: Every plugin tool exercised via real I/O in Phase 8 smoke test (no canned responses).
* NF-7: Multi-comparison-corrected statistical reporting (Holm-Bonferroni at minimum).

## Constraints

* C-1: Claude Pro subscription with usage limits → benchmark must batch and rate-limit-respect.
* C-2: macOS Apple Silicon as the development platform; no NVIDIA GPU.
* C-3: Python 3.11+; uv as the package manager.
* C-4: All work pushed to `SharathSPhD/pratyabhijna` on GitHub; one branch per worktree (`engine`, `plugin`, `bench`, `paper`).

## Key user journeys

### UJ-1: Researcher runs a controlled cascade

```
researcher$ claude --profile pratyabhijna
> /pratyabhijna_run "Write a haiku about a duck that becomes a rabbit"
[engine: cit τ=0.95 → 8 icchā candidates → apohana scoring vs ['paraphrase', 'rephrase'] → BMR(jñāna) selects #3 ΔF=+1.84 → kriyā polish → vimarsa: event=True novelty=0.47]
HAIKU:  ...
researcher$ /pratyabhijna_run "(same prompt) --bypass-vimarsa"
HAIKU:  ...  (vimarsa: event=False)
```

### UJ-2: Practitioner brainstorms AUT

```
poet$ /pratyabhijna_aut "bottle"
[engine: K=8 icchā continuations → apohana vs ['drink', 'water']]
1. ...
2. ...
[vimarsa: event=True for entries 2, 6, 7 — those introduce a fresh aspect]
```

### UJ-3: Phase-9 benchmark driver

```
researcher$ python -m benchmarks.run --domain poetry_interp --n 20 --seed 42
[60 paired Claude Haiku calls; logs to audit/phase9/calls.jsonl]
[stats: H2 supported (g=0.62, p=0.003 paired-perm, BCa CI [0.18, 0.81])]
```

## Success metrics

* SM-1: Plugin loads in Claude Code without error: 100% on macOS Apple Silicon.
* SM-2: Each operator's pytest invariants pass: 100%.
* SM-3: Phase-9 H1+H2+H5 supported under Holm-Bonferroni: minimum acceptable.
* SM-4: Phase-9 absolute Hedges' g for the aggregate composite ≥ 0.30 (small-medium effect target).
* SM-5: Vimarśa specificity (no event on bypass-control): ≥ 95%.
* SM-6: All HTML chart numerics trace cleanly: 100%.

## Out-of-scope reminders for implementer

* Don't add new operators outside the seven specified — extensions go through ADRs.
* Don't substitute models if HuggingFace is blocked — stop and surface.
* Don't synthesize benchmark scores — every score comes from a real Claude CLI call recorded in `audit/phase9/calls.jsonl`.
* Don't generate plots from synthetic / cached / mocked numbers in the HTML or paper.

## Roadmap (post v0.1.0)

* v0.2: multimodal aspect-shift (image + caption).
* v0.3: multi-language (Sanskrit corpus integration).
* v0.4: production deployment as standalone MCP server.
* v0.5: integration with `pramana` valid-cognition layer.
