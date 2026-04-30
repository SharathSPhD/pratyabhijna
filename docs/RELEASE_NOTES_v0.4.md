# PCE v0.4.0 — Release Notes

**Date:** 2026-04-30
**Branch merged:** `v0.4-mechanism-study` → `main`
**Tag:** `v0.4.0`
**Frozen archive:** `paper/v0.4/`

## Summary

PCE v0.4 is a pre-registered mechanism study of a recursive self-reflexivity layer for large language models, built on Abhinavagupta's Pratyabhijñā philosophy and a deliberately small subset of active inference. Where v0.3 asked the holistic question — *does the cascade beat the bare model?* — and reported a null, v0.4 takes the next disciplined step: it treats the cascade as a stack of named sub-mechanisms and asks which ones carry the load.

The headline is mixed and honest. The cascade-vs-bare contrasts (H1.v4–H4.v4, pooled in H5.v4) do not move at the pilot's per-domain *n*; the pooled effect is *g* = 0.14 with a 95 % CI that crosses zero. The recursive revision pass is, however, robustly positive on its own (H8a.v4, *g* = 0.65, *p* < 1e-4, *n* = 27), and a learned commit gate outperforms the v0.3 event-driven gate at predicting *when* a revision is worth committing (H8b.v4, F1 0.65 vs 0.52). The proxy scorer disagrees with a calibrated Sonnet-4.5 LLM-judge at ρ = 0.0 (H9.v4) — a methodological flag, not a refutation.

This is the project's first portable release: PCE now ships as a Cursor plugin, a Claude Code plugin, and a standalone `pce` CLI, all driven by the same cascade module. The Anthropic Python SDK code path is removed (ADR-007); the OAuth Claude CLI is the single supported substrate.

## Headline numbers

| Hypothesis | Reading | Statistic |
|---|---|---|
| H8a.v4 — shadow revision > draft (within-cascade) | **supported** | *g* = 0.649, BCa 95 % CI [0.031, 0.095], *p* < 1e-4, *n* = 27 |
| H8b.v4 — learned gate F1 > event gate F1 | **supported** | learned 0.647 vs event 0.516 |
| H8c.v4 — commit-policy leaderboard | leaderboard reported; pairwise gaps not significant after Holm | `always_revise` ▸ `learned_gate` ▸ `event_gated` ▸ `always_draft` |
| H1.v4–H4.v4 — cascade vs bare per domain | **inconclusive** at this *n* | *g* ∈ [−0.32, +0.32]; retrospective power ≤ 0.24 |
| H5.v4 — fixed-effects pool of H1–H4 | **not supported** | pooled *g* = 0.145, CI [−0.255, 0.544] |
| H9.v4 — judge-vs-proxy agreement | **flagged** as a metric-design issue | ρ = 0.0; sign-agreement 56.5 %, *n* = 23 |
| Pilot total cost | — | $13.21 across 1 277 Bedrock calls |

## What changed since v0.3

### Architecture / methodology
- Five-policy commit-policy multiplexer over the cascade arm: `always_draft`, `always_revise`, `event_gated` (the v0.3 policy), `learned_gate` (ADR-002, logistic head), and an analysis-only `oracle` upper bound. H8c reports the leaderboard against bare; H8b reports the gate calibration.
- Fixed-effects H5 pool (ADR-005). Random-effects DerSimonian–Laird is reported as a sensitivity check.
- Honest active-inference accounting (paper §4): FreeEnergyBudget gates revision, cit_temperature drives best-of-K via prompt perturbations, BMR prunes generative-model components — but the OAuth CLI does not expose the sampler so we do not claim full variational inference. Hopfield ālayavijñāna is wired but multi-session dynamics are deferred to v0.5.
- Sonnet-4.5 LLM-judge bridge with a frozen prompt; per-item judge verdicts published to `benchmarks/results_v0.4/judge.jsonl`.
- AWS Bedrock substrate for the Phase 7 pilot (ADR-006); same `claude --print` interface, different profile.

### Plugin portability
- Cursor plugin manifest at `plugin/.cursor-plugin/plugin.json` mirrors the Claude Code manifest (same MCP tools, slash commands, hooks).
- Standalone `pce` CLI (`src/pce/cli.py`) with `cascade`, `judge-pair`, `smoke`, `config show`, and `showcase generate` subcommands; wired into `pyproject.toml` as `pce` console-script.
- `PCEConfig` (`src/pce/config.py`) with a 5-layer override chain: defaults ▸ `~/.config/pce/config.toml` ▸ repo `pce.toml` ▸ env vars ▸ CLI flags. Configurable to any Anthropic CLI-addressable model (default `haiku` for cascade, `sonnet` for judge). `PCE_HAIKU_MODEL` retained as a deprecated back-compat alias.
- ADR-007: Anthropic Python SDK code path removed. PCE now declares one supported substrate: `claude --print` over OAuth.

### 9-demo showcase
- `benchmarks/showcase_v0.4/` ships nine creative outputs with full cascade traces:
  - 3 Sanskrit chandas (anuṣṭubh, gāyatrī, indravajrā) — curated reference verses validated by `tools/sanskrit_chandas.py` (v0.5 swaps in cascade-generated outputs once a chandas-aware scorer is wired).
  - 3 English poetry styles (Dickinson slant, imagist haiku, traditional pastoral) — real Phase 7 cascade traces.
  - 3 scientific creativity prompts (galaxy arms, ice geometry, unreasonable effectiveness) — real Phase 7 cascade traces.
- The Astro v0.4 site renders all 9 demos with the cit → ānanda → icchā → apohana → jñāna → kriyā → vimarśa → revision pipeline, draft / revised diff view, and validator output.

### Paper rewrite
- New title: *Pratyabhijñā × Active Inference, v0.4: A Mechanism Study of Recursive Self-Reflexivity Layers for LLM Creative Cognition*.
- Abstract, §1 introduction, and §2 related work fully rewritten as academic prose. §10 discussion expanded to eight detailed subsections (mechanism reading, per-operator dissection, gate calibration, H9 flag, philosophical underpinnings, compounding work, threats to validity, unmerged-state context).
- New sections: §7b Substrate and Portability, §10b Honest AI Claims, §10c Showcase Examples.
- 19 new verified bibliography entries (active inference, LLM-as-judge, self-refinement, BMR, computational Sanskrit, Pratyabhijñā philosophy, Hopfield networks). Six unverifiable v0.3 entries removed. Bibliography verification log at `audit/v0.4/lit_verification.jsonl`.
- Frozen archive at `paper/v0.4/main.pdf` + `paper/v0.4/sections/*.tex`.

### HTML overhaul
- `presentation/index.html` and the root `index.html` redirect shim are deleted.
- New Astro v0.4 site at `docs/site/`: dark-mode toggle, sidebar nav, Inter / Source Serif 4 / JetBrains Mono pairing, MDX-driven content, Astro components for HypothesisCard, ForestPlot, CommitPolicyBar, JudgeScorerScatter, CostPanel, CitationsList, CascadeTraceViewer, ShowcaseCard, ChandasMeterDisplay, DiffView.
- GitHub Actions Pages workflow at `.github/workflows/pages.yml` deploys the site on push to `main`. Legacy `/presentation/` URLs 301-redirect to the new site root.

## Migration

| from v0.3 | to v0.4 |
|---|---|
| `make benchmark.pilot` | `python -m benchmarks.driver --version v0.4 ...` (or use the Phase 7 audit results published in this release) |
| `presentation/index.html` | <https://sharathsphd.github.io/pratyabhijna/> (Astro v0.4 site) |
| Anthropic Python SDK (`PCE_USE_SDK=1`) | OAuth Claude CLI on `PATH` only (ADR-007); no SDK path |
| `PCE_HAIKU_MODEL` | still works (back-compat); prefer `PCE_CASCADE_MODEL`, `~/.config/pce/config.toml`, or `--model` |
| `paper/main.tex` (v0.3 narrative) | rewritten as v0.4 mechanism study; v0.3 archive at `paper/v0.3/` |

## Acknowledgements

PCE is the second project in an ongoing program that grounds agent design in classical Indian darśana. The first, Pratyākṣa (direct perception / context-discipline), reports a strong Stouffer pooled signal (Z = 9.114) across ten studies on RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding, and SWE-bench Verified. PCE's smaller, more decomposed effect on the recognition + creativity axis is a calibration data-point for the program: creativity is harder to move with this kind of mechanism than hallucination is.

The Phase 7 mechanism pilot was run on AWS Bedrock; thanks to AWS for the credit envelope that let this experiment parallelise across domains.

## Citation

```bibtex
@misc{sathish2026pce_v04,
  author       = {Sathish, S.},
  title        = {Pratyabhij\~n\=a Creative Engine v0.4: A Mechanism Study of Recursive Self-Reflexivity Layers for LLM Creative Cognition},
  year         = 2026,
  url          = {https://sharathsphd.github.io/pratyabhijna/},
  howpublished = {GitHub release \texttt{v0.4.0}},
  note         = {Frozen archive: \texttt{paper/v0.4/}.}
}
```

## The §0.5 unmerged-state context (carried into the public record)

Phase 7 of the v0.4 mechanism study completed on AWS Bedrock on 2026-04-30 with the result tree pushed to `origin/v0.4-mechanism-study`. From that date until Phase 8 landed, the public `main` branch and the GitHub Pages site told the v0.3 story; a reader arriving on May 1 2026 would have seen the v0.3 negative-result summary at the headline level. The branch-only stance was a defensible choice — the v0.4 paper, Astro site, and release notes were not ready, and a premature merge would have surfaced raw stats without their academic interpretation — but it was not costless. The Phase 8 mitigation is not a defensive squash but an explicit acknowledgement, recorded in the paper's §1 introduction, in §10.8 of the discussion, on the [reproducibility page](https://sharathsphd.github.io/pratyabhijna/reproducibility) of the new site, and in the PR body for the v0.4.0 mega-merge. v0.5 introduces a "preliminary results" PR window that lands within 48 hours of pilot completion to prevent the same gap from recurring.

## License

MIT. Substrate models are accessed via the OAuth Claude CLI; no SDK calls.
