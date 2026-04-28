# Pratyabhijñā Creative Engine (PCE)

A Claude Code plugin that operationalises Abhinavagupta's Pratyabhijñā five-*śakti* generative cascade as typed operators over an active-inference / Bayesian Model Reduction substrate, with a recursive *vimarśa* self-reflexivity layer.

> **v0.2 in flight.** v0.1 shipped a directional null (see [docs/AS_SHIPPED_v0.1.md](docs/AS_SHIPPED_v0.1.md) and [docs/reviews/2026-04-28-adversarial-plugin-review.md](docs/reviews/2026-04-28-adversarial-plugin-review.md)). v0.2 makes `vimarśa` causal (two-pass-always revision), adds Haiku as a first-class generative substrate (apples-to-apples ablation), and runs a four-arm pilot. Frozen scope: [docs/SPEC_v0.2.md](docs/SPEC_v0.2.md), [docs/PRD_v0.2.md](docs/PRD_v0.2.md). v0.1 paper preserved at `paper/v0.1/`.

> **Research vector.** Pairing Pratyabhijñā's architecture with active-inference mathematics, with an explicit *vimarśa* meta-module that detects aspect-shifts (Wittgenstein). The gap that visible repositories in the space (`attractor-flow`, `pramana`) imply but do not yet contain.

This repository holds the engine, the plugin wrapper (**17 MCP tools, 5 skills, 5 agents, 5 slash commands, 3 hooks** — v0.2 adds `pce.pce_cascade(arm=…)` and `pce.haiku_bare`), the benchmark harness with paired statistics, the HTML presentation, and the arxiv preprint.

* Formal operator specification (v0.1): [`docs/SPEC.md`](docs/SPEC.md). v0.2 contract: [`docs/SPEC_v0.2.md`](docs/SPEC_v0.2.md).
* Completion contracts (v0.1): [`docs/COMPLETION_PROMISES.md`](docs/COMPLETION_PROMISES.md). v0.2: [`docs/COMPLETION_PROMISES_v0.2.md`](docs/COMPLETION_PROMISES_v0.2.md).
* Architecture decision records: [`docs/`](docs/) (v0.1) + [`docs/adr/v0.2/`](docs/adr/v0.2/) (v0.2).
* Plugin manifest: [`plugin/.claude-plugin/plugin.json`](plugin/.claude-plugin/plugin.json) (v0.2.0).
* Pre-registered hypotheses (H1–H6 v0.1, H1.v2–H8.v2 v0.2): [`docs/SPEC.md#2-hypotheses-pre-registered`](docs/SPEC.md), [`docs/SPEC_v0.2.md#2-hypotheses-pre-registered-for-v02`](docs/SPEC_v0.2.md).

## Quickstart

```bash
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv venv && uv sync --extra dev
uv run python scripts/verify_real_model.py     # downloads HF models (no mocks)
make smoke                                     # plugin manifest + in-process MCP smoke

# v0.2 prove-gate (single-case validation before pilot):
uv run python scripts/prove_gate.py --strict

# v0.2 pilot (~$15 Haiku envelope; ~50 min on Apple Silicon):
make benchmark.pilot                           # 3-arm pilot writes benchmarks/results_v2/<domain>.json
make stats.pilot                               # haiku_cascade vs haiku_bare contrast, H1-H6, BCa, Holm
uv run python benchmarks/figures.py --results-dir benchmarks/results_v2
uv run python benchmarks/autoreport.py --stats benchmarks/results_v2/stats.json --paper-dir paper

# v0.1 (preserved for the v0.1 paper / HTML; 70-100 min on Apple Silicon):
make bench && make stats && make figures && make autoreport
```

The v0.2 pilot runs three arms per item (the four-arm matrix's `local_cascade` is deferred to v0.3 due to throughput on the pilot host):

| arm | model | PCE? |
|---|---|---|
| `haiku_bare` | Anthropic Claude Haiku (via Claude Code CLI, wrapped by `HaikuLM`) | no — **v0.2 primary control** |
| `haiku_cascade` | Anthropic Claude Haiku via two-pass-always PCE cascade (K=3) | yes — **v0.2 primary treatment** |
| `local_bare` | Qwen2-1.5B-Instruct (raw `cit` only) | no — substrate baseline |

Outputs land in `benchmarks/results_v2/<domain>.json` (resumable). The statistics produce `benchmarks/results_v2/stats.json` with the v0.2 contrast set: `primary` (`haiku_cascade` vs `haiku_bare`), `local_ablation` (empty in pilot), `substrate_baseline` (`haiku_bare` vs `local_bare`), and the six hypotheses H1–H6.

## Engine

```
cit  →  iccha (×K)  →  apohana  →  jnana (BMR ΔF)  →  kriya  →  surface
                                          │
                                       vimarsa
                                          │
                       (re-enter cascade if event)
```

The Hopfield-attractor *ālayavijñāna* (storehouse) supports SWS abstraction (`consolidate_sws`) and REM-style replay (`consolidate_rem`); both are exposed as MCP tools.

## Plugin surface

* **Skills** (5): `pce-poetry-generation`, `pce-poetry-interpretation`, `pce-divergent-thinking`, `pce-scientific-creativity`, `pce-vimarsa-self-reflection`.
* **Agents** (5): `pce-poet`, `pce-interpreter`, `pce-ideator`, `pce-scientist`, `pce-vimarsa-auditor`.
* **Slash commands** (5): `/pce-compose`, `/pce-interpret`, `/pce-aut`, `/pce-bbh`, `/pce-trace`.
* **Hooks** (3): `SessionStart`, `PreToolUse` (audit-stamp every PCE MCP call), `PostToolUse` (consolidation tick).
* **MCP tools** (17): `pce.cit`, `pce.ananda`, `pce.iccha`, `pce.apohana`, `pce.jnana`, `pce.kriya`, `pce.vimarsa`, `pce.cascade`, `pce.embed`, `pce.lm.generate`, `pce.lm.entropy`, `pce.store.add`, `pce.store.recall`, `pce.store.consolidate_sws`, `pce.store.consolidate_rem`, plus v0.2 additions `pce.pce_cascade(arm=local|haiku)` and `pce.haiku_bare`.

## Hypotheses (pre-registered)

| H | claim | domain | direction | α | power |
|---|---|---|---|---|---|
| H1 | PCE > Haiku on AUT (CreativityPrism) | aut | one-sided | 0.05 | 0.80 |
| H2 | PCE > Haiku on Wittgenstein aspect-shift | poetry_interp | one-sided | 0.05 | 0.80 |
| H3 | PCE > Haiku on POEMetric poetry-gen composite | poetry_gen | one-sided | 0.05 | 0.80 |
| H4 | PCE > Haiku on BBH-style sci-creativity | sci_creativity | one-sided | 0.05 | 0.80 |
| H5 | aggregate composite z-blend > 0 | aggregate | one-sided | 0.05 | 0.85 |
| H6 | within-PCE: vimarśa-fired > vimarśa-not-fired | within-PCE | one-sided | 0.05 | 0.70 |

Statistical protocol: paired permutation (sign-flip; exact for n ≤ 18, MC otherwise), Hedges' *g* with small-sample correction, BCa bootstrap 95% CI (10k resamples), Wilcoxon signed-rank (one-sided), Holm-Bonferroni across {H1..H4}, a-priori (g=0.5) + retrospective (observed g) power. Negative-result obligation: any rejected hypothesis is reported in the abstract.

### Headline result — v0.2 pilot (apples-to-apples, n_paired = 19)

**Sign reversal vs v0.1.** The v0.2 four-arm design swaps the broken v0.1 contrast (which conflated cascade contribution with substrate gap) for the apples-to-apples `haiku_cascade` vs `haiku_bare` test. Three of four pre-registered hypotheses (H1, H2, H4) show effect sizes from medium to very large with 95% BCa CIs strictly above zero. *No* hypothesis crosses the strict pre-registered Holm-adjusted *p* < 0.05 threshold — this is a power constraint of the pilot (with n=5 per domain the exact sign-flip permutation floor is *p* = 0.0312 and Holm-Bonferroni with m=4 floors the smallest possible adjusted *p* at 0.125). A properly-powered run at n ≈ 20/domain is the v0.3 next step.

| H | n | Δ (haiku_cascade − haiku_bare) | g | 95% BCa CI | perm p | Holm p | BCa CI > 0 |
|---|---|---|---|---|---|---|---|
| H1 (AUT) | 5 | +0.016 | **+1.26** | [+0.009, +0.024] | 0.0312 | 0.125 | **yes** |
| H2 (poetry_interp) | 5 | +0.165 | **+2.12** | [+0.106, +0.207] | 0.0312 | 0.125 | **yes** |
| H3 (poetry_gen) | 5 | +0.024 | +0.30 | [−0.035, +0.068] | 0.250 | 0.250 | no |
| H4 (sci_creativity) | 4 | +0.053 | **+0.76** | [+0.017, +0.104] | 0.125 | 0.250 | **yes** |
| H5 (z-blend) | 19 | +0.000 | 0.00 | [−0.114, +0.121] | 0.502 | — | no |
| H6 (vimarśa fired vs not, haiku_cascade) | 9/5 | +0.031 | +0.39 | [−0.038, +0.099] | 0.303 | — | no |

**Operator now causal.** v0.1 saw `vimarśa` fire on 0/30 cascade trials; v0.2's two-pass-always cascade (ADR-003) sees it fire on 9/14 `haiku_cascade` trials, and the within-cascade fired-vs-not contrast is directionally positive at the pilot's *n*. All numbers are auto-generated from `benchmarks/results_v2/stats.json` via `benchmarks/autoreport.py`. Cost: $3.60 over 136 Haiku calls (well under the $18 envelope).

The full v0.1 → v0.2 narrative — TRIZ contradictions, ADRs, prove-gate, and the four-arm pilot — is in [`paper/main.tex`](paper/main.tex) and the [HTML presentation](presentation/index.html). The v0.1 paper is preserved at [`paper/v0.1/`](paper/v0.1/) for reference.

## Reproducibility

Every numerical claim in the paper / presentation traces back to JSON artefacts under `audit/` and `benchmarks/results_v2/` (v0.2) or `benchmarks/results/` (v0.1). The driver writes a checkpoint after every call and the audit log records the model checksum, git SHA, seed, and wall-clock per call. v0.2 additionally writes per-Haiku-call records under `audit/haiku/` and a persistent cost ledger at `audit/cost_ledger.json`. `PCE_DEVICE`, `PCE_DTYPE` env vars override the local-LM device autodetection (CUDA → MPS → CPU); `PCE_HAIKU_CLI` and `PCE_HAIKU_MODEL` configure the Haiku CLI binding.

## License

MIT. Substrate models are downloaded from HuggingFace under their respective licences (Apache-2.0 for Qwen2-1.5B-Instruct, MIT for sentence-transformers/all-MiniLM-L6-v2).
