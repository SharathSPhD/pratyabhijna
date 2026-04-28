# Pratyabhijñā Creative Engine (PCE)

A Claude Code plugin that operationalises Abhinavagupta's Pratyabhijñā five-*śakti* generative cascade as typed operators over an active-inference / Bayesian Model Reduction substrate, with a recursive *vimarśa* self-reflexivity layer.

> **Research vector.** Pairing Pratyabhijñā's architecture with active-inference mathematics, with an explicit *vimarśa* meta-module that detects aspect-shifts (Wittgenstein). The gap that visible repositories in the space (`attractor-flow`, `pramana`) imply but do not yet contain.

This repository holds the engine, the plugin wrapper (**15 MCP tools, 5 skills, 5 agents, 5 slash commands, 3 hooks**), the benchmark harness with paired statistics, the HTML presentation, and the arxiv preprint.

* Formal operator specification: [`docs/SPEC.md`](docs/SPEC.md)
* Completion contracts: [`docs/COMPLETION_PROMISES.md`](docs/COMPLETION_PROMISES.md)
* Architecture decision records: [`docs/`](docs/)
* Plugin manifest: [`plugin/.claude-plugin/plugin.json`](plugin/.claude-plugin/plugin.json)
* Pre-registered hypotheses (H1–H6): [`docs/SPEC.md#2-hypotheses-pre-registered`](docs/SPEC.md)

## Quickstart

```bash
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv venv && uv sync --extra dev
uv run python scripts/verify_real_model.py     # downloads HF models (no mocks)
make smoke                                     # plugin manifest + in-process MCP smoke
make bench                                     # 3-arm A/B benchmark (~70-100 min on Apple Silicon)
make stats                                     # paired permutation, BCa, Wilcoxon, Holm, power
make figures                                   # paper/figures + presentation/figures
make autoreport                                # paper/autoreport.tex + main.tex placeholders
```

The benchmark runs three arms per item:

| arm | model | PCE? |
|---|---|---|
| `claude_haiku` | Claude Haiku 4.5 (via Claude Code CLI `-p --model haiku`) | no — headline control |
| `local_bare` | Qwen2-1.5B-Instruct (raw `cit` only) | no — sensitivity control on the same substrate |
| `local_cascade` | Qwen2-1.5B-Instruct via PCE cascade (K=4) | yes |

Outputs land in `benchmarks/results/<domain>.json` (resumable: re-running picks up where it stopped). The statistics produce `benchmarks/results/stats.json` with all six pre-registered hypotheses. The figures + autoreport feed the LaTeX paper and the HTML presentation.

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
* **MCP tools** (15): `pce.cit`, `pce.ananda`, `pce.iccha`, `pce.apohana`, `pce.jnana`, `pce.kriya`, `pce.vimarsa`, `pce.cascade`, `pce.embed`, `pce.lm.generate`, `pce.lm.entropy`, `pce.store.add`, `pce.store.recall`, `pce.store.consolidate_sws`, `pce.store.consolidate_rem`.

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

### Headline result (this run, n_paired = 38)

The pre-registered design returned a **directional null**:

| H | n | Δ (PCE − Haiku) | g | Holm p | supported |
|---|---|---|---|---|---|
| H1 (AUT) | 8 | −0.284 | −2.43 | 1.000 | no |
| H2 (poetry_interp) | 10 | −0.161 | −1.12 | 1.000 | no |
| H3 (poetry_gen) | 12 | −0.031 | −0.31 | 1.000 | no |
| H4 (sci_creativity) | 8 | −0.053 | −1.63 | 1.000 | no |
| H5 (z-blend) | 38 | 0.000 | 0.00 | — | no |
| H6 (vimarśa fired vs not) | 0/30 | n/a | — | — | n/a |

Notable: **vimarśa fired on 0/30 cascade trials** in this run, so H6 is undefined and the cascade reduces to a temperature-sampler at K=4. Substrate-matched sensitivity contrasts (cascade vs Local-Qwen) are all small (\|g\|≤0.4) and non-directional. The paper's §Limitations and §Discussion identify the two specific knobs (predicted-token-on-top criterion, BMR ΔF threshold) whose re-tuning is the obvious next experiment. All numbers above are auto-generated from `benchmarks/results/stats.json`.

## Reproducibility

Every numerical claim in the paper / presentation traces back to JSON artefacts under `audit/` and `benchmarks/results/`. The driver writes a checkpoint after every call and the audit log records the model checksum, git SHA, seed, and wall-clock per call. `PCE_DEVICE` and `PCE_DTYPE` env vars override the device autodetection (CUDA → MPS → CPU).

## License

MIT. Substrate models are downloaded from HuggingFace under their respective licences (Apache-2.0 for Qwen2-1.5B-Instruct, MIT for sentence-transformers/all-MiniLM-L6-v2).
