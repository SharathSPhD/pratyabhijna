# Pratyabhijñā Creative Engine (PCE)

A Claude Code plugin that operationalises Abhinavagupta's Pratyabhijñā five-*śakti* generative cascade as typed operators over an active-inference / Bayesian Model Reduction substrate, with a recursive *vimarśa* self-reflexivity layer.

> **Live results page:** **<https://sharathsphd.github.io/pratyabhijna/>** — auto-loads the latest `benchmarks/results_v0.3/stats.json` from this repository on every visit (cache-busted), so any push to `main` that updates results is reflected immediately on the page.
>
> **v0.3 ships.** Directly addresses the four falsifiable findings of the [v0.2 adversarial review](docs/reviews/2026-04-29-adversarial-v0.2-review.md): (i) clean Haiku CLI substrate (subprocess flag isolation + scrubbed `HOME` + per-item `IntegrityProbe`) eliminates Claude-Code-context contamination of the bare control; (ii) event-gated commit on jñāna ΔF with always-shadow revision lifts vimarśa from prompting to a causal step; (iii) full active-inference uplift on the cascade causal path (aspect-conditioned BMR, Hopfield warm-start, plumbed `cit_temperature`, per-item free-energy budget); (iv) two new control arms (`haiku_bare_2K_scorer` for compute, `haiku_generic_revise_2pass` for protocol) explicitly isolate the architecture from the v0.2 review's confounds. Frozen scope: [docs/SPEC_v0.3.md](docs/SPEC_v0.3.md), [docs/PRD_v0.3.md](docs/PRD_v0.3.md), [docs/RELEASE_NOTES_v0.3.md](docs/RELEASE_NOTES_v0.3.md). v0.2 paper preserved at `paper/v0.2/`; v0.1 at `paper/v0.1/`.

> **Research vector.** Pairing Pratyabhijñā's architecture with active-inference mathematics, with an explicit *vimarśa* meta-module that detects aspect-shifts (Wittgenstein). The gap that visible repositories in the space (`attractor-flow`, `pramana`) imply but do not yet contain.

This repository holds the engine, the plugin wrapper (**19 MCP tools, 5 skills, 5 agents, 5 slash commands, 3 hooks** — v0.3 adds `pce.haiku_clean_substrate_probe` and `pce.hopfield_state` and reshapes `pce.pce_cascade` for the four-arm matrix), the benchmark harness with paired statistics, the HTML presentation, and the arxiv preprint.

* Formal operator specification: v0.3 [`docs/SPEC_v0.3.md`](docs/SPEC_v0.3.md) (latest); v0.2 [`docs/SPEC_v0.2.md`](docs/SPEC_v0.2.md), v0.1 [`docs/SPEC.md`](docs/SPEC.md).
* Completion contracts: v0.3 [`docs/COMPLETION_PROMISES_v0.3.md`](docs/COMPLETION_PROMISES_v0.3.md); v0.2 [`docs/COMPLETION_PROMISES_v0.2.md`](docs/COMPLETION_PROMISES_v0.2.md); v0.1 [`docs/COMPLETION_PROMISES.md`](docs/COMPLETION_PROMISES.md).
* Architecture decision records: [`docs/adr/v0.3/`](docs/adr/v0.3/) (v0.3, ADR-001..ADR-005); [`docs/adr/v0.2/`](docs/adr/v0.2/) (v0.2); [`docs/`](docs/) (v0.1).
* TRIZ contradiction cards: [`docs/triz/v0.3/`](docs/triz/v0.3/) (v0.3, C1..C5); [`docs/triz/`](docs/triz/) (v0.2).
* Plugin manifest: [`plugin/.claude-plugin/plugin.json`](plugin/.claude-plugin/plugin.json) (v0.3.0).
* Pre-registered hypotheses (H1.v3–H8.v3): [`docs/SPEC_v0.3.md`](docs/SPEC_v0.3.md).

## Quickstart

```bash
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv venv && uv sync --extra dev
uv run python scripts/verify_real_model.py     # downloads HF models (no mocks)
make smoke                                     # plugin manifest + in-process MCP smoke

# v0.3 outer-host smoke (must pass before any inner-substrate hardening):
uv run python scripts/verify_outer_host_loads_pce.py

# v0.3 prove-gate (single-case validation; per-item integrity probe + leakage scan):
uv run python scripts/prove_gate.py --strict

# v0.3 pilot (~$8-10 Haiku envelope under the locked $20 cap; ~60-90 min on Apple Silicon):
uv run python benchmarks/driver.py --pilot --out-dir benchmarks/results_v0.3 --cost-cap-usd 20
uv run python benchmarks/stats.py --results-dir benchmarks/results_v0.3 --out benchmarks/results_v0.3/stats.json
uv run python benchmarks/figures.py --results-dir benchmarks/results_v0.3 --stats benchmarks/results_v0.3/stats.json --out-dirs paper/figures
uv run python benchmarks/autoreport.py --stats benchmarks/results_v0.3/stats.json --paper-dir paper

# v0.2 (preserved for the v0.2 paper / HTML):
make benchmark.pilot && make stats.pilot

# v0.1 (preserved for the v0.1 paper / HTML; 70-100 min on Apple Silicon):
make bench && make stats && make figures && make autoreport
```

The v0.3 pilot runs the four-arm matrix per item:

| arm | model | PCE? | role |
|---|---|---|---|
| `haiku_bare` | Anthropic Claude Haiku (clean CLI substrate) | no | architecture-vs-nothing primary control |
| `haiku_cascade` | Haiku via v0.3 event-gated cascade (K=3, active-inference uplift) | yes | **primary treatment** |
| `haiku_bare_2K_scorer` | Haiku, 1 pass, K'=2K candidates + jñāna BMR | no | **+K compute control (H6.v3)** |
| `haiku_generic_revise_2pass` | Haiku, 2 passes, fixed generic revise brief | no (generic 2-pass) | **revision-protocol control (H7.v3)** |

Outputs land in `benchmarks/results_v0.3/<domain>.json` (resumable). The statistics produce `benchmarks/results_v0.3/stats.json` with: `primary` (H1.v3–H4.v3, `haiku_cascade` vs `haiku_bare`), `H5` (random-effects pooled g across primary domains), `H6_v3_extra_compute` (cascade vs +K compute), `H7_v3_generic_revise` (cascade vs generic 2-pass), `H8_v3_revision_vs_draft` (within-cascade revision-vs-draft pair on items where event-gated commit chose revision), and per-arm per-domain means.

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
* **MCP tools** (19 in v0.3): the 15 v0.1/v0.2 tools (`pce.cit`, `pce.ananda`, `pce.iccha`, `pce.apohana`, `pce.jnana`, `pce.kriya`, `pce.vimarsa`, `pce.cascade`, `pce.embed`, `pce.lm.generate`, `pce.lm.entropy`, `pce.store.add`, `pce.store.recall`, `pce.store.consolidate_sws`, `pce.store.consolidate_rem`) plus v0.2's `pce.pce_cascade(arm=…)` and `pce.haiku_bare`, plus v0.3's `pce.haiku_clean_substrate_probe` (live IntegrityProbe against the inner CLI subprocess) and `pce.hopfield_state` (introspect the active-inference ālayavijñāna). The v0.3 `pce.pce_cascade` adds new arms `haiku_bare_2K` and `haiku_generic_revise` plus `commit_policy`, `cit_temperature`, `use_storehouse`, `hopfield_weight` parameters.

## Hypotheses (pre-registered, v0.3)

| H | claim | contrast | domain | direction | α | power |
|---|---|---|---|---|---|---|
| H1.v3 | PCE > bare on AUT | `haiku_cascade` vs `haiku_bare` | aut | one-sided | 0.05 | 0.80 |
| H2.v3 | PCE > bare on Wittgenstein aspect-shift | `haiku_cascade` vs `haiku_bare` | poetry_interp | one-sided | 0.05 | 0.80 |
| H3.v3 | PCE > bare on POEMetric poetry-gen | `haiku_cascade` vs `haiku_bare` | poetry_gen | one-sided | 0.05 | 0.80 |
| H4.v3 | PCE > bare on BBH-style sci-creativity | `haiku_cascade` vs `haiku_bare` | sci_creativity | one-sided | 0.05 | 0.80 |
| **H5.v3** | random-effects pooled Hedges' g across H1.v3–H4.v3 > 0 | DerSimonian–Laird pool | aggregate | one-sided | 0.05 | 0.85 |
| **H6.v3** | PCE > +K-compute control | `haiku_cascade` vs `haiku_bare_2K_scorer` | per-domain | one-sided | 0.05 | 0.80 |
| **H7.v3** | PCE > generic 2-pass control | `haiku_cascade` vs `haiku_generic_revise_2pass` | per-domain | one-sided | 0.05 | 0.80 |
| **H8.v3** | within-cascade: score(revision) > score(draft) | paired within `haiku_cascade` (committed=revision) | within-arm | one-sided | 0.05 | 0.70 |

Statistical protocol: paired permutation (sign-flip; exact for n ≤ 18, MC otherwise), Hedges' *g* with small-sample correction, BCa bootstrap 95% CI (10k resamples), Wilcoxon signed-rank (one-sided), Holm-Bonferroni across {H1..H4}, a-priori (g=0.5) + retrospective (observed g) power, **length-controlled scoring** (per-arm linear word-count effect regressed out before pairing; both raw and length-controlled estimates reported), and **strict JSON output** (`_clean_json` + `allow_nan=False`). Negative-result obligation: any rejected hypothesis is reported in the abstract.

### Headline result — v0.3 pilot

The pilot results are auto-generated from `benchmarks/results_v0.3/stats.json` into [`paper/main.tex`](paper/main.tex) and [`paper/autoreport.tex`](paper/autoreport.tex) via `benchmarks/autoreport.py`. The full v0.1 → v0.2 → v0.3 narrative — TRIZ contradictions, ADRs, prove-gate, and the four-arm pilot — is in [`paper/main.tex`](paper/main.tex) and the [HTML presentation](presentation/index.html). The v0.2 paper is preserved at [`paper/v0.2/`](paper/v0.2/) and v0.1 at [`paper/v0.1/`](paper/v0.1/).

## Reproducibility

Every numerical claim in the paper / presentation traces back to JSON artefacts under `audit/` and `benchmarks/results_v0.3/` (v0.3), `benchmarks/results_v2/` (v0.2), or `benchmarks/results/` (v0.1). The driver writes a checkpoint after every call and the audit log records the model checksum, git SHA, seed, and wall-clock per call. v0.3 additionally writes per-item integrity probes to `audit/v0.3/integrity_probes.jsonl`, per-run cost snapshots to `audit/v0.3/cost_snapshot.json`, and the pilot run log to `audit/v0.3/pilot_run.log`. `PCE_DEVICE`, `PCE_DTYPE` env vars override the local-LM device autodetection (CUDA → MPS → CPU); `PCE_HAIKU_CLI` and `PCE_HAIKU_MODEL` configure the Haiku CLI binding; `PCE_HAIKU_CLEAN_SUBSTRATE=0` falls back to the v0.2 inheriting-env path (only useful for forensics).

## License

MIT. Substrate models are downloaded from HuggingFace under their respective licences (Apache-2.0 for Qwen2-1.5B-Instruct, MIT for sentence-transformers/all-MiniLM-L6-v2).
