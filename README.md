# Pratyabhijñā Creative Engine (PCE)

A portable plugin (Cursor + Claude Code + standalone CLI) that operationalises Abhinavagupta's Pratyabhijñā five-*śakti* generative cascade as typed operators over an active-inference / Bayesian Model Reduction substrate, with a recursive *vimarśa* self-reflexivity layer.

> **v0.4 — mechanism study of recursive self-reflexivity layers for LLM creative cognition.** Cascade-vs-bare null at the pilot's *n*; recursive revision pass robustly positive on its own (H8a, *g* = 0.65, *p* < 1e-4); learned commit gate beats the v0.3 event gate (H8b, F1 0.65 vs 0.52); proxy scorer disagrees with the Sonnet-4.5 LLM-judge at ρ = 0.0 (H9 — flagged as a metric-design problem, not a refutation). Cost ledger split honestly: Haiku cascade `$12.73 / 1,277` calls; Sonnet judge `$0.48 / 23` rows; combined pilot spend `$13.21`. v0.4 frozen at `paper/v0.4/`.
>
> **Live site:** **<https://sharathsphd.github.io/pratyabhijna/>** — Astro Pages site that reads `benchmarks/results_v0.4/stats.json` directly and animates all 9 [showcase demos](https://sharathsphd.github.io/pratyabhijna/showcase) with full cascade traces.
>
> **Companion work.** Pratyākṣa (direct perception / context-discipline) at <https://zenodo.org/records/19680692> and <https://sharathsphd.github.io/context-engineering-harness/>. PCE is the recognition + creativity counterpart in the same author program.

This repository holds the engine, the dual plugin manifests (`plugin/.claude-plugin/plugin.json`, `plugin/.cursor-plugin/plugin.json`), the standalone `pce` CLI, the Phase 7 mechanism pilot's results and audit, the v0.4 paper, and the Astro v0.4 site.

* Frozen scope: v0.4 [`docs/SPEC_v0.4.md`](docs/SPEC_v0.4.md), [`docs/PRD_v0.4.md`](docs/PRD_v0.4.md), [`docs/RELEASE_NOTES_v0.4.md`](docs/RELEASE_NOTES_v0.4.md), [`docs/COMPLETION_PROMISES_v0.4.md`](docs/COMPLETION_PROMISES_v0.4.md).
* ADRs: v0.4 [`docs/adr/v0.4/`](docs/adr/v0.4/) (ADR-001 cit_temperature substrate, ADR-002 learned commit gate, ADR-003 free-energy budget, ADR-004 managed-API fairness lattice, ADR-005 fixed-effects H5, ADR-006 typed Haiku errors, ADR-007 SDK code-path removal).
* TRIZ cards: v0.4 [`docs/triz/v0.4/`](docs/triz/v0.4/).
* Plugin manifests: [`plugin/.claude-plugin/plugin.json`](plugin/.claude-plugin/plugin.json) and [`plugin/.cursor-plugin/plugin.json`](plugin/.cursor-plugin/plugin.json) (both at v0.4.0).
* Hypotheses: H1.v4–H4.v4 (per-domain), H5.v4 (fixed-effects pool), H6.v4 / H7.v4 (fairness controls), H8a/b/c.v4 (mechanism decomposition), H9.v4 (judge agreement).

## Three install paths

```bash
# 1. Standalone CLI (works in any shell with `claude` on PATH)
git clone https://github.com/SharathSPhD/pratyabhijna.git
cd pratyabhijna
uv pip install -e .                       # required: registers the `pce` console script
                                          # and pulls numpy / sentence-transformers
pce smoke                                 # verifies the CLI, OAuth substrate, and cascade module
pce cascade --prompt "Write a haiku about rain on a tin roof" \
            --constraint "imagism" --k 4 --seed 4242

# 2. Cursor plugin
cursor --install-plugin .

# 3. Claude Code plugin
claude plugin install https://github.com/SharathSPhD/pratyabhijna
# or, for a local clone:
ln -s "$(pwd)" "$HOME/.claude/plugins/pce"
```

See [`docs/RUN_LOCAL.md`](docs/RUN_LOCAL.md) for the full operator guide, including the precedence chain (defaults → repo `pce.toml` → user `~/.config/pce/config.toml` → env vars → CLI overrides) and example configurations.

## Reproduce v0.4 numbers

```bash
# Regenerate the figure pack and autoreport against the published stats.json
python -m benchmarks.figures --version v0.4
python -m benchmarks.autoreport --version v0.4 --strict

# Build the paper PDF (tectonic preferred; latexmk also supported)
cd paper && tectonic -X compile main.tex

# Build and serve the Astro site locally
python scripts/prepare_site_data.py
cd docs/site && pnpm install && pnpm build && pnpm preview
```

The v0.4 pilot ran the same four base arms as v0.3 plus a five-policy commit multiplexer over the cascade arm. The arm matrix is:

| arm | role |
|---|---|
| `haiku_bare` | architecture-vs-nothing primary control |
| `haiku_cascade` | **primary treatment** (always-shadow revision; commit per policy) |
| `haiku_bare_2K_scorer` | **+K-compute control (H6.v4)** |
| `haiku_generic_revise_2pass` | **revision-protocol control (H7.v4)** |

The commit-policy multiplex over `haiku_cascade` is `always_draft` / `always_revise` / `event_gated` (the v0.3 policy) / `learned_gate` (ADR-002) / `oracle` (analysis upper bound).

## Pre-registered hypotheses (v0.4)

| H | claim | contrast | domain | reading |
|---|---|---|---|---|
| H1.v4 | cascade > bare on AUT | `haiku_cascade` vs `haiku_bare` | aut | inconclusive (n = 5) |
| H2.v4 | cascade > bare on Wittgenstein aspect-shift | `haiku_cascade` vs `haiku_bare` | poetry_interp | inconclusive (n = 10) |
| H3.v4 | cascade > bare on POEMetric poetry-gen | `haiku_cascade` vs `haiku_bare` | poetry_gen | inconclusive (n = 6) |
| H4.v4 | cascade > bare on BBH-style sci-creativity | `haiku_cascade` vs `haiku_bare` | sci_creativity | inconclusive (n = 4) |
| **H5.v4** | fixed-effects pool of H1–H4 > 0 | inverse-variance pool (ADR-005) | aggregate | **g = 0.14, CI crosses 0; not supported** |
| **H8a.v4** | within-cascade: revision > draft | paired score(revision) − score(draft) | within-arm | **supported, g = 0.65, p < 1e-4** |
| **H8b.v4** | learned gate F1 > event gate F1 | binary classifier metrics | within-arm | **supported, F1 0.65 vs 0.52** |
| H8c.v4 | commit-policy leaderboard | pairwise paired permutations | within-arm | leaderboard reported; pairwise gaps not significant after Holm |
| H9.v4 | judge ≈ proxy on per-item delta | Spearman ρ + sign-agreement | held-out | **flagged: ρ = 0.0, sign-agreement 56.5%** |

Statistical protocol: paired permutation (50 000 permutations), Hedges' *g* with small-sample correction, BCa bootstrap 95 % CI (10 000 resamples), Wilcoxon signed-rank backup, Holm-Bonferroni across primary contrasts, fixed-effects pool for H5 (ADR-005, with random-effects DerSimonian–Laird as a sensitivity check), strict JSON output, length-controlled scoring. Negative-result obligation: every rejected hypothesis is reported in the paper abstract.

## Showcase

Nine creative outputs with full cascade traces (3 Sanskrit chandas + 3 English poetry styles + 3 scientific creativity prompts):

* Sanskrit (3, live cascade output as of v0.4.1): `sanskrit_anustubh`, `sanskrit_gayatri`, `sanskrit_indravajra`. v0.4 has no chandas-aware scorer, so the chandas validator's pass/fail is *informational* — a v0.5 ladder item adds chandas scoring inside the cascade itself.
* English (3, real Phase 7 cascade traces): `english_dickinson_slant`, `english_imagist_haiku`, `english_pastoral_traditional` (draft + shadow revision).
* Science (3, real Phase 7 cascade traces): `science_galaxy_arms`, `science_ice_geometry`, `science_unreasonable_effectiveness`.

Browse the [showcase index](https://sharathsphd.github.io/pratyabhijna/showcase) — every demo page renders the cit → ānanda → icchā → apohana → jñāna → kriyā → vimarśa → revision pipeline with diff view and validator output.

## Engine sketch

```
cit  →  ānanda  →  icchā (×K)  →  apohana  →  jñāna (BMR ΔF)  →  kriyā  →  surface
                                                  │
                                               vimarśa
                                                  │
                                          (commit policy multiplexer)
                                                  │
                                              committed
```

The Hopfield-attractor *ālayavijñāna* (storehouse) is wired in v0.4 (`consolidate_sws`, `consolidate_rem`, `pce.hopfield_state`) but its multi-session dynamics were not exercised in the Phase 7 pilot — that ladder rung is on v0.5.

## Plugin surface

* **Slash commands** (5): `/pce-compose`, `/pce-interpret`, `/pce-aut`, `/pce-bbh`, `/pce-trace` (mirrored across the Cursor and Claude Code manifests).
* **Skills** (5): `pce-poetry-generation`, `pce-poetry-interpretation`, `pce-divergent-thinking`, `pce-scientific-creativity`, `pce-vimarsa-self-reflection`.
* **Agents** (5): `pce-poet`, `pce-interpreter`, `pce-ideator`, `pce-scientist`, `pce-vimarsa-auditor`.
* **MCP tools** (19+): the v0.3 set (cit, ānanda, icchā, apohana, jñāna, kriyā, vimarśa, cascade, embed, lm.generate, lm.entropy, store.add, store.recall, store.consolidate_sws, store.consolidate_rem, pce_cascade, haiku_bare, haiku_clean_substrate_probe, hopfield_state) plus v0.4's commit-policy multiplexer hooks. The standalone CLI exposes `pce config show`, `pce smoke`, `pce cascade`, `pce judge-pair --domain DOM --item-id ID --treatment-text PATH --control-text PATH`, and `pce showcase --regenerate SLUG`.
* **Hooks** (3): `SessionStart`, `PreToolUse` (audit-stamp every PCE MCP call), `PostToolUse` (consolidation tick).

## Reproducibility

Every numerical claim in the paper, the Astro site, and this README traces back to JSON artefacts under `benchmarks/results_v0.4/` and `audit/v0.4/`. The pilot driver writes a checkpoint after every call; the audit log records the model checksum, git SHA, seed, and wall-clock per call. v0.4 publishes per-item integrity probes (`audit/v0.4/integrity_probes_merged.jsonl`), per-domain cost ledgers (`audit/v0.4/cost_ledger_*.json`), the bibliography verification log (`audit/v0.4/lit_verification.jsonl`), and the Phase 8 gate report (`audit/v0.4/phase8_gate_report.json`). The OAuth Claude CLI is the single supported substrate (ADR-007); legacy `PCE_USE_SDK=1` users get a clear deprecation error.

The judge audit metadata in `benchmarks/results_v0.4/judge.jsonl` is replay-auditable via `formatted_prompt_sha256` (unique per row, post-hoc reconstructed for v0.4.1 by `scripts/recover_judge_formatted_sha.py`). The `input_tokens = 9` field on each row is **a placeholder, not a measurement**: the OAuth `claude --print` substrate did not expose per-call token counts in the v0.4 pilot. The placeholder is documented in `benchmarks/results_v0.4/judge_agreement.json` under `input_tokens_provenance` and in `docs/RUN_LOCAL.md` § "Judge audit metadata". v0.5 upgrades the judge bridge to a usage-emitting provider, at which point `input_tokens` becomes a real measurement.

The §0.5 unmerged-state critique (why the Phase 7 results were not landed on `main` until Phase 8 closed the paper, the site, and the release together) is recorded verbatim in `paper/sections/01_introduction.tex`, on the [reproducibility page](https://sharathsphd.github.io/pratyabhijna/reproducibility), and in the Phase 8 PR body.

## License

MIT. Substrate models are accessed via the OAuth Claude CLI; no SDK calls.
