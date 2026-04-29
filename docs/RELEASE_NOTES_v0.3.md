# PCE v0.3 — Release Notes

**Tag**: `v0.3.0`
**Branch**: `v0.3`
**Date**: 2026-04-29

## Summary

PCE v0.3 directly addresses the four falsifiable findings of the v0.2 adversarial review (`docs/reviews/2026-04-29-adversarial-v0.2-review.md`):

1. **Substrate cleanliness**. Every Haiku call now goes through a clean CLI substrate (subprocess flag isolation + scrubbed `HOME` + per-item `IntegrityProbe`), eliminating Claude-Code-context contamination of the bare control.
2. **Causal vimarsa**. The cascade now uses an event-gated commit policy on the jñāna ΔF signal, with an always-shadow revision so H8 (revision-vs-draft within cascade) is measurable on every item — fired or not.
3. **Active-inference uplift on the causal path**. Aspect-conditioned BMR (`jnana` reduces over the right model space, not a constant prior), Hopfield ālayavijñāna as warm-start aspect prior in `apohana`, `cit_temperature` plumbed through the `iccha` fan-out posterior, and a per-item free-energy budget controlling abort/continue.
4. **Compute & protocol fairness arms**. Two new control arms (`haiku_bare_2K_scorer` for the +K compute control, `haiku_generic_revise_2pass` for the protocol control) explicitly isolate the architecture from the confounds the v0.2 review flagged.

## What's new

### Architecture

- **Clean Haiku CLI substrate** (ADR-001).
  - `claude --print --tools "" --strict-mcp-config --disable-slash-commands --setting-sources='' --no-session-persistence --permission-mode bypassPermissions` per inner subprocess.
  - Scrubbed `HOME`: temporary directory with strict permissions; selective symlinking of `~/Library/Keychains/` (macOS) or `~/.config/claude/` (Linux) so OAuth still works while denying access to plugins, skills, settings.
  - Explicit `ENV_ALLOWLIST` (no `os.environ.copy()`); parent-process Claude env vars filtered out.
  - `IntegrityProbe` (`pce.substrate.integrity`) periodically asks the inner subprocess what plugins/skills/MCP it has access to and scans the response with a negation-aware `LEAKAGE_REGEX`.
  - **Two-tier isolation**: only the *inner* `claude --print` subprocess runs in the clean environment; the *outer* Claude Code host that loads PCE keeps its plugins/skills/MCP intact (`scripts/verify_outer_host_loads_pce.py` guards against regressions).

- **Event-gated cascade with always-shadow revision** (ADR-002).
  - `commit_policy: Literal["event_gated", "always_revise", "always_draft"]` replaces the v0.2 `bypass_vimarsa` boolean (kept as a legacy alias mapping to `"always_draft"`).
  - Vimarsa fires when `|ΔF| ≥ DEFAULT_DELTA_F_THRESHOLD` (0.01) and aspect-multiplicity is detected; commits the revision when the event fires, otherwise commits the draft.
  - Both `surface_draft` and `surface_revision` are always populated and scored, regardless of commit outcome.

- **Active-inference uplift** (ADR-003, ADR-004, ADR-005).
  - `jnana` adds `ReductionTarget="aspect_conditioned"`: when aspects are supplied, BMR reduces over `|aspects|` candidate models that softly boost in-support candidates, producing a non-degenerate ΔF.
  - `apohana` accepts an optional `HopfieldStore` (from `pce.active_inference.hopfield`, distinct from the v0.1 store) for a warm-start aspect prior.
  - `iccha` plumbs `cit_temperature` through the parity sampler's `tau`; recorded on each candidate's `sampler` dict for audit.
  - `vimarsa.consolidate` writes the committed surface back to the Hopfield store under REM (append) or SWS (k-means consolidate) modes.
  - `pce.active_inference.budget.FreeEnergyBudget` is a per-item ledger (`earn_jnana`, `earn_aspect`, `earn_tokens`, `should_continue_revision`) that gates the revision pass under depleted budget.

### Plugin surface (19 MCP tools, +2 from v0.2)

- `pce.pce_cascade` — v0.3 arm-switchable cascade. Takes `arm` (one of `haiku`, `haiku_cascade`, `haiku_bare`, `haiku_bare_2K`, `haiku_generic_revise`, `local`, `local_cascade`), `commit_policy`, `cit_temperature`, `use_storehouse`, `hopfield_weight`. Backward-compatible with v0.2 callers via `bypass_vimarsa` alias.
- `pce.haiku_clean_substrate_probe` — live `IntegrityProbe` against the inner subprocess.
- `pce.hopfield_state` — introspect the active-inference ālayavijñāna (per-domain pattern counts, recent L2 norms).

### Benchmark protocol

- **Four-arm matrix** on the v0.2 sample (n=20 items, 5 per domain): `haiku_bare`, `haiku_cascade`, `haiku_bare_2K_scorer`, `haiku_generic_revise_2pass`. Local arms removed from the default; available for legacy audits.
- **Per-item integrity probe** (`audit/v0.3/integrity_probes.jsonl`); halts on leakage unless `--allow-leakage` is set.
- **Length-controlled scoring** in `benchmarks/stats.py`: per-arm linear word-count effect regressed out before pairing; both raw and length-controlled estimates and Hedges' g reported.
- **Strict JSON output**: `_clean_json` maps non-finite floats to `null`; `allow_nan=False` writer.

### Hypotheses (8, vs v0.2's 6)

- **H1.v3 – H4.v3** (per-domain primary): `haiku_cascade` vs `haiku_bare`.
- **H5.v3** (redesigned): random-effects DerSimonian-Laird pooled Hedges' g across the four primary domains, with τ² heterogeneity. Replaces the v0.2 z-blend.
- **H6.v3** (headline fairness): `haiku_cascade` vs `haiku_bare_2K_scorer`. Architecture-vs-more-compute.
- **H7.v3** (protocol fairness): `haiku_cascade` vs `haiku_generic_revise_2pass`. Brief-content vs pass-existence.
- **H8.v3** (revision-causality): paired `score(revision) - score(draft)` within `haiku_cascade` on items where the event-gated commit chose revision.

## Migration from v0.2

| v0.2 thing                              | v0.3 replacement / mapping                                         |
|-----------------------------------------|--------------------------------------------------------------------|
| `bypass_vimarsa: bool`                  | `commit_policy: Literal["event_gated", "always_revise", "always_draft"]` (legacy alias preserved) |
| `LMProtocol`                            | `GeneratorProtocol` (LMProtocol kept as alias, with capability flags `supports_logprobs`, `supports_score`, `supports_entropy` and `length_proxy_logp`) |
| `arm="local"` / `arm="haiku"` only      | adds `arm="haiku_bare_2K"` and `arm="haiku_generic_revise"` for control protocols |
| `H5` z-blend                            | `H5.v3` random-effects DerSimonian-Laird pooled Hedges' g          |
| `H6` (within-PCE event vs no-event)     | `H6.v3` (cascade vs +K compute control) ; v0.2 H6 retired in favour of H8.v3 within-cascade pair |
| Bare `HaikuLM(env=os.environ.copy())`   | `HaikuLM(clean_substrate=True)` with `IntegrityProbe`               |
| `audit/` (flat)                         | `audit/v0.3/` (new artefacts) ; legacy `audit/` preserved          |
| `benchmarks/results_v2/`                | `benchmarks/results_v0.3/`                                         |
| `paper/` (v0.2)                         | `paper/v0.2/` (archived) ; new `paper/` (v0.3)                     |

## Verified gates

- **Phase 2 substrate**: 50/50 leak-free + 10/10 IntegrityProbe pass on real Haiku.
- **Phase 3 active inference**: ΔF non-degenerate (≥ 0.01) on the duck-rabbit fixture.
- **Phase 4 causal vimarsa**: `revision_differs_from_draft ≥ 80%` on prove-gate items.
- **Phase 5 prove-gate**: per-item integrity probe + leakage scan + fixture-specific assertions all pass.
- **Phase 6 plugin**: smoke_plugin --with-haiku passes (19/19 tools loadable, MCP server v0.3.0).
- **Phase 7 pilot**: pilot under $20 budget, strict JSON output, ΔF non-degenerate.

## Cost telemetry

Pilot Haiku spend recorded in `audit/cost_ledger.json` (cumulative across all v0.3 runs) and per-run snapshots in `audit/v0.3/cost_snapshot.json`.

## Files of interest

- SPEC: `docs/SPEC_v0.3.md`
- PRD: `docs/PRD_v0.3.md`
- ADRs: `docs/adr/v0.3/ADR-001..ADR-005`
- TRIZ: `docs/triz/v0.3/C1..C5`
- Completion promises: `docs/COMPLETION_PROMISES_v0.3.md`
- Adversarial review: `docs/reviews/2026-04-29-adversarial-v0.2-review.md`
