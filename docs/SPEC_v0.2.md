# PCE v0.2 — formal specification

Version: 0.2.0 (frozen during planning rounds 1-3, see [docs/AS_SHIPPED_v0.1.md](AS_SHIPPED_v0.1.md)).

This SPEC supersedes [docs/SPEC.md](SPEC.md) for v0.2. The v0.1 SPEC remains as the pre-registration record; this file is the actual contract for v0.2 implementation, benchmark, and acceptance.

## 0. Purpose (delta from v0.1)

v0.2 closes the central gap of v0.1: the cascade did not actually use its recursive `vimarsa` loop and the benchmark did not compare like-for-like substrates. v0.2 makes `vimarsa` causal and adds Haiku as a first-class generative substrate so the apples-to-apples question — does the PCE cascade improve a strong model's creative output? — can be answered.

## 1. Scope (frozen)

In scope (this version):

- All seven operators from v0.1 (`cit -> iccha -> apohana -> ananda -> jnana -> kriya -> vimarsa`) keep their typed signatures, but several change semantics (see §4).
- A pluggable LM substrate via a new `LMProtocol` (see §4.0). Two implementations:
  - `LocalLM` (existing, refactored): `Qwen/Qwen2-1.5B-Instruct` via `transformers` with auto device/dtype.
  - `HaikuLM` (new): wraps `claude -p --model haiku` (and an optional Anthropic SDK path gated by `PCE_USE_SDK=1`) with deterministic seeds, retry, cost telemetry, and per-call audit logs.
- A two-pass-always cascade: `draft = kriya(...)`, `event = vimarsa(draft, ...)`, `revision = kriya(prompt + aspect_brief, ...)`, `state.surface = revision`. Hard cap of one revision. The cascade also exposes a `bypass_vimarsa=True` flag that returns `state.surface = draft` without revising.
- A four-arm benchmark driver: `local_bare`, `local_cascade`, `haiku_bare`, `haiku_cascade`.
- A pilot benchmark targeted at ~$15 wallclock+API cost: n=20-30 paired (5-8 per domain), K=4.
- A prepared (dry-run-tested) `scripts/run_judge_bridge.py` for an offline ~$100 30-pair Sonnet judge bridge. Not invoked in this session.
- TRIZ-resolved operator changes (five contradictions, see [docs/triz/](triz)).
- Updated paper, HTML presentation, and README headline blocks bound to the v0.2 `stats.json`.

Out of scope (this version, explicitly):

- No new domains beyond the four already in v0.1.
- No retraining or fine-tuning of any LM.
- No live Sonnet judge run.
- No new plugin marketplace re-submission beyond version bump.
- The Hopfield ālayavijñāna and consolidation tools are kept but not part of the benchmark causal path.

## 2. Hypotheses (pre-registered for v0.2)

Treatment-vs-control pairings change per arm in v0.2; each hypothesis names its pair explicitly.

| ID | Statement | Treatment | Control | Domain | n target | α |
|----|-----------|-----------|---------|--------|----------|---|
| **H1.v2** | `haiku_cascade` > `haiku_bare` on the AUT composite. | `haiku_cascade` | `haiku_bare` | aut | >=5 | 0.05 |
| **H2.v2** | `haiku_cascade` > `haiku_bare` on the Wittgenstein poetry-interp aspect-multiplicity score. | `haiku_cascade` | `haiku_bare` | poetry_interp | >=5 | 0.05 |
| **H3.v2** | `haiku_cascade` > `haiku_bare` on the POEMetric-style poetry-gen composite. | `haiku_cascade` | `haiku_bare` | poetry_gen | >=5 | 0.05 |
| **H4.v2** | `haiku_cascade` > `haiku_bare` on the BBH-style scientific-creativity composite. | `haiku_cascade` | `haiku_bare` | sci_creativity | >=5 | 0.05 |
| **H5.v2** | Aggregate composite (z-scaled per domain, paired permutation) is positive. | `haiku_cascade` | `haiku_bare` | aggregate | n_total >=20 | 0.05 |
| **H6.v2** | `local_cascade` > `local_bare` on the aggregate composite — same-substrate ablation; isolates the architectural contribution. | `local_cascade` | `local_bare` | aggregate | n_total >=20 | 0.05 |
| **H7.v2** | Within `haiku_cascade`, items where vimarsa fired (`vimarsa_event=True`) score >= items where it did not. Internal-validity test. | event=True | event=False | within-treatment | depends on fire rate | 0.05 |
| **H8.v2** | The two-pass revision delta (`revision_score - draft_score`) has positive median in `haiku_cascade`. Causal contribution test. | revision | draft | within-treatment | n_total | 0.05 |

Statistical method unchanged from v0.1: paired permutation tests (`scipy.stats.permutation_test`), Hedges' g, Wilcoxon signed-rank, BCa bootstrap CI (10,000 resamples), Holm-Bonferroni across `{H1.v2, H2.v2, H3.v2, H4.v2}`. Power computed both a-priori and retrospectively.

H1.v2-H4.v2 are *directional*. The pilot's primary success criterion is **at least one of {H1.v2, H2.v2, H8.v2} supported**. H6.v2 is the secondary internal-validity test; H7.v2 and H8.v2 are causal contribution tests.

## 3. Architecture (engineering view, deltas only)

```
┌─ Substrate layer ────────────────────────────────────────────────┐
│  src/pce/substrate/                                              │
│  ├── lm_protocol.py    LMProtocol(generate, score, embed)         │
│  ├── lm.py             LocalLM(LMProtocol)  -- existing, refit    │
│  ├── haiku_lm.py       HaikuLM(LMProtocol)  -- new                │
│  ├── embed.py          Embedder              -- unchanged         │
│  └── hopfield.py       HopfieldStore         -- unchanged         │
└──────────────────────────────────────────────────────────────────┘

┌─ Cascade two-pass ───────────────────────────────────────────────┐
│  draft   = kriya(jnana_pick(iccha(prompt, K)))                    │
│  event   = vimarsa(draft, aspects)                                │
│  revision = kriya(jnana_pick(iccha(prompt + brief, K)))           │
│  state.surface = revision   (or draft if bypass_vimarsa)         │
│  cap = 1 revision per call                                        │
└──────────────────────────────────────────────────────────────────┘

┌─ Benchmark four arms ────────────────────────────────────────────┐
│  local_bare    : LocalLM .generate(prompt, tau=0.9)               │
│  local_cascade : run_cascade(LocalLM, two-pass, bypass=False)     │
│  haiku_bare    : HaikuLM .generate(prompt)                        │
│  haiku_cascade : run_cascade(HaikuLM, two-pass, bypass=False)     │
│  Primary contrast: haiku_cascade vs haiku_bare                    │
│  Ablation     :    local_cascade vs local_bare                    │
└──────────────────────────────────────────────────────────────────┘
```

## 4. Operator semantics changes

### 4.0 LMProtocol (new)

```python
class LMProtocol(Protocol):
    name: str   # "qwen2-1.5b" | "claude-haiku" | ...
    def generate(self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int) -> Candidate: ...
    def report(self) -> dict[str, Any]: ...
```

`Candidate.embedding` is computed by the cascade via the shared `Embedder`, not by the LM.

### 4.1 iccha

- `_build_prompt` no longer appends a constraint suffix when called from a parity-mode (added kwarg `prompt_mode: Literal["constraint_suffix", "verbatim"] = "verbatim"`).
- `DEFAULT_SAMPLER_GRID` keeps eight entries but the pilot uses K=4 with samplers `[(tau=0.9, top_p=0.95, top_k=50)] * K`. This matches the bare arm exactly; the explore-exploit grid is opt-in via `sampler_grid_mode="grid"`.

### 4.2 apohana

- Adds optional contrast normalization via min-max over candidates (gated by `normalize=True`). Default off to preserve backward compatibility, but `run_cascade` enables it.

### 4.3 jnana

- The `np.clip(apoha, 0.0, None)` is removed. Negative apoha values now penalize candidates close to `must_avoid`.
- Pseudo counts:
  - old: `pseudo = 1 + lambda_a * ananda + lambda_p * max(0, apoha)`.
  - new: `pseudo = 1 + lambda_a * ananda + lambda_p * shifted_apoha`, where `shifted_apoha = (apoha - apoha.min()) / max(apoha.max() - apoha.min(), eps)` so negative apoha pushes a candidate to its minimum without triggering log-domain singularities.

### 4.4 vimarsa

- New kwarg `min_evidence_points: int = 1`. When fewer than `min_evidence_points` trajectory points are supplied, switching is **not gated** (treated as N/A) instead of failing closed.
- `aspect_threshold` becomes domain-driven: the cascade passes `aspect_threshold = max(1, len(aspects) // 2 + 1)` so domains with few aspects can still fire; domains with `aspects=[]` use `aspect_count_required=0` and the gate becomes novelty + aesthetic only.
- Signature gains `return_brief: bool = True` so it returns a short natural-language brief listing the missing aspects (used by the cascade's revision pass).

### 4.5 cascade.run_cascade

- Two-pass-always by default. Returns `state` with new fields `surface_draft`, `surface_revision`, `vimarsa_event_draft`, `vimarsa_event_revision`, `revision_delta_score` (filled by the benchmark scorer post-hoc).
- `cit_temperature` is plumbed: it overrides the first sampler in `iccha`'s grid and is multiplicative on the rest.
- `bypass_vimarsa: bool = False` skips the revision and returns `state.surface = surface_draft`.

## 5. Plugin surface (deltas only)

- Manifest version bumped `0.1.0 -> 0.2.0` in `plugin/.claude-plugin/plugin.json` and `pyproject.toml`.
- `plugin/.mcp.json` no longer hard-pins device/dtype. Auto-detect by default; users can override via `PCE_DEVICE` / `PCE_DTYPE` / `PCE_LM_DEVICE` / `PCE_LM_DTYPE` env vars.
- New env vars:
  - `PCE_HAIKU_CLI` (path to `claude` binary; default `claude`).
  - `PCE_HAIKU_MODEL` (default `haiku`).
  - `PCE_USE_SDK` (default unset; `1` enables the Anthropic SDK code path when `ANTHROPIC_API_KEY` is set).
- New MCP tool name: `pce_cascade(arm: str = "local", ...)` switches the substrate (`local`|`haiku`). Existing tool names are kept for backward compatibility.

## 6. Benchmark protocol (v0.2 pilot)

### 6.1 Subjects

Same four domains and the same item bank as v0.1.

### 6.2 Pilot configuration

- Arms: `local_bare`, `local_cascade`, `haiku_bare`, `haiku_cascade`.
- n per domain: 6 (poetry_gen), 5 (poetry_interp), 5 (aut), 5 (sci_creativity); total >=20 paired observations.
- K = 4.
- `max_tokens` = 200 (raises v0.1's 120 to reduce truncation; documented in ADR `docs/adr/v0.2/ADR-002-jnana-signed-apoha.md`).
- Seed = 4242.
- Cost ledger: `audit/cost_ledger.json` records every Haiku call's `total_cost_usd` from `claude -p --output-format json`. Pilot must total under $20 (10% safety margin over $15 envelope).

### 6.3 Judging

- Pilot uses local proxy scorers only.
- The prepared `scripts/run_judge_bridge.py` adds a 30-pair Sonnet judge layer when run offline. Documented in [docs/HOWTO_JUDGE.md](HOWTO_JUDGE.md).

### 6.4 Statistics

Same machinery as v0.1 (`benchmarks/stats.py`), now reporting:

- H1.v2-H4.v2 per domain: paired (`haiku_cascade` - `haiku_bare`).
- H5.v2 aggregate, Holm-Bonferroni across H1.v2-H4.v2.
- H6.v2 same-substrate ablation: paired (`local_cascade` - `local_bare`).
- H7.v2 within-treatment Wilcoxon (event vs no-event).
- H8.v2 two-pass revision causal contribution: paired (`revision_score` - `draft_score`) within `haiku_cascade`, treated as a sign-test if the revision metric is binary, Wilcoxon otherwise.

## 7. Acceptance criteria for v0.2.0 release

Engineering gates (all must pass):

- All seven operators implemented per §4 with mypy --strict and ruff clean.
- `pytest` passes including new `tests/cascade_two_pass_test.py` and operator-level signed-apoha tests.
- `scripts/prove_gate.py --strict` exits zero on duck-rabbit textual + AUT brick: `vimarsa_event=True` for at least one of the two cases on `haiku_cascade`, `revision != draft` on both cases for `haiku_cascade`, `aspect_count > 0` on duck-rabbit, `haiku_cascade` text differs from `haiku_bare` text on both cases.
- `scripts/validate_paper.py` clean.
- Plugin smokes (`scripts/smoke_plugin.py`, `scripts/verify_plugin.py`, `scripts/verify_real_model.py`) pass with the v0.2 manifest.
- `audit/cost_ledger.json` total under $20.
- `scripts/run_judge_bridge.py --dry-run` succeeds end-to-end on synthetic data.
- CI green; `v0.2` branch merged to `main`; tag `v0.2.0` pushed.

Research-hypothesis gates (any of these is the success signal; report the result honestly either way):

- *Primary success*: at least one of `{H1.v2, H2.v2, H8.v2}` shows directional support after Holm-Bonferroni.
- *Mechanism success* (sufficient even if primary fails): vimarsa fires on at least 30% of `haiku_cascade` items in domains where aspects are supplied.
- *Honest negative*: if neither holds, the paper, presentation, and README report the negative result and propose v0.3 design changes. v0.2 still ships.

## 8. Risk register (deltas)

| Risk | Mitigation |
|------|-----------|
| Haiku CLI rate limit during four-arm pilot | Per-call timeout, 60 s exponential backoff up to 3 retries, checkpoint after every call (already in `driver.py`). |
| Cost overrun beyond $15 pilot envelope | Cost ledger tracks running total; driver aborts gracefully when ledger >= $20 with a clear message. |
| Two-pass revision changes Haiku output unpredictably | Prove-gate cases pin the expected behavior; revision delta is logged per item for audit. |
| Sonnet judge bridge invoked accidentally | `scripts/run_judge_bridge.py` requires explicit `--live` flag; default mode is `--dry-run`. Without `ANTHROPIC_API_KEY` it exits cleanly. |
| Hot-spotting `apohana` normalization breaks v0.1 tests | New behavior gated by `normalize=True`; v0.1 tests that pass `normalize=False` are unaffected. |
| Plugin device autodetect on a new user's CPU-only mac is too slow | Document `PCE_DEVICE=cpu PCE_DTYPE=float32` override in README; do not hard-pin. |
