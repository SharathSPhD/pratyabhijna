# PCE v0.3 — formal specification

Version: 0.3.0 (frozen during planning round 1, see [docs/RELEASE_NOTES_v0.2.md](RELEASE_NOTES_v0.2.md) and [docs/reviews/2026-04-29-adversarial-v0.2-review.md](reviews/2026-04-29-adversarial-v0.2-review.md)).

This SPEC supersedes [docs/SPEC_v0.2.md](SPEC_v0.2.md) for v0.3. The v0.2 SPEC remains as the previous-version contract; this file is the actual contract for v0.3 implementation, benchmark, and acceptance.

## 0. Purpose (delta from v0.2)

v0.2 established the apples-to-apples framing on Haiku and made `vimarsa` causal as a two-pass-always scaffold. The v0.2 adversarial review surfaced three critical gaps that v0.3 must close before the architectural claim can be defended:

- **Apples-to-apples is incomplete.** `haiku_cascade` (~ 2K Haiku calls + an explicit revision prompt) was paired against `haiku_bare` (1 Haiku call, no revision prompt). Any score gain is confounded by extra inference budget and revision scaffolding.
- **Computation is mostly telemetry.** BMR `delta_F` is degenerate, `cit_temperature` is captured but never applied, Hopfield/storehouse memory is outside the cascade causal path, and `vimarsa_event` does not gate any commit decision.
- **Substrate is not clean.** The `claude --print` subprocess inherits Claude Code system context, plugin context, and skill context, so the "Haiku" baseline is actually "Haiku running inside Claude Code."

v0.3 closes all three. **No API key is used; OAuth via `claude` CLI stays the only auth path.**

## 1. Scope (frozen)

In scope (this version):

- **Clean Haiku CLI substrate** for the inner subprocess that `HaikuLM` spawns: `--print --system-prompt "You are a helpful assistant." --disable-slash-commands --strict-mcp-config --setting-sources "" --permission-mode bypassPermissions --no-session-persistence --output-format json --model haiku`, executed with `subprocess.run(env=clean_env, cwd=tmp_clean_dir)` where `clean_env` is built explicitly (allow-list only) and `HOME=/tmp/pce_home_<pid>/` contains only the OAuth credential symlink.
- **Outer host preserved.** The Claude Code session, the MCP server, the `python` benchmark driver, and CLAUDE.md auto-discovery are *not* sanitized. The PCE plugin must keep loading normally so `pce_cascade(...)` and the cascade orchestration can run at all. Isolation is strictly inner-subprocess only.
- **`IntegrityProbe`** at `src/pce/substrate/integrity.py` runs a one-shot `claude --print` inside the cleaned subprocess and asserts the response is leakage-free against a frozen regex (`Claude Code`, `skill`, `plugin`, `MCP`, `I appreciate`, `CLAUDE.md`). Probe outcome is cached and re-keyed by `(env_hash, flags_hash)`.
- **Active-inference uplift on the causal path:**
  - `jnana` BMR enumerates **aspect-conditioned reductions** (priors weighted by Hopfield retrieval per aspect) so the winning reduction reports informative `delta_F > 0` when the surface actually covers must-have aspects.
  - `apohana` queries `HopfieldStore` for nearby aspects (warm-start prior) and `vimarsa` calls `consolidate(state, mode)` to write the committed surface back. Per-domain stores at `audit/storehouse/<domain>.npz` (reset between domains during the benchmark to keep observations independent).
  - `cit_temperature` is plumbed end-to-end: `run_cascade -> iccha`. The parity sampler `tau` becomes `0.9 * cit_temperature`; recorded on `Candidate.sampler` for audit.
  - **Free-energy budget** at `src/pce/active_inference/budget.py` keeps a per-item ledger that earns / pays F based on (a) `delta_F` from jnana, (b) embedding distance to aspect prior, (c) committed token count. The shadow revision pass aborts when the ledger drops below threshold.
- **Causal vimarsa = event-gated commit + always-shadow revision:**
  - Pass 1 always runs (draft + vimarsa brief). Vimarsa receives `delta_F_draft` as evidence.
  - Pass 2 (shadow revision) always runs and is always scored. Persisted on `state.surface_revision`.
  - Commit policy: `if vimarsa_event then state.surface = revision else state.surface = draft`. Recorded as `state.committed in {"draft", "revision"}`.
  - The `bypass_vimarsa` knob is dropped in favor of an explicit `commit_policy: Literal["event_gated", "always_revise", "always_draft"]` so the four benchmark arms share one cascade entry point.
- **Four-arm benchmark on the same v0.2 sample:**
  - `haiku_bare` (1 call, parity sampler).
  - `haiku_cascade` (event-gated commit, always-shadow revision; the architectural arm).
  - `haiku_bare_2K_scorer` (best-of-K=2K with the same embedding scorer, no revise) — the matched-budget control.
  - `haiku_generic_revise_2pass` (2-pass with a generic "make this more vivid and specific" brief, no apohana / jnana / vimarsa) — the matched-revision control.
- **Statistics rebuild:**
  - **H1.v3-H4.v3**: per-domain `haiku_cascade` vs `haiku_bare` (architecture vs nothing).
  - **H5.v3 redesigned**: composite Hedges' g across H1.v3-H4.v3 (paired effect-size meta-aggregate), not z-scored deltas.
  - **H6.v3**: `haiku_cascade` vs `haiku_bare_2K_scorer` (architecture vs more compute) — the headline fairness contrast.
  - **H7.v3**: `haiku_cascade` vs `haiku_generic_revise_2pass` (architecture vs generic 2-pass) — isolates brief content from brief existence.
  - **H8.v3**: paired `revision_score` vs `draft_score` within `haiku_cascade` for items where the event committed revision; Hedges' g + permutation.
- Length-controlled scoring on AUT and sci_creativity composites (per-token normalization + verbosity penalty).
- Strict JSON output (`json.dump(allow_nan=False)`), missing values as `null`.

Out of scope (this version, explicitly):

- No new domains beyond v0.2's four.
- No `local_bare` / `local_cascade` arm (per user constraint).
- No live Sonnet judge run, no Sonnet bridge invocation. (Bridge code stays present for offline use.)
- No new plugin marketplace re-submission beyond version bump.
- No fine-tuning or training of any LM.
- No Anthropic SDK / API-key dependent path; v0.3 OAuth via `claude` CLI only.
- No expansion of the benchmark sample size (frozen at v0.2's `n=5`/domain).

## 2. Hypotheses (pre-registered for v0.3)

| ID | Statement | Treatment | Control | Domain | n target | α |
|----|-----------|-----------|---------|--------|----------|---|
| **H1.v3** | `haiku_cascade` > `haiku_bare` on the AUT composite. | `haiku_cascade` | `haiku_bare` | aut | >=5 | 0.05 |
| **H2.v3** | `haiku_cascade` > `haiku_bare` on the Wittgenstein poetry-interp aspect-multiplicity composite. | `haiku_cascade` | `haiku_bare` | poetry_interp | >=5 | 0.05 |
| **H3.v3** | `haiku_cascade` > `haiku_bare` on the POEMetric-style poetry-gen composite. | `haiku_cascade` | `haiku_bare` | poetry_gen | >=5 | 0.05 |
| **H4.v3** | `haiku_cascade` > `haiku_bare` on the BBH-style scientific-creativity composite. | `haiku_cascade` | `haiku_bare` | sci_creativity | >=5 | 0.05 |
| **H5.v3** | Composite Hedges' g across H1.v3-H4.v3 (fixed-effects meta-aggregate of paired effect sizes) is positive. | `haiku_cascade` | `haiku_bare` | aggregate | n_total >= 20 | 0.05 |
| **H6.v3** | `haiku_cascade` > `haiku_bare_2K_scorer` on the aggregate composite — matched-budget fairness contrast. | `haiku_cascade` | `haiku_bare_2K_scorer` | aggregate | n_total >= 20 | 0.05 |
| **H7.v3** | `haiku_cascade` > `haiku_generic_revise_2pass` on the aggregate composite — matched-revision contrast. | `haiku_cascade` | `haiku_generic_revise_2pass` | aggregate | n_total >= 20 | 0.05 |
| **H8.v3** | Within `haiku_cascade` items where the event committed revision, `revision_score` > `draft_score` paired. | revision | draft | within-treatment | n_event >= 1 | 0.05 |

Statistical method: paired permutation tests (`scipy.stats.permutation_test`), Hedges' g, Wilcoxon signed-rank, BCa bootstrap CI (10,000 resamples), Holm-Bonferroni across `{H1.v3, H2.v3, H3.v3, H4.v3}`. Power computed both a-priori and retrospectively. All JSON output strict (`allow_nan=False`).

H6.v3 is the headline contrast — it directly answers the v0.2 reviewer's "is it just more compute?" critique.

## 3. Architecture (engineering view, deltas from v0.2)

```
+-- Substrate layer (v0.3) -----------------------------------------+
|  src/pce/substrate/                                                |
|  +-- lm_protocol.py    GeneratorProtocol with capability flags    |
|  +-- lm.py             LocalLM (kept; not benchmarked in v0.3)    |
|  +-- haiku_lm.py       HaikuLM via clean inner subprocess         |
|  +-- integrity.py      IntegrityProbe (NEW)                       |
|  +-- embed.py          Embedder (unchanged)                       |
|  +-- hopfield.py       HopfieldStore (now in cascade causal path) |
|  +-- (subprocess env)  scrubbed HOME + tmp cwd + allow-list flags |
+--------------------------------------------------------------------+

+-- Active inference (NEW for v0.3) ---------------------------------+
|  src/pce/active_inference/                                         |
|  +-- budget.py         per-item free-energy ledger                 |
+--------------------------------------------------------------------+

+-- Cascade event-gated commit + always-shadow revision -------------+
|  draft       = kriya(jnana_pick(iccha(prompt, K, cit_temperature)))|
|  brief       = vimarsa(draft, aspects, evidence={delta_F_draft})   |
|  shadow_rev  = kriya(jnana_pick(iccha(prompt + brief, K, ...)))    |
|  if event:                                                         |
|      state.surface = shadow_rev                                    |
|      state.committed = "revision"                                  |
|  else:                                                             |
|      state.surface = draft                                         |
|      state.committed = "draft"                                     |
|  state.surface_draft, state.surface_revision always populated      |
+--------------------------------------------------------------------+

+-- Benchmark four arms (v0.3) --------------------------------------+
|  haiku_bare                : HaikuLM .generate(prompt)              |
|  haiku_cascade             : run_cascade(haiku, commit_policy=event)|
|  haiku_bare_2K_scorer      : best-of-K=2K, same embedding scorer    |
|  haiku_generic_revise_2pass: 2-pass generic brief, no PCE operators |
|  Headline contrasts:                                               |
|    H1.v3-H4.v3, H5.v3 -- haiku_cascade vs haiku_bare                |
|    H6.v3              -- haiku_cascade vs haiku_bare_2K_scorer      |
|    H7.v3              -- haiku_cascade vs haiku_generic_revise_2pass|
|    H8.v3              -- revision vs draft within haiku_cascade     |
+--------------------------------------------------------------------+
```

## 4. Operator semantics changes (deltas from v0.2)

### 4.0 GeneratorProtocol (renamed, kept `LMProtocol` alias)

```python
class GeneratorProtocol(Protocol):
    name: str
    supports_logprobs: bool   # NEW; HaikuLM = False
    supports_score: bool      # NEW; HaikuLM = False
    supports_entropy: bool    # NEW; HaikuLM = False
    def generate(self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int) -> Candidate: ...
    def report(self) -> dict[str, Any]: ...
    def length_proxy_logp(self, candidate: Candidate) -> float: ...   # NEW; calibrated proxy
```

### 4.1 iccha

- `cit_temperature` is plumbed: `parity_sampler["tau"] = 0.9 * cit_temperature`. Recorded on `Candidate.sampler["cit_temperature"]`.
- Otherwise unchanged from v0.2 (`prompt_mode="verbatim"`, `sampler_grid_mode="parity"`).

### 4.2 apohana

- Adds `hopfield_query: HopfieldStore | None = None` kwarg. When present, queries the store for `top-K` nearby aspects and adds their inner-product strengths as a soft warm-start prior on `must_have` and `must_avoid` distances.
- Default off for backward compatibility; `run_cascade` enables it.

### 4.3 jnana

- `_enumerate_reductions` rewritten to enumerate **aspect-conditioned reductions**: each reduction asserts a hypothesis "the surface satisfies aspect subset `S_i`". Priors weighted by Hopfield retrieval strength per aspect when a store is supplied.
- Returns `delta_F` that is informative (>0 when winning reduction has higher posterior on must-have aspects) — verifiable on prove-gate fixtures.
- Falls back to a uniform aspect prior when no aspect dictionary is supplied (e.g., AUT) so AUT items still get a meaningful posterior.

### 4.4 vimarsa

- New `evidence_points` accepts `delta_F_draft` (and any future scalar evidence). When `|delta_F_draft|` exceeds `delta_F_threshold` (default 0.05), it counts as one evidence point.
- New `consolidate(state, mode)` hook: vimarsa calls `HopfieldStore.write_back` at end of cascade to compound the storehouse across prompts.
- `commit_policy` not handled here — that's cascade-level.

### 4.5 cascade.run_cascade

- New kwarg: `commit_policy: Literal["event_gated", "always_revise", "always_draft"] = "event_gated"`.
- New kwarg: `delta_F_threshold: float = 0.05`.
- Drops `bypass_vimarsa`. The four benchmark arms map to:
  - `haiku_bare`: not via run_cascade; goes straight through `HaikuLM.generate` once.
  - `haiku_cascade`: `commit_policy="event_gated"`.
  - `haiku_bare_2K_scorer`: a separate driver path, not run_cascade — emits K candidates via iccha + jnana on Haiku, no revision pass.
  - `haiku_generic_revise_2pass`: `commit_policy="always_revise"`, brief replaced by a generic brief, apohana / jnana / vimarsa knobs minimized.
- Always populates `state.surface_draft` and `state.surface_revision`. `state.committed in {"draft", "revision"}`.
- Free-energy budget consulted before the shadow revision pass; if ledger underwater, abort with a structured `BudgetUnderwaterError` (counted on the run summary; the shadow_revision is recorded as `None` and excluded from H8.v3 for that item only).

## 5. Plugin surface (deltas from v0.2)

- Manifest version bumped `0.2.0 -> 0.3.0` in `plugin/.claude-plugin/plugin.json` and `pyproject.toml`.
- New MCP tool `haiku_clean_substrate_probe()` returns the cached `IntegrityProbe` JSON.
- New MCP tool `hopfield_state(domain: str)` returns the storehouse snapshot for a domain (read-only view).
- `pce_cascade(arm: Literal["haiku", "haiku_bare_2K", "haiku_generic_revise"], commit_policy=..., max_tokens=..., K=..., cit_temperature=...)` — `local` and `local_cascade` arms removed from the tool surface (per scope freeze; LocalLM remains importable for backward compatibility).
- New env vars:
  - `PCE_HAIKU_CLEAN_HOME` (override scrubbed HOME path; default `/tmp/pce_home_<pid>/`).
  - `PCE_INTEGRITY_PROBE_INTERVAL` (every-N items; default 10).
  - `PCE_DELTA_F_THRESHOLD` (default 0.05).

## 6. Benchmark protocol (v0.3 pilot)

### 6.1 Subjects

Same four domains and the *same items + seeds* as the v0.2 pilot (frozen). `n=5`/domain (matches v0.2 pilot scope; re-using v0.2's frozen item bank).

### 6.2 Pilot configuration

- Arms: `haiku_bare`, `haiku_cascade`, `haiku_bare_2K_scorer`, `haiku_generic_revise_2pass`.
- K=4 for `haiku_cascade` and `haiku_generic_revise_2pass`; K=2 (best-of-2 with scorer) for `haiku_bare_2K_scorer` (re-uses the v0.2 K=4 default budget envelope; named "_2K_" to flag matched-budget intent — the actual K is config-driven).
- `max_tokens` = 200.
- Seed = 4242.
- IntegrityProbe runs at the start of each domain plus every 10 items thereafter.
- Cost ledger `audit/cost_ledger.json` records every Haiku call; pilot must total under $20.

### 6.3 Judging

- Pilot uses local proxy scorers + length-controlled scoring layer.
- Sonnet judge bridge (`scripts/run_judge_bridge.py`) NOT invoked in v0.3 (per scope).

### 6.4 Statistics

`benchmarks/stats.py` rewritten:

- H1.v3-H4.v3 per domain: paired (`haiku_cascade` - `haiku_bare`).
- H5.v3 aggregate: fixed-effects composite Hedges' g across H1.v3-H4.v3.
- H6.v3 matched-budget: paired (`haiku_cascade` - `haiku_bare_2K_scorer`).
- H7.v3 matched-revision: paired (`haiku_cascade` - `haiku_generic_revise_2pass`).
- H8.v3 within-treatment revision causal contribution: paired (`revision_score` - `draft_score`) for items where event committed revision.
- All JSON written with `allow_nan=False`; missing values serialized as `null`.

## 7. Acceptance criteria for v0.3.0 release

Engineering gates (all must pass):

- All operator changes per §4 with mypy --strict and ruff clean.
- `pytest` passes including new `tests/test_jnana_aspect_bmr.py`, `tests/test_apohana_hopfield.py`, `tests/test_cit_temperature.py`, `tests/test_free_energy_budget.py`, `tests/cascade_event_gated_test.py`, `tests/integrity_probe_test.py`.
- `scripts/prove_gate.py` exits zero on duck-rabbit textual + AUT brick: `vimarsa_event=True` for at least one of the two cases on `haiku_cascade`, `delta_F` non-degenerate on both cases for `haiku_cascade`, `IntegrityProbe.passes` for every Haiku subprocess call, leakage regex passes on every Haiku subprocess output, `revision_differs_from_draft` whenever commit policy committed revision.
- `scripts/verify_outer_host_loads_pce.py` passes: outer host can still discover and load the PCE plugin.
- `scripts/validate_paper.py` clean.
- Plugin smokes (`scripts/smoke_plugin.py --with-haiku`, `scripts/verify_plugin.py`, `scripts/verify_real_model.py`) pass with the v0.3 manifest.
- `audit/cost_ledger.json` total under $20.
- CI green; `v0.3` branch merged to `main`; tag `v0.3.0` pushed.

Research-hypothesis gates (any of these is the success signal; report the result honestly either way):

- *Primary success*: `H6.v3` directionally supported (matched-budget contrast).
- *Secondary success* (sufficient even if H6.v3 fails): `H7.v3` directionally supported (matched-revision contrast).
- *Tertiary success*: at least one of `{H1.v3, H2.v3, H3.v3, H4.v3, H8.v3}` supported after Holm-Bonferroni.
- *Honest negative*: if none hold, the paper, presentation, and README report the negative result and propose v0.4 design changes. v0.3 still ships.

## 8. Risk register (deltas)

| Risk | Mitigation |
|------|-----------|
| `claude` CLI still injects system context with `--system-prompt` set | IntegrityProbe blocks the run before incurring spend; probe regex frozen in `src/pce/substrate/integrity.py`. |
| OAuth credentials missing from scrubbed HOME (`claude` CLI auth fails) | Explicitly symlink the OAuth credential file into the scrubbed HOME; HaikuLM raises a structured `CleanSubstrateAuthError` if first call returns `401`. |
| Hopfield-in-cascade slows throughput unacceptably | Per-prompt warm-start only, no hot loops; size-bounded store per domain. |
| BMR aspect reductions on items without aspect dictionaries | Uniform aspect-prior fallback per ADR-003; AUT items get a flat reduction grid. |
| Free-energy budget aborts too many revision passes | Soft default threshold (`PCE_FE_BUDGET_FLOOR=-2.0`); env-var override; abort rate logged in `state.audit`. |
| H6.v3 budget-matched control inflates cost above $15 envelope | `haiku_bare_2K_scorer` uses the same K as `haiku_cascade` draft pass (K=4), not literally 2K samples; "_2K_" labels the *intent* of matched budget against cascade's revision call. |
| Outer Claude Code session may set `CLAUDE_CODE_*` env vars that leak to subprocess | HaikuLM constructs `clean_env` from a frozen allow-list; never inherits `os.environ` blindly. Warning emitted at `__init__` if parent env contains `CLAUDE_CODE_*`. |
