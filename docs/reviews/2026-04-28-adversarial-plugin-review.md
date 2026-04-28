# Adversarial Review: Pratyabhijna Creative Engine v0.1

Date: 2026-04-28  
Reviewer: Cursor agent, with parallel specialist subagents  
Scope: research background, `docs/` plan/spec/PRD/ADRs, engine implementation, plugin surface, benchmark raw data, statistics, paper/presentation claims, and live Claude Code CLI plugin behavior.

## Executive Summary

This review finds that the repository is **not faking its reported benchmark numbers**: the per-domain JSON files, `benchmarks/results/stats.json`, `paper/autoreport.tex`, and README headline table are internally consistent. Selected rows recompute exactly from raw output text. The negative result is real.

However, the negative result should not be read as "Pratyabhijna + active inference cannot help creativity." The stronger adversarial conclusion is:

1. The shipped benchmark is **not the same experiment described by the SPEC/PRD**. The SPEC repeatedly says "PCE-Haiku vs no-PCE Haiku"; the executed benchmark tests `local_cascade` (Qwen2-1.5B + PCE) vs `claude_haiku`, plus sensitivity vs `local_bare`.
2. The plugin trails bare/local and Haiku mostly because the actual cascade is **not activating its central recursive mechanism**. In the benchmark, `vimarsa_event` is 0/30, and the cascade implementation makes that outcome structurally likely.
3. Even if `vimarsa` fires, it is currently **post-hoc telemetry**, not a causal revision loop. It does not alter the returned surface text.
4. The selection objective (`jnana` over `ananda` + clipped `apohana`) is **not aligned with the benchmark scoring objective** and can choose candidates that score worse than the bare sample.
5. The `iccha` prompt wrapper and sampler grid create a **distribution shift** relative to `local_bare`, so the sensitivity comparison is not a clean "same model, PCE only" ablation.
6. The live Claude Code plugin works at the MCP level, but it ships with `.mcp.json` hard-pinning `PCE_LM_DEVICE=cpu` and `PCE_LM_DTYPE=float32`, while the benchmark path used auto-detected `mps`/`float16`. That makes the live plugin slower and operationally different from the benchmarked engine.

The most important v0.2 fix is therefore not more documentation. It is to make `vimarsa` causal and measurable:

- fix the cascade's impossible switching gate;
- define domain-specific aspect evidence for domains without explicit aspects;
- re-enter `iccha`/`kriya` when `vimarsa` fails or fires;
- run a same-substrate, prompt-matched, sampler-matched ablation before comparing to Haiku;
- use judge validation to determine whether the proxy metrics measure the intended creative nuance.

## Clarifying Questions

These questions are not blockers for this review, but they determine the correct v0.2 target:

1. Should v0.2 optimize for **beating `local_bare` on the same local substrate**, or for **beating Claude Haiku**? Those are different research claims.
2. Is the desired plugin a **local-model cognitive layer**, or should it wrap/steer **Claude/Haiku/Sonnet as the generative substrate**?
3. Should benchmark success prioritize **surface literary quality**, **aspect-shift multiplicity**, or **visible recursive self-revision traces**?
4. Are you willing to pay for a small but real **Sonnet/Opus or human-judge validation set** to calibrate the local proxy metrics?
5. Should `vimarsa` be allowed to **rewrite the output**, or should it remain only an auditor/detector?

## Methodology

I used both direct inspection and parallel specialist review:

- Document adversarial review of `research1.md`, `docs/SPEC.md`, `docs/PRD.md`, `docs/plan.md`, `docs/operator-spec.md`, `docs/ADR-*.md`, README, and paper claims.
- Python implementation review of `src/pce/cascade.py`, `src/pce/operators/{vimarsa,jnana,iccha,ananda,apohana,kriya}.py`, and `src/pce/substrate/lm.py`.
- Benchmark/statistics review of `benchmarks/{driver,scoring,stats,items}.py`, `benchmarks/results/*.json`, `paper/autoreport.tex`, and README tables.
- Plugin/CLI review of `plugin/.claude-plugin/plugin.json`, `plugin/.mcp.json`, `plugin/mcp/server.py`, `plugin/commands`, `plugin/skills`, `plugin/agents`, and hooks.
- Live Claude Code CLI checks with `--plugin-dir ./plugin`.
- Targeted Python probes that recomputed raw scores and reproduced `vimarsa` gating behavior.

## Tests and Probes Run

### 1. Claude plugin manifest validation

Command:

```bash
claude plugin validate ./plugin
```

Result:

```text
Validating marketplace manifest: .../plugin/.claude-plugin/marketplace.json
✔ Validation passed
```

Interpretation: the plugin manifest is valid for Claude Code's plugin loader.

### 2. Installed plugin inventory

Command:

```bash
claude plugin list
```

Relevant result:

```text
triz-engine@triz-arena        enabled
pratyaksha-context-eng-harness@pratyaksha-context-eng-harness enabled
```

Interpretation: relevant companion plugins are present, but `pratyabhijna-creative-engine` is not globally installed. The live tests below used `--plugin-dir ./plugin`, which is appropriate for session-local validation.

### 3. Real Claude Code CLI tool call: `reset_state`

Command:

```bash
claude -p --model haiku --plugin-dir ./plugin \
  --permission-mode bypassPermissions \
  --allowedTools pratyabhijna_mcp__reset_state \
  --output-format json \
  "Use the pratyabhijna_mcp__reset_state tool exactly once..."
```

Result summary:

```text
success; tool result summarized as:
"The Hopfield store was reset with no patterns stored before or after."
```

Interpretation: Claude Code can load the plugin directory, expose the MCP server, and call a real tool.

### 4. Real Claude Code CLI tool call: standalone `vimarsa`

Command:

```bash
claude -p --model haiku --plugin-dir ./plugin \
  --permission-mode bypassPermissions \
  --allowedTools pratyabhijna_mcp__vimarsa \
  --output-format json \
  "Call pratyabhijna_mcp__vimarsa exactly once with prompt='Interpret the duck-rabbit figure', ..."
```

Result summary from Claude:

```text
event=true
novelty=0.512
aspect_count=2
switching=0
ananda=0.9
```

Audit confirmation in `audit/phase8/mcp_calls.jsonl`:

```json
{"tool":"vimarsa","result":{"event":true,"novelty":0.5115566849708557,"diagnostics":{"aspect_count":2.0,"switching":0.0,"ananda":0.9}}}
```

Interpretation: the standalone MCP `vimarsa` tool can fire because `plugin/mcp/server.py` passes `iccha_apoha_trajectory=None`. The cascade path does not.

### 5. Real Claude Code CLI tool call: `report`

Command:

```bash
claude -p --model haiku --plugin-dir ./plugin \
  --permission-mode bypassPermissions \
  --allowedTools pratyabhijna_mcp__report \
  --output-format json \
  "Call pratyabhijna_mcp__report exactly once..."
```

Result summary:

```text
LM model: Qwen/Qwen2-1.5B-Instruct
Device: cpu
Dtype: float32
Embedder: sentence-transformers/all-MiniLM-L6-v2
Hopfield patterns: 0
```

Interpretation: live plugin runtime is hard-pinned to CPU/float32 by `plugin/.mcp.json`, unlike the benchmark path that auto-detected MPS/float16. This is a real operational mismatch.

### 6. Stats recomputation from raw JSON

Probe recomputed paired means from `benchmarks/results/*.json` and compared them to `benchmarks/results/stats.json`.

Result:

```text
aut            n=8   mean_delta=-0.283813029089 stats=-0.283813029089
poetry_interp  n=10  mean_delta=-0.161013267934 stats=-0.161013267934
poetry_gen     n=12  mean_delta=-0.030537809204 stats=-0.030537809204
sci_creativity n=8   mean_delta=-0.053009173988 stats=-0.053009173988
```

Interpretation: reported primary estimates match raw data exactly.

### 7. Selected benchmark row recomputation

Recomputed selected cases from raw text:

```text
poetry_gen p01:
  claude_haiku   reported=0.547660347 recomputed=0.547660347
  local_bare     reported=0.486098004 recomputed=0.486098004
  local_cascade  reported=0.349375012 recomputed=0.349375012

poetry_interp i01:
  claude_haiku   reported=0.517406483 recomputed=0.517406483
  local_bare     reported=0.257566929 recomputed=0.257566929
  local_cascade  reported=0.260355850 recomputed=0.260355850

aut a01:
  claude_haiku   reported=0.920170397 recomputed=0.920170397
  local_bare     reported=0.661180468 recomputed=0.661180468
  local_cascade  reported=0.630663788 recomputed=0.630663788

sci_creativity s01:
  claude_haiku   reported=0.513406659 recomputed=0.513406659
  local_bare     reported=0.501642873 recomputed=0.501642873
  local_cascade  reported=0.469583829 recomputed=0.469583829
```

Interpretation: selected benchmark rows are not fabricated or manually altered.

### 8. Raw benchmark gap summary

Probe over all committed benchmark rows:

```text
domain,n,mean_cascade,mean_bare,mean_haiku,delta_cascade_minus_bare,delta_cascade_minus_haiku,events,near_token_cap_local_bare,near_token_cap_cascade
poetry_gen,12,0.529071,0.555012,0.559608,-0.025941,-0.030538,0,6,11
poetry_interp,10,0.416650,0.434818,0.577664,-0.018167,-0.161013,0,8,9
aut,8,0.636996,0.624991,0.920809,+0.012005,-0.283813,0,7,7
sci_creativity,8,0.465231,0.474756,0.518240,-0.009525,-0.053009,0,7,8
```

Interpretation:

- `local_cascade` only weakly beats `local_bare` on AUT.
- `vimarsa_event` is 0 in every committed benchmark row.
- Many local outputs look near token cap/truncation, especially cascade outputs. The current `max_tokens=120` cap is likely too tight for domains asking for paragraphs/lists.

### 9. Direct `vimarsa` structural gate probe

Using the same surface and aspects:

```text
standalone trajectory=None:
  event=True, novelty=0.53598, aspect_count=2, switching=0

cascade-like one-point trajectory:
  event=False, novelty=0.53598, aspect_count=2, switching=0

no aspects:
  event=False, novelty=0.53598, aspect_count=0, switching=0
```

Interpretation: the same duck-rabbit surface fires in standalone mode but fails under cascade-like conditions solely because the cascade supplies a one-point trajectory, making `switching >= 2` impossible. Domains without explicit aspects fail the aspect-count conjunct by construction.

### 10. `jnana` negative apohana clipping probe

Probe:

```text
apoha=[-10, 0] => selected=0, posterior=[0.5, 0.5]
apoha=[0, 0]   => selected=0, posterior=[0.5, 0.5]
apoha=[10, 0]  => selected=0, posterior=[0.916667, 0.083333]
```

Interpretation: strongly bad `apohana` evidence is equivalent to neutral evidence because `jnana` uses `np.clip(apoha, 0.0, None)`. Contrastive exclusion currently rewards positive avoidance but does not penalize negative avoidance.

### 11. One real local benchmark probe

Probe: poetry interpretation item `i01`, Qwen2-1.5B, `K=2`, `max_tokens=32`.

Result:

```text
lm_device=mps, lm_dtype=float16
bare_meta={'ok': True, 'elapsed_s': 3.79}
cascade_meta={'ok': True, 'elapsed_s': 8.69, 'vimarsa_event': False, 'novelty': 0.51598, 'delta_F': 0.0, 'selected_idx': 0}
bare_score=0.237439
cascade_score=0.207817
```

Interpretation: even on a tiny fresh probe, the cascade takes more than 2x wall-clock, does not fire `vimarsa`, and scores lower than bare under the local proxy metric.

### 12. Verification scripts

Command:

```bash
uv run python scripts/verify_plugin.py
uv run pytest tests/test_stats.py tests/test_pipeline.py -q
```

Result:

```text
verify_plugin.py: ok=true
pytest: 9 passed
```

Interpretation: the plugin structure and benchmark pipeline tests pass. The issue is not basic brokenness; it is experiment identity and mechanism inactivity.

## Findings

### P0-1: The primary benchmark is not the SPEC's stated experiment

Evidence:

- `docs/SPEC.md` describes H1-H4 as **PCE-Haiku vs no-PCE Haiku**.
- `benchmarks/driver.py` actually runs:
  - `claude_haiku`: Claude Haiku via CLI;
  - `local_bare`: Qwen2-1.5B raw;
  - `local_cascade`: Qwen2-1.5B through PCE.
- `benchmarks/stats.py` sets treatment to `local_cascade` and primary control to `claude_haiku`.

Why it matters:

This confounds architecture and model substrate. A small Qwen2 model plus PCE losing to Haiku does not answer whether a PCE layer improves Haiku outputs. The result is still useful, but it is a different result:

> PCE on Qwen2-1.5B did not beat Claude Haiku under local proxy metrics.

That is not:

> PCE-Haiku did not beat Haiku.

Recommendation:

- Rewrite SPEC/PRD/ADR-004 to match actual arms, or implement the original same-model Haiku A/B.
- For v0.2, make `local_cascade` vs `local_bare` the primary architecture test, and Haiku only the external quality reference.

### P0-2: `vimarsa` cannot fire in the cascade as currently wired

Evidence:

`src/pce/cascade.py` constructs:

```python
trajectory = [(e_iccha, e_apoha)]
```

and passes that non-`None` trajectory to `vimarsa`.

`src/pce/operators/vimarsa.py` then requires:

```python
switching_ok = switching >= int(switching_threshold)
```

with `switching_threshold = 2`.

A one-point trajectory has zero transitions. Therefore `switching_ok` is always false in `run_cascade`.

Additionally:

- poetry generation and AUT pass `aspects=[]`, so `aspect_count=0` and `aspect_ok` is always false.
- the direct probe showed the same surface fires when `trajectory=None` and fails when a one-point trajectory is supplied.

Why it matters:

The central novel mechanism is not merely "underperforming." It is structurally blocked in the main cascade path. H6 being 0/30 is not an ordinary null. It is a measurement/integration failure.

Recommendation:

- Treat `iccha_apoha_trajectory=None` as the current cascade default until there is a real multi-step trajectory.
- Or change switching semantics so a one-shot cascade has no switching requirement.
- Domain-profile `vimarsa`: require aspects only where the domain supplies aspects, and use alternate multiplicity signals for poetry generation/AUT.
- Add a regression test: a duck-rabbit surface that fires via standalone `vimarsa` should also fire through `run_cascade` when given the same aspects/retrieval set.

### P0-3: `vimarsa` does not affect output even if it fires

Evidence:

`run_cascade` computes:

1. candidates via `iccha`;
2. selected candidate via `jnana`;
3. final surface via `kriya`;
4. `vimarsa` after the final surface.

The returned `state.surface` is not revised based on `vimarsa_event`.

Why it matters:

The benchmark scores surface text. A detector that runs after text selection cannot improve the text unless it gates resampling, revision, or rendering. Current `vimarsa` is audit telemetry, not a recursive self-reflexivity layer in the generative sense promised by the research narrative.

Recommendation:

Implement a causal loop:

```text
iccha -> apohana/ananda -> jnana -> kriya draft -> vimarsa
if no event and task expects multiplicity:
    revise constraint / resample / expand aspects / re-run kriya
if event:
    optionally preserve or amplify the aspect-shift surface
```

Record both pre- and post-`vimarsa` surfaces so the causal contribution is testable.

### P0-4: Release acceptance criteria are not met but documents still imply v0.1 success

Evidence:

`docs/SPEC.md` acceptance criteria include:

- H1 + H2 + H5 directional support after Holm-Bonferroni;
- `vimarsa_event=True` for at least 9/10 duck-rabbit textual probes;
- paper compiles cleanly.

Actual state:

- H1-H5 unsupported;
- H6 undefined (`n_fired=0`);
- `pdflatex` was unavailable locally; only a structural validator was run;
- `docs/plan.md` still shows phases 3-11 pending/stale.

Why it matters:

The repo is honest in README/paper about the negative result, but the SPEC still encodes success gates that the project did not clear.

Recommendation:

Split acceptance criteria:

- engineering release criteria;
- research-hypothesis success criteria;
- negative-result publication criteria.

Then update `docs/plan.md`, `docs/SPEC.md`, and PRD status to "as shipped."

### P1-1: Live plugin runtime differs from benchmark runtime

Evidence:

`plugin/.mcp.json` pins:

```json
"PCE_LM_DEVICE": "cpu",
"PCE_LM_DTYPE": "float32"
```

The live Claude CLI `report` tool confirmed:

```text
Device: cpu
Dtype: float32
```

But `LocalLM()` in the benchmark path auto-detects MPS/float16, confirmed by the probe:

```text
lm_device=mps, lm_dtype=float16
```

Why it matters:

The actual installed plugin is slower and operationally different from the benchmarked system. Users may experience timeouts or far worse latency than the paper/README imply.

Recommendation:

- Remove forced CPU/float32 from `.mcp.json`.
- Use `PCE_DEVICE`/`PCE_DTYPE` auto-detection consistently.
- Provide a documented override for deterministic CPU mode.

### P1-2: `cit_temperature` is a dead public parameter in `run_cascade`

Evidence:

`run_cascade` accepts `cit_temperature`, stores it in `CascadeState`, but does not pass it into `iccha` or `cit`.

Why it matters:

Skills/docs advise changing temperature, but the cascade public API silently ignores it. That makes tuning misleading and could hide the actual source of underperformance.

Recommendation:

- Plumb `cit_temperature` into `iccha`'s sampler grid or remove it from the public API.
- Add a test that different cascade temperatures change candidate token distributions under a fixed seed.

### P1-3: `iccha` changes the prompt relative to the bare baseline

Evidence:

`local_bare` generates from the benchmark prompt verbatim.

`iccha` transforms it into:

```python
f"{prompt.rstrip()}\nWrite a response that is {constraint.text}.\n\n"
```

For many benchmark prompts the prompt already includes explicit instructions. This duplicates or slightly changes the instruction surface.

Why it matters:

The `local_cascade` vs `local_bare` sensitivity contrast is not prompt-matched. On a 1.5B model, extra instruction prose can degrade instruction following or cause repetition.

Recommendation:

Add an ablation:

- `local_bare_verbatim`
- `local_bare_with_constraint_suffix`
- `local_cascade_no_suffix`
- `local_cascade_current`

Only compare architecture after prompt parity.

### P1-4: `jnana` ignores negative contrastive evidence

Evidence:

`src/pce/operators/jnana.py` uses:

```python
np.clip(apoha, 0.0, None)
```

Probe:

```text
apoha=[-10, 0] and apoha=[0, 0] produce identical posterior=[0.5, 0.5]
```

Why it matters:

`apohana` is supposed to exclude "all-other-than-this." Current `jnana` only rewards positive exclusion; it does not penalize candidates close to avoid regions.

Recommendation:

Use a signed or shifted transform:

- min-max normalize `apoha` over candidates;
- use `softplus(scale * apoha)` with a negative branch;
- or subtract a penalty for `apoha < 0`.

Then test that a candidate semantically close to `must_avoid` loses posterior mass.

### P1-5: `jnana`/`ananda` objective is not the benchmark objective

Evidence:

`jnana` selects using `ananda` and `apohana`. The benchmark scorers reward different proxies:

- poetry: keyword imagery, emotional words, lexical diversity, literary-device markers;
- AUT: fluency, originality, elaboration, flexibility;
- science: frame coverage, novelty, specificity;
- interpretation: aspect coverage and retrieval novelty.

Why it matters:

The cascade can rationally choose a candidate that scores worse under the benchmark. That is exactly what appears to happen in examples like `poetry_gen p01`.

Recommendation:

Add an "oracle over candidates" analysis:

1. generate K candidates using current `iccha`;
2. score all K with the domain scorer;
3. compare `jnana` selection to scorer argmax;
4. quantify regret.

If regret is high, tune `ananda`/`jnana` or expose a domain reward hook.

### P1-6: Max token budget likely truncates local outputs

Evidence:

Heuristic truncation counts:

```text
poetry_gen:     local_bare 6/12, cascade 11/12
poetry_interp:  local_bare 8/10, cascade 9/10
aut:            local_bare 7/8,  cascade 7/8
sci_creativity: local_bare 7/8,  cascade 8/8
```

Direct JSON examples show many local responses ending mid-sentence.

Why it matters:

Haiku produces complete polished outputs while local arms are token-capped. This affects fluency, coverage, elaboration, and aspect count. Cascade is worse because it spends multiple samples at the same per-candidate cap and then selects one incomplete candidate.

Recommendation:

- Raise `max_tokens` for domains that request paragraphs or 8-item lists.
- Add `truncation_rate` to the benchmark report.
- Score with and without truncated rows.

### P1-7: H5 pooling is statistically loose

Evidence:

H5 z-scores domain deltas and concatenates 38 observations with weights. This treats per-item z-scores as one pooled sample.

Why it matters:

The domains are different tasks with different score distributions and different n. H5 is useful as an exploratory composite but not a strong confirmatory aggregate.

Recommendation:

Use one of:

- domain-level meta-analysis of mean deltas;
- hierarchical bootstrap over domain then item;
- multivariate permutation preserving domain structure.

### P1-8: Metrics are internally consistent but not the SPEC's judges

Evidence:

`benchmarks/scoring.py` says:

```text
No LLM-as-judge.
```

But SPEC/PRD describe Sonnet/Opus-style judging and POEMetric/CreativityPrism-inspired rubrics.

Why it matters:

The proxy metrics may not measure "creative nuance." Some axes reward keyword overlap, length, and embedding thresholds rather than literary merit or aspectual transformation.

Recommendation:

Run a judge bridge:

- sample 30 pairs;
- score with local proxies and Sonnet/Opus/human rubric;
- compute Spearman correlation;
- only trust proxy metrics where correlation is adequate.

### P1-9: Plugin public API does not match SPEC public API

Evidence:

SPEC names tools like `cascade_run`, `op_cit`, `bench_score`, `audit_log`.

`plugin/mcp/server.py` exposes:

```text
cit, iccha, apohana, ananda, jnana, kriya, vimarsa, cascade,
hopfield_store, hopfield_recall, consolidate_sws, consolidate_rem,
consolidate_cycle, report, reset_state
```

Commands and skills similarly use `pce-*` names rather than the SPEC's `/pratyabhijna_*` names.

Why it matters:

The plugin works, but "matches SPEC" is false by identifier and lifecycle contract.

Recommendation:

Pick one canonical API:

- rename/wrap server tools to match SPEC; or
- regenerate SPEC §5 from the actual plugin.

### P2-1: `vimarsa` aspect requirement excludes entire domains

Evidence:

`benchmarks/driver.py` passes `aspects=[]` for `poetry_gen` and `aut`.

`vimarsa` requires `aspect_count >= 2`.

Why it matters:

`vimarsa` can never fire in those domains unless aspects are provided or the criterion is domain-specific.

Recommendation:

For domains without explicit aspect targets, compute aspects dynamically:

- cluster candidate interpretations;
- extract line-level semantic facets;
- use LLM or embedding topic labels;
- define AUT category diversity as aspect multiplicity.

### P2-2: `delta_F` is mostly uninformative in benchmark metadata

Evidence:

The direct local probe produced:

```text
delta_F=0.0
```

Several committed rows show `delta_F: 0.0`.

Why it matters:

BMR is the formal active-inference heart of the system, but current logging may not reveal useful posterior switching. If `delta_F` often collapses to zero, the "insight" mechanism is not doing much work.

Recommendation:

- histogram `delta_F` across all candidate sets;
- assert non-degenerate posterior entropy changes in tests;
- compare `jnana` against simple `argmax(ananda)` and `argmax(score_proxy)`.

### P2-3: The Hopfield store/consolidation exists but is not part of benchmark causality

Evidence:

The plugin exposes Hopfield and SWS/REM tools, but Phase 9 driver does not use multi-run memory or consolidation as part of generation.

Why it matters:

Research background emphasizes storehouse/consolidation, but benchmark output does not test it.

Recommendation:

Move storehouse claims to "implemented but not benchmarked" unless v0.2 includes a memory-conditioned benchmark.

### P2-4: Paper/README negative-result framing is honest but incomplete

Evidence:

README and paper report the directional null and 0/30 `vimarsa`.

Missing:

- the SPEC arm mismatch;
- the impossible one-point switching gate;
- the plugin CPU/float32 runtime mismatch;
- candidate-selection regret analysis.

Recommendation:

Add an `AS_SHIPPED.md` or update README with:

- actual arms;
- actual n;
- actual runtime config;
- which SPEC claims were not met;
- known mechanism gaps.

## Why the New Plugin Trails Bare Versions

The evidence supports a multi-cause explanation. Ranked by likely impact:

### 1. No causal `vimarsa` benefit

The central claimed advantage is recursive self-reflexivity. In the actual benchmark:

- it never fires;
- it is blocked by the one-point trajectory;
- it is absent from domains with no aspects;
- it cannot change the final text even if it fires.

Therefore `local_cascade` is mostly `iccha` multi-sampling + `jnana` selection, not a self-reflexive creative loop.

### 2. Prompt and sampler mismatch versus bare

`local_bare` receives the task prompt directly with `tau=0.9`.

`local_cascade` receives a suffix-mutated prompt and candidates from a wide sampler grid starting at `tau=0.40`. This can bias toward generic, low-temperature outputs or duplicated prompt content.

### 3. Selection objective mismatch

The cascade optimizes `ananda + clipped apohana` through BMR. The benchmark rewards lexical diversity, list count, elaboration, aspect cosine hits, and specificity. Those objectives are not equivalent.

### 4. Negative `apohana` does not penalize

Bad candidates are not sufficiently down-weighted when they violate `must_avoid`. This weakens the contrastive exclusion operator.

### 5. Token budget truncation

Local arms, especially cascade, frequently look truncated. The metrics reward completion/coverage, so token budget is a direct score depressor.

### 6. Substrate gap dominates Haiku comparison

Haiku is far stronger than Qwen2-1.5B. The primary Haiku contrast is useful as an external reference, but it overwhelms architectural signal.

### 7. Live plugin runtime config is worse than benchmark runtime

The actual plugin is CPU/float32 by default. This does not explain the benchmark result, but it will make real CLI cascade use slower and less reliable.

## Critical Gaps for v0.2

### A. Make `vimarsa` causal

Minimum implementation:

```text
draft = kriya(selected)
event, diag = vimarsa(draft)
if task_requires_aspect_shift and not event:
    revise prompt with explicit missing aspects
    resample or polish
return revised_surface
```

Measure:

- `surface_before_vimarsa`;
- `surface_after_vimarsa`;
- `event_before`;
- `event_after`;
- score delta caused by revision.

### B. Fix the impossible switching gate

Options:

1. Pass `iccha_apoha_trajectory=None` until a real trajectory exists.
2. Compute trajectory over candidate rank order or revision passes, not a single point.
3. Make `switching_threshold=0` for one-shot cascade.

### C. Add prompt/sampler parity ablations

Required experiment matrix:

| Arm | Prompt | Sampler | Selection |
|---|---|---|---|
| local_bare | raw | tau=0.9 | first sample |
| local_bare_suffix | raw + constraint suffix | tau=0.9 | first sample |
| cascade_no_suffix | raw | cascade grid | jnana |
| cascade_matched | raw | tau=0.9 for all K | jnana |
| cascade_current | suffix | cascade grid | jnana |

This isolates whether PCE loses because of the cascade or because of prompt/sampler drift.

### D. Add candidate oracle regret analysis

For every cascade run:

```text
regret = score(best_candidate_by_benchmark_proxy) - score(jnana_selected)
```

If regret is high, tune `jnana`.

### E. Fix signed `apohana`

Make negative evidence matter. Add tests where a must-avoid-near candidate loses posterior mass.

### F. Raise token budgets or change tasks

Use enough tokens for requested format:

- poetry generation: enough for full form;
- poetry interpretation: two complete paragraphs;
- AUT: 8 complete uses;
- science: 4-6 complete sentences.

Add truncation metrics.

### G. Reconcile docs with shipped reality

Create `docs/AS_SHIPPED.md` with:

- shipped plugin tool names;
- actual benchmark arms;
- actual n;
- actual supported/unsupported hypotheses;
- known gaps;
- exact v0.2 target.

### H. Validate metrics against a judge

Run Sonnet/Opus/human scoring on a frozen subset and compare with local proxies. Do not assume local embedding proxies capture "creative nuance."

### I. Remove CPU pin from plugin config

Default plugin runtime should match engine runtime:

```json
"env": {}
```

or:

```json
"PCE_DEVICE": "",
"PCE_DTYPE": ""
```

and let `LMConfig` auto-detect MPS/CUDA/CPU.

## Recommended v0.2 Experiment

Use the same Qwen2 substrate first. Do not compare to Haiku until PCE beats its own base.

### Stage 1: Mechanism calibration

- Run 20 duck-rabbit/aspect probes.
- Target `vimarsa_event` rate 30-60%.
- Verify zero/low false positives on controls.
- Assert cascade path and standalone path agree.

### Stage 2: Same-substrate ablation

Run 20 items:

- `local_bare`;
- `local_bare_suffix`;
- `cascade_no_vimarsa`;
- `cascade_with_vimarsa_revision`;
- `oracle_best_of_K`.

Primary success:

```text
cascade_with_vimarsa_revision > local_bare
```

Secondary:

```text
jnana regret <= 20% of oracle_best_of_K lift
```

### Stage 3: Judge validation

Score the same subset with:

- local proxy;
- Sonnet judge;
- blind human if possible.

Report correlation. If correlation is poor, replace the proxy before full benchmarking.

### Stage 4: Full run

Only after same-substrate PCE beats bare should Haiku be reintroduced as an external reference.

## Verdict

The project is **engineering-real but research-inconclusive**.

What is solid:

- the plugin manifest and MCP server load through Claude Code;
- real tools can be called from the Claude CLI;
- raw benchmark JSON and published stats are internally consistent;
- the negative result is honestly reported in top-level README/paper.

What is not solid:

- SPEC/PRD do not describe the actual experiment;
- `vimarsa` is structurally inert in the cascade benchmark;
- `vimarsa` is non-causal even when it detects;
- objective, prompt, sampler, and metric misalignment are sufficient to explain trailing bare versions;
- the live plugin runtime config differs from the benchmark runtime.

The next version can plausibly beat bare versions, but only after the recursive layer becomes a real text-producing loop rather than a post-hoc diagnostic flag.
