# PCE v0.2 Adversarial Review — Haiku Substrate, Causal `vimarsa`, and Benchmark Validity

Date: 2026-04-29  
Reviewed branch: `v0.2`  
Reviewed release: `v0.2.0` / PR #1 (`3798f0d` merge commit on `main`)  
Review mode: adversarial, evidence-first, post-release audit  
Live plugin trace: `audit/adversarial_v0_2_plugin_trace.jsonl`  
Live plugin trace summary: `audit/adversarial_v0_2_plugin_trace_summary.json`

## Executive Verdict

PCE v0.2 is a real improvement over v0.1, but it does **not yet establish the full computational claim** implied by the release narrative. The engineering objectives are mostly met: Haiku is callable through the plugin, `pce_cascade(arm="haiku")` works, `vimarsa` is no longer inert telemetry, the cascade produces a draft and revision, and the operator chain (`iccha -> apohana -> ananda -> jnana -> kriya -> vimarsa -> revision`) is active in code.

The research objective is only **partially met**. The results are directionally encouraging, but they are not a clean proof that the PCE computational architecture improves Haiku creativity. The primary comparison is confounded by extra inference budget, extra context, and a second revision prompt. The shipped pilot also omits the `local_cascade` ablation, does not compute the pre-registered H8 revision-vs-draft test, and uses a Haiku CLI pathway that appears to inherit Claude Code/plugin context in at least some benchmark outputs. That last point is serious: several benchmark rows contain tool/skill meta-chatter, which means the "Haiku" substrate may not be a clean raw Haiku endpoint.

Short version: **v0.2 is not merely prompting, but the evidence still does not isolate a computational architecture effect from multi-call prompt-and-revise scaffolding.**

## Questions I Would Ask Before v0.3

1. Should `haiku_bare` be a clean Anthropic SDK call rather than `claude -p`, so no Claude Code plugin/skill context can leak into benchmark outputs?
2. Should the primary baseline be budget-matched, e.g. `haiku_bare_best_of_2K`, `haiku_bare_revise_once`, or `haiku_bare_self_consistency`, rather than a single Haiku call?
3. Is the core claim "PCE improves creativity per dollar/call" or "PCE improves creativity at unconstrained inference budget"? These are different hypotheses.
4. Should `vimarsa_event` actually gate revision, or is "two-pass-always" intentionally a revision scaffold whose event is only telemetry?
5. Should v0.2 release wording be amended to say "three-arm pilot with four-arm-capable driver" rather than "four-arm pilot"?
6. Should H8 (`revision_score - draft_score`) be mandatory before claiming `vimarsa` is causally effective?
7. Should the paper keep H5 as an aggregate hypothesis if the current z-blend mathematically collapses the mean to near zero after per-domain centering?
8. Do we want the Hopfield/alayavijnana layer to become causally active in generation, or remain a separate plugin demo surface?

## Scope and Evidence Reviewed

Primary files inspected:

- `docs/SPEC_v0.2.md`
- `docs/RELEASE_NOTES_v0.2.md`
- `docs/reviews/2026-04-28-prove-gate.md`
- `src/pce/cascade.py`
- `src/pce/operators/iccha.py`
- `src/pce/operators/jnana.py`
- `src/pce/operators/vimarsa.py`
- `src/pce/substrate/haiku_lm.py`
- `src/pce/substrate/lm_protocol.py`
- `plugin/mcp/server.py`
- `benchmarks/driver.py`
- `benchmarks/stats.py`
- `benchmarks/scoring.py`
- `benchmarks/results_v2/stats.json`
- `benchmarks/results_v2/*.json`
- `scripts/prove_gate.py`
- `scripts/smoke_plugin.py`

I also ran a live plugin smoke with Haiku enabled:

```bash
PCE_HAIKU_COST_CAP_USD=100 \
uv run python scripts/smoke_plugin.py \
  --with-haiku \
  --out-jsonl audit/adversarial_v0_2_plugin_trace.jsonl \
  --out-json audit/adversarial_v0_2_plugin_trace_summary.json
```

Result:

```json
{
  "ok": true,
  "pass": 18,
  "fail": 0,
  "skipped": 0,
  "skip_lm": false,
  "with_haiku": true,
  "expected_total": 18
}
```

## Live Plugin Trace

The plugin did trigger successfully from the in-process MCP tool manager. This confirms the plugin surface is not merely documented; tools execute and return structured results.

| # | Tool | Status | Key result |
|---:|---|---|---|
| 1 | `report` | PASS | v0.2.0, Qwen2-1.5B on `mps`, MiniLM embedder dim 384 |
| 2 | `ananda` | PASS | score `0.800979` |
| 3 | `apohana` | PASS | scores `[0.323825, -0.020556]`; negative apoha is observable |
| 4 | `jnana` | PASS | selected idx `2`, posterior peak `0.501706`, `delta_F = 0.0` |
| 5 | `kriya` | PASS | verbatim surface |
| 6 | `vimarsa` | PASS | event `true`, novelty `0.807269`, aspect count `2` |
| 7 | `hopfield_store` | PASS | pattern count `1` |
| 8 | `hopfield_recall` | PASS | recall cosine `0.630131` |
| 9 | `consolidate_sws` | PASS | `3` patterns after SWS |
| 10 | `consolidate_rem` | PASS | `8` REM steps |
| 11 | `consolidate_cycle` | PASS | pattern count `3 -> 5` |
| 12 | `reset_state` | PASS | pattern count reset to `0` |
| 13 | `cit` | PASS | local LM generated non-empty sentence |
| 14 | `iccha` | PASS | `K=3`, all parity samplers `tau=0.9/top_p=0.95/top_k=50` |
| 15 | `cascade` | PASS | local bypass cascade; `vimarsa_event=true`, `two_pass=false` |
| 16 | `pce_cascade` local | PASS | local substrate switch works; bypass path works |
| 17 | `haiku_bare` | PASS | Haiku text non-empty; ledger total `$4.0293`, calls `151` |
| 18 | `pce_cascade` haiku | PASS | two-pass true; draft/revision differ; `vimarsa_event=true`; ledger calls increased |

Key live Haiku bare output:

> The duck-rabbit illusion displays both a duck and a rabbit, depending on whether one focuses on the protruding part as a bill or as ears.

Key live Haiku cascade output:

```json
{
  "arm": "haiku",
  "two_pass": true,
  "surface_draft": "A duck and a rabbit are the two animals you can see in the duck-rabbit illusion, depending on which way you interpret the ambiguous image.",
  "vimarsa_brief": "Tighten imagery and intensify the contrast between the named aspects.",
  "surface_revision": "Duck or rabbit—the illusion hinges on whether the pivotal line reads as a beak or ears, forcing your eye to choose between two incompatible creatures.",
  "vimarsa_event": true,
  "vimarsa_event_draft": true,
  "revision_differs_from_draft": true,
  "posterior": [0.4567045569, 0.2900301814, 0.2532652617],
  "delta_F_draft": 0.0,
  "delta_F_revision": 0.0,
  "aspect_count_revision": 2.0,
  "aspect_max_cosine_revision": 0.5806168318
}
```

This trace is important because it proves both sides of the critique:

- The plugin really runs the cascade and `vimarsa` is causally placed before the final output.
- The final improvement is also clearly mediated by a natural-language revision prompt plus extra Haiku calls.

## Findings

### P1 — The primary benchmark is not apples-to-apples on inference budget or prompt context

The release correctly frames `haiku_cascade` vs `haiku_bare` as same-substrate. It is not same-budget.

Evidence:

- `haiku_bare` runs one `HaikuLM.generate()` call in `benchmarks/driver.py`.
- `haiku_cascade` runs `_one_pass()` twice in `src/pce/cascade.py`.
- Each `_one_pass()` calls `iccha()`, which calls the LM `K` times.
- Therefore the cascade uses approximately `2K` Haiku completions per item, versus `1` for bare.
- The second pass prompt includes `Reviser brief`, the previous draft, and "Now produce the revised response."

For the pilot setting, that means a typical `haiku_cascade` item gets roughly six or eight Haiku generations, plus a revision prompt containing the selected draft. A single `haiku_bare` response is not a budget-matched baseline.

Impact:

The observed gains may be due to search, reranking, and revision budget rather than the Pratyabhijna/active-inference architecture specifically. The effect may still be valuable, but the scientific claim needs to become "PCE as an inference-time compute procedure improves outputs" rather than "the architecture improves Haiku under equal compute."

Recommended fix:

Add baselines:

- `haiku_bare_best_of_2K`: same number of Haiku completions, scored/reranked by the same local scorer.
- `haiku_bare_revise_once`: draft plus generic revise prompt, no `apohana/jnana/vimarsa`.
- `haiku_bare_self_consistency`: same call count, choose best by scorer.
- `haiku_cascade_no_brief`: two-pass call budget, but no aspect-derived brief.

Only after these controls can the cascade contribution be isolated.

### P1 — The Haiku CLI path is probably not a clean raw Haiku endpoint

The benchmark outputs show Claude Code/plugin/system-context contamination. This goes beyond a normal "model sometimes prefaces answer" issue.

Evidence from `benchmarks/results_v2/aut.json`:

- `haiku_bare` for brick begins: "I appreciate the skill being loaded, but I notice this is a mismatch—the brainstorming skill is designed..."
- `haiku_cascade` for brick begins: "Understood—the old brainstorm command is deprecated. You're already using the superpowers brainstorming skill..."

These phrases are not caused by the user benchmark prompt. They are artifacts of the execution environment or Claude Code CLI context. That means the Haiku substrate used in the pilot may not be equivalent to a clean Anthropic API call.

Impact:

This undermines the claim that the pilot measures Haiku as a neutral substrate. The benchmark may partly measure Claude Code's current tool/skill instruction stack interacting with prompt content. It also makes the release less reproducible outside this user's Claude Code environment.

Recommended fix:

Run the primary benchmark through the Anthropic SDK path with explicit `ANTHROPIC_API_KEY`, clean system prompt, and logged model ID. Treat the `claude -p` CLI as an integration smoke path, not the research path, unless its system context can be proven clean.

### P1 — H8 is pre-registered but not implemented

`docs/SPEC_v0.2.md` pre-registers H8.v2: revision score minus draft score should have positive median in `haiku_cascade`.

Evidence:

- `src/pce/cascade.py` keeps `surface_draft` and `surface_revision`.
- `benchmarks/driver.py` only persists final `text`, final `composite`, and compact metadata.
- `benchmarks/stats.py` has no H8 block.
- `benchmarks/results_v2/stats.json` contains no H8 result.

Impact:

This is the most direct causal test of `vimarsa` as an effective revision mechanism. Without it, v0.2 can show "the final cascade output scored higher than bare," but it cannot quantify whether the revision pass improved over the cascade draft.

Recommended fix:

Persist draft and revision surfaces in benchmark rows, score both, and add H8:

- `draft_composite`
- `revision_composite`
- `revision_delta = revision_composite - draft_composite`
- paired sign test / Wilcoxon
- per-domain and aggregate reporting

### P1 — `local_cascade` ablation is still absent from shipped pilot results

This is documented as deferred in the release notes, so it is not hidden. But it remains a gap beyond "needs more power." It means the architecture is not yet isolated on a weak local substrate either.

Evidence:

- `benchmarks/results_v2/stats.json` has `local_ablation.primary.*.n = 0`.
- `H6_local_cascade` has `n_fired = 0`, `n_not_fired = 0`.
- Release notes say arms run were `local_bare`, `haiku_bare`, `haiku_cascade`.

Impact:

The driver supports four arms, and the prove-gate ran four arms, but the pilot result is three-arm. The phrasing "four-arm pilot" should be softened to "four-arm-capable design; shipped pilot ran three arms due to local cascade throughput."

Recommended fix:

Run `local_cascade` on a GPU host or reduce it to a small ablation table explicitly separated from the primary Haiku result.

### P1 — `vimarsa_event` does not gate revision

`vimarsa` is causally placed in the loop because its brief feeds the second pass. But the event boolean itself does not decide whether revision happens.

Evidence:

- `run_cascade()` always builds `revision_prompt` and calls `_one_pass()` again unless `bypass_vimarsa=True`.
- `_build_brief()` always returns a brief, even if `event=False`.
- `event_d` is recorded, not used as a branch condition.

Impact:

The claim "`vimarsa` is causal" is true only in the sense that the brief is causally inserted into the revision prompt. It is false if read as "the detected event controls whether or how the system revises." The current mechanism is closer to "always revise, using a brief generated by the same function that also emits an event flag."

Recommended fix:

Either:

- explicitly define `vimarsa` as a universal reviser, not an event-gated controller, or
- add event-conditioned modes: `revise_only_if_event`, `generic_revision_if_no_event`, and `aspect_revision_if_event`.

### P2 — `delta_F` remains uninformative in live traces and benchmark metadata

The live plugin trace shows `delta_F_draft = 0.0` and `delta_F_revision = 0.0` for the Haiku cascade. The benchmark rows also frequently record `delta_F = 0.0`.

Evidence:

- Live `pce_cascade_haiku`: both draft and revision `delta_F` are `0.0`.
- `docs/reviews/2026-04-28-prove-gate.md` already notes `delta_F == 0` on all cascade arms.
- `jnana` still selects via posterior, so selection is not inactive, but the Bayesian model reduction telemetry is not currently a meaningful insight signal.

Impact:

This weakens the active-inference claim. The implementation has a mathematical posterior, but the advertised "Bayesian Model Reduction insight detection" is not producing informative evidence in the shipped traces.

Recommended fix:

Rework BMR reductions so identity/no-reduction cannot trivially dominate, or report a separate "selection posterior" and stop treating `delta_F` as an insight signal until calibrated.

### P2 — Haiku sampler parity is overstated under the default CLI backend

The cascade and bare arms carry the same sampler dictionary. But in the default CLI path, `HaikuLM` does not pass temperature, top-p, or top-k to `claude -p`.

Evidence:

- `_call_cli_once()` constructs `[cli_bin, "-p", "--model", model, "--output-format", "json", prompt]`.
- The SDK path uses `temperature` and `top_p`, but `PCE_USE_SDK` is opt-in.
- The audit log records sampler settings even when the CLI backend cannot enforce them.

Impact:

The release can claim sampler intent parity, but not necessarily effective sampler parity. For the default pilot path, parity is only in local metadata.

Recommended fix:

Use SDK for benchmarks, or prove the CLI supports and receives equivalent sampling flags. Otherwise update docs to state: "Sampler parity is enforced for local/SDK substrates; CLI Haiku records requested sampler but may use CLI defaults."

### P2 — The SPEC and implementation disagree on `cit_temperature`

`docs/SPEC_v0.2.md` says `cit_temperature` overrides the first sampler and multiplicatively affects the rest. The code stores `cit_temperature` in `CascadeState` but does not pass it into `_one_pass()` or `iccha()`.

Evidence:

- `run_cascade()` accepts `cit_temperature`.
- `_one_pass()` has no `cit_temperature` parameter.
- `iccha()` in parity mode hard-codes `PARITY_SAMPLER`.

Impact:

This is a dead or decorative parameter in the main cascade path. It also echoes one of the v0.1 critique themes: telemetry/parameter surfaces that look meaningful but do not alter generation.

Recommended fix:

Either remove the parameter from the v0.2 claim, or actually thread it into the sampler grid with tests proving output sampler metadata changes.

### P2 — Prove-gate assertions are weaker than the SPEC text

The prove-gate is useful, but it lets "any cascade arm" satisfy conditions that the acceptance criteria imply should hold on `haiku_cascade`.

Evidence:

- `scripts/prove_gate.py` checks `vimarsa_event_at_least_one_arm` across `local_cascade` and `haiku_cascade`.
- It checks `revision_differs_from_draft` across either cascade arm.
- It checks aspect/novelty floors across any arm.

Impact:

The gate can pass even if the main target arm (`haiku_cascade`) fails a causal requirement, as long as `local_cascade` or another arm clears it. The actual recorded prove-gate appears to show `haiku_cascade` passing, but the code does not require that invariant.

Recommended fix:

Add fixture fields like:

- `haiku_cascade_vimarsa_event_required`
- `haiku_cascade_revision_differs_required`
- `haiku_cascade_aspect_floor`

Then test the gate itself with mocked/fixed rows.

### P2 — H5 aggregate is mathematically suspect

`stats.py` z-scores each domain's item-level deltas by subtracting that domain's mean, then concatenates weighted z-scored deltas. This makes the overall mean nearly zero by construction, which is exactly what appears in `stats.json`.

Evidence:

- `stats.json` H5 estimate is `7.304e-18`, despite all four primary domain deltas being positive.
- `stats.py` `_zscore()` computes `(arr - mean(arr)) / sd`, then concatenates weighted z-score arrays.

Impact:

H5 currently cannot represent "aggregate positive effect" as most readers would understand it. It is not just underpowered; it is centered away from the mean effect.

Recommended fix:

Use one of:

- raw paired deltas scaled by pre-declared domain standardizers from baseline data,
- domain-level Stouffer combination of H1-H4 p-values,
- weighted mean of domain estimates with bootstrap over domains/items,
- mixed-effects model with domain as a random or fixed factor.

### P2 — Benchmark scoring proxies can be gamed by verbosity and format

The cascade often produces longer, more elaborated responses, especially after the revision prompt. Scorers reward length/specificity/elaboration in several domains.

Evidence:

- AUT scoring gives elaboration `min(1, avg_words / 18)`.
- Scientific creativity gives specificity `min(1, n_words / 80)`.
- The revision prompt asks for novelty, vividness, and surprise; the outputs often become longer and formatted.

Impact:

Some of the observed score gain may be "more words and more structure" rather than deeper creativity. This is especially problematic because bare is single-pass and cascade is revise-pass.

Recommended fix:

Add length-matched scoring, verbosity penalties, and pairwise blind judge evaluation. Also add a `haiku_bare_verbose` baseline with the same instruction to be vivid/specific.

### P3 — `LMProtocol` is narrower than planned and does not expose scoring/logprobs

The plan/SPEC discussion originally gestured at `generate_with_logprobs`, `score`, and embedding callbacks. The actual protocol is `generate()` and `report()`.

Evidence:

- `src/pce/substrate/lm_protocol.py` defines only `name`, `generate`, and `report`.
- `HaikuLM` uses a placeholder length-based `logp_proxy = -output_tokens * log(2)`.

Impact:

The substrate abstraction is useful, but not enough to support a full active-inference substrate with scoring, uncertainty, or calibrated likelihood. Haiku is integrated as a generator, not a probabilistic substrate.

Recommended fix:

Rename this layer `GeneratorProtocol` unless/until it supports scoring/logprobs. Or add explicit optional capabilities:

- `supports_logprobs`
- `score(prompt, completion)`
- `entropy(prompt)`
- `embed(text)` or a shared embedding interface

### P3 — Hopfield/storehouse is not part of the benchmark causal path

This is explicitly out of scope in the v0.2 SPEC, so it is not a hidden failure. It does matter for the larger "Pratyabhijna x active inference computational system" claim.

Evidence:

- `src/pce/cascade.py` has no Hopfield calls.
- `benchmarks/driver.py` has no Hopfield calls.
- Hopfield tools pass in the plugin smoke, but they are separate MCP utilities.

Impact:

The alayavijnana/storehouse layer exists as a plugin subsystem, not as part of creative generation. The "computational system" is therefore not fully established as an integrated Pratyabhijna architecture.

Recommended fix:

For v0.3 or v0.4, make retrieval memory participate in `vimarsa` or `apohana`, and benchmark with memory on/off.

### P3 — JSON artifacts contain non-standard `NaN`

`stats.json` includes `NaN` values in empty local ablation blocks.

Impact:

Strict JSON parsers will reject the artifact. This matters because presentation, paper, external reviewers, and reproducibility tools may parse it.

Recommended fix:

Write JSON with `allow_nan=False` and serialize missing values as `null`.

## Does v0.2 Meet Its Objectives?

### Objective: Add Haiku as first-class substrate

Verdict: **Partially met.**

The plugin can call Haiku through `HaikuLM`, the MCP tool `haiku_bare` works, and `pce_cascade(arm="haiku")` works. But the default CLI pathway is not a clean research substrate because sampler knobs are not enforced and benchmark outputs show Claude Code skill-context leakage.

### Objective: Make `vimarsa` causal

Verdict: **Partially met.**

The brief generated by `vimarsa` is inserted into the second pass. The live trace proves the final revision differs from draft. However, `vimarsa_event` does not gate or control revision, H8 is not computed, and the brief can be generic when there are no aspects.

### Objective: Resolve v0.1 operator failures

Verdict: **Mostly met.**

The v0.1 failure where `vimarsa` never fired is fixed. Signed `apohana` is observable. Prompt/sampler parity is improved for local and logically improved for Haiku, but the CLI backend weakens effective sampler parity.

### Objective: Apples-to-apples Haiku comparison

Verdict: **Not fully met.**

It is same substrate and same scorer. It is not same inference budget, not same prompt context, not necessarily same effective sampler settings, and possibly not clean Haiku API context.

### Objective: Establish a computational system beyond prompting

Verdict: **Partially met.**

There is real computation:

- embedding geometry,
- contrastive `apohana`,
- `ananda` scoring,
- posterior selection in `jnana`,
- `vimarsa` novelty/aspect detection,
- audit-state capture,
- Hopfield tools outside the cascade.

But the system is not yet a fully established active-inference creative-cognition system:

- BMR `delta_F` is degenerate in traces,
- Haiku has no real logprobs/score interface,
- `cit_temperature` is inert,
- Hopfield memory is not in the cascade,
- `vimarsa_event` does not gate action,
- benchmark gains are entangled with extra LLM calls and revision prompting.

## What Gaps Still Exist Beyond the Deferred v0.3 Items?

The release notes already defer:

- full `local_cascade` ablation,
- live Sonnet judge bridge,
- properly powered n≈20/domain run,
- per-domain K / early-exit.

Additional gaps not fully covered by that list:

1. Clean Haiku substrate via SDK, with no Claude Code skill/system leakage.
2. Budget-matched and revision-matched baselines.
3. H8 implementation and reporting.
4. Effective sampler parity for Haiku, not just recorded sampler parity.
5. `cit_temperature` plumbing or removal from claims.
6. `delta_F` calibration or demotion from active-inference evidence.
7. Prove-gate tightening to require `haiku_cascade` specifically.
8. Strict JSON output without `NaN`.
9. H5 aggregate redesign.
10. Length/verbosity-controlled scoring.
11. Memory/storehouse integration into the actual generation path.
12. Naming cleanup: SPEC H7 is reported as `H6_haiku_cascade`, while SPEC H8 is absent.

## Strongest Positive Evidence

The live trace proves v0.2 is not imaginary. `pce_cascade(arm="haiku")` produced:

- a draft,
- a `vimarsa` brief,
- a different revision,
- an active posterior over candidates,
- aspect coverage,
- a fired `vimarsa` event,
- structured audit metadata.

This is materially better than v0.1, where `vimarsa` was effectively post-hoc or structurally gated out. The trace also shows signed `apohana` and posterior selection are active enough to choose among candidates.

## Strongest Negative Evidence

The benchmark result most likely measures a compound treatment:

```text
PCE treatment =
  Haiku call budget multiplier
  + K-sampling
  + local embedding scorer/reranker
  + natural-language revision brief
  + previous draft in context
  + final revision prompt
  + possible Claude Code skill-context contamination
  + Pratyabhijna/active-inference operators
```

The current data does not isolate the last term.

## Recommended Next Experiment

Before running a bigger benchmark, run a small but controlled 2-case or 8-case ablation matrix:

| Arm | Calls | Prompt context | Rerank | Revision | Purpose |
|---|---:|---|---|---|---|
| `haiku_bare_1x` | 1 | original | no | no | current baseline |
| `haiku_bare_2K_random` | 2K | original | random pick | no | call-budget control |
| `haiku_bare_2K_scorer` | 2K | original | same scorer | no | reranker control |
| `haiku_generic_revise` | 2K | generic brief + draft | scorer | yes | revision-prompt control |
| `haiku_vimarsa_revise` | 2K | `vimarsa` brief + draft | scorer | yes | true PCE arm |
| `haiku_vimarsa_event_gated` | variable | event-gated | scorer | conditional | causal controller arm |

Use the SDK backend, strict JSON, stored draft/revision scores, and a blind judge subset. This would answer whether `vimarsa` adds something beyond extra inference and generic revision.

## Final Assessment

PCE v0.2 is a strong engineering iteration and a credible scaffold for a real computational creativity system. It does not yet justify the stronger research claim that the Pratyabhijna x active-inference computation independently beats bare Haiku. The pilot demonstrates that "Haiku plus the PCE two-pass search-and-revise pipeline" can beat a single bare Haiku call under local proxy scorers. That is useful, but it is not the same as establishing an apples-to-apples architectural effect.

The next version should be less ambitious in prose and more ruthless in controls: clean substrate, equal call budget, revision baseline, H8, and mechanism-specific ablations.
