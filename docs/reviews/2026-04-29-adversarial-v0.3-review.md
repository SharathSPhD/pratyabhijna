# PCE v0.3 Adversarial Review — Why the Cascade Still Trails

Date: 2026-04-29  
Reviewed branch: `main`  
Reviewed HEAD at start of review: `a09795e`  
Review mode: adversarial, evidence-first, post-release audit  
Fresh plugin trace: `audit/adversarial_v0_3_plugin_trace.jsonl`  
Fresh plugin trace summary: `audit/adversarial_v0_3_plugin_trace_summary.json`

## Executive Verdict

PCE v0.3 is a serious methodological improvement over v0.2, but it still does **not** prove the basic research hypothesis that a Pratyabhijna + active-inference architecture increases creativity. It does prove several narrower engineering claims:

- the inner Haiku CLI substrate can be isolated from Claude Code/plugin/skill leakage;
- the cascade executes a real multi-operator path rather than only a prompt wrapper;
- `vimarsa` now gates which surface is committed;
- draft and shadow revision surfaces are both preserved, enabling counterfactual analysis;
- the v0.3 benchmark honestly reports a directional-negative result rather than hiding it.

The key adversarial finding is sharper than "PCE trails because active inference failed." In the v0.3 pilot, the cascade often generated **better shadow revisions**, but the event gate failed to commit them. A post-hoc rescoring of all 20 shadow revisions gives:

```text
all shadow revision - draft: n=20, mean=+0.0458, median=+0.0323, positive=15, negative=5
event-committed only:        n=3,  mean=+0.0037, median=+0.0036, positive=2,  negative=1
not committed:               n=17, mean=+0.0532, median=+0.0373, positive=13, negative=4
```

That means the registered H8.v3 test is structurally underpowered and mis-targeted: it tests only the three revisions the current gate chose, while most of the apparent revision value is in revisions the gate discarded. In several high-value cases, the cascade kept the draft even though the shadow revision scored much higher:

```text
poetry_interp/i01: delta=+0.2114, committed=draft, event=false
poetry_interp/i03: delta=+0.1963, committed=draft, event=false
poetry_interp/i02: delta=+0.1838, committed=draft, event=false
sci_creativity/s01: delta=+0.0486, committed=draft, event=false
poetry_gen/p03: delta=+0.0476, committed=draft, event=false
```

Counterfactual scoring of "always commit the shadow revision" on the same generated artifacts is directionally positive against the main controls:

```text
shadow revision vs haiku_bare:                mean delta +0.0289
shadow revision vs haiku_bare_2K_scorer:      mean delta +0.0344
shadow revision vs haiku_generic_revise_2pass: mean delta +0.0131
```

This is post-hoc and must not be treated as proof. But it strongly suggests the next version should focus less on adding philosophical machinery and more on calibrating the action policy that decides whether the generated revision is worth committing.

Short version: **v0.3 may already contain a useful revision generator, but its event gate is throwing away much of the gain. Several "active-inference" claims are also overstated because `cit_temperature` and the free-energy budget are not causally effective in the Haiku CLI benchmark path.**

## Questions I Would Ask Before v0.4

1. Is the core claim **superiority** ("PCE beats bare/control arms") or **mechanistic utility** ("PCE discovers better candidate revisions, even if the current gate is wrong")?
2. Should H8.v4 be split into two hypotheses: **H8a all shadow revisions vs drafts** and **H8b event-gate accuracy**?
3. Should v0.4 commit policy be optimized against the scorer/judge, or should the commit policy remain theory-pure even if it discards empirically better outputs?
4. Is OAuth-only via `claude --print` still a hard constraint for the research benchmark, even though it prevents true sampler control and clean API reproducibility?
5. Should the paper retract or soften claims that the per-item free-energy budget "gates abort/continue decisions" until that branch is actually wired?
6. Should `cit_temperature` remain a headline active-inference mechanism if the Haiku CLI backend ignores it, or should it be restricted to SDK/local substrates?
7. What result would terminate the research program? A powered null against human judges? A null after all action-policy calibration? A null after each operator ablation?
8. What is the minimum acceptable proof of "creativity" beyond local proxy composites: blind human preference, frontier-judge consensus, or task-specific behavioral metrics?
9. Should the v0.4 study be powered for all four domains, or should it first narrow to the domain where the shadow revisions show the strongest signal (`poetry_interp`)?
10. Should "Pratyabhijna + active inference" be decomposed into separately testable modules (`vimarsa` brief, BMR candidate selection, Hopfield memory, FE budget, temperature) rather than defended as one bundled treatment?

## Scope and Evidence Reviewed

Primary source files inspected:

- `src/pce/cascade.py`
- `src/pce/operators/iccha.py`
- `src/pce/operators/vimarsa.py`
- `src/pce/substrate/haiku_lm.py`
- `src/pce/substrate/integrity.py`
- `src/pce/active_inference/budget.py`
- `plugin/mcp/server.py`
- `benchmarks/driver.py`
- `benchmarks/stats.py`
- `benchmarks/scoring.py`
- `benchmarks/results_v0.3/*.json`
- `paper/main.tex`
- `paper/sections/07_methods.tex`
- `paper/sections/09_results.tex`
- `paper/sections/10_discussion.tex`
- `paper/sections/11_limitations.tex`
- `docs/SPEC_v0.3.md`
- `docs/PRD_v0.3.md`
- `docs/adr/v0.3/*.md`
- `docs/triz/v0.3/*.md`
- `docs/reviews/2026-04-29-adversarial-v0.2-review.md`

Fresh validation commands run:

```bash
PCE_HAIKU_COST_CAP_USD=20 \
uv run python scripts/smoke_plugin.py \
  --with-haiku \
  --out-jsonl audit/adversarial_v0_3_plugin_trace.jsonl \
  --out-json audit/adversarial_v0_3_plugin_trace_summary.json
```

Result:

```json
{
  "ok": false,
  "pass": 20,
  "fail": 2,
  "skipped": 0,
  "skip_lm": false,
  "with_haiku": true,
  "expected_total": 22
}
```

The smoke run should be read carefully. It confirms that the plugin surface loads and that the main Haiku cascade path works on a live sample. It also discovered a live operational problem: the later Haiku control-arm probes failed after the Claude CLI returned `429` quota exhaustion. A direct CLI probe confirmed:

```json
{
  "is_error": true,
  "api_error_status": 429,
  "result": "You're out of extra usage · resets 3pm (Europe/London)",
  "total_cost_usd": 0
}
```

Therefore the fresh sample run is **partially successful**: it validates the live PCE path and clean-substrate behavior up to quota exhaustion, but it does not provide a clean fresh all-arms smoke pass. The previously published v0.3 pilot remains the complete all-arm result set; this review records the new smoke failure as an operational reproducibility risk.

## Fresh Live Plugin Trace

The live trace confirms these Haiku-path outputs before the quota failure:


| Tool                               | Status | Key observation                                                                                          |
| ---------------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| `haiku_clean_substrate_probe`      | PASS   | Probe returned successfully; no leakage failure surfaced.                                                |
| `haiku_bare`                       | PASS   | Returned: "In a duck-rabbit illusion, one might see a duck or a rabbit..."                               |
| `pce_cascade_haiku`                | PASS   | Returned a revision: "In a duck-rabbit illusion, one sees a beaked waterfowl or a fuzzy-eared mammal..." |
| `pce_cascade_haiku_bare_2k`        | FAIL   | `HaikuLM CLI rc=1`; direct CLI later showed 429 quota exhaustion.                                        |
| `pce_cascade_haiku_generic_revise` | FAIL   | Same `HaikuLM CLI rc=1`; likely same quota exhaustion.                                                   |


The successful `pce_cascade_haiku` trace matters because it proves the plugin is not inert:

```json
{
  "surface": "In a duck-rabbit illusion, one sees a beaked waterfowl or a fuzzy-eared mammal—starkly contrasting creatures sharing a single form.",
  "committed": "revision",
  "commit_policy": "event_gated",
  "vimarsa_event": true
}
```

The recent Haiku audit files also show the clean-substrate flags in actual CLI calls:

```json
{
  "backend": "cli",
  "clean_substrate": true,
  "isolation_flags": [
    "--tools",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources",
    "",
    "--permission-mode",
    "bypassPermissions",
    "--no-session-persistence"
  ]
}
```

This validates the clean-substrate claim as an integration mechanism, while leaving open whether `claude --print` is an ideal research substrate.

## Numerical Pattern in v0.3 Results

The v0.3 pilot has `n=5` items/domain and four Haiku arms:

- `haiku_bare`
- `haiku_cascade`
- `haiku_bare_2K_scorer`
- `haiku_generic_revise_2pass`

Mean composite scores rank as follows:

```text
poetry_gen:
  generic_2pass=0.613 > bare=0.606 > 2K=0.590 > cascade=0.578

poetry_interp:
  generic_2pass=0.534 > bare=0.507 > 2K=0.497 > cascade=0.493

aut:
  2K=0.939 > generic_2pass=0.934 > bare=0.930 > cascade=0.913

sci_creativity:
  generic_2pass=0.538 > bare=0.514 > 2K=0.509 > cascade=0.508
```

The cascade is the lowest mean arm in all four domains.

Primary pre-registered contrasts (`haiku_cascade - haiku_bare`) are all negative:


| Hypothesis | Domain         | Mean delta | Hedges' g | BCa CI             | One-sided permutation p |
| ---------- | -------------- | ---------- | --------- | ------------------ | ----------------------- |
| H1.v3      | AUT            | -0.0166    | -0.636    | [-0.0332, -0.0001] | 0.90625                 |
| H2.v3      | poetry_interp  | -0.0145    | -0.899    | [-0.0262, -0.0050] | 1.0                     |
| H3.v3      | poetry_gen     | -0.0279    | -0.243    | [-0.0945, +0.0488] | 0.75                    |
| H4.v3      | sci_creativity | -0.0062    | -0.216    | [-0.0276, +0.0092] | 0.6875                  |


Aggregate contrasts:


| Contrast                          | Pooled g | 95% CI           | Interpretation                               |
| --------------------------------- | -------- | ---------------- | -------------------------------------------- |
| H5.v3 cascade vs bare             | -0.463   | [-0.932, +0.006] | Directional negative, near zero upper bound. |
| H6.v3 cascade vs 2K scorer        | -0.273   | [-0.731, +0.185] | Directional negative, inconclusive.          |
| H7.v3 cascade vs generic 2-pass   | -0.570   | [-1.048, -0.093] | Directional negative with CI below zero.     |
| H8.v3 committed revision vs draft | +0.207   | n=3, p=0.375     | Tiny positive but underpowered.              |


## Findings

### P0 — The event gate appears to be the immediate reason v0.3 trails

Files:

- `src/pce/cascade.py`
- `benchmarks/results_v0.3/*.json`
- `benchmarks/stats.py`

Evidence:

- `run_cascade()` always generates `surface_draft` and `surface_revision` for `event_gated` policy.
- The committed output is `revision` iff `event_d` is true; otherwise the draft is scored as the final cascade output.
- The pilot committed revision only 3/20 times:
  - `aut`: 0/5 revisions committed
  - `poetry_gen`: 0/5 revisions committed
  - `poetry_interp`: 2/5 revisions committed
  - `sci_creativity`: 1/5 revisions committed
- Rescoring all shadow revisions shows `revision - draft` is positive on 15/20 items, with mean `+0.0458`.
- The largest gains were mostly discarded because `event=false`.

Impact:

The negative result may be less about "the PCE revision pass is bad" and more about "the gate does not know when the revision is good." The current H8 only tests the gate-approved subset, which is the wrong place to look if the gate itself is the suspect mechanism.

Recommended action:

For v0.4, split the mechanism:

- `H8a`: all shadow revisions vs drafts, regardless of commit.
- `H8b`: gate accuracy: does `event=true` predict positive `revision - draft`?
- `H8c`: calibrated commit policy: commit revision when predicted improvement exceeds threshold.

Then run `always_draft`, `event_gated`, `always_revise`, and `oracle_commit` on the same generated artifacts. If `oracle_commit` or `always_revise` beats controls while `event_gated` trails, the system has a policy-calibration problem, not a generation problem.

### P0 — The free-energy budget is not behavior-gating despite docs and paper claims

Files:

- `src/pce/active_inference/budget.py`
- `src/pce/cascade.py`
- `docs/adr/v0.3/ADR-005-free-energy-budget.md`
- `paper/sections/09_results.tex`
- `paper/sections/10_discussion.tex`

Evidence:

- `FreeEnergyBudget.should_continue_revision()` exists and says it decides whether the revision pass should continue.
- `src/pce/cascade.py` constructs a ledger and calls:
  - `ledger.earn_jnana(...)`
  - `ledger.earn_tokens(...)`
  - `ledger.earn_aspect(...)`
  - `ledger.to_audit()`
- `run_cascade()` never calls `should_continue_revision()`.
- The cascade always runs the shadow revision whenever `commit_policy` is `event_gated` or `always_revise`.
- The only skip branch is `commit_policy == "always_draft"`.
- `paper/sections/10_discussion.tex` states: "the per-item free-energy budget gates abort/continue decisions." That is not true in the current code.

Impact:

One headline active-inference mechanism is audit-only. It may help explain the result only retrospectively, not causally. This weakens the claim that v0.3 fully moved active inference onto the causal path.

Recommended action:

Either wire the budget or demote it:

```python
if not ledger.should_continue_revision():
    commit draft
    audit["revision_skipped_reason"] = "free_energy_budget_underwater"
    return state
```

Then pre-register:

- abort rate,
- cost saved,
- score loss/gain from aborts,
- whether budget balance predicts `revision - draft`.

### P0 — `cit_temperature` is not causally active for the Haiku CLI benchmark path

Files:

- `src/pce/operators/iccha.py`
- `src/pce/substrate/haiku_lm.py`
- `tests/test_cit_temperature.py`
- `paper/main.tex`
- `paper/sections/10_discussion.tex`

Evidence:

- `iccha()` correctly multiplies parity `tau` by `cit_temperature`.
- `HaikuLM.generate()` receives that sampler and records it in the audit JSON.
- The optional SDK path uses:

```python
temperature=float(sampler.get("tau", 1.0))
top_p=float(sampler.get("top_p", 0.95))
```

- The default CLI path calls `_call_cli(seeded_prompt)` and `_build_cmd(prompt)`.
- `_build_cmd()` includes `--print`, `--output-format json`, `--model`, clean-substrate flags, and the prompt. It does **not** pass temperature, top-p, or top-k to `claude --print`.
- The v0.3 benchmark uses the CLI path by scope; API/SDK is out of scope.

Impact:

The paper can honestly say `cit_temperature` modulates the **requested sampler dictionary**, but not that it modulates Haiku generation in the actual benchmark. For Haiku CLI, this is recorded intent, not causal control.

Recommended action:

For v0.4, choose one:

1. If Claude CLI supports sampling flags, pass them and add an integration test that inspects the command.
2. Use SDK only for benchmark runs where sampler control matters.
3. Remove `cit_temperature` from the headline Haiku claim and keep it only for LocalLM/SDK substrates.

### P1 — H8.v3 is correctly implemented as registered, but the registration misses the most diagnostic signal

Files:

- `benchmarks/stats.py`
- `benchmarks/results_v0.3/*.json`

Evidence:

- H8.v3 only includes items where `committed == "revision"`.
- The pilot had only 3 such items.
- All 20 cascade rows have both `surface_draft` and `surface_revision`.
- All-shadow rescoring shows 15/20 revisions beat drafts, but most were not included in H8.

Impact:

The current H8 tells us little about whether the revision pass works. It tells us only that the gate-approved subset is tiny and weakly positive.

Recommended action:

Report both:

- `H8_policy`: committed revisions only, as currently defined.
- `H8_shadow_all`: all shadow revisions vs drafts.
- `H8_gate_calibration`: event as classifier for positive revision delta.

### P1 — The main paper overstates active-inference completion

Files:

- `paper/sections/10_discussion.tex`
- `paper/sections/09_results.tex`
- `paper/main.tex`

Evidence:

The discussion says:

> `cit_temperature` now modulates icchā's sampler-grid posterior, the Hopfield store contributes a warm-start prior to apohana, and the per-item free-energy budget gates abort/continue decisions.

This bundles three mechanisms with different truth values:

- Hopfield can be causally active when the store has patterns.
- `cit_temperature` is causally active for LocalLM/SDK, but not Haiku CLI.
- free-energy budget is not gating anything.

Impact:

The paper's negative-result honesty is strong, but the mechanistic-methods claim is too generous. An adversarial reviewer would flag this as "active inference remains partially telemetry."

Recommended action:

Revise the paper language before any public arXiv claim:

- "budget ledger is audited but not yet action-gating" unless wired;
- "cit_temperature is recorded for Haiku CLI but only enforced in SDK/local substrates" unless CLI flags are wired;
- "Hopfield contributes only after store warm-up and only when aspects are present."

### P1 — H5 pre-registration drift: SPEC says fixed-effects; stats/paper use random-effects

Files:

- `docs/SPEC_v0.3.md`
- `benchmarks/stats.py`
- `paper/sections/07_methods.tex`
- `paper/sections/09_results.tex`

Evidence:

- `docs/SPEC_v0.3.md` defines H5 as "fixed-effects meta-aggregate."
- `benchmarks/stats.py` implements DerSimonian-Laird random-effects pooling.
- The paper describes random-effects.

Impact:

The numeric result is not the core reason PCE trails, but this is a pre-registration hygiene issue. It gives a skeptical reviewer an avoidable target.

Recommended action:

Either update the SPEC with an explicit amendment or change stats to fixed-effects. For v0.4, freeze the exact aggregation formula before running the benchmark.

### P1 — v0.3 sample smoke now fails under quota, so "all tools pass" is time-sensitive

Files:

- `scripts/smoke_plugin.py`
- `audit/adversarial_v0_3_plugin_trace_summary.json`
- `audit/adversarial_v0_3_plugin_trace.jsonl`

Evidence:

- Fresh smoke result: 20 pass, 2 fail.
- Direct CLI probe shows `api_error_status=429`.
- The tool error collapses to `HaikuLM CLI rc=1` with an empty stderr tail, hiding the real JSON error in stdout.

Impact:

The plugin can be working yet appear broken because `_call_cli_once()` only surfaces stderr on non-zero exit. Claude CLI may return useful JSON on stdout even when exit code is 1.

Recommended action:

Improve `_call_cli_once()` error handling:

- if rc != 0, attempt to parse stdout JSON;
- surface `api_error_status` and `result`;
- classify quota/rate-limit as `HaikuRateLimitError`;
- make smoke summaries distinguish implementation failure from external quota failure.

### P2 — Hopfield is real but mostly cold-start in this pilot

Files:

- `src/pce/cascade.py`
- `src/pce/operators/apohana.py`
- `src/pce/active_inference/hopfield.py`
- `benchmarks/driver.py`

Evidence:

- `run_cascade()` only forms `aspect_priors` if `hopfield.n_patterns > 0`.
- Early items in each domain have no storehouse advantage.
- Domains are reset for independence, so memory has limited opportunity to accumulate.

Impact:

The benchmark is a weak test of the memory hypothesis. It is closer to "cold-start PCE" than "PCE with an alayavijnana storehouse."

Recommended action:

Add a burn-in/held-out design:

- burn in the Hopfield store on non-scored prompts from the same domain;
- score held-out items;
- compare `full` vs `no_hopfield` vs `preloaded_hopfield`;
- pre-register the expectation that memory helps aspect-rich domains more than AUT.

### P2 — The discrete MCP tools do not individually recreate the full cascade semantics

Files:

- `plugin/mcp/server.py`

Evidence:

- The `apohana` MCP tool does not use the active Hopfield store by default.
- The `iccha` MCP tool does not expose every v0.3 cascade context.
- The full v0.3 semantics are mostly represented by `pce_cascade`, not by manually chaining individual MCP tools.

Impact:

An external user may call the tools one-by-one and believe they are exercising the same causal graph, when they are only exercising component demos.

Recommended action:

Document `pce_cascade` as the research-grade path. Add a `trace=true` option returning every internal operator result so users do not have to reconstruct the graph manually.

## Why the Plugin Still Trails

### 1. The commit policy is discarding useful revisions

This is the strongest explanation supported by the data. The cascade's shadow revisions are often better than its drafts under the same scorers, but the current event detector commits revision only 15% of the time. The negative primary result uses the committed surface, so it scores many drafts when better revisions were available.

This reframes the failure:

- Bad framing: "PCE cannot generate more creative outputs."
- Better framing: "PCE may generate better revisions, but its self-recognition policy does not recognize them."

This is exactly the kind of failure a Pratyabhijna-inspired system should care about: recognition is the missing step.

### 2. Generic revision is a very strong baseline

H7.v3 is the harshest result: generic 2-pass dominates the cascade on this pilot. That means `vimarsa` brief content is not yet better than a simple "make it more vivid/specific" instruction.

Possible reasons:

- `vimarsa` briefs are too generic ("tighten imagery", "intensify contrast").
- the brief is based on embedding geometry rather than a robust critique of the task;
- aspects are absent in AUT and poetry-gen, so the brief collapses toward generic revision anyway;
- the generic revise arm always commits revision, while the cascade often discards its revision.

### 3. The active-inference components are partially non-causal

BMR candidate selection is real. Hopfield is conditionally real. But:

- free-energy budget is not gating behavior;
- `cit_temperature` is not enforced in Haiku CLI;
- switching trajectory in `vimarsa` is not supplied by the cascade and is treated as N/A.

This means the benchmark is not yet a full test of "active inference improves creativity." It is a test of a BMR/rerank + embedding-gated revision system with audit-only budget and CLI-default Haiku sampling.

### 4. The v0.3 controls are correctly harsher than v0.2

The clean substrate removed the leakage that likely distorted v0.2. The 2K and generic-revise controls are real controls, not strawmen. A negative result under those controls is not embarrassing; it is the first honest measurement of the hard problem.

### 5. The local proxy metrics may not reward philosophical richness

The local composites reward measurable features: aspect multiplicity, keyword/embedding alignment, unusual-use count, coherence, anti-cliche distance, and length-controlled proxies. They may not capture literary surprise, interpretive depth, or "creative nuance" in the human sense. But this cuts both ways: until human/judge validation is added, the project should not claim broad creativity gains.

## What Would Prove the Basic Hypothesis?

The basic hypothesis should be decomposed before it is tested:

> A Pratyabhijna-style recognitional cascade, implemented with active-inference control signals, produces more creative outputs than matched Haiku controls.

To prove that, v0.4 needs a chain of evidence:

1. **Generation signal**: PCE-generated revisions or candidates beat drafts and matched controls.
2. **Recognition signal**: `vimarsa`/event policy correctly predicts which candidate or revision is better.
3. **Mechanism signal**: removing BMR, Hopfield, `vimarsa` brief, or budget control weakens the result.
4. **Construct signal**: human or frontier-judge ratings agree with local proxy improvements.
5. **Budget signal**: the effect remains under equal calls/tokens/cost, or the claim is reframed as quality-per-dollar.

### Proposed v0.4 Experiment Pack

#### Experiment A — Shadow Revision Value

Question:

> Does the PCE shadow revision improve the draft before policy gating?

Arms:

- `haiku_cascade_draft`
- `haiku_cascade_shadow_revision`
- `haiku_bare`
- `haiku_generic_revise`

Primary endpoint:

- paired `score(shadow_revision) - score(draft)` across **all** cascade items.

Success criterion:

- positive BCa lower bound and positive blind-judge preference on a stratified subset.

Why it matters:

This tests whether the PCE revision generator has value independent of the broken gate.

#### Experiment B — Gate Calibration

Question:

> Does `vimarsa_event` predict positive revision value?

Data:

- For every item, compute `revision_delta = score(revision) - score(draft)`.
- Treat `vimarsa_event` as a binary classifier.

Metrics:

- precision/recall for positive `revision_delta`;
- AUROC if event score becomes continuous;
- calibration curve over `delta_F`, novelty, aspect count, ananda, and budget balance.

Success criterion:

- event-positive items have significantly higher `revision_delta` than event-negative items, or the gate is replaced.

Why it matters:

This tests recognition, not generation.

#### Experiment C — Commit Policy Arms

Question:

> Which policy should actually ship?

Arms using the same generated draft/revision artifacts:

- `always_draft`
- `always_revise`
- `event_gated`
- `oracle_commit` (post-hoc upper bound: choose revision iff it scores higher)
- `learned_gate` (train on half, evaluate on held-out half)

Success criterion:

- `learned_gate` or theory-gate beats `always_draft` and approaches `oracle_commit` without overfitting.

Why it matters:

This directly addresses the observed failure mode.

#### Experiment D — Mechanism Ablations

Question:

> Which part of Pratyabhijna + active inference carries any gain?

Arms:

- full PCE;
- no BMR (`jnana` uniform/random or ananda-only);
- no `apohana`;
- no `vimarsa` brief (generic revise);
- no Hopfield;
- no free-energy budget;
- no event gate (`always_revise`);
- no clean-substrate relaxation allowed except as a negative control.

Success criterion:

- a pre-registered ordering, e.g. `full > no_vimarsa > generic_revise` on aspect-rich tasks, or `preloaded_hopfield > no_hopfield` on aspect-memory tasks.

Why it matters:

Without ablations, "Pratyabhijna + active inference" remains too bundled to interpret.

#### Experiment E — Human / Frontier Judge Construct Validation

Question:

> Do humans or a fixed judge prefer PCE outputs when local proxies say they improve?

Design:

- blind pairwise comparisons;
- balanced arm labels hidden;
- at least two human raters or a frozen LLM-judge prompt plus a human audit subset;
- report inter-rater agreement.

Success criterion:

- local proxy deltas correlate with blind preference;
- PCE wins on a pre-specified creativity rubric, not only on proxy composites.

Why it matters:

This is required before saying "more creative" rather than "higher proxy score."

#### Experiment F — Powered Narrow-Domain Replication

Question:

> Is the strongest observed signal real?

Start with `poetry_interp`, because it shows the largest discarded shadow-revision gains.

Design:

- `n=30` or power-calculated target;
- fixed K and cost cap;
- compare `shadow_revision`, `event_gated`, `always_revise`, `generic_revise`, and bare;
- include human/LLM judge subset.

Success criterion:

- PCE shadow revision or calibrated PCE commit beats generic revise and bare with positive CI.

Why it matters:

Proving one domain cleanly is better than claiming all four domains prematurely.

## Required Corrections Before the Next Public Claim

1. Change paper language saying the free-energy budget gates abort/continue unless it is wired.
2. Clarify that `cit_temperature` is not enforced for the Haiku CLI backend.
3. Amend SPEC H5 fixed/random-effects mismatch.
4. Add a sample-smoke note that quota/rate-limit failures are externally caused and should be surfaced as such.
5. Add all-shadow H8 analysis to the paper or an appendix, clearly labeled post-hoc.
6. Reframe v0.3 as "negative for committed event-gated cascade, promising post-hoc for shadow revision value" rather than simply "active-inference uplift did not beat bare."

## Final Assessment

PCE v0.3 meets the strongest part of its engineering objective: it creates a clean, auditable, falsifiable benchmark substrate and no longer hides behind v0.2's confounds. But it does not yet meet the research objective.

The most useful next step is not a larger philosophical architecture. It is a disciplined mechanism study:

- prove whether the revision generator is useful;
- prove whether recognition/gating can identify useful revisions;
- only then claim that Pratyabhijna-style recognition improves creativity.

The surprising positive signal is that all-shadow revision looks better than draft on 15/20 items under the existing proxies. The hard negative signal is that the recognitional event gate failed to recognize most of those improvements. For a project named after recognition, that is both the core bug and the most promising next experiment.