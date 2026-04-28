# PCE — formal specification

Version: 0.1.0 (Phase 2 lock; will evolve via ADRs).

## 0. Purpose

PCE is a Claude Code plugin that operationalizes the Pratyabhijñā 5-śakti generative cascade as typed operators over an active-inference / Bayesian-Model-Reduction substrate, with a recursive *vimarśa* meta-module that detects aspect-shifts in the cascade output. The engineering bet is: a typed cascade that tracks both the *content* of a candidate and the *meta-trajectory* of the candidate-selection process can produce textual outputs that current chat models do not — specifically, candidates that exhibit Wittgensteinian aspect-shifts and high-creative-divergence under controlled benchmark conditions.

## 1. Scope

In scope (this version):

* The 7-operator cascade described in [docs/operator-spec.md](operator-spec.md) running over a 1.5–4 B local LM substrate (Phi-3-mini-4k-instruct or Qwen2-1.5B-Instruct) plus `sentence-transformers/all-MiniLM-L6-v2` embeddings and an in-process Hopfield store.
* A Claude Code plugin wrapping the engine: 15 MCP tools, 5 skills, 5 agents, 5 slash commands, 3 hooks.
* A 4-domain paired-A/B benchmark of Haiku-with-PCE versus Haiku-without-PCE: poetry-generation, poetry-interpretation (Wittgenstein aspect-shift), Alternative Uses Task (CreativityPrism subset), scientific-creativity probes (BBH-style).
* A statistically-rigorous report (paired permutation tests, Hedges' g, Wilcoxon signed-rank, BCa bootstrap CIs, Holm-Bonferroni, retrospective power), an HTML presentation, and an arxiv-format preprint.

Out of scope (this version):

* Multi-agent / population-level emulation of DMN/ECN dynamics beyond the segregation-integration counter.
* Physical thermodynamic-computing hardware.
* Affective `rasa` modeling, phenomenological `prakāśa`.
* Languages other than English (corpus, prompts, judges all English).
* Image / audio / multimodal extension.

## 2. Hypotheses (pre-registered)

These six hypotheses are the gating contract for "did the plugin work" and are recorded *here* before any benchmark is run; Phase 9's `benchmarks/results/*.json` reports must contain a record per hypothesis with: estimate, 95% BCa CI, paired permutation p-value (one-sided where directional), Wilcoxon signed-rank p-value, Hedges' g, and a-priori + retrospective power.

| ID | Statement | Domain | Effect direction | α | Power target |
|----|-----------|--------|------------------|---|--------------|
| **H1** | PCE-Haiku achieves higher CreativityPrism aggregate (`Quality·Novelty·Diversity`) than no-PCE Haiku on the AUT-8 + TTCT-7 slice. | aut | PCE > no-PCE | 0.05 | 0.80 |
| **H2** | PCE-Haiku achieves higher aspect-multiplicity score on the n=20 Wittgenstein poetry-interp probe than no-PCE Haiku. | poetry_interp | PCE > no-PCE | 0.05 | 0.80 |
| **H3** | PCE-Haiku achieves higher POEMetric advanced-creative-abilities composite (mean of creativity, lexical diversity, idiosyncrasy, emotional resonance, literary devices, imagery) on the n=20 stratified poetry-gen slice than no-PCE Haiku, judged by Sonnet/Opus. | poetry_gen | PCE > no-PCE | 0.05 | 0.80 |
| **H4** | PCE-Haiku achieves higher creative-novelty score on the n=15 BBH-style scientific creativity probes than no-PCE Haiku. | sci_creativity | PCE > no-PCE | 0.05 | 0.80 |
| **H5** | The aggregate composite C₀ = ½ · z(H1) + ⅙ · z(H2) + ⅙ · z(H3) + ⅙ · z(H4) (z-scaled per domain) is positive at α=0.05 (one-sided paired permutation, n=70 paired observations). | aggregate | C₀ > 0 | 0.05 | 0.85 |
| **H6** | The within-PCE *vimarśa-event-fired* trials score higher on H2 + H3 + H4 dimensions than within-PCE *vimarśa-event-not-fired* trials. (Internal-validity test: did the recursive layer actually cause the lift?) | within-PCE | event > no-event | 0.05 | 0.70 |

Pre-registered statistical method: paired permutation test (`scipy.stats.permutation_test` with `statistic=lambda a, b: np.mean(a - b)` and `permutation_type='samples'`) on the per-prompt paired score deltas. Hedges' g via the standard small-sample-correction formula. BCa bootstrap CIs with 10,000 resamples. Holm-Bonferroni correction across {H1, H2, H3, H4} (H5/H6 are derived from these and not double-counted). Power computed both a-priori (under the assumed effect size of g=0.5 on each domain with n_domain=15-20) and retrospectively (using the observed effect).

Negative-result obligation: if any hypothesis is rejected against PCE, the paper must report it in the abstract.

## 3. Architecture (engineering view)

```
┌────────────────────────────────────────────────────────────────────┐
│                      Claude Code CLI / Cursor MCP                    │
│                                ▲                                     │
│                         (15 MCP tools, 5 skills,                     │
│                          5 agents, 5 commands)                       │
│                                │                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    pratyabhijna-creative-engine                 │ │
│  │  plugin/                                                         │ │
│  │  ├── .claude-plugin/plugin.json                                  │ │
│  │  ├── .mcp.json                                                   │ │
│  │  ├── marketplace.json                                            │ │
│  │  ├── mcp/server.py        ◄── FastMCP server, imports `pce`      │ │
│  │  ├── skills/              ◄── 5 SKILL.md skills                  │ │
│  │  ├── agents/              ◄── 5 specialist agents                │ │
│  │  ├── commands/            ◄── 5 slash commands                   │ │
│  │  └── hooks/               ◄── 3 lifecycle hooks                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                │                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                          src/pce/                                │ │
│  │  ├── types.py               typed state                          │ │
│  │  ├── substrate/                                                   │ │
│  │  │   ├── lm.py            local LM wrapper (Phi-3-mini-4k)      │ │
│  │  │   ├── embed.py         sentence-transformers wrapper         │ │
│  │  │   └── hopfield.py      ālayavijñāna store                    │ │
│  │  ├── operators/                                                   │ │
│  │  │   ├── cit.py                                                  │ │
│  │  │   ├── ananda.py                                               │ │
│  │  │   ├── iccha.py                                                │ │
│  │  │   ├── apohana.py                                              │ │
│  │  │   ├── jnana.py        BMR-based posterior selector            │ │
│  │  │   ├── kriya.py                                                │ │
│  │  │   └── vimarsa.py      aspect-shift detector                   │ │
│  │  ├── consolidation/                                               │ │
│  │  │   ├── sws.py          deterministic schema abstraction        │ │
│  │  │   └── rem.py          stochastic cross-basin replay           │ │
│  │  └── cascade.py          orchestrator                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  ~/.cache/huggingface/hub/                                       │ │
│  │   models--microsoft--Phi-3-mini-4k-instruct                       │ │
│  │   models--sentence-transformers--all-MiniLM-L6-v2                 │ │
│  │   (verified by scripts/verify_real_model.py)                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

## 4. Operators

The seven operators with their full type signatures, semantics, and invariants are defined in [docs/operator-spec.md](operator-spec.md). The SPEC inherits all of those invariants by reference.

## 5. Plugin surface (Phase 7)

### 5.1 MCP tools (15)

Namespace: `pratyabhijna_mcp__<name>` (Cursor convention).

| # | Tool | Surface | Dispatches to |
|---|------|---------|---------------|
| 1 | `cascade_run` | `(prompt, constraint_text, K=8, render="verbatim")` → CascadeState | `cascade.run_cascade` |
| 2 | `op_cit` | `(prompt, temperature, seed)` → Candidate | `operators.cit` |
| 3 | `op_iccha` | `(prompt, constraint_text, K)` → list[Candidate] | `operators.iccha` |
| 4 | `op_jnana` | `(candidates, ananda_scores, apoha_scores)` → (idx, ΔF, posterior) | `operators.jnana` |
| 5 | `op_apohana` | `(candidates, must_avoid)` → scores | `operators.apohana` |
| 6 | `op_ananda` | `(candidate, constraint_text)` → score | `operators.ananda` |
| 7 | `op_kriya` | `(candidate, render_mode)` → text | `operators.kriya` |
| 8 | `op_vimarsa` | `(prompt, surface, retrieval_set, aspects)` → event/novelty | `operators.vimarsa` |
| 9 | `consolidate_sws` | `(traces)` → centroids | `consolidation.sws` |
| 10 | `consolidate_rem` | `(n_steps, temperature)` → replay traces | `consolidation.rem` |
| 11 | `engine_state` | `()` → engine diagnostic snapshot | `cascade.snapshot` |
| 12 | `engine_reset` | `()` → ack | `cascade.reset` |
| 13 | `model_loaded` | `()` → which models are loaded, sizes | `substrate.report` |
| 14 | `audit_log` | `(phase)` → JSONL contents | `audit.read_phase` |
| 15 | `bench_score` | `(domain, prompt, response_with_pce, response_without_pce)` → scores | `benchmarks.score` |

Every tool is required to return a non-canned, engine-touched response in the Phase-8 smoke test (`audit/phase8/smoke.json`).

### 5.2 Skills (5)

Each ships as a `SKILL.md` with YAML frontmatter (name, description) and a body.

| Skill | Trigger / use |
|-------|---------------|
| `pratyabhijna_apply_cascade` | Use when generating creative text under constraint and the user wants `vimarśa`-aware output. |
| `pratyabhijna_interpret_aspect_shift` | Use when interpreting a poem / image-description / ambiguous text in N qualitatively-distinct ways. |
| `pratyabhijna_aut_brainstorm` | Use for AUT-style alternative-uses brainstorming under explicit constraint vector. |
| `pratyabhijna_scientific_analogy` | Use to propose cross-domain scientific or mathematical analogies / hypotheses. |
| `pratyabhijna_audit_trace` | Use to read `audit/phase*` outputs and summarize a cascade run. |

### 5.3 Agents (5)

| Agent | Role |
|-------|------|
| `pratyabhijna_engineer` | Implementation help inside the engine (operators, substrate). |
| `pratyabhijna_pratyabhijna_scholar` | Sanskrit-text-aware question answering grounded in research-extended.md. |
| `pratyabhijna_critic` | Read a cascade output and propose specific revisions. |
| `pratyabhijna_benchmark_runner` | Drive Phase-9 paired A/B from inside Claude Code. |
| `pratyabhijna_consolidation_manager` | Manage SWS + REM phases on the in-process Hopfield store. |

### 5.4 Slash commands (5)

| Command | Behaviour |
|---------|-----------|
| `/pratyabhijna_run` | invoke `cascade_run` on a freeform prompt |
| `/pratyabhijna_aspects` | run `op_vimarsa` against the active document |
| `/pratyabhijna_aut` | run AUT brainstorm with K=8 candidates |
| `/pratyabhijna_consolidate` | run an SWS + REM consolidation pass |
| `/pratyabhijna_status` | summarize loaded models, last cascade audit, version |

### 5.5 Hooks (3)

| Hook | Event | Behaviour |
|------|-------|-----------|
| `cascade_audit_hook` | `PostToolUse` for any `pratyabhijna_mcp__cascade_*` | append the cascade record to `audit/cascade.jsonl` |
| `prompt_constraint_hook` | `UserPromptSubmit` | extract any explicit constraints from the user prompt and persist into `.pce/last_constraint.json` |
| `consolidate_at_idle_hook` | `Stop` | trigger SWS + REM consolidation when the cascade has run > N times since last consolidation |

## 6. Benchmark protocol (Phase 9)

### 6.1 Subjects

The unit of analysis is *prompt*. Each domain has a fixed prompt slice (Phase 1):

| Domain | n | Source | Stratification |
|--------|---|--------|----------------|
| poetry_gen | 20 | POEMetric public CSV | sonnet/villanelle/ghazal/ballad/limerick/haiku |
| poetry_interp | 20 | 10 Project Gutenberg + 10 POEMetric | aspect-pair difficulty (subjective 1-3) |
| aut | 15 | CreativityPrism public release (8 AUT + 4 TTCT + 3 short-story) | task type |
| sci_creativity | 15 | MacGyver/SciBench/BBH composite | analogy / hypothesis / math-creativity |

### 6.2 Pairing

Each prompt is run through Claude Haiku twice in randomized order (per-prompt random seed):

* Condition `pce`: prompt is wrapped through the `cascade_run` MCP tool and the cascade's `kriyā` output is taken as the response.
* Condition `nopce`: prompt is sent directly to Haiku via the `claude` CLI with no plugin loaded.

Both conditions log to `audit/phase9/calls.jsonl` with: timestamp, prompt SHA-256, condition, raw Haiku output, judge score per metric, `vimarsa_event` (PCE only).

### 6.3 Judging

* Sonnet/Opus as judge (chosen *not* Haiku to avoid same-model self-judging).
* Per-domain rubric stored in `benchmarks/judge_prompts/<domain>.txt`. CrEval pairwise rubric (Cao et al. 2025) used as the default for free-form creativity scoring.
* Each judging call emits a JSON object with the per-metric scores; we average over 3 judge runs per response (different judge prompt seeds) to control judge variance.

### 6.4 Statistics

For each hypothesis Hi:

1. compute the per-prompt paired delta `δ_i = score_pce - score_nopce`;
2. paired permutation test (`scipy.stats.permutation_test`, n_resamples=10_000, one-sided): p-value;
3. Hedges' g with small-sample correction;
4. Wilcoxon signed-rank p (sanity / non-parametric backup);
5. BCa bootstrap CI (10_000 resamples) for the paired-mean delta;
6. retrospective power via the observed effect size and n.

Multi-comparison: Holm-Bonferroni across {H1, H2, H3, H4}. H5 (composite) is reported as a single test on the combined-z paired delta. H6 (within-PCE event vs no-event) is a within-condition Wilcoxon.

A hypothesis is considered "supported" if the Holm-corrected p < 0.05 and the BCa CI is strictly positive.

## 7. Acceptance criteria for v0.1.0 release

* All 7 operators implemented with operator-spec invariants holding under pytest (Phase 5).
* All 4 connectors (anti-stub, real-model, artifact, remote-push) green for every phase 0–11.
* On the duck-rabbit textual probe, `vimarsa_event=True` for at least 9 of 10 runs at temperature 1.0; `vimarsa_event=False` for all bypass-control runs.
* H1 + H2 + H5 directional support after Holm-Bonferroni (the *minimum* claim).
* H3, H4, H6 reported regardless of outcome.
* Plugin smoke test in Phase 8 produces a non-canned response from each of the 15 tools, 5 skills, 5 agents, 5 commands, 3 hooks.
* Paper compiles cleanly; HTML presentation renders standalone with all charts sourced from real `benchmarks/results/*.json`.

## 8. Risk register

| Risk | Mitigation |
|------|-----------|
| HF download blocked / gated | Stop and surface to user, don't substitute. |
| Claude CLI rate limits during Phase 9 | Batch with rate-limit aware retry (p99 wait < 60 s); checkpoint per-prompt. |
| Judge variance > effect size | 3-run averaging + judge-prompt frozen via SHA. |
| `vimarsa` event over-fires | Threshold tuned in Phase 6 against the bypass-control specificity test. |
| pymdp v1 / JAX incompatibility on macOS | Fall back to `pymdp.legacy` (numpy backend) for the BMR step; recorded in ADR-003. |
| Phi-3 too small for cascade quality | Plan B is Qwen2-1.5B-Instruct under same interface; ADR-001. |
| Local LM too slow | Cap max_tokens=64 per icchā candidate; cache LM logits; concurrent K via `torch.compile` in Phase 6. |

## 9. Glossary

* **Aspect-shift** (Wittgenstein): perceiving the same stimulus under a *qualitatively different* aspect (duck-rabbit). For PCE, an interpretation that introduces a feature absent from prior interpretations *and* coherent with the source.
* **Apohana**: Buddhist categorical-exclusion semantics — "X is anti-non-X." For PCE, contrastive scoring against negative exemplars.
* **BMR**: Bayesian Model Reduction — recompute posterior under a simpler prior, accept if free-energy gain is positive.
* **Cit-substrate**: the luminous-ground generative prior, implemented as a temperature-scheduled token sampler over the local LM's logits.
* **Cascade-state**: the typed dictionary passed between operators; the audit-of-record.
* **Vimarśa**: recursive self-touching; in PCE, the meta-detector that flags aspect-shifts in the cascade output.
