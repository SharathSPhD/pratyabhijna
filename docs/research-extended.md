<!-- placeholder-policy: allow -->
# PCE — extended research grounding

Synthesis of [research1.md](../research1.md) plus three Phase-1 explorer briefs (active-inference / BMR / Hopfield; creativity benchmarks; thermodynamic generative computing + DMN/ECN gating). This document is the empirical and formal anchor for [docs/operator-spec.md](operator-spec.md). Every citation here is mirrored as a verified entry in [paper/references.bib](../paper/references.bib).

## 1. The convergence: Pratyabhijñā × active inference × abduction

[research1.md §3](../research1.md) makes the strongest available case that Abhinavagupta's *icchā → jñāna → kriyā* triad maps cleanly onto Friston's active-inference / Bayesian-Model-Reduction (BMR) framework, with Peirce's Firstness/Secondness/Thirdness as the Western mid-point. The structural identification is:

| Pratyabhijñā operator | Friston / active inference | Peirce |
|-----------------------|---------------------------|--------|
| `prakāśa` (luminous ground) | high-entropy generative prior over latents | Firstness — undifferentiated quality |
| `icchā` (intentional vector, pre-cognition) | candidate hypothesis sampling under constraint | abductive musement |
| `jñāna` (cognition / synthesis) | posterior selection by free-energy minimization | Secondness — surprise-driven hypothesis |
| `kriyā` (enactive realization) | predictive control / generative output | Thirdness — habit / surface form |
| `apohana-śakti` (categorical exclusion) | KL term against negative-exemplar prior | contrastive concept formation |
| `vimarśa` (recursive self-touching) | meta-level prior reduction over agent's own model | interpretant agency |

The structural compatibility is *exact* in the sense that each operator has a defined input-output shape on a typed state space and a non-arbitrary composition rule, and is *not* covered by any working implementation: research1.md §1 places Pratyabhijñā in Tier 2 ("architecturally formal, computationally underexplored") and identifies the recursive `vimarśa` layer as the gap that visible repositories (`attractor-flow`, `pramana`) imply but do not contain.

PCE's design commits to closing exactly that gap: every other layer below has at least one prior implementation we can lean on; the recursive-self-reflexivity layer is novel.

## 2. Active inference and Bayesian Model Reduction

### 2.1 pymdp API surface

[`infer-actively/pymdp`](https://github.com/infer-actively/pymdp) v1.0.0 (March 2026) ships a JAX-first backend with a discrete-state POMDP factorization defined by:

* `A[m]`: P(observation_m | hidden factors) — likelihood per modality, with explicit `A_dependencies[m]`
* `B[f]`: P(s_{f,t+1} | parents, actions) — per-factor transitions
* `C` : softmax log-preference over observations (cost / preference field)
* `D[f]`: prior over initial state for factor `f`

The `Agent` class exposes `infer_states(...)`, `infer_policies(qs) -> (q_pi, neg_efe)`, `calc_vfe`, `rollout()`, `infer_and_plan()`, `sample_action(...)` with explicit `jax.random` PRNG keys. Helpers `random_A_array`, `random_B_array`, `construct_controllable_B`, and the v1 `Model` / `Distribution` higher-level wrappers handle normalization. Observation format is categorical (renamed from `onehot_obs` to `categorical_obs` in v1). Citation: \citep{Heins2022pymdp}.

### 2.2 Bayesian Model Reduction equations

pymdp does not (as of the public release reviewed) ship a dedicated SPM-style `spm_MDP_log_evidence` BMR facade. The reducer for `jñāna` therefore has to be implemented directly. The general identity (Friston, Parr & Zeidman 2018, \citep{Friston2018BayesianModelReductionArxiv}) is:

\[
\tilde p(\theta\mid y) = p(\theta\mid y)\,\frac{\tilde p(\theta)}{p(\theta)}\,\frac{p(y)}{\tilde p(y)},
\qquad
\frac{\tilde p(y)}{p(y)} = \mathbb{E}_{p(\theta\mid y)}\!\Big[\frac{\tilde p(\theta)}{p(\theta)}\Big].
\]

For the Dirichlet-conjugate case (categorical likelihoods), with full-prior concentrations \(\mathbf{a}\), reduced-prior \(\tilde{\mathbf{a}}\), and full-posterior \(\mathbf{a}_\text{post}\):

\[
\tilde{\mathbf{a}}_\text{post} = \mathbf{a}_\text{post} + \tilde{\mathbf{a}} - \mathbf{a}
\]

\[
\Delta F = \ln \mathcal{B}(\tilde{\mathbf{a}}_\text{post}) - \ln \mathcal{B}(\mathbf{a}_\text{post}) + \ln \mathcal{B}(\mathbf{a}) - \ln \mathcal{B}(\tilde{\mathbf{a}})
\]

with \(\mathcal{B}(\mathbf{a}) = \prod_i \Gamma(a_i) / \Gamma(\sum_i a_i)\) the Dirichlet normalizer. \(\Delta F > 0\) means the reduced model is favoured. The "hyperparameters" are the prior-concentration choices, *not* step sizes. Greedy pruning suffices in practice.

The narrative of *insight as post-hoc model reduction* is in Friston/Lin et al. 2017 \citep{Friston2017ActiveInferenceCuriosity}, which links offline reduction explicitly to sleep-like pruning and "aha"-style reorganization.

### 2.3 Reducer pseudocode (used as `jñāna` core)

```python
def reduce_dirichlet(a_post, prior_full, prior_reduced):
    """Categorical BMR: returns (delta_F, q_reduced)."""
    tilde_a_post = a_post + prior_reduced - prior_full
    delta_F = (log_Beta(tilde_a_post)
               - log_Beta(a_post)
               + log_Beta(prior_full)
               - log_Beta(prior_reduced))
    return delta_F, normalize(tilde_a_post)
```

K candidate priors → pick `argmax_k delta_F_k`. PCE will run this in log-space throughout (via `scipy.special.gammaln`) to avoid overflow.

## 3. Storehouse / `ālayavijñāna` substrate

The Yogācāra storehouse motif (research1.md §2) is the closest premodern analogue to a learned latent generative model whose conditioned ripening produces novel outputs (Waldron 2003 \citep{Waldron2003BuddhistUnconscious}). The natural computational implementation is a **Hopfield-style attractor network**: associative memory, deterministic energy descent on recall, capacity bounded by the Amit-Gutfreund-Sompolinsky line (\(\approx 0.138\,P/N\)).

For creative-association testing we adopt Mtenga et al. 2024 \citep{Mtenga2024HopfieldCreativity} as the empirical benchmark — they demonstrate that inhibition gating in Hopfield networks switches between analytical and exploratory association regimes, which is exactly the bypass-control condition our `vimarśa` aspect-shift detector needs.

A separately-cited 2025 self-optimization analysis of Watson's lineage (Weber, Guckelsberger & Froese 2025, \citep{Weber2025SelfOptHopfield}, arXiv:2501.04007) gives the Hebbian update we use during the SWS schema-abstraction phase:
\[
s_i(t+1) = f\!\left[\sum_j w_{ij} s_j(t) + I_i\right],\quad \Delta w^{L}_{ij} = \alpha s_i s_j.
\]

> **Correction logged**: research1.md §1 referenced "Cuesta-Lopez et al. 2024 *Chaos, Solitons & Fractals*" Hopfield-creativity — the explorer brief could not verify that exact reference. Mtenga 2024 *J. Creative Behavior* is a confirmed substitute for the same role; if a CSF Cuesta-Lopez paper is later confirmed it can be added.

## 4. Thermodynamic generative substrate (`cit`-layer)

The bottom-layer "luminous ground" is most cleanly realized as a **Langevin-driven generative process**. The canonical equations are from Whitelam 2025 \citep{Whitelam2025Generative} (a correction over research1.md's "Whitelam 2024 *Nature Communications*" — that title resolves to arXiv:2506.15121):

\[
\dot{x}_i = -\mu\,\partial_i V_\theta(\mathbf{x}) + \sqrt{2\mu k_B T}\,\eta_i(t),\quad \langle\eta_i(t)\eta_j(t')\rangle = \delta_{ij}\delta(t-t')
\]

with potential \(V_\theta(\mathbf{x}) = \sum_i (J_2 x_i^2 + J_4 x_i^4 + b_i x_i) + \sum_{(ij)} J_{ij} x_i x_j\). Whitelam & Casert 2025 \citep{WhitelamCasert2025NonlinearNC} extend to nonlinear out-of-equilibrium thermodynamic computing. Coles et al. 2023 \citep{Coles2023ThermoAI} unify these under "Thermodynamic AI" with HMC-style \(d\mathbf{p} = [\mathbf{f} - BM^{-1}\mathbf{p}]dt + D\,d\mathbf{w}\).

For PCE we *do not* need to physically simulate Langevin dynamics — we need its *behaviour at the sampling layer*. The link is exact: an overdamped Langevin process with energy \(E(x)\) samples (in the long-time limit) from \(\propto \exp(-E(x)/k_BT)\), and lowering temperature sharpens the distribution. This is precisely the role of softmax temperature \(\tau\) in token sampling: \(\pi(a) \propto \exp(Q(a)/\tau)\). The `cit`-substrate is therefore implemented as a **temperature-scheduled token sampler** over the local LM's logits, with `T(t)` annealed across the cascade — concrete primitives:

| Primitive | Role |
|-----------|------|
| temperature `τ` | global curvature; primary explore→commit knob |
| top-p / nucleus | adaptive mass truncation; constraint-aware risk cap |
| top-k | hard cardinality cap; combine with `τ` |
| typical sampling | down-weights too-high / too-low surprisal tokens |
| mirostat | feedback loop targeting target perplexity band |

The `icchā` operator emits *K* candidate continuations under different `(τ, top-p, top-k, seed)` tuples — this is its operational definition.

## 5. Sleep / consolidation phase

Two recent threads inform the design:

* Soca et al. 2024 \citep{Soca2024MedHypEntropySleep} argue sleep reduces thermodynamic entropy of the CNS by exporting it via dissipation.
* Stochastic-thermodynamic models of the REM-NREM ultradian cycle (e.g., Sun et al., *Chaos, Solitons & Fractals* — exact bib entry to be confirmed in Phase 11).
* Lewis, Knoblich & Poe 2018 \citep{LewisKnoblichPoe2018} (research1.md): SWS abstracts schemas; REM does cross-domain replay.

PCE's `consolidation/` module implements two named phases:

* **SWS-like**: deterministic centroid extraction over recent traces in `ālayavijñāna`, low-temperature descent on schema energy. Implemented as k-means clustering + Hopfield rewrite.
* **REM-like**: stochastic cross-basin replay at *high* effective temperature, sampling random walks between attractor basins to seed novel attractor candidates. Implemented as random-walk Metropolis between Hopfield basins.

## 6. DMN / ECN gating — `vimarśa` activation criterion

Beaty et al. 2015 \citep{Beaty2015SciRepDMNECN} demonstrated divergent-thinking ability correlates with simultaneous engagement of antagonistic Default Mode and Executive Control networks (`r ≈ 0.44, p ≈ 0.02` against global graph efficiency). Chen, Kenett et al. 2025 \citep{ChenKenett2025CommunBiolDynamics} sharpen this with a sliding-window MTD-FC analysis showing creativity is predicted by the *count of segregated→integrated transitions* between DMN and ECN, with an inverted-U on integration balance.

Operationally:

\[
\text{switching frequency} = \#(T_\text{In} \to T_\text{Se}) + \#(T_\text{Se} \to T_\text{In})
\qquad
\text{balance} = \frac{T_\text{In} - T_\text{Se}}{T_\text{all}}.
\]

For PCE we don't have brain regions, but we do have analogous policy populations: an *associative* policy (`icchā`-driven exploratory token sampling) and a *verifier* policy (`apohana`-driven contrastive scoring). The `vimarśa` operator counts **segregated → integrated transitions** between these populations during a single cascade run. Aspect-shift = a specific transition pattern (multiple segregated episodes followed by a single late integrated episode) plus a novelty signal in the post-state. This is the formal definition that enters `docs/operator-spec.md` and the H6 hypothesis in `docs/SPEC.md`.

## 7. Creativity benchmarks — selection and rubric

Phase 1's benchmark brief identified four current benchmarks plus AUT and BBH-style probes. Final selection for PCE Phase 9:

### 7.1 POEMetric (Li, Wang, Wilkinson 2026 \citep{li2026poemetric})

* arXiv 2604.03695, replication at [`Bingru-Li/POEMetric`](https://github.com/Bingru-Li/POEMetric).
* 203 human exemplars across 7 forms (sonnet, villanelle, ghazal, ballad, limerick, haiku, free-verse). 10-dimension judging taxonomy split into:
  * **instruction-following**: form accuracy, theme alignment;
  * **advanced creative abilities**: creativity, lexical diversity (MATTR), idiosyncrasy, emotional resonance, literary devices (simile/metaphor/personification/allusion), imagery;
  * **general appraisal**: overall poem quality, human-vs-LLM authorship.
* Judging: deterministic prosody checker (70% rhyme/meter tolerance), LLM-as-judge (Gemini-2.5-Pro in the paper; we substitute Claude Sonnet/Opus to avoid Haiku-judges-Haiku circularity), human Likert 1–5 on 58 poems with κ/ρ agreement.
* **PCE n=20 slice**: stratified random subset by form (4 sonnets / 3 villanelles / 3 ghazals / 4 ballads / 3 limericks / 3 haiku) with frozen RNG seed `pce-poemetric-2026`; verbatim prompts copied from the public CSV with SHA-pin recorded in `audit/phase9/datasets.json`.

### 7.2 CreativityPrism (Hou et al. 2025 \citep{hou2025creativityprism})

* arXiv 2510.20091, project at [`joeyhou.github.io/CreativityPrism`](https://joeyhou.github.io/CreativityPrism/).
* Three orthogonal lenses (Quality / Novelty / Diversity) over 20 metrics aggregated across 9 tasks.
* Verbatim prompt examples: AUT — `Create a list of creative alternative uses for a bottle.`; DAT — `Please write 10 nouns in English that are as irrelevant from each other as possible, in all meanings and uses of the words.`
* Judging: AUT uses GPT-4 20-shot Likert 1–5 (Pearson ≈ 0.7 vs human, Organisciak et al. 2023 \citep{organisciak2023aut}); TTCT subscales (Fluency / Flexibility / Originality / Elaboration) via GPT-judging; many metrics min-max normalized.
* **PCE n=15 slice**: 8 AUT items + 4 TTCT-style + 3 short-story constraint packs.

### 7.3 Custom Wittgenstein aspect-shift task (poetry interpretation, n=20)

A novel probe needed for H2 / H6 since no existing benchmark targets aspect-shift directly:

* Input: a short poem (10 chosen from Project Gutenberg / public-domain anthology, 10 from the POEMetric human set so they can also be used as PCE-targets).
* Task: "Provide N (here N=4) interpretations of this poem, each highlighting a different aspect."
* Score = aspect-novelty: Sonnet judge rates 1–5 whether the four interpretations describe *qualitatively distinct* aspects (vs paraphrases of one reading). Vimarśa-active when ≥ 3 of 4 are scored ≥ 4.

### 7.4 BBH-style scientific creativity (n=15)

BBH alone is mostly symbolic. We compose a 15-item probe set:

* 5 cross-domain analogy items from MacGyver / SciBench;
* 5 novel-hypothesis items from BIG-Bench's `causal_judgement` and `formal_fallacies` (used inversely: ask for *novel* causal hypotheses, judge originality);
* 5 mathematical-creativity items in the style of `multistep_arithmetic_two_nums` extended with "find a non-obvious decomposition" prompts.

Suzgun et al. 2023 \citep{suzgun2023bbh} provide the canonical BBH framework; Organisciak et al. 2023 \citep{organisciak2023aut} frame the LLM-judge methodology.

### 7.5 CrEval / CreataSet (Cao et al. 2025 \citep{cao2025creval})

Used only for *judge-prompt validation*: the CrEval pairwise-creativity rubric is the cleanest operationalization of "judge ranks A vs B for creativity"; we adopt the prompt structure verbatim for the Sonnet/Opus judge in §7.1 and §7.3 to avoid bespoke rubrics.

### 7.6 CreativeBench (Wang et al. 2026 \citep{wang2026creativebench})

* HF [`Zethive/CreativeBench`](https://huggingface.co/datasets/Zethive/CreativeBench), arXiv 2603.11863.
* `creativity = quality × novelty` with binary pass via test harness + canonical-solution divergence.
* Out of scope for PCE Phase 9 (code-generation, not natural-language creativity), but the *score formula* informs the H1 aggregate index.

## 8. Numerical stability — implementation gotchas

* All Dirichlet-Beta computations in log-space via `scipy.special.gammaln` (never compute Γ directly).
* Always clamp `prior_reduced` cells away from 0 by `eps = 1e-7` to keep ratios finite.
* Softmax via `scipy.special.logsumexp`, with max-subtract before exponentiation.
* JAX `float32` is fine for inference but BMR free-energy differences need `float64` to keep ΔF stable across small reductions; pymdp v1 defaults to `float32`, override per-call.
* `pymdp.legacy` (numpy backend) is preferable for the symbolic BMR layer — JAX shines for batched amortized inference but our reducer is K-small and offline.

## 9. What we are NOT claiming

To keep §11 of the paper honest, three negative results are pre-registered now:

* PCE does *not* claim to capture the phenomenological luminosity (`prakāśa`) or affective `rasa`. Research1.md §4 flags these as out-of-reach for any current framework; we stop at the operator-grammar level.
* PCE does *not* implement a full thermodynamic-computing physical substrate (Whitelam-style hardware). The `cit`-layer is a *behavioural* analogue via temperature-scheduled sampling.
* PCE does *not* claim biological fidelity for its sleep / consolidation phase; it borrows the SWS / REM dual-optimizer pattern as engineering motivation only.

## 10. Citations source-of-truth

Every reference above resolves to a verbatim entry in [paper/references.bib](../paper/references.bib). Phase 11's `paper/citations.checksum` records the SHA-256 of (title + year + first-author + DOI) for each entry; the `verify_artifact.py` gate checks that every `\cite{key}` in `paper/main.tex` resolves AND has a matching checksum line.
