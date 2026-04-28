# PCE — operator specification

Formal type signatures and invariants for the seven Pratyabhijñā operators that Phase 5 will implement. This document is the contract that every operator's tests in `tests/operators/` must enforce. It is derived from [docs/research-extended.md](research-extended.md) and is the upstream of [docs/SPEC.md](SPEC.md) (Phase 2).

The complete cascade is:

```
prompt + constraint
        │
        ▼
   ┌────────┐
   │  cit   │   high-entropy generative prior over a token-distribution
   └───┬────┘   under temperature-scheduled sampling.
       │
       ▼
   ┌────────┐
   │ ānanda │   aesthetic-coherence reward; assigns a scalar to candidate states.
   └───┬────┘
       │
       ▼
   ┌────────┐
   │ icchā  │   pre-cognitive directional vector: produces K candidate continuations
   └───┬────┘   under (τ, top-p, top-k, seed_k) tuples.
       │
       ▼      ┌──────────┐
   ┌────────┐ │ apohana  │  contrastive exclusion: anti-non-X scoring vs negative
   │ jñāna  │◄│  (side)  │  exemplars in embedding space.
   └───┬────┘ └──────────┘
       │      Bayesian Model Reduction over icchā's K candidates.
       ▼
   ┌────────┐
   │ kriyā  │   surface enaction: turn the selected jñāna candidate into text.
   └───┬────┘
       │
       ▼
   ┌────────┐
   │vimarśa │   recursive aspect-shift detector: did the cascade produce a new aspect?
   └───┬────┘
       │
       ▼
  CascadeRecord (with vimarśa_event flag, novelty score, audit trail)
```

## 0. Shared types

All types live in `src/pce/types.py` (Phase 5):

```python
from dataclasses import dataclass
from typing import NewType
import numpy as np
import numpy.typing as npt

Embedding = NewType("Embedding", npt.NDArray[np.float32])  # shape (d,), d=384 for MiniLM-L6-v2
LogitVec = NewType("LogitVec", npt.NDArray[np.float32])    # shape (V,), V = LM vocab size
Token = NewType("Token", int)
TokenSeq = NewType("TokenSeq", list[Token])

@dataclass(frozen=True)
class Constraint:
    """A constraint vector for icchā: pulls candidate generation toward an axis."""
    text: str                                # natural-language description (e.g. "haiku, 5-7-5")
    embedding: Embedding                     # MiniLM embedding of `text`
    weight: float = 1.0                      # multiplier on the constraint pull
    must_avoid: list[str] | None = None     # apohana negative exemplars

@dataclass(frozen=True)
class Candidate:
    """One icchā-generated candidate continuation."""
    seed: int
    sampler: dict[str, float]                # {"tau": float, "top_p": float, "top_k": int}
    tokens: TokenSeq
    text: str
    logp: float                              # sum log-prob under cit
    embedding: Embedding                     # MiniLM embedding of `text`

@dataclass(frozen=True)
class CascadeState:
    """The full state passed between operators during one cascade run."""
    prompt: str
    constraint: Constraint
    cit_temperature: float
    candidates: tuple[Candidate, ...]        # produced by icchā
    posterior: npt.NDArray[np.float32]       # shape (K,), softmax over candidates from jñāna
    selected: Candidate | None
    surface: str | None                      # produced by kriyā
    vimarsa_event: bool
    vimarsa_novelty: float
    aspects: tuple[str, ...]                 # extracted aspects (for vimarśa)
    audit: dict[str, float | int | str]      # arbitrary diagnostic floats
```

Substrate handles live in `src/pce/substrate/`:

```python
class CitSubstrate(Protocol):
    """The luminous-ground generative prior. Wraps the local LM."""
    def generate(self, prompt: str, *, max_tokens: int, sampler: dict[str, float],
                 seed: int) -> Candidate: ...

class EmbeddingSubstrate(Protocol):
    """sentence-transformers wrapper; deterministic on (text)."""
    def encode(self, text: str | list[str]) -> Embedding | list[Embedding]: ...

class HopfieldStore(Protocol):
    """ālayavijñāna storehouse."""
    def store(self, pattern: Embedding) -> None: ...
    def recall(self, cue: Embedding, *, max_iter: int = 50) -> Embedding: ...
    def consolidate_sws(self, traces: list[Embedding]) -> list[Embedding]: ...
    def consolidate_rem(self, n_steps: int = 100, temperature: float = 1.5) -> list[Embedding]: ...
```

## 1. `cit` — luminous-ground generative prior

`src/pce/operators/cit.py`

```python
def cit(prompt: str, *, lm: CitSubstrate, temperature: float, max_tokens: int = 64,
        top_p: float = 0.95, top_k: int = 50, seed: int = 0) -> Candidate
```

**Semantics.** Sample one continuation from the local LM's joint distribution under (`temperature`, `top_p`, `top_k`, `seed`). This is a thin wrapper that confirms the LM's logits exist, are non-degenerate, and obey the temperature relation: \(\pi(a) \propto \exp(Q(a)/\tau)\).

**Invariants (test-enforceable):**

* `cit(prompt, ..., temperature=0.001)` must return a near-greedy continuation: the top-token argmax under temperature 0.001 should equal the unconstrained argmax of the LM.
* `cit(prompt, ..., temperature=1.0)` and `cit(prompt, ..., temperature=2.0)` on the same seed must have non-identical token sequences with probability ≥ 0.95 over a 10-prompt sample.
* The Shannon entropy of the per-step token distribution must be monotonically non-decreasing in `temperature` over a 10-prompt × 5-temperature grid (regression test).
* The candidate's `logp` field must equal `sum_t log p(t | prefix, sampler)` to numerical tolerance (1e-4) under the LM's own scoring.

## 2. `ānanda` — aesthetic-coherence scorer

`src/pce/operators/ananda.py`

```python
def ananda(candidate: Candidate, *, constraint: Constraint,
           embed: EmbeddingSubstrate, reward: CrossEncoder | None = None) -> float
```

**Semantics.** Returns a scalar in `[0, 1]` aggregating four axes:

* `coherence`: cosine(MiniLM(candidate.text), MiniLM(constraint.text))
* `diversity`: distinct-2 / distinct-3 over candidate tokens (lexical diversity, MATTR-aligned)
* `form_fidelity`: fraction of constraint's literal demands met (e.g., haiku 5-7-5 syllable check, regex match for required tokens)
* `reward_model`: optional cross-encoder ranking score in `[0, 1]`

`ananda = w_c · coherence + w_d · diversity + w_f · form_fidelity + w_r · reward_model` with `(w_c, w_d, w_f, w_r) = (0.40, 0.20, 0.20, 0.20)` (defaults; tunable by Phase 6).

**Invariants:**

* Identical candidate twice → identical score (determinism).
* `ananda(empty_candidate, ...)` = 0.0 exactly.
* `ananda` ≥ 0.5 ⇒ `coherence` ≥ 0.30 (no aesthetic-without-coherence score allowed).
* Strictly monotone in `coherence` when other axes are held fixed (perturb test).

## 3. `icchā` — pre-cognitive directional vector

`src/pce/operators/iccha.py`

```python
def iccha(prompt: str, constraint: Constraint, *, lm: CitSubstrate,
          K: int = 8, sampler_grid: list[dict] | None = None,
          base_seed: int = 0) -> tuple[Candidate, ...]
```

**Semantics.** Emit K candidate continuations under K different sampler tuples drawn from `sampler_grid`. The default grid varies `(τ, top_p, top_k, seed)` to span exploit→explore. icchā is *pre-cognitive*: candidates are generated in parallel, none committed.

**Invariants:**

* Length-K output, all distinct seeds.
* Embedding diversity: `mean_pairwise_cosine(candidates) ≤ 0.85` (the K candidates are not all near-duplicates).
* For the default 8-candidate grid, the spread of `logp` across candidates must be ≥ 0.5 nats; the spread of `temperature` across the grid must be ≥ 0.5 (regression on grid choice).
* Reproducible: same `base_seed` → same candidate texts.

## 4. `apohana-śakti` — contrastive exclusion (side channel)

`src/pce/operators/apohana.py`

```python
def apohana(candidates: tuple[Candidate, ...], constraint: Constraint, *,
            embed: EmbeddingSubstrate) -> npt.NDArray[np.float32]
```

**Semantics.** For each candidate, return a contrastive score `s_k ∈ [-1, 1]`:

\[
s_k = \cos(c_k, q) - \max_{n \in \mathcal{N}} \cos(c_k, n)
\]

where `c_k` is the candidate embedding, `q` is `constraint.embedding`, and `𝒩` is the set of `constraint.must_avoid` exemplars. Operationalizes Buddhist *apoha*: "X is anti-non-X."

**Invariants:**

* Output shape `(K,)` matching `candidates`.
* If `must_avoid` is empty, `s_k = cos(c_k, q)` (degenerate-but-correct case).
* If a candidate equals a negative exemplar verbatim, `s_k ≤ 0` (anti-correct-on-pure-overlap test).
* Reproducible across calls.

## 5. `jñāna` — Bayesian Model Reduction posterior selection

`src/pce/operators/jnana.py`

```python
def jnana(candidates: tuple[Candidate, ...], apoha_scores: npt.NDArray[np.float32],
          ananda_scores: npt.NDArray[np.float32], *,
          full_prior: npt.NDArray[np.float32] | None = None,
          reduction_target: str = "halve") -> tuple[int, float, npt.NDArray[np.float32]]
```

**Semantics.** Maps K candidates → (`selected_index`, `delta_F`, `posterior`).

The full-prior is constructed from a flat Dirichlet `Dir(1, 1, ..., 1)` of length K updated with pseudo-counts `α_k = 1 + λ_a · ananda_scores_k + λ_p · max(0, apoha_scores_k)` to give the *full-model posterior* `a_post`. K reduced priors are then enumerated:

* `reduction_target="halve"`: each reduced prior keeps half the candidates (those with highest pseudo-counts) and zeros the rest;
* `reduction_target="single"`: K reductions, each keeping exactly one candidate (`a_k = 1.0`, others = ε);
* `reduction_target="custom"`: caller supplies the K reduced priors.

For each reduction, compute ΔF as in §2.2 of [research-extended.md](research-extended.md):

```python
delta_F_k = (gammaln(sum(tilde_a_post)).item() - sum(gammaln(tilde_a_post))
             - gammaln(sum(a_post)).item() + sum(gammaln(a_post))
             + gammaln(sum(a)).item() - sum(gammaln(a))
             - gammaln(sum(tilde_a)).item() + sum(gammaln(tilde_a)))
```

(implementation uses log-Beta differences; the line above is illustrative).

The selected reduction maximizes ΔF; `selected_index` is the candidate that survives. `posterior` is the normalized `tilde_a_post` of the winning reduction (length K, with surviving candidates at non-trivial mass and others at ε).

**Invariants:**

* `delta_F` strictly increases when a clearly-better reduction is added vs a worse one (regression on toy 3-candidate setup).
* The selected candidate's `apoha_scores[selected]` is in the top 50% of the K scores (BMR cannot pick a clearly-anti-constraint candidate).
* For `reduction_target="single"` the posterior has exactly one entry > 0.9 and the rest < 0.05 / (K-1) (strong reduction).
* Numerical: `delta_F` finite for all K∈[2,32] on a randomized stress test with 100 seeds.

## 6. `kriyā` — surface enaction

`src/pce/operators/kriya.py`

```python
def kriya(selected: Candidate, *, lm: CitSubstrate | None = None,
          render_mode: str = "verbatim",
          claude_renderer: Callable[[str], str] | None = None) -> str
```

**Semantics.** Turn the selected `Candidate` into a final surface text.

* `render_mode="verbatim"`: return `selected.text` unchanged (the cascade has already done the cognitive work).
* `render_mode="polish"`: re-pass through the local LM with a low-temperature polish prompt (`"Refine this preserving its meaning: ..."`).
* `render_mode="claude_polish"`: delegate the polish step to `claude_renderer` (Claude Haiku via CLI). Phase-9 default for poetry generation; off for AUT and scientific creativity (where surface form is irrelevant).

**Invariants:**

* `render_mode="verbatim"` is identity.
* `render_mode="polish"` preserves embedding cosine-similarity ≥ 0.85 with `selected.text` (semantic fidelity).
* Output is a non-empty string.

## 7. `vimarśa` — recursive aspect-shift detector

`src/pce/operators/vimarsa.py`

```python
def vimarsa(prompt: str, surface: str, *, embed: EmbeddingSubstrate,
            retrieval_set: list[str], aspects: list[str],
            ananda_score: float,
            iccha_apoha_trajectory: list[tuple[float, float]] | None = None
            ) -> tuple[bool, float, dict[str, float]]
```

**Semantics.** The novel-contribution operator. Returns `(vimarsa_event, novelty, diagnostic)`.

* **Aspect-novelty**: novelty = max\((0, 1 - \max_{r \in \text{retrieval\_set}} \cos(\text{embed}(\text{surface}), \text{embed}(r)))\). I.e., surface is novel iff dissimilar from every retrieval-set item.
* **Aspect-multiplicity**: at least 2 of the supplied `aspects` must be present in `surface` (substring-or-paraphrase check via embedding cosine ≥ 0.55).
* **Switching frequency** (when `iccha_apoha_trajectory` is provided): count segregated→integrated transitions between the icchā-explore policy (high entropy) and the apohana-verify policy (low entropy) during this cascade run; require ≥ 2 transitions for an `event` flag.
* **Aesthetic gate**: `ananda_score ≥ 0.4`.

`vimarsa_event = (novelty ≥ 0.30) AND (aspect_multiplicity ≥ 2) AND (switching ≥ 2 if available else True) AND (ananda_score ≥ 0.4)`.

**Invariants:**

* Idempotent on identical input.
* `vimarsa_event = False` whenever `surface ∈ retrieval_set` (cannot recognize what we already had).
* On the duck-rabbit textual probe (a poem with two known readings, `aspects=["duck", "rabbit"]`), at temperature 1.0 the event must fire (acceptance criterion in Phase 6).
* On a paraphrase-only control (`surface` is a near-paraphrase of an item in `retrieval_set`), the event must NOT fire (specificity test).
* `novelty ∈ [0, 1]`, `ananda_score ∈ [0, 1]` always.

## 8. Cascade orchestrator

`src/pce/cascade.py`

```python
def run_cascade(prompt: str, *, constraint: Constraint, substrate: Substrate,
                K: int = 8, render_mode: str = "verbatim",
                retrieval_set: list[str], aspects: list[str],
                seed: int = 0) -> CascadeState
```

**Semantics.** Compose `cit → icchā → (apohana, ānanda) → jñāna → kriyā → vimarśa` into one typed pipeline. All intermediate artifacts go into `CascadeState` for audit (`audit/phase6/probes.jsonl` records this in Phase 6).

**Invariants:**

* `run_cascade(... bypass_vimarsa=True)` produces identical output up through `kriyā` but `vimarsa_event = False` and `vimarsa_novelty = 0.0`. Used as the bypass-control in Phase 6.
* For the duck-rabbit probe, `run_cascade` with PCE on must show `vimarsa_event=True` while `bypass_vimarsa=True` shows `vimarsa_event=False`.
* Total cascade runtime ≤ 30 s wall-clock on the Phi-3-mini-4k substrate at K=8 (operational SLA for Phase 9 benchmarks).

## 9. Bypass-control mode

The `bypass_vimarsa` flag (and equivalently the `--no-pce` Phase-9 condition) bypasses the full cascade and emits the prompt to the LM directly with a single (canonical) sampler tuple. This is exactly the "no-PCE" baseline for the H1-H6 hypothesis tests.

## 10. Audit obligations

Every operator records its inputs, parameters, and outputs into a per-cascade audit dict that is appended to `audit/phaseN/cascade.jsonl`. The audit log is the single source of truth for Phase 9 statistics; `verify_artifact.py` and the per-phase tests both read it.
