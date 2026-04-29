# ADR-002 (v0.4) — `LearnedGate` logistic regression commit policy

Status: Accepted (frozen at end of Phase 1).
Date: 2026-04-29.
Related TRIZ card: [docs/triz/v0.4/C1-theory-purity-vs-measurable-utility.md](../../triz/v0.4/C1-theory-purity-vs-measurable-utility.md).

## Context

The v0.3 adversarial review showed that the `vimarsa_event` gate committed `revision` on 3/20 items even though shadow revisions outscored drafts on 15/20 items. The recognitional gate, in its v0.3 form, throws away most of the latent gain. The C1 card resolves the contradiction by adding a learned policy alongside the event gate, not by replacing it.

## Decision

A new module `src/pce/policies/commit.py` defines a `CommitPolicy` ABC and five concrete policies:

```python
class CommitPolicy(ABC):
    name: str
    @abstractmethod
    def decide(self, state: CascadeState, features: PolicyFeatures, vimarsa_event: bool) -> bool: ...

class AlwaysDraft(CommitPolicy):    # never commits revision
class AlwaysRevise(CommitPolicy):   # always commits revision (current haiku_generic_revise_2pass)
class EventGated(CommitPolicy):     # current v0.3 policy via vimarsa_event
class LearnedGate(CommitPolicy):    # logistic regression over PolicyFeatures
class OracleCommit(CommitPolicy):   # offline analysis ONLY: picks higher-scoring surface
```

`PolicyFeatures` is a dataclass with five scalar features:

```python
@dataclass
class PolicyFeatures:
    delta_F: float          # jnana BMR delta_F on the draft
    novelty: float          # ananda novelty signal
    aspect_count: float     # vimarsa diagnostics aspect_count (0 if unsupplied)
    ananda: float           # vimarsa diagnostics ananda
    budget_balance: float   # FreeEnergyBudget.balance just before the revision pass
```

`LearnedGate` wraps a scikit-learn `LogisticRegression(class_weight="balanced", solver="liblinear")` over `PolicyFeatures.as_vector()`. The decision threshold is fixed at 0.5; the model is loaded from `artifacts/learned_gate_v0_4.joblib` at instantiation.

## Training data and leakage control

Training data is sampled exclusively from the v0.3 audit traces in `audit/v0_3_traces/*.jsonl` (the v0.3 pilot artifacts moved into a stable subdirectory at the start of Phase 3). The v0.4 evaluation set never enters training.

Each row in the training set is one `haiku_cascade` item from the v0.3 pilot:

- features: the five scalars above (taken from the v0.3 item's audit JSON).
- label: `1` if `score(shadow_revision) > score(draft) + epsilon` else `0`, where `epsilon = 0.0` (strict), with ties broken to label `0`.

Cross-validation: **leave-one-domain-out**. Train on three domains, evaluate on the held-out fourth, repeat for each held-out domain. The reported AUROC is the mean across folds. The deployed model is then retrained on all four domains (still v0.3 only — never v0.4 evaluation data).

`scripts/train_learned_gate.py` runs the cross-validation and the final fit; emits `artifacts/learned_gate_v0_4.joblib` and `artifacts/learned_gate_v0_4.metadata.json` with mean fold AUROC, per-fold AUROC, feature coefficients, and training data sha256.

## Pre-registration

H8c.v4 in [docs/SPEC_v0.4.md](../../SPEC_v0.4.md) tests `LearnedGate` vs `EventGated` paired across all v0.4 cascade items. Because the model was trained exclusively on v0.3 traces, the v0.4 comparison is held-out and unbiased.

`OracleCommit` is reported as a non-arm post-hoc upper bound. The gap between `LearnedGate` and `OracleCommit` quantifies "what better recognition could still buy."

## Consequences

Positive:

- Resolves the v0.3 review's headline finding (event gate discards useful revisions) without abandoning the event-gated theoretical motivation.
- Pluggable policy layer makes future ablations (Experiment D in v0.5) straightforward.
- Leakage controlled by domain split — no v0.4 evaluation data in training.

Negative:

- Adds scikit-learn `LogisticRegression` to the inference path. Already an existing dependency.
- The trained model is a small (≤ 100 KB) artifact committed to the repo. Tradeoff: reproducibility vs repo size.
- Acceptance criterion (mean AUROC > 0.55 on leave-one-domain-out CV) might fail on a small v0.3 training set; the gate falls back to `EventGated` and `H8c.v4` reports the failure honestly.

## Implementation files

- `src/pce/policies/__init__.py` — re-exports `CommitPolicy`, `PolicyFeatures`, all five policies.
- `src/pce/policies/commit.py` — ABC and concrete policies.
- `scripts/train_learned_gate.py` — leave-one-domain-out CV training + final fit.
- `artifacts/learned_gate_v0_4.joblib` — pickled `LogisticRegression`.
- `artifacts/learned_gate_v0_4.metadata.json` — fold AUROC, coefficients, training sha256.
- `tests/test_commit_policies.py` — all policies decide consistently on a frozen feature fixture.
- `tests/test_learned_gate_training.py` — CV AUROC ≥ 0.55 on v0.3 traces.

## Acceptance gate (Phase 3)

- `tests/test_commit_policies.py` passes (each policy returns the expected decision on the frozen fixture).
- `tests/test_learned_gate_training.py` passes (mean leave-one-domain-out CV AUROC ≥ 0.55).
- `artifacts/learned_gate_v0_4.joblib` and `artifacts/learned_gate_v0_4.metadata.json` exist and validate.
- `benchmarks/driver.py` cascade arm multiplexes `commit_policy ∈ {event_gated, always_draft, always_revise, learned_gate}` on the same draft/revision artifacts (no extra Haiku spend).
- Prove-gate v0.4 fixture: `learned_gate` differs from `event_gated` on at least one fixture item.
