# ADR-002 (v0.2) — Signed `apohana` and shifted `jnana` pseudo-counts

Status: Accepted (frozen during planning round 3, TRIZ five-pack).
Date: 2026-04-28.
Related TRIZ card: [docs/triz/C2-coverage-vs-novelty.md](../../triz/C2-coverage-vs-novelty.md).

## Context

The v0.1 `jnana` operator builds Dirichlet pseudo-counts as:

```python
pseudo = 1 + lambda_a * ananda + lambda_p * np.clip(apoha, 0., None)
```

The clip discards negative `apoha` evidence — i.e. candidates that are very close to a `must_avoid` exemplar score the same as neutral candidates. The adversarial review's probe confirms `apoha=[-10, 0]` produces the same posterior as `apoha=[0, 0]`. This is a P1-4 finding.

## Decision

Replace the clip with a min-max-normalized shift that maps `apoha` into `[0, 1]` over the K candidates of the current call:

```python
def _shift_apoha(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)

pseudo = 1 + lambda_a * ananda + lambda_p * _shift_apoha(apoha)
```

`apohana` gains an opt-in normalize flag:

```python
def apohana(candidates, constraint, *, embed, normalize: bool = False) -> npt.NDArray[np.float32]:
    raw = pos - neg_max  # unchanged
    if normalize:
        return _shift_apoha(raw)
    return raw
```

Default `normalize=False` preserves backward compatibility for unit tests and direct callers; `run_cascade` passes `normalize=True`. Inside `jnana`, the shift is applied unconditionally on the input regardless of whether `apohana` already normalized — the shift is idempotent on `[0, 1]`-shaped inputs (it returns them unchanged) so this is safe.

When `must_avoid=[]`, `apohana` returns raw positive cosines; the shift then renormalizes within candidate set, which is harmless because the BMR posterior is invariant to global re-scaling of `apoha` after addition with `ananda`.

## Consequences

Positive:

- Candidates close to `must_avoid` lose posterior mass, fixing the v0.1 silent-failure mode.
- The shift is bounded so `pseudo` cannot blow up (`pseudo <= 1 + lambda_a + lambda_p`).
- Idempotent on already-normalized input; safe to apply twice.

Negative:

- The shift loses absolute magnitude information across calls. We mitigate this by reporting the *raw* `apoha` distribution in the audit log alongside the shifted values, so post-hoc diagnostics are unaffected.
- Existing v0.1 unit tests that assumed `apoha=[0, 0]` and `apoha=[-10, 0]` produce equal posteriors will fail. Those tests are updated to assert the new (correct) behavior.

## Alternatives considered

- `softplus(scale * apoha)` with a negative branch: rejected because it has a free `scale` hyperparameter we would have to tune per domain.
- Direct subtraction `pseudo - max(0, -apoha)`: rejected because pseudo can go negative and break Dirichlet normalization.
- Min-max over all calls' history: rejected because the shift would be non-deterministic per call.

## Implementation pointers

- `src/pce/operators/apohana.py` — add `normalize` kwarg + `_shift_apoha` helper.
- `src/pce/operators/jnana.py` — replace the clip with `_shift_apoha(apoha)` (imported from `apohana`).
- `src/pce/cascade.py` — pass `normalize=True` to `apohana`.
- `tests/operators/test_apohana.py` — assert `pos - neg_max` semantics unchanged when `normalize=False`; assert shifted output in `[0, 1]` when `normalize=True`.
- `tests/operators/test_jnana.py` — replace the v0.1 negative-clip test with a positive test that `apoha=[-10, 0]` yields lower posterior on index 0 than `apoha=[0, 0]`.
