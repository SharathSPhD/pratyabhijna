"""v0.4 commit policies (ADR-002).

Five commit policies decide, per-item, whether the cascade commits the shadow
revision or the draft surface:

* :class:`AlwaysDraft`   — never commit revision (control / ablation).
* :class:`AlwaysRevise`  — always commit revision.
* :class:`EventGated`    — v0.3 default: commit revision iff ``vimarsa_event``.
* :class:`LearnedGate`   — logistic regression over five scalar features
                            extracted from the cascade audit, trained on v0.3
                            traces with leave-one-domain-out CV.
* :class:`OracleCommit`  — *post-hoc* analysis: pick the higher-scoring
                            surface. Reported as an upper bound but NEVER an
                            evaluation arm (would leak label).

The policies form a uniform :class:`CommitPolicy` ABC with a single
``decide(features, vimarsa_event) -> bool`` method that returns ``True``
when the policy chooses revision, ``False`` for draft. The cascade applies
the chosen policy *after* both passes have completed (``always_draft`` is
the only case that skips the revision pass; that is handled in
``run_cascade`` directly so the LM-call savings are real).

This module deliberately keeps the abstract feature dataclass
(:class:`PolicyFeatures`) detached from :class:`pce.types.CascadeState` so
unit tests can assemble fixtures by hand and the training script can
re-construct features from v0.3 audit JSON without instantiating a full
cascade.

LearnedGate persistence: the trained model is loaded from
``artifacts/learned_gate_v0_4.joblib``. If the artifact is missing or
malformed, ``LearnedGate`` falls back to the v0.3 ``EventGated`` policy
and records the fallback reason in ``last_fallback_reason``. The
benchmark driver and prove-gate surface that field so a missing artifact
is not silently masked.

Per ADR-002 the feature vector ordering is frozen:
``[delta_F, novelty, aspect_count, ananda, budget_balance]``.
"""
from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEARNED_GATE_PATH = REPO_ROOT / "artifacts" / "learned_gate_v0_4.joblib"

PolicyName = Literal[
    "always_draft", "always_revise", "event_gated", "learned_gate", "oracle"
]


@dataclass(frozen=True)
class PolicyFeatures:
    """Frozen 5-D feature vector consumed by every commit policy.

    Ordering is locked by ADR-002 and must match the LearnedGate training
    script. ``as_vector()`` returns a ``list[float]`` for sklearn input.
    """

    delta_F: float
    novelty: float
    aspect_count: float
    ananda: float
    budget_balance: float

    FEATURE_ORDER: ClassVar[tuple[str, ...]] = (
        "delta_F",
        "novelty",
        "aspect_count",
        "ananda",
        "budget_balance",
    )

    def as_vector(self) -> list[float]:
        """Return [delta_F, novelty, aspect_count, ananda, budget_balance]."""
        return [
            float(self.delta_F),
            float(self.novelty),
            float(self.aspect_count),
            float(self.ananda),
            float(self.budget_balance),
        ]

    def to_audit(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls.FEATURE_ORDER)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(f) or math.isinf(f):
        return float(default)
    return f


def extract_features_from_audit(audit: Mapping[str, Any]) -> PolicyFeatures:
    """Build :class:`PolicyFeatures` from a v0.3 / v0.4 cascade audit dict.

    The audit dict is the union of fields written by ``run_cascade`` and
    ``benchmarks/driver.py`` for a cascade item. Missing or non-finite
    values fall back to deterministic defaults so the feature vector is
    always well-formed and finite (sklearn rejects NaN by default).

    Recognised keys (v0.3 + v0.4):

    * ``delta_F`` / ``delta_F_draft`` — preferred draft ΔF.
    * ``novelty`` / ``vimarsa_novelty`` — vimarsa novelty score.
    * ``vimarsa_diag.aspect_count`` / ``aspect_count`` — diagnostic aspect count.
    * ``vimarsa_diag.ananda`` / ``ananda`` — diagnostic ananda score.
    * ``budget_ledger.balance_bits`` / ``budget_balance`` — FE ledger balance.
    """
    delta_F = _safe_float(
        audit.get("delta_F_draft", audit.get("delta_F")),
        default=0.0,
    )
    novelty = _safe_float(
        audit.get("novelty", audit.get("vimarsa_novelty")),
        default=0.0,
    )
    diag = audit.get("vimarsa_diag_draft") or audit.get("vimarsa_diag") or {}
    if not isinstance(diag, Mapping):
        diag = {}
    aspect_count = _safe_float(
        audit.get("aspect_count", diag.get("aspect_count")),
        default=0.0,
    )
    ananda_val = _safe_float(
        audit.get("ananda", diag.get("ananda")),
        default=0.0,
    )
    ledger = audit.get("budget_ledger") or {}
    if not isinstance(ledger, Mapping):
        ledger = {}
    budget_balance = _safe_float(
        audit.get("budget_balance", ledger.get("balance_bits")),
        default=0.0,
    )
    return PolicyFeatures(
        delta_F=delta_F,
        novelty=novelty,
        aspect_count=aspect_count,
        ananda=ananda_val,
        budget_balance=budget_balance,
    )


class CommitPolicy(ABC):
    """ABC: decide whether to commit revision (True) or draft (False)."""

    name: ClassVar[PolicyName]

    @abstractmethod
    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        """Return True to commit revision, False to commit draft."""

    def to_audit(self) -> dict[str, Any]:
        return {"name": self.name}


class AlwaysDraft(CommitPolicy):
    """Never commit revision. Used as the v0.4 negative-control arm."""

    name: ClassVar[PolicyName] = "always_draft"

    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        return False


class AlwaysRevise(CommitPolicy):
    """Always commit revision. Used as the v0.2-style positive control."""

    name: ClassVar[PolicyName] = "always_revise"

    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        return True


class EventGated(CommitPolicy):
    """v0.3 default: commit revision iff vimarsa fired."""

    name: ClassVar[PolicyName] = "event_gated"

    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        return bool(vimarsa_event)


class LearnedGate(CommitPolicy):
    """Logistic regression over :class:`PolicyFeatures`.

    On instantiation, attempts to load a pickled ``LogisticRegression`` plus
    optional ``StandardScaler`` from ``model_path``. If loading fails, falls
    back to :class:`EventGated` and records the reason in
    ``self.last_fallback_reason``. The fallback path keeps the cascade
    runnable in environments where the artifact has not been built yet
    (e.g. fresh clones, CI before the training step).

    The underlying model exposes ``predict_proba`` so a soft probability is
    available alongside the binary decision. The probability is recorded
    on every ``decide`` call in ``self.last_proba`` so the audit log can
    surface it without needing a second forward pass.
    """

    name: ClassVar[PolicyName] = "learned_gate"

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        threshold: float = 0.5,
        fallback_event_gated: bool = True,
    ) -> None:
        self.model_path: Path = Path(model_path) if model_path else DEFAULT_LEARNED_GATE_PATH
        self.threshold = float(threshold)
        self.fallback_event_gated = bool(fallback_event_gated)
        self._model: Any | None = None
        self._scaler: Any | None = None
        self.last_proba: float | None = None
        self.last_fallback_reason: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import joblib
        except ImportError as exc:
            self.last_fallback_reason = f"joblib import failed: {exc}"
            return
        if not self.model_path.exists():
            self.last_fallback_reason = (
                f"artifact missing: {self.model_path} (run scripts/train_learned_gate.py)"
            )
            return
        try:
            blob = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            self.last_fallback_reason = f"joblib.load failed: {type(exc).__name__}: {exc}"
            return
        if isinstance(blob, dict):
            self._model = blob.get("model")
            self._scaler = blob.get("scaler")
        else:
            self._model = blob
            self._scaler = None
        if self._model is None or not hasattr(self._model, "predict_proba"):
            self.last_fallback_reason = (
                f"loaded artifact has no predict_proba: type={type(self._model).__name__}"
            )
            self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        if self._model is None:
            if not self.fallback_event_gated:
                # Strict mode: refuse to decide without a model.
                self.last_proba = None
                return False
            decision = bool(vimarsa_event)
            self.last_proba = None
            return decision
        x = [features.as_vector()]
        if self._scaler is not None:
            x = self._scaler.transform(x)
        proba_arr = self._model.predict_proba(x)
        # sklearn LogisticRegression returns shape (n_samples, n_classes).
        # We need P(revision_better) — the positive class.
        try:
            classes = list(self._model.classes_)
            pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
        except (AttributeError, ValueError):
            pos_idx = -1
        proba = float(proba_arr[0][pos_idx])
        self.last_proba = proba
        return proba >= self.threshold

    def to_audit(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_path": str(self.model_path),
            "threshold": float(self.threshold),
            "is_loaded": bool(self.is_loaded()),
            "last_proba": self.last_proba,
            "last_fallback_reason": self.last_fallback_reason,
        }


class OracleCommit(CommitPolicy):
    """Post-hoc oracle: commits whichever surface scores higher.

    Requires ``draft_score`` and ``revision_score`` to be supplied via
    :meth:`set_scores` *before* :meth:`decide` is called (the cascade
    cannot know either score until both passes finish, and the scoring
    happens in the benchmark driver post-hoc). NEVER use OracleCommit as
    an evaluation arm — it leaks the label. Reported only as an upper
    bound for the policy comparison in H8c.v4.
    """

    name: ClassVar[PolicyName] = "oracle"

    def __init__(self) -> None:
        self._draft_score: float | None = None
        self._revision_score: float | None = None
        self.last_choice: bool | None = None

    def set_scores(self, draft_score: float, revision_score: float) -> None:
        self._draft_score = float(draft_score)
        self._revision_score = float(revision_score)

    def decide(self, features: PolicyFeatures, vimarsa_event: bool) -> bool:
        if self._draft_score is None or self._revision_score is None:
            raise RuntimeError(
                "OracleCommit.decide called before set_scores; this should never "
                "appear in an evaluation arm. Use it only post-hoc with both scores."
            )
        choice = bool(self._revision_score > self._draft_score)
        self.last_choice = choice
        return choice

    def to_audit(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "draft_score": self._draft_score,
            "revision_score": self._revision_score,
            "last_choice": self.last_choice,
        }


_POLICY_REGISTRY: dict[str, type[CommitPolicy]] = {
    "always_draft": AlwaysDraft,
    "always_revise": AlwaysRevise,
    "event_gated": EventGated,
    "learned_gate": LearnedGate,
    "oracle": OracleCommit,
}


def policy_for_name(name: str, **kwargs: Any) -> CommitPolicy:
    """Factory: build a :class:`CommitPolicy` instance by name.

    ``kwargs`` are forwarded to the policy's constructor when supported
    (only :class:`LearnedGate` accepts kwargs today).
    """
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown commit policy: {name!r}; "
            f"valid={sorted(_POLICY_REGISTRY)}"
        )
    if cls is LearnedGate:
        return cls(**kwargs)
    if kwargs:
        raise TypeError(
            f"policy {name!r} does not accept kwargs (got {sorted(kwargs)})"
        )
    return cls()


def write_policy_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write LearnedGate training metadata as JSON. Used by the training script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
