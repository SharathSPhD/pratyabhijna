"""Shared types for the PCE engine.

These dataclasses are frozen and exhaustive: every operator's input and output
shape is described here, and `tests/test_types.py` verifies the round-trip
JSON serialization used by the audit log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

EmbeddingArray = npt.NDArray[np.float32]
PosteriorArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Constraint:
    """A constraint vector pulling icchā candidate generation toward an axis."""

    text: str
    embedding: EmbeddingArray
    weight: float = 1.0
    must_avoid: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Constraint.text must be a non-empty string")
        if not isinstance(self.embedding, np.ndarray):
            raise TypeError("Constraint.embedding must be a numpy array")
        if self.embedding.ndim != 1:
            raise ValueError(f"Constraint.embedding must be 1D, got {self.embedding.ndim}D")
        if self.weight < 0:
            raise ValueError(f"Constraint.weight must be >= 0, got {self.weight}")


@dataclass(frozen=True)
class Candidate:
    """One icchā-generated candidate continuation."""

    seed: int
    sampler: dict[str, float]
    tokens: tuple[int, ...]
    text: str
    logp: float
    embedding: EmbeddingArray

    def __post_init__(self) -> None:
        if "tau" not in self.sampler:
            raise ValueError("Candidate.sampler must include 'tau'")
        if self.embedding.ndim != 1:
            raise ValueError(f"Candidate.embedding must be 1D, got {self.embedding.ndim}D")

    def to_audit(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "sampler": dict(self.sampler),
            "n_tokens": len(self.tokens),
            "text": self.text,
            "logp": float(self.logp),
        }


@dataclass(frozen=True)
class CascadeState:
    """The full state of one cascade run; the audit-of-record.

    v0.2 (ADR-003): the cascade is two-pass-always, so ``surface`` carries
    the *revision* output by default and the additional ``surface_draft`` /
    ``surface_revision`` fields preserve both passes for the H8.v2
    revision-vs-draft contribution test. When ``bypass_vimarsa=True`` is
    passed to ``run_cascade`` the cascade collapses to a single pass and
    ``surface == surface_draft`` with ``surface_revision is None``.
    """

    prompt: str
    constraint: Constraint
    cit_temperature: float
    candidates: tuple[Candidate, ...]
    posterior: PosteriorArray
    selected: Candidate | None
    surface: str | None
    vimarsa_event: bool
    vimarsa_novelty: float
    aspects: tuple[str, ...]
    surface_draft: str | None = None
    surface_revision: str | None = None
    vimarsa_event_draft: bool = False
    vimarsa_brief: str | None = None
    audit: dict[str, Any] = field(default_factory=dict)

    def to_audit(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "constraint_text": self.constraint.text,
            "constraint_must_avoid": list(self.constraint.must_avoid),
            "cit_temperature": float(self.cit_temperature),
            "n_candidates": len(self.candidates),
            "posterior": [float(p) for p in self.posterior.tolist()],
            "selected_idx": (
                self.candidates.index(self.selected) if self.selected is not None else -1
            ),
            "surface": self.surface,
            "surface_draft": self.surface_draft,
            "surface_revision": self.surface_revision,
            "vimarsa_event": self.vimarsa_event,
            "vimarsa_event_draft": self.vimarsa_event_draft,
            "vimarsa_novelty": float(self.vimarsa_novelty),
            "vimarsa_brief": self.vimarsa_brief,
            "aspects": list(self.aspects),
            "audit": dict(self.audit),
        }
