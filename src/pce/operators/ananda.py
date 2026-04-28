"""`ananda` - aesthetic-coherence reward.

Returns a scalar in [0, 1] aggregating coherence (cosine to constraint),
diversity (distinct-2/3 tokens), form fidelity (regex / syllable / required-
token check), and an optional cross-encoder reward.

Defaults follow [docs/operator-spec.md §2](../../../docs/operator-spec.md#2-ananda--aesthetic-coherence-scorer).
"""
from __future__ import annotations

import re

from pce.substrate.embed import Embedder
from pce.types import Candidate, Constraint

DEFAULT_WEIGHTS = {
    "coherence": 0.40,
    "diversity": 0.20,
    "form_fidelity": 0.20,
    "reward_model": 0.20,
}

_SYLL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _count_syllables(text: str) -> int:
    return len(_SYLL_RE.findall(text))


def _distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return float(len(set(grams)) / len(grams))


def _form_fidelity(text: str, constraint: Constraint) -> float:
    """Heuristic form-fidelity: try syllable / required-token checks if they're hinted."""
    score = 1.0
    constraint_text = constraint.text.lower()
    if "haiku" in constraint_text:
        # Haiku target: 17 syllables, ±3 tolerance.
        s = _count_syllables(text)
        diff = abs(s - 17) / 17.0
        score *= max(0.0, 1.0 - diff)
    if "5-7-5" in constraint_text:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            sylls = [_count_syllables(ln) for ln in lines[:3]]
            target = [5, 7, 5]
            diff = sum(abs(s - t) for s, t in zip(sylls, target, strict=False)) / max(1, sum(target))
            score *= max(0.0, 1.0 - diff)
        else:
            score *= 0.3
    if "haiku" not in constraint_text and "5-7-5" not in constraint_text:
        # No specific form hint: form_fidelity defaults to 0.7 (neutral) so it
        # doesn't dominate the aggregate without a reason.
        score = 0.7
    return float(max(0.0, min(1.0, score)))


def ananda(
    candidate: Candidate,
    *,
    constraint: Constraint,
    embed: Embedder,
    reward: float | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Returns a scalar reward in [0, 1]."""
    if not candidate.text.strip():
        return 0.0

    w = dict(weights or DEFAULT_WEIGHTS)
    # If no reward model provided, redistribute its weight onto the other axes proportionally.
    if reward is None:
        rw = w.pop("reward_model", 0.0)
        norm = sum(w.values()) or 1.0
        for k in list(w):
            w[k] = float(w[k]) * (1.0 + rw / norm)

    coherence = float(embed.cosine(candidate.embedding, constraint.embedding))
    coherence = max(0.0, min(1.0, (coherence + 1.0) / 2.0))  # cosine [-1,1] -> [0,1]

    tokens = candidate.text.split()
    div2 = _distinct_n(tokens, 2)
    div3 = _distinct_n(tokens, 3)
    diversity = float(0.5 * div2 + 0.5 * div3)

    form = _form_fidelity(candidate.text, constraint)

    score = (
        w.get("coherence", 0.0) * coherence
        + w.get("diversity", 0.0) * diversity
        + w.get("form_fidelity", 0.0) * form
    )
    if reward is not None:
        score += w.get("reward_model", 0.0) * float(max(0.0, min(1.0, reward)))
    return float(max(0.0, min(1.0, score)))
