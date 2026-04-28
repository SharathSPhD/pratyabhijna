"""`vimarśa` - recursive aspect-shift detector.

The single most novel operator in PCE. Returns (event, novelty, diagnostic).

Activation criterion (conjunctive, ADR-002):

* novelty       >= τ_n (default 0.30)
* aspect_count  >= k   (default 2)
* switching     >= 2   (when an icchā/apohana trajectory is supplied)
* aesthetic     >= τ_a (default 0.40)
"""
from __future__ import annotations

import numpy as np

from pce.substrate.embed import Embedder

DEFAULT_NOVELTY_THRESHOLD = 0.30
DEFAULT_ASPECT_THRESHOLD = 2
DEFAULT_SWITCHING_THRESHOLD = 2
DEFAULT_ASPECT_COSINE_HIT = 0.55
DEFAULT_AESTHETIC_FLOOR = 0.40


def _count_switching(trajectory: list[tuple[float, float]]) -> int:
    """Count segregated->integrated transitions.

    Each entry is (entropy_iccha, entropy_apohana). A step is `integrated` iff
    both entropies are within 25% of each other; `segregated` otherwise. We
    count every Segregated->Integrated boundary crossing.
    """
    if not trajectory:
        return 0
    states: list[str] = []
    for e_i, e_a in trajectory:
        if max(e_i, e_a) <= 0:
            states.append("seg")
            continue
        ratio = min(e_i, e_a) / max(e_i, e_a)
        states.append("int" if ratio >= 0.75 else "seg")
    transitions = 0
    for i in range(1, len(states)):
        if states[i - 1] == "seg" and states[i] == "int":
            transitions += 1
    return transitions


def vimarsa(
    prompt: str,
    surface: str,
    *,
    embed: Embedder,
    retrieval_set: list[str],
    aspects: list[str],
    ananda_score: float,
    iccha_apoha_trajectory: list[tuple[float, float]] | None = None,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    aspect_threshold: int = DEFAULT_ASPECT_THRESHOLD,
    switching_threshold: int = DEFAULT_SWITCHING_THRESHOLD,
    aspect_cosine_hit: float = DEFAULT_ASPECT_COSINE_HIT,
    aesthetic_floor: float = DEFAULT_AESTHETIC_FLOOR,
) -> tuple[bool, float, dict[str, float]]:
    if not surface.strip():
        return False, 0.0, {"empty_surface": 1.0}
    surf_emb = embed.encode(surface)
    # Novelty against retrieval set.
    if retrieval_set:
        retr_embs = embed.encode(list(retrieval_set))
        if retr_embs.ndim == 1:
            retr_embs = retr_embs[None, :]
        sims = retr_embs @ surf_emb
        max_sim = float(sims.max())
    else:
        max_sim = 0.0
    novelty = float(max(0.0, 1.0 - max_sim))
    # Aspect multiplicity.
    if aspects:
        asp_embs = embed.encode(list(aspects))
        if asp_embs.ndim == 1:
            asp_embs = asp_embs[None, :]
        a_sims = asp_embs @ surf_emb
        aspect_count = int(np.sum(a_sims >= aspect_cosine_hit))
    else:
        aspect_count = 0
    # Switching frequency.
    switching = _count_switching(iccha_apoha_trajectory or [])
    diag = {
        "novelty": novelty,
        "max_retrieval_cosine": max_sim,
        "aspect_count": float(aspect_count),
        "switching": float(switching),
        "ananda": float(ananda_score),
    }
    novelty_ok = novelty >= float(novelty_threshold)
    aspect_ok = aspect_count >= int(aspect_threshold)
    aesthetic_ok = float(ananda_score) >= float(aesthetic_floor)
    if iccha_apoha_trajectory is None:
        switching_ok = True
    else:
        switching_ok = switching >= int(switching_threshold)
    event = bool(novelty_ok and aspect_ok and aesthetic_ok and switching_ok)
    return event, novelty, diag
