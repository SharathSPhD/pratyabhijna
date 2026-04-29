"""`vimarśa` - recursive aspect-shift detector and revision brief.

Returns ``(event, novelty, diagnostic)`` by default and
``(event, novelty, diagnostic, brief)`` when ``return_brief=True`` so the
two-pass cascade in ``pce.cascade.run_cascade`` can feed the brief back into
``iccha`` as a revision instruction (ADR-003).

Activation criterion (conjunctive):

* novelty       >= τ_n (default 0.30)
* aspect_count  >= k   (default 1, was 2 in v0.1)
* aesthetic     >= τ_a (default 0.40)
* switching     >= s   (default 1) - only enforced when the caller supplies
  an iccha/apohana trajectory of at least ``min_evidence_points`` entries;
  treated as N/A otherwise (per ADR-003 the two-pass cascade always passes
  ``iccha_apoha_trajectory=None``, so switching does not gate the event).

v0.2 changes vs v0.1:

* ``min_evidence_points`` (default 1) replaces the old hardcoded
  ``switching_threshold=2`` which required at least two trajectory points
  to fire. The two-pass cascade only has one observation per pass so the
  v0.1 gate was structurally closed (P0-2 in the adversarial review).
* ``aspect_threshold`` defaults to 1 (was 2). The duck-rabbit textual probe
  has two aspects but small models often only realize one of them in a
  short surface; the v0.1 floor of 2 was too strict.
* ``return_brief=True`` makes ``vimarsa`` emit a structured revision brief
  the cascade can feed back into the next pass. When the caller has not
  supplied any aspects (poetry_gen, AUT) we emit a generic creative
  revision brief instead of failing the gate on aspects.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from pce.active_inference.hopfield import HopfieldStore, WriteMode
from pce.substrate.embed import Embedder

DEFAULT_NOVELTY_THRESHOLD = 0.30
DEFAULT_ASPECT_THRESHOLD = 1
DEFAULT_SWITCHING_THRESHOLD = 1
DEFAULT_MIN_EVIDENCE_POINTS = 1
# Tuned in v0.1 phase 6 against Qwen2-1.5B surfaces and the duck-rabbit /
# river-clock / wave-particle / candlestick-faces probe battery: 0.55 was
# too strict for avg-pooled MiniLM embeddings on long surfaces; 0.40 fires
# on 2/4 probes without firing on the bypass control.
DEFAULT_ASPECT_COSINE_HIT = 0.40
DEFAULT_AESTHETIC_FLOOR = 0.40
# v0.3 ADR-002: vimarsa is now event-gated by ΔF as well as the v0.2 criteria.
# When ``delta_F`` is supplied (the cascade always supplies it from jñāna's
# draft-pass posterior), the event additionally requires
# ``delta_F >= delta_F_threshold``. Default threshold of 0.05 bits is a
# weak floor: any reduction with informative evidence above noise. ΔF below
# 0.05 means jñāna couldn't pick a winner with confidence -- in that case
# the revision pass is unlikely to help so we skip the commit-revision
# branch and surface the draft instead.
DEFAULT_DELTA_F_THRESHOLD = 0.05

GENERIC_BRIEF = (
    "Refine the previous draft for novelty, vividness, and surprise. "
    "Push at least one image or claim further than a baseline reading would. "
    "Keep the same form and length."
)


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


def _build_brief(
    *,
    aspects: list[str],
    aspect_sims: npt.NDArray[np.float32] | None,
    aspect_cosine_hit: float,
    novelty: float,
    novelty_threshold: float,
) -> str:
    """Compose a revision brief listing missing aspects + novelty pressure.

    When the caller has no aspect dictionary (poetry_gen, AUT) we return the
    generic creative brief.
    """
    if not aspects or aspect_sims is None or aspect_sims.size == 0:
        return GENERIC_BRIEF
    sims = np.asarray(aspect_sims, dtype=np.float32)
    missing: list[str] = [
        aspects[i] for i in range(len(aspects)) if float(sims[i]) < float(aspect_cosine_hit)
    ]
    parts: list[str] = []
    if missing:
        formatted = "; ".join(missing[:4])
        parts.append(f"Surface the following aspects that the draft did not realize: {formatted}.")
    if novelty < float(novelty_threshold):
        parts.append("Move further from common readings - the draft was too close to obvious priors.")
    if not parts:
        # Aspects all hit and novelty already cleared the bar; ask for sharpening.
        parts.append("Tighten imagery and intensify the contrast between the named aspects.")
    return " ".join(parts)


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
    min_evidence_points: int = DEFAULT_MIN_EVIDENCE_POINTS,
    aspect_cosine_hit: float = DEFAULT_ASPECT_COSINE_HIT,
    aesthetic_floor: float = DEFAULT_AESTHETIC_FLOOR,
    delta_F: float | None = None,
    delta_F_threshold: float = DEFAULT_DELTA_F_THRESHOLD,
    return_brief: bool = False,
) -> (
    tuple[bool, float, dict[str, float]]
    | tuple[bool, float, dict[str, float], str]
):
    """Detect an aspect-shift event over the surface text.

    When ``return_brief=True`` the return is extended with a string brief
    that the cascade can feed back into ``iccha`` for a revision pass.
    """
    if not surface.strip():
        diag_empty: dict[str, float] = {"empty_surface": 1.0}
        if return_brief:
            return False, 0.0, diag_empty, GENERIC_BRIEF
        return False, 0.0, diag_empty
    surf_emb = embed.encode(surface)
    if retrieval_set:
        retr_embs = embed.encode(list(retrieval_set))
        if retr_embs.ndim == 1:
            retr_embs = retr_embs[None, :]
        sims = retr_embs @ surf_emb
        max_sim = float(sims.max())
    else:
        max_sim = 0.0
    novelty = float(max(0.0, 1.0 - max_sim))
    aspect_sims: npt.NDArray[np.float32] | None
    if aspects:
        asp_embs = embed.encode(list(aspects))
        if asp_embs.ndim == 1:
            asp_embs = asp_embs[None, :]
        aspect_sims = np.asarray(asp_embs @ surf_emb, dtype=np.float32)
        aspect_count = int(np.sum(aspect_sims >= aspect_cosine_hit))
    else:
        aspect_sims = None
        aspect_count = 0
    switching = _count_switching(iccha_apoha_trajectory or [])
    diag: dict[str, float] = {
        "novelty": novelty,
        "max_retrieval_cosine": max_sim,
        "aspect_count": float(aspect_count),
        "aspect_threshold": float(aspect_threshold),
        "switching": float(switching),
        "switching_threshold": float(switching_threshold),
        "min_evidence_points": float(min_evidence_points),
        "ananda": float(ananda_score),
    }
    if aspect_sims is not None:
        diag["aspect_max_cosine"] = float(aspect_sims.max())
    novelty_ok = novelty >= float(novelty_threshold)
    # When the domain has no aspect dictionary we treat the aspect gate as N/A
    # (the brief becomes the generic creative brief). This unblocks
    # poetry_gen and AUT, where v0.1 silently failed on `aspects=[]`.
    aspect_ok = (not aspects) or aspect_count >= int(aspect_threshold)
    aesthetic_ok = float(ananda_score) >= float(aesthetic_floor)
    # Switching gate is only enforced when caller supplied enough trajectory
    # evidence; otherwise treated as N/A per ADR-003.
    if iccha_apoha_trajectory is None or len(iccha_apoha_trajectory) < int(min_evidence_points):
        switching_ok = True
        diag["switching_gate"] = 0.0  # N/A
    else:
        switching_ok = switching >= int(switching_threshold)
        diag["switching_gate"] = 1.0
    # v0.3 ADR-002: ΔF gate. When the cascade supplies a draft-pass ΔF, the
    # event must clear `delta_F_threshold` (default 0.05 bits) on top of the
    # v0.2 criteria. ΔF is the only signal here that comes from BMR rather
    # than embedding heuristics, so it carries meaningful active-inference
    # information about whether jñāna actually had evidence to commit to a
    # winning reduction. When ``delta_F is None`` (legacy callers) the gate
    # is N/A.
    if delta_F is None:
        delta_F_ok = True
        diag["delta_F_gate"] = 0.0  # N/A
    else:
        delta_F_ok = float(delta_F) >= float(delta_F_threshold)
        diag["delta_F_gate"] = 1.0
        diag["delta_F"] = float(delta_F)
        diag["delta_F_threshold"] = float(delta_F_threshold)
    event = bool(novelty_ok and aspect_ok and aesthetic_ok and switching_ok and delta_F_ok)
    if return_brief:
        brief = _build_brief(
            aspects=list(aspects),
            aspect_sims=aspect_sims,
            aspect_cosine_hit=aspect_cosine_hit,
            novelty=novelty,
            novelty_threshold=novelty_threshold,
        )
        return event, novelty, diag, brief
    return event, novelty, diag


def consolidate(
    *,
    surface: str,
    aspects: list[str],
    embed: Embedder,
    hopfield: HopfieldStore,
    mode: WriteMode = "rem",
    label_strategy: Literal["best_aspect", "first_aspect", "domain_only"] = "best_aspect",
) -> dict[str, object]:
    """Write the committed surface back to the per-domain storehouse.

    v0.3 ADR-004: this is the cascade-end hook that closes the
    iccha-apoha-jnana-kriya-vimarsa loop. The cascade calls ``consolidate``
    after committing a surface; the storehouse then carries warm-start mass
    for the next prompt in the same domain.

    ``mode="rem"`` (default) appends the surface as a new pattern (fast,
    REM-style). ``mode="sws"`` consolidates against the nearest existing
    pattern when cosine ≥ threshold (slow-wave-style merge).

    ``label_strategy``:

    * ``"best_aspect"``: label = the aspect with the highest cosine to the
      surface (or ``""`` if no aspects supplied).
    * ``"first_aspect"``: label = ``aspects[0]`` (or ``""``).
    * ``"domain_only"``: label = ``hopfield.domain``.

    Returns a small audit dict (label, mode, n_patterns_after) for inclusion
    on :class:`pce.types.CascadeState.audit`.
    """
    if not surface.strip():
        return {"written": False, "reason": "empty_surface"}
    surf_emb = embed.encode(surface)
    if surf_emb.ndim != 1:
        surf_emb = np.asarray(surf_emb).reshape(-1)
    label: str
    if label_strategy == "domain_only" or not aspects:
        label = hopfield.domain if label_strategy == "domain_only" else ""
    elif label_strategy == "first_aspect":
        label = str(aspects[0])
    else:  # best_aspect
        asp_embs = embed.encode(list(aspects))
        if asp_embs.ndim == 1:
            asp_embs = asp_embs[None, :]
        sims = np.asarray(asp_embs @ surf_emb, dtype=np.float32)
        label = str(aspects[int(np.argmax(sims))]) if sims.size else ""
    hopfield.write(surf_emb.astype(np.float32), label=label, mode=mode)
    return {
        "written": True,
        "label": label,
        "mode": mode,
        "n_patterns_after": int(hopfield.n_patterns),
        "domain": hopfield.domain,
    }
