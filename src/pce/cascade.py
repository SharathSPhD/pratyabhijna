"""End-to-end Pratyabhijñā cascade orchestrator.

`run_cascade(prompt, constraint, ...)` executes the full
`cit -> icchā -> apohana -> ananda -> jñāna -> kriyā -> vimarśa` pipeline and
returns a `CascadeState` capturing every intermediate.

The orchestrator is deliberately simple: each operator is invoked with concrete
substrate handles passed in by the caller, and we audit-log everything to the
returned `CascadeState.audit` dictionary.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from pce.operators.ananda import ananda
from pce.operators.apohana import apohana
from pce.operators.iccha import iccha
from pce.operators.jnana import jnana
from pce.operators.kriya import kriya
from pce.operators.vimarsa import vimarsa
from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.types import CascadeState, Constraint


def _entropy_of(scores: np.ndarray) -> float:  # type: ignore[type-arg]
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = arr - arr.max()
    p = np.exp(arr)
    p /= p.sum() + 1e-30
    p = np.clip(p, 1e-30, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_cascade(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LocalLM,
    embed: Embedder,
    K: int = 8,
    cit_temperature: float = 1.0,
    max_tokens: int = 64,
    base_seed: int = 0,
    retrieval_set: list[str] | None = None,
    aspects: list[str] | None = None,
    render_mode: str = "verbatim",
    claude_renderer: Callable[[str], str] | None = None,
    lambda_a: float = 2.0,
    lambda_p: float = 2.0,
) -> CascadeState:
    if not prompt.strip():
        raise ValueError("run_cascade: prompt must be non-empty")
    t0 = time.time()
    candidates = iccha(
        prompt,
        constraint,
        lm=lm,
        K=K,
        base_seed=base_seed,
        max_tokens=max_tokens,
    )
    apoha = apohana(candidates, constraint, embed=embed)
    ananda_scores = np.array(
        [ananda(c, constraint=constraint, embed=embed) for c in candidates],
        dtype=np.float32,
    )
    sel_idx, delta_F, posterior = jnana(
        candidates, apoha, ananda_scores, lambda_a=lambda_a, lambda_p=lambda_p
    )
    selected = candidates[sel_idx]
    surface = kriya(
        selected,
        render_mode=render_mode,  # type: ignore[arg-type]
        lm=lm if render_mode == "polish" else None,
        embed=embed if render_mode == "polish" else None,
        claude_renderer=claude_renderer,
    )
    # vimarśa.
    e_iccha = _entropy_of(ananda_scores)
    e_apoha = _entropy_of(apoha)
    trajectory = [(e_iccha, e_apoha)]
    event, novelty, vim_diag = vimarsa(
        prompt,
        surface,
        embed=embed,
        retrieval_set=list(retrieval_set or []),
        aspects=list(aspects or []),
        ananda_score=float(ananda_scores[sel_idx]),
        iccha_apoha_trajectory=trajectory,
    )
    state = CascadeState(
        prompt=prompt,
        constraint=constraint,
        cit_temperature=float(cit_temperature),
        candidates=candidates,
        posterior=posterior,
        selected=selected,
        surface=surface,
        vimarsa_event=event,
        vimarsa_novelty=novelty,
        aspects=tuple(aspects or []),
        audit={
            "elapsed_s": float(time.time() - t0),
            "delta_F": float(delta_F),
            "selected_idx": int(sel_idx),
            "ananda_scores": [float(s) for s in ananda_scores.tolist()],
            "apoha_scores": [float(s) for s in apoha.tolist()],
            "vimarsa_diag": vim_diag,
            "entropy_iccha": e_iccha,
            "entropy_apoha": e_apoha,
        },
    )
    return state
