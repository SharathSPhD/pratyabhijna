"""End-to-end Pratyabhijñā cascade orchestrator (v0.2: two-pass-always).

``run_cascade(prompt, constraint, ...)`` executes the full
``cit -> icchā -> apohana -> jñāna -> kriyā -> vimarśa -> kriyā(revision)``
pipeline and returns a :class:`CascadeState` capturing every intermediate.

v0.2 (ADR-003) makes ``vimarśa`` causal:

#. Pass 1 (draft) runs ``iccha -> apohana -> jnana -> kriya`` and emits a
   draft surface.
#. ``vimarsa`` is invoked with ``return_brief=True`` to produce a structured
   revision brief listing missing aspects (or a generic creative brief when
   the domain has no aspect dictionary).
#. Pass 2 (revision) runs ``iccha`` again with the brief appended to the
   prompt, then re-runs ``apohana -> jnana -> kriya`` to emit the final
   surface. ``state.surface = revision``; the draft is preserved on
   ``state.surface_draft`` for the H8.v2 contribution test.

``bypass_vimarsa=True`` collapses the cascade to a single pass and returns
the draft as ``state.surface`` - the explicit ablation control. Default is
``False`` (two-pass-always).

Per ADR-005 ``iccha`` is invoked with ``prompt_mode="verbatim"`` and
``sampler_grid_mode="parity"`` so the bare-vs-cascade contrast is purely
architectural rather than confounded by prompt or sampler drift. Per
ADR-002 ``apohana`` is invoked with ``normalize=True`` so ``jnana`` sees
the shifted apoha that lets must-avoid penalties affect the posterior.

The substrate is :class:`pce.substrate.lm_protocol.LMProtocol` so the
cascade can run against either ``LocalLM`` (Qwen2-1.5B) or ``HaikuLM``
(Anthropic Haiku via ``claude`` CLI). Both substrates honor ``seed`` so
candidate diversity in parity mode is purely seed-driven.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from pce.operators.ananda import ananda
from pce.operators.apohana import apohana
from pce.operators.iccha import iccha
from pce.operators.jnana import jnana
from pce.operators.kriya import kriya
from pce.operators.vimarsa import vimarsa
from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, CascadeState, Constraint

# Per-pass seed offset so the revision pass draws from a distinct sampler
# subspace from the draft pass while still being deterministic from
# ``base_seed``.
_REVISION_SEED_OFFSET = 17


def _entropy_of(scores: np.ndarray) -> float:  # type: ignore[type-arg]
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = arr - arr.max()
    p = np.exp(arr)
    p /= p.sum() + 1e-30
    p = np.clip(p, 1e-30, 1.0)
    return float(-np.sum(p * np.log(p)))


def _one_pass(
    *,
    prompt: str,
    constraint: Constraint,
    lm: LMProtocol,
    embed: Embedder,
    K: int,
    max_tokens: int,
    base_seed: int,
    render_mode: str,
    polish_lm: LocalLM | None,
    claude_renderer: Callable[[str], str] | None,
    lambda_a: float,
    lambda_p: float,
) -> tuple[
    tuple[Candidate, ...],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    int,
    float,
    npt.NDArray[np.float32],
    str,
]:
    """Run one cascade pass; returns (cands, apoha, ananda, sel, dF, post, surface)."""
    candidates = iccha(
        prompt,
        constraint,
        lm=lm,
        K=K,
        base_seed=base_seed,
        max_tokens=max_tokens,
        prompt_mode="verbatim",
        sampler_grid_mode="parity",
    )
    apoha = apohana(candidates, constraint, embed=embed, normalize=True)
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
        lm=polish_lm if render_mode == "polish" else None,
        embed=embed if render_mode == "polish" else None,
        claude_renderer=claude_renderer,
    )
    return candidates, apoha, ananda_scores, sel_idx, delta_F, posterior, surface


def run_cascade(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LMProtocol,
    embed: Embedder,
    K: int = 4,
    cit_temperature: float = 0.9,
    max_tokens: int = 200,
    base_seed: int = 0,
    retrieval_set: list[str] | None = None,
    aspects: list[str] | None = None,
    render_mode: str = "verbatim",
    claude_renderer: Callable[[str], str] | None = None,
    lambda_a: float = 2.0,
    lambda_p: float = 2.0,
    bypass_vimarsa: bool = False,
    polish_lm: LocalLM | None = None,
) -> CascadeState:
    """Run the two-pass-always cascade on a single prompt.

    Returns the revision in ``state.surface`` (the cascade's output for
    benchmarking). When ``bypass_vimarsa=True`` the cascade collapses to a
    single pass and ``state.surface == state.surface_draft``.
    """
    if not prompt.strip():
        raise ValueError("run_cascade: prompt must be non-empty")
    t0 = time.time()
    aspects_list = list(aspects or [])
    retrieval_list = list(retrieval_set or [])

    cands_d, apoha_d, anan_d, sel_d, dF_d, post_d, draft = _one_pass(
        prompt=prompt,
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=K,
        max_tokens=max_tokens,
        base_seed=base_seed,
        render_mode=render_mode,
        polish_lm=polish_lm,
        claude_renderer=claude_renderer,
        lambda_a=lambda_a,
        lambda_p=lambda_p,
    )
    e_iccha_d = _entropy_of(anan_d)
    e_apoha_d = _entropy_of(apoha_d)
    vim_out_d = vimarsa(
        prompt,
        draft,
        embed=embed,
        retrieval_set=retrieval_list,
        aspects=aspects_list,
        ananda_score=float(anan_d[sel_d]),
        iccha_apoha_trajectory=None,
        return_brief=True,
    )
    assert len(vim_out_d) == 4
    event_d, novelty_d, diag_d, brief = vim_out_d

    if bypass_vimarsa:
        state = CascadeState(
            prompt=prompt,
            constraint=constraint,
            cit_temperature=float(cit_temperature),
            candidates=cands_d,
            posterior=post_d,
            selected=cands_d[sel_d],
            surface=draft,
            vimarsa_event=bool(event_d),
            vimarsa_novelty=float(novelty_d),
            aspects=tuple(aspects_list),
            surface_draft=draft,
            surface_revision=None,
            vimarsa_event_draft=bool(event_d),
            vimarsa_brief=brief,
            audit={
                "elapsed_s": float(time.time() - t0),
                "delta_F_draft": float(dF_d),
                "selected_idx_draft": int(sel_d),
                "ananda_scores_draft": [float(s) for s in anan_d.tolist()],
                "apoha_scores_draft": [float(s) for s in apoha_d.tolist()],
                "vimarsa_diag_draft": diag_d,
                "entropy_iccha_draft": e_iccha_d,
                "entropy_apoha_draft": e_apoha_d,
                "two_pass": False,
                "bypassed": True,
            },
        )
        return state

    revision_prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Reviser brief: {brief}\n\n"
        f"Previous draft:\n{draft.strip()}\n\n"
        "Now produce the revised response."
    )
    cands_r, apoha_r, anan_r, sel_r, dF_r, post_r, revision = _one_pass(
        prompt=revision_prompt,
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=K,
        max_tokens=max_tokens,
        base_seed=base_seed + _REVISION_SEED_OFFSET,
        render_mode=render_mode,
        polish_lm=polish_lm,
        claude_renderer=claude_renderer,
        lambda_a=lambda_a,
        lambda_p=lambda_p,
    )
    e_iccha_r = _entropy_of(anan_r)
    e_apoha_r = _entropy_of(apoha_r)
    vim_out_r = vimarsa(
        prompt,
        revision,
        embed=embed,
        retrieval_set=retrieval_list,
        aspects=aspects_list,
        ananda_score=float(anan_r[sel_r]),
        iccha_apoha_trajectory=None,
        return_brief=False,
    )
    assert len(vim_out_r) == 3
    event_r, novelty_r, diag_r = vim_out_r

    state = CascadeState(
        prompt=prompt,
        constraint=constraint,
        cit_temperature=float(cit_temperature),
        candidates=cands_r,
        posterior=post_r,
        selected=cands_r[sel_r],
        surface=revision,
        vimarsa_event=bool(event_r),
        vimarsa_novelty=float(novelty_r),
        aspects=tuple(aspects_list),
        surface_draft=draft,
        surface_revision=revision,
        vimarsa_event_draft=bool(event_d),
        vimarsa_brief=brief,
        audit={
            "elapsed_s": float(time.time() - t0),
            "two_pass": True,
            "bypassed": False,
            "delta_F_draft": float(dF_d),
            "delta_F_revision": float(dF_r),
            "delta_F": float(dF_r),
            "selected_idx_draft": int(sel_d),
            "selected_idx_revision": int(sel_r),
            "selected_idx": int(sel_r),
            "ananda_scores_draft": [float(s) for s in anan_d.tolist()],
            "ananda_scores_revision": [float(s) for s in anan_r.tolist()],
            "ananda_scores": [float(s) for s in anan_r.tolist()],
            "apoha_scores_draft": [float(s) for s in apoha_d.tolist()],
            "apoha_scores_revision": [float(s) for s in apoha_r.tolist()],
            "apoha_scores": [float(s) for s in apoha_r.tolist()],
            "vimarsa_diag_draft": diag_d,
            "vimarsa_diag_revision": diag_r,
            "vimarsa_diag": diag_r,
            "vimarsa_brief": brief,
            "entropy_iccha_draft": e_iccha_d,
            "entropy_apoha_draft": e_apoha_d,
            "entropy_iccha": e_iccha_r,
            "entropy_apoha": e_apoha_r,
            "revision_differs_from_draft": bool(revision.strip() != draft.strip()),
        },
    )
    return state
