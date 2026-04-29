"""End-to-end Pratyabhijñā cascade orchestrator.

``run_cascade(prompt, constraint, ...)`` executes the full
``cit -> icchā -> apohana -> jñāna -> kriyā -> vimarśa -> kriyā(revision)``
pipeline and returns a :class:`CascadeState` capturing every intermediate.

v0.3 (ADR-002 / ADR-003 / ADR-004 / ADR-005) makes the cascade a real
inference loop:

1. **Always** run the draft pass (P1): ``iccha -> apohana -> jnana -> kriya``.
2. **Always** run vimarsa with ``return_brief=True`` and feed
   ``delta_F_draft`` as the ΔF evidence; vimarsa's event additionally requires
   ``delta_F >= delta_F_threshold`` (default 0.05).
3. **Always** run the shadow revision pass (P2) when ``commit_policy`` is
   ``"event_gated"`` or ``"always_revise"``, regardless of whether vimarsa
   fired -- so H8 (revision-vs-draft contribution) is *always* measurable
   on every paired item.
4. **Commit** according to ``commit_policy``:

   * ``"event_gated"`` (v0.3 default): commit revision iff vimarsa fired,
     otherwise commit draft.
   * ``"always_revise"`` (v0.2 behaviour): commit revision unconditionally.
   * ``"always_draft"`` (explicit ablation, formerly ``bypass_vimarsa=True``):
     commit draft unconditionally; revision pass is skipped to save cost.

   ``state.committed`` records ``"revision"`` or ``"draft"``; the audit
   records ``commit_policy``, ``vimarsa_event``, ``delta_F_draft``,
   ``revision_differs_from_draft``, and the per-item free-energy ledger.

Per ADR-005 ``iccha`` is invoked with ``prompt_mode="verbatim"`` and
``sampler_grid_mode="parity"`` so the bare-vs-cascade contrast is purely
architectural rather than confounded by prompt or sampler drift. Per
ADR-002 ``apohana`` is invoked with ``normalize=True`` so ``jnana`` sees
the shifted apoha that lets must-avoid penalties affect the posterior. Per
ADR-003 ``jnana`` runs in ``aspect_conditioned`` mode whenever the caller
supplies aspects, so the BMR posterior reflects which candidates actually
realize the must-have aspects. Per ADR-004 the cascade can thread an
optional :class:`pce.active_inference.HopfieldStore` so apohana gets a
warm-start prior and vimarsa.consolidate writes the committed surface
back at the end.

The substrate is :class:`pce.substrate.lm_protocol.LMProtocol` so the
cascade can run against either ``LocalLM`` (Qwen2-1.5B) or ``HaikuLM``
(Anthropic Haiku via ``claude`` CLI). Both substrates honor ``seed`` so
candidate diversity in parity mode is purely seed-driven.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt

from pce.active_inference.budget import FreeEnergyBudget
from pce.active_inference.hopfield import HopfieldStore
from pce.operators.ananda import ananda
from pce.operators.apohana import apohana
from pce.operators.iccha import iccha
from pce.operators.jnana import jnana
from pce.operators.kriya import kriya
from pce.operators.vimarsa import (
    DEFAULT_ASPECT_COSINE_HIT,
    DEFAULT_DELTA_F_THRESHOLD,
    consolidate,
    vimarsa,
)
from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.substrate.lm_protocol import LMProtocol
from pce.types import Candidate, CascadeState, Constraint

# Per-pass seed offset so the revision pass draws from a distinct sampler
# subspace from the draft pass while still being deterministic from
# ``base_seed``.
_REVISION_SEED_OFFSET = 17

CommitPolicy = Literal["event_gated", "always_revise", "always_draft"]


def _entropy_of(scores: np.ndarray) -> float:  # type: ignore[type-arg]
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr = arr - arr.max()
    p = np.exp(arr)
    p /= p.sum() + 1e-30
    p = np.clip(p, 1e-30, 1.0)
    return float(-np.sum(p * np.log(p)))


def _per_candidate_apoha_iccha_trajectory(
    apoha: npt.NDArray[np.float32],
    ananda: npt.NDArray[np.float32],
) -> list[tuple[float, float]]:
    """v0.4 (FR-4.v4 / ADR-005 in PRD): per-candidate (e_iccha, e_apoha) trajectory.

    `vimarsa.switching_ok` was always trivially True in v0.3 because the
    cascade passed `iccha_apoha_trajectory=None`. v0.4 supplies a real
    trajectory: each candidate contributes one point
    ``(softplus(ananda_i), softplus(apoha_i))``. The trajectory has length K
    so vimarsa's switching gate is exercised on the same evidence the
    cascade just consumed.

    Using softplus rather than raw scores keeps both axes positive (so the
    ratio test in :func:`vimarsa._count_switching` is well-defined) without
    losing relative ordering between candidates.
    """
    a = np.asarray(ananda, dtype=np.float64)
    b = np.asarray(apoha, dtype=np.float64)
    n = int(min(a.size, b.size))
    out: list[tuple[float, float]] = []
    for i in range(n):
        e_i = float(np.log1p(np.exp(np.clip(a[i], -50.0, 50.0))))
        e_a = float(np.log1p(np.exp(np.clip(b[i], -50.0, 50.0))))
        out.append((e_i, e_a))
    return out


def _aspect_membership_matrix(
    candidates: tuple[Candidate, ...],
    aspects: list[str],
    *,
    embed: Embedder,
) -> npt.NDArray[np.float32]:
    """Cosine matrix between K candidate embeddings and A aspect embeddings."""
    if not aspects:
        return np.zeros((len(candidates), 0), dtype=np.float32)
    asp_embs = embed.encode(list(aspects))
    if asp_embs.ndim == 1:
        asp_embs = asp_embs[None, :]
    cand_emb = np.stack([c.embedding for c in candidates], axis=0).astype(np.float32)
    return (cand_emb @ asp_embs.T).astype(np.float32)


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
    cit_temperature: float,
    aspects: list[str],
    aspect_priors: npt.NDArray[np.float32] | None,
    hopfield: HopfieldStore | None,
    hopfield_weight: float,
) -> tuple[
    tuple[Candidate, ...],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    int,
    float,
    npt.NDArray[np.float32],
    str,
    npt.NDArray[np.float32],
]:
    """Run one cascade pass; returns (cands, apoha, ananda, sel, dF, post, surface, aspect_mem)."""
    candidates = iccha(
        prompt,
        constraint,
        lm=lm,
        K=K,
        base_seed=base_seed,
        max_tokens=max_tokens,
        prompt_mode="verbatim",
        sampler_grid_mode="parity",
        cit_temperature=cit_temperature,
    )
    apoha = apohana(
        candidates,
        constraint,
        embed=embed,
        normalize=True,
        hopfield=hopfield,
        hopfield_weight=hopfield_weight,
    )
    ananda_scores = np.array(
        [ananda(c, constraint=constraint, embed=embed) for c in candidates],
        dtype=np.float32,
    )
    aspect_membership = _aspect_membership_matrix(candidates, aspects, embed=embed)
    if aspects:
        sel_idx, delta_F, posterior = jnana(
            candidates,
            apoha,
            ananda_scores,
            reduction_target="aspect_conditioned",
            aspect_membership=aspect_membership,
            aspect_priors=aspect_priors,
            lambda_a=lambda_a,
            lambda_p=lambda_p,
        )
    else:
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
    return (
        candidates,
        apoha,
        ananda_scores,
        sel_idx,
        delta_F,
        posterior,
        surface,
        aspect_membership,
    )


def _resolve_commit_policy(
    commit_policy: CommitPolicy | None, bypass_vimarsa: bool
) -> CommitPolicy:
    """Resolve the v0.3 ``commit_policy`` from kwargs.

    ``bypass_vimarsa=True`` is the v0.2 deprecated alias for
    ``commit_policy="always_draft"``; if both are set with conflicting
    values we raise so the caller resolves the ambiguity rather than
    silently picking one.
    """
    if commit_policy is None:
        return "always_draft" if bypass_vimarsa else "event_gated"
    if bypass_vimarsa and commit_policy != "always_draft":
        raise ValueError(
            f"run_cascade: bypass_vimarsa=True conflicts with commit_policy={commit_policy!r}; "
            "drop bypass_vimarsa or set commit_policy='always_draft'"
        )
    return commit_policy


def run_cascade(
    prompt: str,
    constraint: Constraint,
    *,
    lm: LMProtocol,
    embed: Embedder,
    K: int = 4,
    cit_temperature: float = 1.0,
    max_tokens: int = 200,
    base_seed: int = 0,
    retrieval_set: list[str] | None = None,
    aspects: list[str] | None = None,
    render_mode: str = "verbatim",
    claude_renderer: Callable[[str], str] | None = None,
    lambda_a: float = 2.0,
    lambda_p: float = 2.0,
    bypass_vimarsa: bool = False,
    commit_policy: CommitPolicy | None = None,
    delta_F_threshold: float = DEFAULT_DELTA_F_THRESHOLD,
    aspect_cosine_hit: float = DEFAULT_ASPECT_COSINE_HIT,
    polish_lm: LocalLM | None = None,
    hopfield: HopfieldStore | None = None,
    hopfield_weight: float = 0.25,
    budget: FreeEnergyBudget | None = None,
    brief_override: str | None = None,
) -> CascadeState:
    """Run the v0.3 event-gated, always-shadow-revision cascade on one prompt.

    ``commit_policy``:

    * ``"event_gated"`` (default): commit revision iff vimarsa fires;
      otherwise commit draft. Always runs both passes so H8 is measurable.
    * ``"always_revise"`` (v0.2 behaviour): commit revision unconditionally.
    * ``"always_draft"``: skip the revision pass; commit draft. Equivalent
      to the v0.2 ``bypass_vimarsa=True`` ablation.

    ``brief_override``: when set, replaces the vimarsa-generated revision
    brief with this fixed string. This is the lever the
    ``haiku_generic_revise_2pass`` benchmark control arm uses to isolate
    the *content* of the brief from the *existence* of a revision pass.
    Default ``None`` -> use the vimarsa brief.

    ``bypass_vimarsa=True`` is kept as a deprecated alias for
    ``commit_policy="always_draft"`` so the v0.2 prove-gate scripts and
    benchmark drivers keep working until they are migrated.
    """
    if not prompt.strip():
        raise ValueError("run_cascade: prompt must be non-empty")
    policy = _resolve_commit_policy(commit_policy, bypass_vimarsa)
    t0 = time.time()
    aspects_list = list(aspects or [])
    retrieval_list = list(retrieval_set or [])
    ledger = budget if budget is not None else FreeEnergyBudget()

    # Per-aspect prior from the storehouse, when available. Falls back to
    # uniform when the store is empty or no aspects are supplied.
    aspect_priors: npt.NDArray[np.float32] | None = None
    storehouse_aspect_attention: list[float] = []
    if hopfield is not None and aspects_list and hopfield.n_patterns > 0:
        # Query the storehouse with the constraint embedding to get a
        # per-aspect mass vector.
        res = hopfield.query(constraint.embedding, aspect_labels=aspects_list)
        storehouse_aspect_attention = [float(p) for p in res.aspect_priors.tolist()]
        if float(res.aspect_priors.sum()) > 0.0:
            aspect_priors = res.aspect_priors

    # ---- Pass 1: draft ---------------------------------------------------
    cands_d, apoha_d, anan_d, sel_d, dF_d, post_d, draft, asp_mem_d = _one_pass(
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
        cit_temperature=cit_temperature,
        aspects=aspects_list,
        aspect_priors=aspect_priors,
        hopfield=hopfield,
        hopfield_weight=hopfield_weight,
    )
    e_iccha_d = _entropy_of(anan_d)
    e_apoha_d = _entropy_of(apoha_d)
    ledger.earn_jnana(dF_d, note="draft pass")
    ledger.earn_tokens(len(cands_d[sel_d].text.split()), note="draft tokens")

    # ---- vimarsa (always run; ΔF-gated event) ---------------------------
    # v0.4 (FR-4.v4): supply a real per-candidate trajectory so vimarsa's
    # switching gate is exercised. v0.3 always passed None, which made
    # ``switching_ok`` trivially True for every cascade item.
    apoha_iccha_trajectory_d = _per_candidate_apoha_iccha_trajectory(apoha_d, anan_d)
    vim_out_d = vimarsa(
        prompt,
        draft,
        embed=embed,
        retrieval_set=retrieval_list,
        aspects=aspects_list,
        ananda_score=float(anan_d[sel_d]),
        iccha_apoha_trajectory=apoha_iccha_trajectory_d,
        delta_F=float(dF_d),
        delta_F_threshold=float(delta_F_threshold),
        aspect_cosine_hit=float(aspect_cosine_hit),
        return_brief=True,
    )
    assert len(vim_out_d) == 4
    event_d, novelty_d, diag_d, brief = vim_out_d

    # Aspect cost: 1 - max aspect cosine on the draft (closer = cheaper).
    if aspects_list and asp_mem_d.size:
        max_asp = float(asp_mem_d[sel_d].max())
        ledger.earn_aspect(max(0.0, 1.0 - max_asp), note="draft aspect distance")

    # ---- v0.4 (ADR-003): FE budget hard gate before shadow revision pass.
    # The cascade still always commits the draft when the budget is
    # underwater; ``commit_policy`` never overrides a budget abort. The two
    # tier hierarchy is:
    #     1. budget gate (generation-level) -> generate revision at all?
    #     2. commit policy (selection-level) -> commit revision over draft?
    fe_budget_underwater = not ledger.should_continue_revision()

    def _draft_only_state(reason: str) -> CascadeState:
        return CascadeState(
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
            committed="draft",
            commit_policy=policy,
            audit={
                "elapsed_s": float(time.time() - t0),
                "delta_F_draft": float(dF_d),
                "delta_F": float(dF_d),
                "selected_idx_draft": int(sel_d),
                "selected_idx": int(sel_d),
                "ananda_scores_draft": [float(s) for s in anan_d.tolist()],
                "ananda_scores": [float(s) for s in anan_d.tolist()],
                "apoha_scores_draft": [float(s) for s in apoha_d.tolist()],
                "apoha_scores": [float(s) for s in apoha_d.tolist()],
                "vimarsa_diag_draft": diag_d,
                "vimarsa_diag": diag_d,
                "entropy_iccha_draft": e_iccha_d,
                "entropy_apoha_draft": e_apoha_d,
                "two_pass": False,
                "bypassed": reason == "commit_policy=always_draft",
                "revision_skipped": True,
                "revision_skipped_reason": reason,
                "fe_budget_underwater": bool(fe_budget_underwater),
                "budget_ledger": ledger.to_audit(),
                "storehouse_aspect_attention": storehouse_aspect_attention,
                "n_storehouse_patterns": int(hopfield.n_patterns) if hopfield else 0,
            },
        )

    if policy == "always_draft":
        return _draft_only_state("commit_policy=always_draft")
    if fe_budget_underwater:
        return _draft_only_state("fe_budget_underwater")

    effective_brief = brief_override if brief_override is not None else brief
    revision_prompt = (
        f"{prompt.rstrip()}\n\n"
        f"Reviser brief: {effective_brief}\n\n"
        f"Previous draft:\n{draft.strip()}\n\n"
        "Now produce the revised response."
    )
    cands_r, apoha_r, anan_r, sel_r, dF_r, post_r, revision, asp_mem_r = _one_pass(
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
        cit_temperature=cit_temperature,
        aspects=aspects_list,
        aspect_priors=aspect_priors,
        hopfield=hopfield,
        hopfield_weight=hopfield_weight,
    )
    e_iccha_r = _entropy_of(anan_r)
    e_apoha_r = _entropy_of(apoha_r)
    ledger.earn_jnana(dF_r, note="revision pass")
    ledger.earn_tokens(len(cands_r[sel_r].text.split()), note="revision tokens")
    if aspects_list and asp_mem_r.size:
        max_asp_r = float(asp_mem_r[sel_r].max())
        ledger.earn_aspect(max(0.0, 1.0 - max_asp_r), note="revision aspect distance")

    apoha_iccha_trajectory_r = _per_candidate_apoha_iccha_trajectory(apoha_r, anan_r)
    vim_out_r = vimarsa(
        prompt,
        revision,
        embed=embed,
        retrieval_set=retrieval_list,
        aspects=aspects_list,
        ananda_score=float(anan_r[sel_r]),
        iccha_apoha_trajectory=apoha_iccha_trajectory_r,
        delta_F=float(dF_r),
        delta_F_threshold=float(delta_F_threshold),
        aspect_cosine_hit=float(aspect_cosine_hit),
        return_brief=False,
    )
    assert len(vim_out_r) == 3
    event_r, novelty_r, diag_r = vim_out_r

    # ---- Commit policy --------------------------------------------------
    if policy == "always_revise":
        committed = "revision"
    elif policy == "event_gated":
        committed = "revision" if bool(event_d) else "draft"
    else:  # always_draft handled above; keep mypy exhaustive
        raise ValueError(f"run_cascade: unreachable commit_policy={policy!r}")
    final_surface = revision if committed == "revision" else draft

    # ---- Storehouse consolidation ---------------------------------------
    consolidate_audit: dict[str, object] | None = None
    if hopfield is not None:
        consolidate_audit = consolidate(
            surface=final_surface,
            aspects=aspects_list,
            embed=embed,
            hopfield=hopfield,
            mode="rem",
        )

    selected_for_state = cands_r[sel_r] if committed == "revision" else cands_d[sel_d]
    posterior_for_state = post_r if committed == "revision" else post_d
    candidates_for_state = cands_r if committed == "revision" else cands_d
    state = CascadeState(
        prompt=prompt,
        constraint=constraint,
        cit_temperature=float(cit_temperature),
        candidates=candidates_for_state,
        posterior=posterior_for_state,
        selected=selected_for_state,
        surface=final_surface,
        vimarsa_event=bool(event_d),  # event from draft drives the commit decision
        vimarsa_novelty=float(novelty_r),
        aspects=tuple(aspects_list),
        surface_draft=draft,
        surface_revision=revision,
        vimarsa_event_draft=bool(event_d),
        vimarsa_brief=brief,
        committed=committed,
        commit_policy=policy,
        audit={
            "elapsed_s": float(time.time() - t0),
            "two_pass": True,
            "bypassed": False,
            "revision_skipped": False,
            "delta_F_draft": float(dF_d),
            "delta_F_revision": float(dF_r),
            "delta_F": float(dF_r if committed == "revision" else dF_d),
            "selected_idx_draft": int(sel_d),
            "selected_idx_revision": int(sel_r),
            "selected_idx": int(sel_r if committed == "revision" else sel_d),
            "ananda_scores_draft": [float(s) for s in anan_d.tolist()],
            "ananda_scores_revision": [float(s) for s in anan_r.tolist()],
            "ananda_scores": [
                float(s) for s in (anan_r if committed == "revision" else anan_d).tolist()
            ],
            "apoha_scores_draft": [float(s) for s in apoha_d.tolist()],
            "apoha_scores_revision": [float(s) for s in apoha_r.tolist()],
            "apoha_scores": [
                float(s) for s in (apoha_r if committed == "revision" else apoha_d).tolist()
            ],
            "vimarsa_diag_draft": diag_d,
            "vimarsa_diag_revision": diag_r,
            "vimarsa_diag": diag_r if committed == "revision" else diag_d,
            "vimarsa_brief": brief,
            "vimarsa_brief_effective": effective_brief,
            "brief_override_used": bool(brief_override is not None),
            "vimarsa_event_draft": bool(event_d),
            "vimarsa_event_revision": bool(event_r),
            "entropy_iccha_draft": e_iccha_d,
            "entropy_apoha_draft": e_apoha_d,
            "entropy_iccha": e_iccha_r,
            "entropy_apoha": e_apoha_r,
            "revision_differs_from_draft": bool(revision.strip() != draft.strip()),
            "delta_F_threshold": float(delta_F_threshold),
            "fe_budget_underwater": False,
            "budget_ledger": ledger.to_audit(),
            "storehouse_aspect_attention": storehouse_aspect_attention,
            "n_storehouse_patterns": int(hopfield.n_patterns) if hopfield else 0,
            "storehouse_consolidate": consolidate_audit,
        },
    )
    return state
