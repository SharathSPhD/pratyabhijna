"""Pratyabhijna Creative Engine - MCP server (v0.3).

Exposes the seven cascade operators, the cascade orchestrator, consolidation
routines, and a handful of audit/diagnostic utilities. v0.3 (ADR-001 .. ADR-005)
adds:

* a clean Haiku CLI substrate (subprocess flag + scrubbed HOME isolation;
  see ``haiku_clean_substrate_probe`` for the live integrity probe);
* event-gated, always-shadow-revision cascade (``pce_cascade`` now takes
  ``commit_policy`` and ``cit_temperature``);
* two new control arms in ``_resolve_arm`` -- ``haiku_bare_2K`` and
  ``haiku_generic_revise`` -- for the four-arm benchmark fairness contrast;
* a per-cascade Hopfield ālayavijñāna with introspection via the new
  ``hopfield_state`` tool (the ``HopfieldStore`` from
  ``pce.active_inference.hopfield``, distinct from the legacy
  v0.1 store under ``pce.substrate.hopfield``).

All v0.1 / v0.2 tool names are preserved for backward compatibility.

Substrate singletons (LocalLM, HaikuLM, Embedder, HopfieldStore) are held as
process-level state so each model is loaded only once per session.

Env vars:

* ``PCE_LM_MODEL``   - HF id for the local LM (default ``Qwen/Qwen2-1.5B-Instruct``).
* ``PCE_LM_DEVICE``  - ``cpu``/``cuda``/``mps`` override; default autodetect.
* ``PCE_LM_DTYPE``   - ``float32``/``float16`` override; default autodetect.
* ``PCE_HAIKU_CLI``  - path to the ``claude`` binary (default ``claude``).
* ``PCE_HAIKU_MODEL``- Haiku model alias (default ``haiku``).
* ``PCE_HAIKU_COST_CAP_USD`` - per-process budget cap for Haiku calls.
* ``PCE_USE_SDK=1``  - prefer the Anthropic SDK path when ``ANTHROPIC_API_KEY``
  is set; otherwise the CLI is used.

Run via::

    uv run python plugin/mcp/server.py

Or via Claude Code's plugin loader (see plugin/.mcp.json).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from pce.active_inference.hopfield import HopfieldStore as ActiveHopfieldStore  # noqa: E402
from pce.cascade import run_cascade  # noqa: E402
from pce.consolidation.sleep import (  # noqa: E402
    is_consolidated,
    run_rem,
    run_sleep_cycle,
    run_sws,
)
from pce.operators.ananda import ananda as ananda_op  # noqa: E402
from pce.operators.apohana import apohana as apohana_op  # noqa: E402
from pce.operators.cit import cit as cit_op  # noqa: E402
from pce.operators.iccha import iccha as iccha_op  # noqa: E402
from pce.operators.jnana import jnana as jnana_op  # noqa: E402
from pce.operators.kriya import kriya as kriya_op  # noqa: E402
from pce.operators.vimarsa import vimarsa as vimarsa_op  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM  # noqa: E402
from pce.substrate.hopfield import HopfieldStore  # noqa: E402
from pce.substrate.integrity import IntegrityProbe  # noqa: E402
from pce.substrate.lm import LMConfig, LocalLM  # noqa: E402
from pce.substrate.lm_protocol import LMProtocol  # noqa: E402
from pce.types import Constraint  # noqa: E402

mcp = FastMCP("pratyabhijna")

_LM: LocalLM | None = None
_HAIKU: HaikuLM | None = None
_EMBED: Embedder | None = None
_HOPFIELD: HopfieldStore | None = None
_ACTIVE_HOPFIELD: ActiveHopfieldStore | None = None
_INTEGRITY_PROBE: IntegrityProbe | None = None
_AUDIT_PATH = REPO_ROOT / "audit" / "phase8" / "mcp_calls.jsonl"

GENERIC_REVISE_BRIEF = (
    "Revise the previous draft to be more creative, specific, and surprising. "
    "Add concrete sensory detail, remove cliches, and strengthen the most "
    "interesting move you can find."
)


def _get_lm() -> LocalLM:
    """Local Qwen2-1.5B substrate. Device/dtype autodetect by default; users
    can pin via ``PCE_LM_DEVICE`` and ``PCE_LM_DTYPE`` env vars (or the
    short ``PCE_DEVICE`` / ``PCE_DTYPE`` aliases).
    """
    global _LM
    if _LM is None:
        cfg_kwargs: dict[str, Any] = {
            "model_id": os.environ.get("PCE_LM_MODEL", "Qwen/Qwen2-1.5B-Instruct"),
        }
        # Only pin device/dtype when the user explicitly asks. The v0.1 plugin
        # hard-pinned cpu/float32 in .mcp.json which was wasteful on Apple
        # Silicon; v0.2 lets LocalLM autodetect (mps -> fp16 on Macs).
        dtype = os.environ.get("PCE_LM_DTYPE") or os.environ.get("PCE_DTYPE")
        device = os.environ.get("PCE_LM_DEVICE") or os.environ.get("PCE_DEVICE")
        if dtype:
            cfg_kwargs["dtype"] = dtype
        if device:
            cfg_kwargs["device"] = device
        _LM = LocalLM(LMConfig(**cfg_kwargs))
    return _LM


def _get_haiku() -> HaikuLM:
    """Haiku substrate (claude CLI / Anthropic SDK). Lazy-loaded."""
    global _HAIKU
    if _HAIKU is None:
        _HAIKU = HaikuLM(config=HaikuConfig.from_env(), embedder=_get_embed())
    return _HAIKU


def _resolve_arm(arm: str) -> tuple[str, LMProtocol]:
    """Map ``arm`` argument to a substrate. Accepts shortforms and long forms.

    ``"local"``, ``"local_bare"``, ``"local_cascade"`` -> LocalLM.
    ``"haiku"``, ``"haiku_bare"``, ``"haiku_cascade"``,
    ``"haiku_bare_2K"``, ``"haiku_generic_revise"`` -> HaikuLM (the cascade
    entry point branches on arm to wire the right ``commit_policy`` /
    K multiplier / brief override per ADR-002 / Phase 7 driver design).
    """
    a = (arm or "local").strip().lower()
    if a in {"local", "local_bare", "local_cascade", "qwen", "qwen2"}:
        return "local", _get_lm()
    if a in {
        "haiku",
        "haiku_bare",
        "haiku_cascade",
        "haiku_bare_2k",
        "haiku_bare_2k_scorer",
        "haiku_generic_revise",
        "haiku_generic_revise_2pass",
        "claude_haiku",
    }:
        return "haiku", _get_haiku()
    raise ValueError(
        f"unknown arm={arm!r}; expected one of: local, haiku, "
        "local_bare, local_cascade, haiku_bare, haiku_cascade, "
        "haiku_bare_2K, haiku_generic_revise"
    )


CommitPolicyLit = Literal[
    "event_gated", "always_revise", "always_draft", "learned_gate"
]


def _v3_arm_overrides(
    arm: str,
    *,
    K: int,
    commit_policy: str | None,
) -> tuple[int, CommitPolicyLit, str | None]:
    """Resolve (K, commit_policy, brief_override) for v0.3 control arms.

    The benchmark driver normally passes ``arm="haiku_cascade"`` and lets
    ``commit_policy`` default to ``"event_gated"``. The two new control
    arms force their own protocol so the driver does not need to know
    the wiring details.
    """
    a = (arm or "").strip().lower()
    if a in {"haiku_bare_2k", "haiku_bare_2k_scorer"}:
        return (max(1, K) * 2, "always_draft", None)
    if a in {"haiku_generic_revise", "haiku_generic_revise_2pass"}:
        return (K, "always_revise", GENERIC_REVISE_BRIEF)
    if a in {"haiku_bare"}:
        return (K, "always_draft", None)
    cp = commit_policy or "event_gated"
    if cp not in {"event_gated", "always_revise", "always_draft", "learned_gate"}:
        raise ValueError(
            f"_v3_arm_overrides: commit_policy must be one of "
            f"'event_gated'|'always_revise'|'always_draft'|'learned_gate'; got {cp!r}"
        )
    return (K, cp, None)  # type: ignore[return-value]


def _get_embed() -> Embedder:
    global _EMBED
    if _EMBED is None:
        _EMBED = Embedder()
    return _EMBED


def _get_hopfield() -> HopfieldStore:
    global _HOPFIELD
    if _HOPFIELD is None:
        _HOPFIELD = HopfieldStore(dim=_get_embed().dim)
    return _HOPFIELD


def _get_active_hopfield(_embed: Embedder) -> ActiveHopfieldStore:
    """Lazy-load the v0.3 active-inference Hopfield ālayavijñāna.

    Distinct from the legacy v0.1 ``_HOPFIELD`` (which lives under
    ``pce.substrate.hopfield`` and powers the v0.1 ``hopfield_store`` /
    ``hopfield_recall`` MCP tools). This is the storehouse the v0.3
    cascade reads (warm-start aspect prior in apohana) and writes
    (vimarsa.consolidate at commit time). The MCP-process store uses a
    single shared domain (``"mcp_session"``) so cross-prompt continuity
    works for any MCP caller; the benchmark driver creates per-domain
    stores instead.
    """
    global _ACTIVE_HOPFIELD
    if _ACTIVE_HOPFIELD is None:
        _ACTIVE_HOPFIELD = ActiveHopfieldStore(domain="mcp_session")
    return _ACTIVE_HOPFIELD


def _get_integrity_probe() -> IntegrityProbe:
    """Singleton :class:`IntegrityProbe` so cached per-(env,flags) results are reused."""
    global _INTEGRITY_PROBE
    if _INTEGRITY_PROBE is None:
        _INTEGRITY_PROBE = IntegrityProbe()
    return _INTEGRITY_PROBE


def _audit(tool: str, args: dict[str, Any], result_summary: dict[str, Any]) -> None:
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "tool": tool,
            "args": args,
            "result": result_summary,
        }, ensure_ascii=False) + "\n")


# Tool 1
@mcp.tool()
def cit(
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 64,
    top_p: float = 0.95,
    top_k: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample one continuation from the local LM (cit substrate)."""
    cand = cit_op(
        prompt,
        lm=_get_lm(),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    out = {"text": cand.text, "logp": cand.logp, "n_tokens": len(cand.tokens), "sampler": cand.sampler}
    _audit("cit", {"prompt": prompt[:200], "temperature": temperature, "seed": seed}, out)
    return out


# Tool 2
@mcp.tool()
def iccha(
    prompt: str,
    constraint_text: str,
    K: int = 6,
    base_seed: int = 0,
    max_tokens: int = 48,
) -> dict[str, Any]:
    """Generate K candidate continuations spanning explore-exploit."""
    embed = _get_embed()
    constraint = Constraint(text=constraint_text, embedding=embed.encode(constraint_text))
    cands = iccha_op(
        prompt,
        constraint,
        lm=_get_lm(),
        K=K,
        base_seed=base_seed,
        max_tokens=max_tokens,
    )
    out = {
        "candidates": [
            {"text": c.text, "logp": c.logp, "sampler": c.sampler} for c in cands
        ],
        "K": K,
    }
    _audit("iccha", {"prompt": prompt[:200], "K": K, "base_seed": base_seed}, {"K": K})
    return out


# Tool 3
@mcp.tool()
def apohana(
    candidate_texts: list[str],
    constraint_text: str,
    must_avoid: list[str] | None = None,
) -> dict[str, Any]:
    """Score candidates by contrastive exclusion against constraint and must_avoid set."""
    embed = _get_embed()
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid or ()),
    )
    from pce.types import Candidate

    cands = tuple(
        Candidate(
            seed=i,
            sampler={"tau": 1.0},
            tokens=(),
            text=t,
            logp=0.0,
            embedding=embed.encode(t),
        )
        for i, t in enumerate(candidate_texts)
    )
    scores = apohana_op(cands, constraint, embed=embed)
    out = {"scores": [float(s) for s in scores.tolist()]}
    _audit("apohana", {"n": len(candidate_texts)}, out)
    return out


# Tool 4
@mcp.tool()
def ananda(
    candidate_text: str,
    constraint_text: str,
    reward: float | None = None,
) -> dict[str, Any]:
    """Aesthetic-coherence score in [0, 1] for a single candidate against a constraint."""
    embed = _get_embed()
    constraint = Constraint(text=constraint_text, embedding=embed.encode(constraint_text))
    from pce.types import Candidate

    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(),
        text=candidate_text,
        logp=0.0,
        embedding=embed.encode(candidate_text),
    )
    score = ananda_op(cand, constraint=constraint, embed=embed, reward=reward)
    out = {"score": float(score)}
    _audit("ananda", {"text": candidate_text[:100]}, out)
    return out


# Tool 5
@mcp.tool()
def jnana(
    candidate_texts: list[str],
    apoha_scores: list[float],
    ananda_scores: list[float],
    reduction_target: str = "halve",
    lambda_a: float = 2.0,
    lambda_p: float = 2.0,
) -> dict[str, Any]:
    """Bayesian Model Reduction across the K candidates; returns selected_idx + posterior + ΔF."""
    if len(candidate_texts) != len(apoha_scores) or len(candidate_texts) != len(ananda_scores):
        raise ValueError("jnana: candidate_texts/apoha_scores/ananda_scores must have the same length")
    apoha = np.array(apoha_scores, dtype=np.float32)
    anan = np.array(ananda_scores, dtype=np.float32)
    cands: tuple[object, ...] = tuple(range(len(candidate_texts)))
    valid_targets = {"halve", "single", "custom", "aspect_conditioned"}
    if reduction_target not in valid_targets:
        raise ValueError(
            f"jnana: reduction_target must be one of {sorted(valid_targets)}; got {reduction_target!r}"
        )
    rt: Literal["halve", "single", "custom", "aspect_conditioned"] = (
        reduction_target  # type: ignore[assignment]
    )
    sel, dF, post = jnana_op(
        cands,
        apoha,
        anan,
        reduction_target=rt,
        lambda_a=lambda_a,
        lambda_p=lambda_p,
    )
    out = {
        "selected_idx": int(sel),
        "selected_text": candidate_texts[int(sel)],
        "delta_F": float(dF),
        "posterior": [float(p) for p in post.tolist()],
    }
    _audit("jnana", {"K": len(candidate_texts), "reduction": reduction_target}, {
        "selected_idx": int(sel), "delta_F": float(dF),
    })
    return out


# Tool 6
@mcp.tool()
def kriya(
    selected_text: str,
    render_mode: str = "verbatim",
) -> dict[str, Any]:
    """Render the selected candidate to a final surface (verbatim or polished)."""
    from pce.types import Candidate

    embed = _get_embed()
    cand = Candidate(
        seed=0,
        sampler={"tau": 1.0},
        tokens=(),
        text=selected_text,
        logp=0.0,
        embedding=embed.encode(selected_text),
    )
    valid_render_modes = {"verbatim", "polish", "claude_polish"}
    if render_mode not in valid_render_modes:
        raise ValueError(
            f"kriya: render_mode must be one of {sorted(valid_render_modes)}; got {render_mode!r}"
        )
    rm: Literal["verbatim", "polish", "claude_polish"] = render_mode  # type: ignore[assignment]
    surface = kriya_op(
        cand,
        render_mode=rm,
        lm=_get_lm() if render_mode == "polish" else None,
        embed=embed if render_mode == "polish" else None,
    )
    out = {"surface": surface}
    _audit("kriya", {"render_mode": render_mode}, {"len": len(surface)})
    return out


# Tool 7
@mcp.tool()
def vimarsa(
    prompt: str,
    surface: str,
    retrieval_set: list[str],
    aspects: list[str],
    ananda_score: float,
    aspect_cosine_hit: float = 0.40,
    novelty_threshold: float = 0.30,
    aesthetic_floor: float = 0.40,
) -> dict[str, Any]:
    """Detect a vimarsa aspect-shift event and return novelty + diagnostics."""
    embed = _get_embed()
    out_tuple = vimarsa_op(
        prompt=prompt,
        surface=surface,
        embed=embed,
        retrieval_set=retrieval_set,
        aspects=aspects,
        ananda_score=ananda_score,
        iccha_apoha_trajectory=None,
        novelty_threshold=novelty_threshold,
        aspect_cosine_hit=aspect_cosine_hit,
        aesthetic_floor=aesthetic_floor,
    )
    # vimarsa returns a 3-tuple when return_brief=False (the default).
    assert len(out_tuple) == 3
    event, novelty, diag = out_tuple
    out = {"event": bool(event), "novelty": float(novelty), "diagnostics": diag}
    _audit("vimarsa", {"prompt": prompt[:200]}, out)
    return out


# Tool 8 - legacy single-arm cascade (kept for v0.1 backward compatibility).
@mcp.tool()
def cascade(
    prompt: str,
    constraint_text: str,
    must_avoid: list[str] | None = None,
    aspects: list[str] | None = None,
    retrieval_set: list[str] | None = None,
    K: int = 4,
    max_tokens: int = 200,
    base_seed: int = 0,
    render_mode: str = "verbatim",
    bypass_vimarsa: bool = False,
) -> dict[str, Any]:
    """Run the full cascade against the local Qwen2 substrate (v0.1-compatible alias).

    Kept for backward compatibility with v0.1 / v0.2 callers. New code should
    call :func:`pce_cascade` directly with ``arm="local"`` (or any of the
    v0.3 ``haiku*`` arms) and explicit ``commit_policy``. ``bypass_vimarsa``
    here maps to ``commit_policy="always_draft"`` (legacy alias).
    """
    embed = _get_embed()
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid or ()),
    )
    state = run_cascade(
        prompt=prompt,
        constraint=constraint,
        lm=_get_lm(),
        embed=embed,
        K=K,
        max_tokens=max_tokens,
        base_seed=base_seed,
        retrieval_set=list(retrieval_set or []),
        aspects=list(aspects or []),
        render_mode=render_mode,
        bypass_vimarsa=bypass_vimarsa,
    )
    out: dict[str, Any] = state.to_audit()
    out["surface"] = state.surface
    out["selected_text"] = state.selected.text if state.selected is not None else None
    _audit("cascade", {"prompt": prompt[:200], "K": K, "bypass_vimarsa": bypass_vimarsa}, {
        "vimarsa_event": state.vimarsa_event,
        "novelty": state.vimarsa_novelty,
        "selected_idx": out["selected_idx"],
    })
    return out


# v0.3 tool: arm-switchable cascade with event-gated commit + active-inference uplift.
@mcp.tool()
def pce_cascade(
    prompt: str,
    constraint_text: str,
    arm: str = "haiku",
    must_avoid: list[str] | None = None,
    aspects: list[str] | None = None,
    retrieval_set: list[str] | None = None,
    K: int = 4,
    cit_temperature: float = 1.0,
    max_tokens: int = 200,
    base_seed: int = 0,
    render_mode: str = "verbatim",
    commit_policy: str | None = None,
    bypass_vimarsa: bool = False,
    use_storehouse: bool = True,
    hopfield_weight: float = 0.25,
) -> dict[str, Any]:
    """Run the v0.3 event-gated, always-shadow-revision cascade.

    Args:
        arm: substrate + protocol selector. Options:

            * ``"haiku"`` / ``"haiku_cascade"`` (default): Haiku substrate,
              event-gated commit, always-shadow revision.
            * ``"haiku_bare"``: Haiku substrate, single pass (always_draft).
              For benchmark control parity; for true bare scoring use the
              ``haiku_bare`` MCP tool which skips the cascade entirely.
            * ``"haiku_bare_2K"``: doubles K, single pass, no revision.
              Controls the "extra compute" confound (H6).
            * ``"haiku_generic_revise"``: 2-pass always_revise with the
              vimarsa brief replaced by a fixed generic creative-revise
              prompt. Isolates brief content from brief existence (H7).
            * ``"local"`` / ``"local_cascade"``: legacy Qwen2-1.5B path.

        commit_policy: one of ``"event_gated"`` (default for cascade arms),
            ``"always_revise"``, ``"always_draft"``, or ``"learned_gate"``
            (v0.4 logistic-regression gate trained per ADR-002; falls
            back to ``"event_gated"`` when the trained model is missing
            or its CV AUROC is below 0.55). The arm dispatch forces
            ``commit_policy`` for the control arms; explicit
            ``commit_policy`` on a cascade arm overrides the default.

        cit_temperature: posterior temperature for ``iccha`` exploration.
            Plumbed through ``parity_sampler`` per ADR-003.

        use_storehouse: when True (default) the per-process active-inference
            Hopfield store is threaded into the cascade so apohana gets a
            warm-start prior and vimarsa.consolidate writes the committed
            surface back at the end. Inspect via ``hopfield_state``.

    The returned payload mirrors :class:`pce.types.CascadeState.to_audit()`
    plus top-level ``surface``, ``surface_draft``, ``surface_revision``,
    ``substrate``, ``arm``, and ``arm_overrides``.
    """
    embed = _get_embed()
    arm_resolved, lm = _resolve_arm(arm)
    # Legacy alias: v0.2 callers pass `bypass_vimarsa=True` instead of
    # `commit_policy="always_draft"`. Honour it when commit_policy was not
    # explicitly set; otherwise let _resolve_commit_policy raise so the
    # caller resolves the ambiguity.
    cp_for_dispatch = commit_policy
    if commit_policy is None and bypass_vimarsa:
        cp_for_dispatch = "always_draft"
    K_eff, commit_eff, brief_override = _v3_arm_overrides(
        arm, K=K, commit_policy=cp_for_dispatch
    )
    # When the arm dispatch picked a commit policy, suppress the legacy
    # bypass alias so we don't trip the conflict guard.
    bypass_for_run = bypass_vimarsa and commit_eff == "always_draft" and commit_policy is None
    hop = _get_active_hopfield(embed) if use_storehouse else None
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid or ()),
    )
    state = run_cascade(
        prompt=prompt,
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=K_eff,
        cit_temperature=cit_temperature,
        max_tokens=max_tokens,
        base_seed=base_seed,
        retrieval_set=list(retrieval_set or []),
        aspects=list(aspects or []),
        render_mode=render_mode,
        commit_policy=commit_eff,
        bypass_vimarsa=bypass_for_run,
        hopfield=hop,
        hopfield_weight=hopfield_weight,
        brief_override=brief_override,
    )
    out: dict[str, Any] = state.to_audit()
    out["surface"] = state.surface
    out["surface_draft"] = state.surface_draft
    out["surface_revision"] = state.surface_revision
    out["substrate"] = arm_resolved
    out["arm"] = arm
    out["arm_overrides"] = {
        "K_effective": K_eff,
        "commit_policy_effective": commit_eff,
        "brief_override_used": brief_override is not None,
    }
    _audit(
        "pce_cascade",
        {
            "prompt": prompt[:200],
            "arm": arm,
            "K": K,
            "commit_policy": commit_policy,
            "cit_temperature": cit_temperature,
        },
        {
            "substrate": arm_resolved,
            "K_effective": K_eff,
            "commit_policy_effective": commit_eff,
            "brief_override_used": brief_override is not None,
            "vimarsa_event": state.vimarsa_event,
            "vimarsa_event_draft": state.vimarsa_event_draft,
            "novelty": state.vimarsa_novelty,
            "committed": state.committed,
            "two_pass": state.audit.get("two_pass", False),
            "revision_differs_from_draft": state.audit.get(
                "revision_differs_from_draft", False
            ),
        },
    )
    return out


@mcp.tool()
def haiku_bare(
    prompt: str,
    max_tokens: int = 200,
    seed: int = 0,
    tau: float = 0.9,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Single Haiku call with no cascade. The bare arm of the v0.2 four-pack."""
    haiku = _get_haiku()
    cand = haiku.generate(
        prompt,
        max_tokens=max_tokens,
        sampler={"tau": tau, "top_p": top_p, "top_k": 50.0},
        seed=seed,
    )
    rep = haiku.report()
    out = {
        "text": cand.text,
        "logp": cand.logp,
        "n_tokens": len(cand.tokens),
        "sampler": cand.sampler,
        "ledger_total_usd": rep["ledger_total_usd"],
        "ledger_n_calls": rep["ledger_n_calls"],
    }
    _audit(
        "haiku_bare",
        {"prompt": prompt[:200], "seed": seed, "max_tokens": max_tokens},
        {
            "text_len": len(cand.text),
            "ledger_total_usd": rep["ledger_total_usd"],
        },
    )
    return out


# Tool 9
@mcp.tool()
def hopfield_store(text: str) -> dict[str, Any]:
    """Store a text's embedding in the Hopfield ālayavijñāna."""
    embed = _get_embed()
    h = _get_hopfield()
    h.store(embed.encode(text))
    out = {"n_patterns": int(h.n_patterns)}
    _audit("hopfield_store", {"text": text[:100]}, out)
    return out


# Tool 10
@mcp.tool()
def hopfield_recall(cue_text: str) -> dict[str, Any]:
    """Recall the closest stored pattern for a given cue text and report cosine similarity."""
    embed = _get_embed()
    h = _get_hopfield()
    cue = embed.encode(cue_text)
    rec = h.recall(cue)
    out = {
        "n_patterns": int(h.n_patterns),
        "recall_cosine": float(np.dot(cue, rec) / (np.linalg.norm(cue) * np.linalg.norm(rec) + 1e-12)),
    }
    _audit("hopfield_recall", {"text": cue_text[:100]}, out)
    return out


# Tool 11
@mcp.tool()
def consolidate_sws(
    trace_texts: list[str], n_centroids: int = 4, n_iter: int = 25, seed: int = 0
) -> dict[str, Any]:
    """SWS k-means consolidation over a list of trace texts; centroids are stored."""
    embed = _get_embed()
    h = _get_hopfield()
    traces = [embed.encode(t) for t in trace_texts]
    centroids = run_sws(h, traces, n_centroids=n_centroids, n_iter=n_iter, seed=seed)
    out = {"n_centroids": len(centroids), "n_patterns": int(h.n_patterns)}
    _audit("consolidate_sws", {"n_traces": len(trace_texts)}, out)
    return out


# Tool 12
@mcp.tool()
def consolidate_rem(n_steps: int = 100, temperature: float = 1.5, seed: int = 0) -> dict[str, Any]:
    """REM Metropolis replay over the current Hopfield store."""
    h = _get_hopfield()
    traj = run_rem(h, n_steps=n_steps, temperature=temperature, seed=seed)
    out = {"n_steps": len(traj), "n_patterns": int(h.n_patterns)}
    _audit("consolidate_rem", {"n_steps": n_steps}, out)
    return out


# Tool 13
@mcp.tool()
def consolidate_cycle(
    trace_texts: list[str],
    sws_centroids: int = 4,
    rem_steps: int = 50,
    rem_temperature: float = 1.5,
    seed: int = 0,
) -> dict[str, Any]:
    """One full SWS + REM consolidation cycle. Returns counts and is_consolidated probe."""
    embed = _get_embed()
    h = _get_hopfield()
    traces = [embed.encode(t) for t in trace_texts]
    diag = run_sleep_cycle(
        h,
        traces,
        sws_centroids=sws_centroids,
        rem_steps=rem_steps,
        rem_temperature=rem_temperature,
        seed=seed,
    )
    consolidated = (
        is_consolidated(h, traces[0]) if traces else False
    )
    out = {**diag, "first_trace_consolidated": bool(consolidated)}
    _audit("consolidate_cycle", {"n_traces": len(trace_texts)}, out)
    return out


# Tool 14
@mcp.tool()
def report() -> dict[str, Any]:
    """Substrate report: model ids, vocab, dim, n_patterns, Haiku cost ledger.

    The Haiku block is only populated if the Haiku substrate has been
    instantiated this session (lazy-loaded on the first ``arm="haiku"`` call
    or any direct ``haiku_bare`` invocation).
    """
    lm = _get_lm()
    embed = _get_embed()
    h = _get_hopfield()
    out: dict[str, Any] = {
        "version": "0.4.0",
        "lm": lm.report(),
        "embedder": {"model_id": embed.model_id, "dim": embed.dim},
        "hopfield": {"dim": h.dim, "beta": h.beta, "n_patterns": h.n_patterns},
        "active_hopfield": (
            {
                "domain": _ACTIVE_HOPFIELD.domain,
                "dim": _ACTIVE_HOPFIELD.dim,
                "n_patterns": _ACTIVE_HOPFIELD.n_patterns,
            }
            if _ACTIVE_HOPFIELD is not None
            else None
        ),
        "audit_path": str(_AUDIT_PATH),
    }
    if _HAIKU is not None:
        out["haiku"] = _HAIKU.report()
    return out


# Tool 15
@mcp.tool()
def reset_state() -> dict[str, Any]:
    """Reset both the v0.1 and v0.3 Hopfield stores (LM/embedder stay loaded)."""
    global _HOPFIELD, _ACTIVE_HOPFIELD, _INTEGRITY_PROBE
    n_before_legacy = _HOPFIELD.n_patterns if _HOPFIELD is not None else 0
    n_before_active = (
        _ACTIVE_HOPFIELD.n_patterns if _ACTIVE_HOPFIELD is not None else 0
    )
    _HOPFIELD = None
    _ACTIVE_HOPFIELD = None
    if _INTEGRITY_PROBE is not None:
        _INTEGRITY_PROBE.invalidate()
    h = _get_hopfield()
    return {
        "n_patterns_before_legacy": n_before_legacy,
        "n_patterns_after_legacy": h.n_patterns,
        "n_patterns_before_active": n_before_active,
        "n_patterns_after_active": 0,
        "integrity_probe_cache_invalidated": _INTEGRITY_PROBE is not None,
    }


# Tool 16 (v0.3 NEW)
@mcp.tool()
def haiku_clean_substrate_probe(force: bool = False) -> dict[str, Any]:
    """Run the v0.3 :class:`IntegrityProbe` against the inner Haiku CLI subprocess.

    Spawns a single ``claude --print`` subprocess via the same path the
    cascade uses, asks the model to enumerate any active plugins / skills /
    system instructions, and asserts the response contains no
    :data:`pce.substrate.integrity.LEAKAGE_REGEX` matches (with
    negation-context filter, so "no plugins loaded" is not a leak).

    Args:
        force: bypass the per-(env_hash, flags_hash) cache and always
            re-spawn the inner subprocess. Default False.

    Returns the full :class:`IntegrityResult` payload as JSON.
    """
    haiku = _get_haiku()
    probe = _get_integrity_probe()
    res = probe.run(haiku, force=force)
    out: dict[str, Any] = {
        "passed": bool(res.passed),
        "response": res.response,
        "leak_matches": list(res.leak_matches),
        "positive_hint": bool(res.positive_hint),
        "env_hash": res.env_hash,
        "flags_hash": res.flags_hash,
        "probe_at_iso": res.probe_at_iso,
        "cost_usd": float(res.cost_usd),
    }
    _audit(
        "haiku_clean_substrate_probe",
        {"force": force},
        {
            "passed": out["passed"],
            "leak_count": len(out["leak_matches"]),
            "cost_usd": out["cost_usd"],
        },
    )
    return out


# Tool 17 (v0.3 NEW)
@mcp.tool()
def hopfield_state(last_n: int = 5) -> dict[str, Any]:
    """Inspect the v0.3 active-inference Hopfield ālayavijñāna.

    Returns ``n_patterns``, ``dim``, the per-aspect-label pattern count,
    and the L2 norms of the last ``last_n`` stored patterns -- enough to
    confirm the storehouse is non-degenerate and warming the cascade
    apohana prior. Distinct from the v0.1 ``hopfield_store`` /
    ``hopfield_recall`` legacy store.
    """
    embed = _get_embed()
    h = _get_active_hopfield(embed)
    last_norms: list[float] = []
    label_counts: dict[str, int] = {}
    if h.n_patterns > 0:
        for lab in h._labels:  # noqa: SLF001 — read-only introspection
            label_counts[lab] = label_counts.get(lab, 0) + 1
        n = int(max(0, min(last_n, h.n_patterns)))
        if n > 0:
            tail = h._patterns[-n:]  # noqa: SLF001 — read-only introspection
            last_norms = [float(np.linalg.norm(row)) for row in tail]
    out: dict[str, Any] = {
        "dim": int(h.dim),
        "n_patterns": int(h.n_patterns),
        "domain": h.domain,
        "label_counts": label_counts,
        "last_norms": last_norms,
    }
    _audit("hopfield_state", {"last_n": last_n}, {"n_patterns": out["n_patterns"]})
    return out


# Tool 19 (v0.4 NEW)
@mcp.tool()
def judge_pair(
    prompt: str,
    text_a: str,
    text_b: str,
    model: str = "sonnet",
    cli_bin: str = "claude",
    timeout_s: int = 120,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the v0.4 Sonnet LLM-judge on a single ad-hoc A/B pair.

    Wraps :mod:`scripts.judge_subset` so any MCP caller can request a
    single judge verdict without scheduling a full stratified subset.
    Uses the same frozen prompt (``scripts/judge_prompt_v0_4.txt``)
    and the same OAuth-only ``claude --print --model sonnet``
    substrate the powered pilot uses, so single-call verdicts are
    directly comparable to ``judge.jsonl`` rows.

    Args:
        prompt: the original user prompt the two candidates are
            answering.
        text_a: candidate A's response.
        text_b: candidate B's response.
        model: Sonnet model alias (default ``"sonnet"``).
        cli_bin: path to the ``claude`` binary.
        timeout_s: per-call timeout.
        dry_run: when True, use the deterministic fake responder
            (longer block wins) so the round-trip can be verified
            without spending Sonnet quota. Default False.

    Returns: a dict with ``winner`` (A/B/tie), ``confidence``,
        ``rationale``, ``input_tokens``, ``output_tokens``,
        ``cost_usd``, ``elapsed_s``, ``prompt_sha256``, and ``model``.
        The prompt sha256 matches the one written to every
        ``judge.jsonl`` row from the powered pilot.
    """
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import judge_subset  # noqa: PLC0415 — lazy load so the MCP server imports cheaply

    template, prompt_sha = judge_subset._load_prompt_template(
        judge_subset.DEFAULT_PROMPT_PATH
    )
    pair = judge_subset.JudgePair(
        domain="adhoc",
        item_id="adhoc",
        item_prompt=prompt,
        treatment_arm="A",
        control_arm="B",
        treatment_text=text_a,
        control_text=text_b,
        treatment_composite=0.0,
        control_composite=0.0,
        proxy_delta=0.0,
        quartile=-1,
    )
    formatted = judge_subset._format_judge_prompt(template, pair=pair, swap=False)
    if dry_run:
        verdict = judge_subset._fake_responder(formatted)
        model_label = "fake-responder"
    else:
        try:
            verdict = judge_subset._call_sonnet_cli(
                formatted, model=model, timeout_s=timeout_s, cli_bin=cli_bin
            )
            model_label = model
        except RuntimeError as exc:
            return {
                "winner": "tie",
                "confidence": 0.0,
                "rationale": f"[error] {str(exc)[:200]}",
                "prompt_sha256": prompt_sha,
                "model": model,
                "error": True,
            }
    in_tok = int(verdict.get("_input_tokens", judge_subset.INPUT_TOKEN_ESTIMATE))
    out_tok = int(verdict.get("_output_tokens", judge_subset.OUTPUT_TOKEN_ESTIMATE))
    cost = (
        in_tok * judge_subset.SONNET_INPUT_USD_PER_TOK
        + out_tok * judge_subset.SONNET_OUTPUT_USD_PER_TOK
    )
    out: dict[str, Any] = {
        "winner": verdict.get("winner", "tie"),
        "confidence": float(verdict.get("confidence") or 0.0),
        "rationale": str(verdict.get("rationale", ""))[:500],
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": float(cost),
        "elapsed_s": float(verdict.get("_elapsed_s", 0.0) or 0.0),
        "prompt_sha256": prompt_sha,
        "model": model_label,
        "error": False,
    }
    _audit(
        "judge_pair",
        {"prompt_excerpt": prompt[:200], "model": model_label, "dry_run": dry_run},
        {
            "winner": out["winner"],
            "confidence": out["confidence"],
            "cost_usd": out["cost_usd"],
            "prompt_sha256": prompt_sha,
        },
    )
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
