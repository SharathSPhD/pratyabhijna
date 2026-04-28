"""Pratyabhijna Creative Engine - MCP server (v0.2).

Exposes the seven cascade operators, the cascade orchestrator, consolidation
routines, and a handful of audit/diagnostic utilities. v0.2 (ADR-001 / ADR-004)
adds a pluggable substrate so the cascade can run against either the local
Qwen2-1.5B (`arm="local"`) or Anthropic Haiku via the `claude` CLI
(`arm="haiku"`); the new `pce_cascade` tool exposes that switch. All v0.1 tool
names are preserved for backward compatibility.

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

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
from pce.substrate.lm import LMConfig, LocalLM  # noqa: E402
from pce.substrate.lm_protocol import LMProtocol  # noqa: E402
from pce.types import Constraint  # noqa: E402

mcp = FastMCP("pratyabhijna")

_LM: LocalLM | None = None
_HAIKU: HaikuLM | None = None
_EMBED: Embedder | None = None
_HOPFIELD: HopfieldStore | None = None
_AUDIT_PATH = REPO_ROOT / "audit" / "phase8" / "mcp_calls.jsonl"


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
    ``"haiku"``, ``"haiku_bare"``, ``"haiku_cascade"``  -> HaikuLM.
    """
    a = (arm or "local").strip().lower()
    if a in {"local", "local_bare", "local_cascade", "qwen", "qwen2"}:
        return "local", _get_lm()
    if a in {"haiku", "haiku_bare", "haiku_cascade", "claude_haiku"}:
        return "haiku", _get_haiku()
    raise ValueError(
        f"unknown arm={arm!r}; expected one of: local, haiku, "
        "local_bare, local_cascade, haiku_bare, haiku_cascade"
    )


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
    sel, dF, post = jnana_op(
        cands,
        apoha,
        anan,
        reduction_target=reduction_target,  # type: ignore[arg-type]
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
    surface = kriya_op(
        cand,
        render_mode=render_mode,  # type: ignore[arg-type]
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
    """Run the full v0.2 two-pass-always cascade against the local Qwen2 substrate.

    For the four-arm v0.2 setup (`local_bare`, `local_cascade`, `haiku_bare`,
    `haiku_cascade`) use :func:`pce_cascade(arm=...)` instead; this tool is the
    drop-in v0.1 replacement that always runs `local_cascade`.
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
    out = state.to_audit()
    out["surface"] = state.surface
    out["selected_text"] = state.selected.text if state.selected is not None else None
    _audit("cascade", {"prompt": prompt[:200], "K": K, "bypass_vimarsa": bypass_vimarsa}, {
        "vimarsa_event": state.vimarsa_event,
        "novelty": state.vimarsa_novelty,
        "selected_idx": out["selected_idx"],
    })
    return out


# v0.2 tool: arm-switchable cascade.
@mcp.tool()
def pce_cascade(
    prompt: str,
    constraint_text: str,
    arm: str = "local",
    must_avoid: list[str] | None = None,
    aspects: list[str] | None = None,
    retrieval_set: list[str] | None = None,
    K: int = 4,
    max_tokens: int = 200,
    base_seed: int = 0,
    render_mode: str = "verbatim",
    bypass_vimarsa: bool = False,
) -> dict[str, Any]:
    """Run the v0.2 two-pass cascade against an arm-selected substrate.

    Args:
        arm: ``"local"`` (Qwen2-1.5B), ``"haiku"`` (Anthropic Haiku via
            ``claude`` CLI), or any of the long-form arm names from the
            benchmark driver (``local_bare``/``local_cascade``/``haiku_bare``/
            ``haiku_cascade``). The cascade itself is the same two-pass
            pipeline regardless of arm; ``arm`` only chooses the substrate.
        bypass_vimarsa: when True the cascade collapses to a single pass and
            returns the draft as ``surface`` (the explicit ablation control
            for H6.v2 / H8.v2). Default False (two-pass-always).

    The returned payload mirrors :class:`pce.types.CascadeState.to_audit()`
    plus a top-level ``surface`` and a ``substrate`` field naming the
    resolved substrate.
    """
    embed = _get_embed()
    arm_resolved, lm = _resolve_arm(arm)
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
        K=K,
        max_tokens=max_tokens,
        base_seed=base_seed,
        retrieval_set=list(retrieval_set or []),
        aspects=list(aspects or []),
        render_mode=render_mode,
        bypass_vimarsa=bypass_vimarsa,
    )
    out = state.to_audit()
    out["surface"] = state.surface
    out["surface_draft"] = state.surface_draft
    out["surface_revision"] = state.surface_revision
    out["substrate"] = arm_resolved
    out["arm"] = arm
    _audit(
        "pce_cascade",
        {"prompt": prompt[:200], "arm": arm, "K": K, "bypass_vimarsa": bypass_vimarsa},
        {
            "substrate": arm_resolved,
            "vimarsa_event": state.vimarsa_event,
            "vimarsa_event_draft": state.vimarsa_event_draft,
            "novelty": state.vimarsa_novelty,
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
        "version": "0.2.0",
        "lm": lm.report(),
        "embedder": {"model_id": embed.model_id, "dim": embed.dim},
        "hopfield": {"dim": h.dim, "beta": h.beta, "n_patterns": h.n_patterns},
        "audit_path": str(_AUDIT_PATH),
    }
    if _HAIKU is not None:
        out["haiku"] = _HAIKU.report()
    return out


# Tool 15
@mcp.tool()
def reset_state() -> dict[str, Any]:
    """Reset the Hopfield store (LM and embedder remain loaded)."""
    global _HOPFIELD
    n_before = _HOPFIELD.n_patterns if _HOPFIELD is not None else 0
    _HOPFIELD = None
    h = _get_hopfield()
    return {"n_patterns_before": n_before, "n_patterns_after": h.n_patterns}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
