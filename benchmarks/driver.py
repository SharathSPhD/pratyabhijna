#!/usr/bin/env python3
"""PCE v0.4 benchmark driver: four-arm Haiku matrix with per-item integrity probe.

v0.4 = v0.3 + commit-policy multiplex (ADR-002). The four base Haiku
arms below are unchanged; on top of every ``haiku_cascade`` row we
synthesise four extra "commit-policy arms" (always_draft /
always_revise / event_gated / learned_gate) plus a post-hoc
``oracle`` analysis arm — at *zero extra Haiku cost* — by re-scoring
the draft / shadow-revision pair under each policy's decision rule
(``_multiplex_commit_policies``).

The four v0.4 base arms are unchanged from v0.3:

* ``haiku_bare``                - 1 Haiku call with the parity sampler.
                                  Architecture-free baseline.
* ``haiku_cascade``             - run_cascade with ``commit_policy="event_gated"``
                                  and the active-inference uplift (aspect-conditioned
                                  BMR, Hopfield warm-start, free-energy budget).
* ``haiku_bare_2K_scorer``      - ``iccha`` with K' = 2K candidates, single pass
                                  (commit_policy="always_draft"). Controls the
                                  *extra-compute* confound (H6).
* ``haiku_generic_revise_2pass`` - ``run_cascade`` with ``commit_policy="always_revise"``
                                  but ``brief_override`` set to a fixed generic
                                  creative-revise prompt. Isolates the
                                  *content* of the brief from the *existence*
                                  of a revision pass (H7).

Per-item integrity probe (ADR-001 / Phase 5 v0.3): before processing every
item we run :class:`pce.substrate.integrity.IntegrityProbe` and assert
``passed=True``. Probe results are cached per-(env_hash, flags_hash) so the
real-time cost is negligible after the first call. Per-item probe rows are
written to ``audit/v0.4/integrity_probes.jsonl`` for forensics; if any
probe fails the driver halts unless ``--allow-leakage`` is set.

Each call's response is scored locally with ``benchmarks.scoring.*`` and
the raw text + axis dict + composite score is appended to a per-domain
JSON file under ``--out-dir`` (default ``benchmarks/results_v0.4``).

Cost telemetry: HaikuLM owns a shared cost ledger that is snapshotted to
``audit/v0.4/cost_snapshot.json`` after every Haiku-touching call so the
run can be budget-capped at the $30 v0.4 envelope.

Robustness:
* Per-call timeout. If a call fails (rate limit, network), the row is recorded
  with ``error`` set and ``composite=null`` (JSON; in-memory it's None).
* Resumable: if ``--out-dir`` already contains a file for a domain, skip items
  whose ``id`` is already present for the requested arms.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from benchmarks import items as bench_items  # noqa: E402
from benchmarks import scoring as bench_scoring  # noqa: E402
from pce.cascade import run_cascade  # noqa: E402
from pce.policies import (  # noqa: E402
    OracleCommit,
    PolicyFeatures,
    extract_features_from_audit,
    policy_for_name,
)
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.haiku_lm import HaikuBudgetExceededError, HaikuLM  # noqa: E402
from pce.substrate.integrity import IntegrityProbe  # noqa: E402
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.substrate.lm_protocol import LMProtocol  # noqa: E402
from pce.types import Constraint  # noqa: E402

# v0.3 default Haiku-only four-arm matrix.
ARMS_V3 = (
    "haiku_bare",
    "haiku_cascade",
    "haiku_bare_2K_scorer",
    "haiku_generic_revise_2pass",
)

# v0.4 commit-policy multiplex (ADR-002): each entry is a synthetic arm
# computed *post-hoc* from the haiku_cascade draft/revision pair, so the
# four cascade-policy arms cost zero extra Haiku calls.
COMMIT_POLICY_ARMS_V4 = (
    "haiku_cascade_event_gated",
    "haiku_cascade_always_draft",
    "haiku_cascade_always_revise",
    "haiku_cascade_learned_gate",
)
ORACLE_ANALYSIS_ARM = "haiku_cascade_oracle"  # post-hoc upper bound, never an evaluation arm
# v0.2 / v0.1 arm aliases retained for backward compatibility.
ARMS_LEGACY = ("local_bare", "local_cascade")
ARM_ALIASES = {
    "claude_haiku": "haiku_bare",
    "haiku_bare_2k": "haiku_bare_2K_scorer",
    "haiku_bare_2K": "haiku_bare_2K_scorer",
    "haiku_generic_revise": "haiku_generic_revise_2pass",
}

DEFAULT_DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
# v0.4 default output dir (Phase 7 powered pilot lands here). Override via
# --out-dir for backward-compat v0.3 runs targeting benchmarks/results_v0.3.
DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "results_v0.4"
COST_SNAPSHOT_PATH = REPO_ROOT / "audit" / "v0.4" / "cost_snapshot.json"
INTEGRITY_LOG_PATH = REPO_ROOT / "audit" / "v0.4" / "integrity_probes.jsonl"

PARITY_SAMPLER: dict[str, float] = {"tau": 0.9, "top_p": 0.95, "top_k": 50.0}

GENERIC_REVISE_BRIEF = (
    "Revise the previous draft to be more creative, specific, and surprising. "
    "Add concrete sensory detail, remove cliches, and strengthen the most "
    "interesting move you can find."
)


def _build_prompt(
    domain: str, item: dict[str, Any]
) -> tuple[str, str, list[str], list[str], list[str]]:
    """Return (user_prompt, constraint_text, must_avoid, aspects, retrieval_set)."""
    if domain == "poetry_gen":
        prompt = (
            f"Compose a {item['form']} about: {item['topic']}.\n"
            f"Avoid: {', '.join(item['must_avoid'])}.\n"
            f"Output only the poem.\n"
        )
        constraint = f"a {item['form']} about {item['topic']}"
        return prompt, constraint, list(item["must_avoid"]), [], []
    if domain == "poetry_interp":
        prompt = (
            "Interpret this line in two short paragraphs, naming each reading:\n\n"
            f"\"{item['surface']}\"\n\nReading A:\n"
        )
        constraint = f"two readings of: {item['surface']}"
        return prompt, constraint, [], list(item["aspects"]), list(item["retrieval_set"])
    if domain == "aut":
        prompt = (
            f"List 8 unusual, non-obvious uses of a {item['object']}. "
            "Be concrete and specific. Avoid the standard everyday use. "
            "Format: one use per line.\n"
        )
        constraint = f"unusual, non-obvious uses of a {item['object']}"
        return (
            prompt,
            constraint,
            [f"the standard everyday use of a {item['object']}"],
            [],
            [],
        )
    if domain == "sci_creativity":
        prompt = (
            f"{item['question']} Give a non-obvious explanation in 4-6 sentences, "
            "naming at least two different framings. Avoid the textbook one-liner.\n"
        )
        constraint = f"non-obvious explanation of: {item['question']}"
        return (
            prompt,
            constraint,
            [f"the standard textbook explanation of {item['question']}"],
            list(item.get("framings", [])),
            [],
        )
    raise ValueError(f"unknown domain: {domain}")


def _snapshot_cost_ledger(haiku_lm: HaikuLM | None) -> None:
    if haiku_lm is None:
        return
    rep = haiku_lm.report()
    COST_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_SNAPSHOT_PATH.write_text(
        json.dumps({"haiku": rep, "ts": time.time()}, indent=2),
        encoding="utf-8",
    )


def _normalise_arm(arm: str) -> str:
    return ARM_ALIASES.get(arm, arm)


CommitPolicyLit = Literal["event_gated", "always_revise", "always_draft"]


def _arm_overrides_for_run_cascade(
    arm: str, *, K: int
) -> tuple[int, CommitPolicyLit, str | None]:
    """Resolve (K_eff, commit_policy, brief_override) for the v0.3 arm dispatch.

    Mirror of :func:`plugin.mcp.server._v3_arm_overrides` so the benchmark
    driver and the MCP layer share the same arm semantics.
    """
    a = _normalise_arm(arm).lower()
    if a == "haiku_bare_2k_scorer":
        return (max(1, K) * 2, "always_draft", None)
    if a == "haiku_generic_revise_2pass":
        return (K, "always_revise", GENERIC_REVISE_BRIEF)
    if a == "haiku_cascade":
        return (K, "event_gated", None)
    if a in {"haiku_bare", "local_bare"}:
        return (K, "always_draft", None)
    if a == "local_cascade":
        return (K, "event_gated", None)
    raise ValueError(f"unknown v0.3 arm: {arm}")


def _call_local_bare(
    prompt: str, *, lm: LocalLM, max_tokens: int, seed: int
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    out = lm.generate(prompt, max_tokens=max_tokens, sampler=PARITY_SAMPLER, seed=seed)
    return out.text, {"ok": True, "elapsed_s": time.time() - started}


def _call_haiku_bare(
    prompt: str, *, haiku_lm: HaikuLM, max_tokens: int, seed: int
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    try:
        out = haiku_lm.generate(
            prompt, max_tokens=max_tokens, sampler=PARITY_SAMPLER, seed=seed
        )
    except HaikuBudgetExceededError as e:
        return "", {
            "ok": False,
            "error": f"budget: {e}",
            "elapsed_s": time.time() - started,
        }
    except Exception as e:  # noqa: BLE001
        return "", {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.time() - started,
        }
    rep = haiku_lm.report()
    return out.text, {
        "ok": True,
        "elapsed_s": time.time() - started,
        "haiku_total_usd": float(rep["ledger_total_usd"]),
        "haiku_n_calls": int(rep["ledger_n_calls"]),
    }


def _call_cascade_arm(
    *,
    prompt: str,
    arm: str,
    lm: LMProtocol,
    embed: Embedder,
    constraint_text: str,
    must_avoid: list[str],
    aspects: list[str],
    retrieval_set: list[str],
    K: int,
    max_tokens: int,
    seed: int,
    haiku_lm: HaikuLM | None,
) -> tuple[str, dict[str, Any]]:
    """Run the v0.3 cascade with arm-specific overrides applied."""
    started = time.time()
    K_eff, commit_policy, brief_override = _arm_overrides_for_run_cascade(arm, K=K)
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid),
    )
    try:
        state = run_cascade(
            prompt=prompt,
            constraint=constraint,
            lm=lm,
            embed=embed,
            K=K_eff,
            max_tokens=max_tokens,
            base_seed=seed,
            retrieval_set=retrieval_set,
            aspects=aspects,
            commit_policy=commit_policy,
            brief_override=brief_override,
        )
    except HaikuBudgetExceededError as e:
        return "", {
            "ok": False,
            "error": f"budget: {e}",
            "elapsed_s": time.time() - started,
        }
    except Exception as e:  # noqa: BLE001
        return "", {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.time() - started,
        }
    aud = state.audit
    diag_draft = aud.get("vimarsa_diag_draft") or aud.get("vimarsa_diag") or {}
    if not isinstance(diag_draft, dict):
        diag_draft = {}
    ledger_audit = aud.get("budget_ledger") or {}
    if not isinstance(ledger_audit, dict):
        ledger_audit = {}
    policy_features_audit = aud.get("policy_features") or {}
    if not isinstance(policy_features_audit, dict):
        policy_features_audit = {}
    meta: dict[str, Any] = {
        "ok": True,
        "elapsed_s": time.time() - started,
        "K_effective": K_eff,
        "commit_policy_effective": commit_policy,
        "brief_override_used": brief_override is not None,
        "committed": str(state.committed),
        "vimarsa_event": bool(state.vimarsa_event),
        "vimarsa_event_draft": bool(state.vimarsa_event_draft),
        "novelty": float(state.vimarsa_novelty),
        "delta_F": _coerce_float_or_none(aud.get("delta_F")),
        "delta_F_draft": _coerce_float_or_none(aud.get("delta_F_draft")),
        "delta_F_revision": _coerce_float_or_none(aud.get("delta_F_revision")),
        "selected_idx": int(aud.get("selected_idx", -1)),
        "two_pass": bool(aud.get("two_pass", False)),
        "revision_differs_from_draft": bool(
            aud.get("revision_differs_from_draft", False)
        ),
        "surface_draft": str(state.surface_draft or ""),
        "surface_revision": str(state.surface_revision or ""),
        # v0.4 feature plumbing for the commit-policy multiplex (ADR-002).
        "aspect_count": _coerce_float_or_none(
            aud.get("aspect_count", diag_draft.get("aspect_count"))
        ),
        "ananda": _coerce_float_or_none(
            aud.get("ananda", diag_draft.get("ananda"))
        ),
        "budget_balance": _coerce_float_or_none(
            aud.get("budget_balance", ledger_audit.get("balance_bits"))
        ),
        "policy_features": policy_features_audit,
        "revision_skipped_reason": aud.get("revision_skipped_reason"),
    }
    if haiku_lm is not None:
        rep = haiku_lm.report()
        meta["haiku_total_usd"] = float(rep["ledger_total_usd"])
        meta["haiku_n_calls"] = int(rep["ledger_n_calls"])
    return state.surface or "", meta


def _coerce_float_or_none(v: object) -> float | None:
    """Convert numeric to float; return None for NaN / non-finite. Used by the
    v0.3 stats ``allow_nan=False`` JSON serialiser pipeline."""
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


SCORERS = {
    "poetry_gen": bench_scoring.score_poetry_gen,
    "poetry_interp": bench_scoring.score_poetry_interp,
    "aut": bench_scoring.score_aut,
    "sci_creativity": bench_scoring.score_sci_creativity,
}


def _domain_items(domain: str, n: int | None = None) -> list[dict[str, Any]]:
    if domain == "poetry_gen":
        out: list[dict[str, Any]] = [dict(x) for x in bench_items.POETRY_GEN]
    elif domain == "poetry_interp":
        out = [dict(x) for x in bench_items.POETRY_INTERP]
    elif domain == "aut":
        out = [dict(x) for x in bench_items.AUT]
    elif domain == "sci_creativity":
        out = [dict(x) for x in bench_items.SCI_CREATIVITY]
    else:
        raise ValueError(domain)
    if n is not None:
        out = out[:n]
    return out


def _load_existing(out_path: Path) -> dict[str, dict[str, Any]]:
    if not out_path.exists():
        return {}
    data = json.loads(out_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = data.get("rows", {})
    return rows


def _save(out_path: Path, rows: dict[str, dict[str, Any]], domain: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"domain": domain, "version": "v0.4", "rows": rows}
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _features_from_cascade_meta(
    meta: dict[str, Any],
) -> PolicyFeatures:
    """Extract :class:`PolicyFeatures` from a cascade arm's audit `meta`.

    Falls back to the v0.3-compatible top-level fields when the explicit
    ``policy_features`` block (added in v0.4) is missing — keeps replay over
    older traces working.
    """
    return extract_features_from_audit(meta)


def _multiplex_commit_policies(
    *,
    domain: str,
    item: dict[str, Any],
    item_rows: dict[str, Any],
    embed: Embedder,
    scorer: Callable[..., bench_scoring.ItemScore],
) -> None:
    """Synthesize per-commit-policy rows from the existing ``haiku_cascade`` row.

    No extra LM calls are issued — both ``surface_draft`` and ``surface_revision``
    are already present in the cascade audit. We re-score both surfaces, then
    let each :class:`CommitPolicy` pick which one to commit.
    """
    cascade_row = item_rows.get("haiku_cascade")
    if not isinstance(cascade_row, dict) or cascade_row.get("skipped"):
        return
    meta = cascade_row.get("meta") or {}
    if not isinstance(meta, dict):
        return
    audit = meta.get("audit") or {}
    if not isinstance(audit, dict):
        audit = {}
    surface_draft = meta.get("surface_draft") or audit.get("surface_draft")
    surface_revision = meta.get("surface_revision") or audit.get("surface_revision")
    if not isinstance(surface_draft, str) or not surface_draft:
        return

    score_draft = scorer(surface_draft, item=item, embed=embed)
    score_revision = (
        scorer(surface_revision, item=item, embed=embed)
        if isinstance(surface_revision, str) and surface_revision
        else None
    )

    features = _features_from_cascade_meta(meta)
    vimarsa_event = bool(meta.get("vimarsa_event_draft", meta.get("vimarsa_event", False)))

    def _commit(decision_revision: bool) -> tuple[str, bench_scoring.ItemScore]:
        if decision_revision and score_revision is not None:
            assert isinstance(surface_revision, str)
            return surface_revision, score_revision
        return surface_draft, score_draft

    for arm_name in COMMIT_POLICY_ARMS_V4:
        if arm_name in item_rows:
            continue
        policy_name = arm_name.removeprefix("haiku_cascade_")
        policy = policy_for_name(policy_name)
        decision = policy.decide(features, vimarsa_event)
        text, score = _commit(decision)
        composite = (
            float(score.composite) if not np.isnan(score.composite) else None
        )
        item_rows[arm_name] = {
            "text": text,
            "axes": dict(score.axes),
            "composite": composite,
            "n_chars": len(text),
            "n_words": len(text.split()) if text else 0,
            "meta": {
                "synthesized_from": "haiku_cascade",
                "commit_policy": policy_name,
                "commit_decision_revision": bool(decision),
                "policy_features": features.to_audit(),
                "score_draft": (
                    float(score_draft.composite)
                    if not np.isnan(score_draft.composite)
                    else None
                ),
                "score_revision": (
                    float(score_revision.composite)
                    if score_revision is not None
                    and not np.isnan(score_revision.composite)
                    else None
                ),
            },
        }

    if ORACLE_ANALYSIS_ARM not in item_rows and score_revision is not None:
        oracle = OracleCommit()
        oracle.set_scores(
            float(score_draft.composite),
            float(score_revision.composite),
        )
        decision = oracle.decide(features, vimarsa_event)
        text, score = _commit(decision)
        composite = (
            float(score.composite) if not np.isnan(score.composite) else None
        )
        item_rows[ORACLE_ANALYSIS_ARM] = {
            "text": text,
            "axes": dict(score.axes),
            "composite": composite,
            "n_chars": len(text),
            "n_words": len(text.split()) if text else 0,
            "meta": {
                "synthesized_from": "haiku_cascade",
                "commit_policy": "oracle",
                "commit_decision_revision": bool(decision),
                "policy_features": features.to_audit(),
                "score_draft": float(score_draft.composite),
                "score_revision": float(score_revision.composite),
                "is_analysis_only": True,
            },
        }


def _per_item_integrity_probe(
    probe: IntegrityProbe,
    *,
    haiku_lm: HaikuLM,
    domain: str,
    item_id: str,
    allow_leakage: bool,
) -> dict[str, Any]:
    """Run + log a per-item :class:`IntegrityProbe`. Halts the run if the probe
    fails unless ``allow_leakage`` is True (in which case the failing rows are
    still recorded and the run continues)."""
    res = probe.run(haiku_lm)  # cached unless env/flags change
    record = {
        "domain": domain,
        "item_id": item_id,
        "passed": bool(res.passed),
        "leak_matches": list(res.leak_matches),
        "positive_hint": bool(res.positive_hint),
        "env_hash": res.env_hash,
        "flags_hash": res.flags_hash,
        "probe_at_iso": res.probe_at_iso,
        "ts": time.time(),
    }
    INTEGRITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INTEGRITY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not res.passed and not allow_leakage:
        raise SystemExit(
            f"[bench] integrity probe FAILED on {domain}/{item_id}: "
            f"leaks={res.leak_matches}; pass --allow-leakage to ignore"
        )
    return record


def run_domain(
    *,
    domain: str,
    arms: tuple[str, ...],
    out_path: Path,
    n: int | None,
    K: int,
    max_tokens: int,
    seed: int,
    embed: Embedder,
    lm: LocalLM | None,
    haiku_lm: HaikuLM | None,
    cost_cap_usd: float | None,
    integrity_probe: IntegrityProbe | None,
    allow_leakage: bool,
) -> int:
    items = _domain_items(domain, n=n)
    rows = _load_existing(out_path)
    scorer = SCORERS[domain]
    print(
        f"[bench] domain={domain}  items={len(items)}  arms={list(arms)}",
        flush=True,
    )
    for i, item in enumerate(items):
        item_id = item["id"]
        item_rows = rows.setdefault(item_id, {"item": item})
        prompt, constraint_text, must_avoid, aspects, retrieval = _build_prompt(
            domain, item
        )
        if integrity_probe is not None and haiku_lm is not None:
            ip_record = _per_item_integrity_probe(
                integrity_probe,
                haiku_lm=haiku_lm,
                domain=domain,
                item_id=item_id,
                allow_leakage=allow_leakage,
            )
            probes_list = item_rows.setdefault("_integrity_probes", [])
            assert isinstance(probes_list, list)
            probes_list.append(
                {
                    "passed": ip_record["passed"],
                    "leak_matches": ip_record["leak_matches"],
                    "env_hash": ip_record["env_hash"],
                    "flags_hash": ip_record["flags_hash"],
                }
            )
        for raw_arm in arms:
            arm = _normalise_arm(raw_arm)
            if arm in item_rows:
                continue
            if (
                cost_cap_usd is not None
                and haiku_lm is not None
                and arm.startswith("haiku")
                and float(haiku_lm.report()["ledger_total_usd"]) >= cost_cap_usd
            ):
                print(
                    f"  [{domain}] {item_id} :: {arm} SKIPPED (cost cap reached)",
                    flush=True,
                )
                item_rows[arm] = {"skipped": True, "reason": "cost_cap"}
                _save(out_path, rows, domain)
                continue
            print(f"  [{domain}] {item_id} :: {arm} ...", flush=True)
            text = ""
            meta: dict[str, Any] = {}
            if arm == "local_bare":
                assert lm is not None
                text, meta = _call_local_bare(
                    prompt, lm=lm, max_tokens=max_tokens, seed=seed + i
                )
            elif arm == "local_cascade":
                assert lm is not None
                text, meta = _call_cascade_arm(
                    prompt=prompt,
                    arm=arm,
                    lm=lm,
                    embed=embed,
                    constraint_text=constraint_text,
                    must_avoid=must_avoid,
                    aspects=aspects,
                    retrieval_set=retrieval,
                    K=K,
                    max_tokens=max_tokens,
                    seed=seed + i,
                    haiku_lm=None,
                )
            elif arm == "haiku_bare":
                assert haiku_lm is not None
                text, meta = _call_haiku_bare(
                    prompt,
                    haiku_lm=haiku_lm,
                    max_tokens=max_tokens,
                    seed=seed + i,
                )
                _snapshot_cost_ledger(haiku_lm)
            elif arm in {
                "haiku_cascade",
                "haiku_bare_2K_scorer",
                "haiku_generic_revise_2pass",
            }:
                assert haiku_lm is not None
                text, meta = _call_cascade_arm(
                    prompt=prompt,
                    arm=arm,
                    lm=haiku_lm,
                    embed=embed,
                    constraint_text=constraint_text,
                    must_avoid=must_avoid,
                    aspects=aspects,
                    retrieval_set=retrieval,
                    K=K,
                    max_tokens=max_tokens,
                    seed=seed + i,
                    haiku_lm=haiku_lm,
                )
                _snapshot_cost_ledger(haiku_lm)
            else:
                raise ValueError(arm)
            if text:
                score = scorer(text, item=item, embed=embed)
                composite = (
                    float(score.composite) if not np.isnan(score.composite) else None
                )
                axes = score.axes
            else:
                composite = None
                axes = {}
            item_rows[arm] = {
                "text": text,
                "axes": axes,
                "composite": composite,
                "n_chars": len(text),
                "n_words": len(text.split()) if text else 0,
                "meta": meta,
            }
            _save(out_path, rows, domain)
        if (
            "haiku_cascade" in item_rows
            and not item_rows["haiku_cascade"].get("skipped")
        ):
            _multiplex_commit_policies(
                domain=domain,
                item=item,
                item_rows=item_rows,
                embed=embed,
                scorer=scorer,
            )
            _save(out_path, rows, domain)
    print(f"[bench] domain={domain} complete -> {out_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help="One or more domain ids",
    )
    parser.add_argument("--n-poetry-gen", type=int, default=15)
    parser.add_argument("--n-poetry-interp", type=int, default=15)
    parser.add_argument("--n-aut", type=int, default=10)
    parser.add_argument("--n-sci-creativity", type=int, default=10)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS_V3),
        help=f"Subset of: {ARMS_V3 + ARMS_LEGACY + tuple(ARM_ALIASES.keys())}",
    )
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--cost-cap-usd",
        type=float,
        default=30.0,
        help="Hard stop on Haiku-arms when ledger exceeds this. 0 disables. "
        "Default 30 = v0.4 powered-pilot cap.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="v0.3 pilot preset: n=5/domain, K=3, all four v0.3 arms, $20 cap.",
    )
    parser.add_argument(
        "--v04-pilot",
        action="store_true",
        help="v0.4 powered-pilot preset (Phase 7): n=20/domain × 4 domains, "
        "all four base arms, K=4, $30 cost cap. The cascade arm's commit-"
        "policy multiplex (always_draft / always_revise / event_gated / "
        "learned_gate + oracle) is computed post-hoc at zero extra Haiku "
        "cost via _multiplex_commit_policies.",
    )
    parser.add_argument(
        "--no-integrity-probe",
        action="store_true",
        help="Skip per-item IntegrityProbe (use only for debugging the driver itself).",
    )
    parser.add_argument(
        "--allow-leakage",
        action="store_true",
        help="Continue even if IntegrityProbe detects leakage; default halts.",
    )
    args = parser.parse_args()

    arms = tuple(_normalise_arm(a) for a in args.arms)
    n_map = {
        "poetry_gen": args.n_poetry_gen,
        "poetry_interp": args.n_poetry_interp,
        "aut": args.n_aut,
        "sci_creativity": args.n_sci_creativity,
    }
    if args.pilot:
        n_map = {
            "poetry_gen": 5,
            "poetry_interp": 5,
            "aut": 5,
            "sci_creativity": 5,
        }
        arms = ARMS_V3
    if args.v04_pilot:
        n_map = {
            "poetry_gen": 20,
            "poetry_interp": 20,
            "aut": 20,
            "sci_creativity": 20,
        }
        arms = ARMS_V3
        if args.K == 4:  # respect explicit user override
            args.K = 4

    embed = Embedder()
    need_local = any(a in arms for a in ARMS_LEGACY)
    need_haiku = any(a.startswith("haiku") for a in arms)
    lm = LocalLM() if need_local else None
    haiku_lm = HaikuLM() if need_haiku else None
    cost_cap = args.cost_cap_usd if args.cost_cap_usd > 0 else None
    integrity_probe = (
        IntegrityProbe()
        if (need_haiku and not args.no_integrity_probe)
        else None
    )

    if integrity_probe is not None and haiku_lm is not None:
        boot = integrity_probe.run(haiku_lm)
        print(
            f"[bench] integrity_probe@boot passed={boot.passed} "
            f"leaks={boot.leak_matches} positive_hint={boot.positive_hint}",
            flush=True,
        )
        if not boot.passed and not args.allow_leakage:
            raise SystemExit(
                "[bench] BOOT integrity probe FAILED; aborting. "
                "Pass --allow-leakage to override."
            )

    for domain in args.domains:
        out_path = args.out_dir / f"{domain}.json"
        run_domain(
            domain=domain,
            arms=arms,
            out_path=out_path,
            n=n_map.get(domain),
            K=args.K,
            max_tokens=args.max_tokens,
            seed=args.seed,
            embed=embed,
            lm=lm,
            haiku_lm=haiku_lm,
            cost_cap_usd=cost_cap,
            integrity_probe=integrity_probe,
            allow_leakage=args.allow_leakage,
        )
    if haiku_lm is not None:
        _snapshot_cost_ledger(haiku_lm)
        rep = haiku_lm.report()
        print(
            f"[bench] haiku cost: ${float(rep['ledger_total_usd']):.4f} over "
            f"{int(rep['ledger_n_calls'])} calls",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
