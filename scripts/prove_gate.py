#!/usr/bin/env python3
"""Phase 5 prove-gate v0.3: validate two cases against the haiku_cascade arm.

Per the user's frozen v0.3 scope ("no need to run local llm arm or sonnet …
the same benchmark sample as v0.2 is the scope"), the prove-gate now runs
*only* the two production Haiku arms on the two prove-gate fixtures:

* ``haiku_bare``    - one-shot Haiku via :class:`HaikuLM.generate` (parity sampler)
* ``haiku_cascade`` - Haiku through :func:`pce.cascade.run_cascade` with
                      ``commit_policy="event_gated"`` (always-shadow revision)

The local Qwen2-1.5B arm has been dropped from the gate; it lives only as a
ceiling reference in legacy v0.2 audits. The two new control arms
(``haiku_bare_2K_scorer``, ``haiku_generic_revise_2pass``) are exercised by
the Phase 7 benchmark driver, not the prove-gate.

For every Haiku call the gate now verifies the *clean inner-subprocess
substrate* (ADR-001):

* :class:`pce.substrate.integrity.IntegrityProbe` is run once at boot and the
  result must be ``passed=True``; cached for the rest of the run.
* :data:`pce.substrate.integrity.LEAKAGE_REGEX` is applied (with the same
  negation-context filter the probe uses) to every ``haiku_bare`` and
  ``haiku_cascade`` surface (and, for the cascade, both shadow draft and
  shadow revision). Any leak match fails the case.

For ``haiku_cascade`` it additionally asserts (ADR-002 / ADR-003):

* ``haiku_cascade.delta_F_draft`` is non-degenerate (``|ΔF| >= delta_F_floor``)
  whenever the fixture supplies aspects, so the BMR aspect-conditioned
  posterior is doing real work, not collapsing to the prior.
* ``haiku_cascade.vimarsa_event_draft`` fires iff the fixture sets
  ``haiku_cascade_vimarsa_event_required=true`` (duck-rabbit must fire;
  aut_brick must NOT fire because it has no aspects).
* When commit policy committed *revision*, ``revision_differs_from_draft``
  is true (the revision pass actually changed the surface).

Per-arm payload is written to ``audit/prove_gate/<case>/<arm>/result.json``;
overall summary lands at ``audit/prove_gate/overall.json``.

Exit codes:
* 0 - all gates passed.
* 1 - a hard gate failed (and ``--strict`` was set).
* 2 - claude CLI not present.

Cost envelope: ~$0.10-$0.20 of Haiku
(2 cases × 2 arms; cascade = 2 passes × K=3 = 6 generations + 1 probe).

Usage::

    uv run python scripts/prove_gate.py [--strict]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pce.cascade import run_cascade  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM  # noqa: E402
from pce.substrate.integrity import IntegrityProbe, _scan_leakage  # noqa: E402
from pce.types import Constraint  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
AUDIT_DIR = REPO_ROOT / "audit" / "prove_gate"
ARMS = ("haiku_bare", "haiku_cascade")
K = 3
MAX_TOKENS = 220


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _bare_call_haiku(haiku: HaikuLM, prompt: str, *, seed: int) -> tuple[str, dict[str, Any]]:
    t = time.time()
    out = haiku.generate(
        prompt,
        max_tokens=MAX_TOKENS,
        sampler={"tau": 0.9, "top_p": 0.95, "top_k": 50.0},
        seed=seed,
    )
    return out.text, {"elapsed_s": float(time.time() - t)}


def _cascade_call(
    haiku: HaikuLM,
    embed: Embedder,
    fixture: dict[str, Any],
    *,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    t = time.time()
    constraint = Constraint(
        text=str(fixture["constraint_text"]),
        embedding=embed.encode(str(fixture["constraint_text"])),
        must_avoid=tuple(fixture.get("must_avoid", []) or ()),
    )
    state = run_cascade(
        prompt=str(fixture["prompt"]),
        constraint=constraint,
        lm=haiku,
        embed=embed,
        K=K,
        max_tokens=MAX_TOKENS,
        base_seed=seed,
        retrieval_set=list(fixture.get("retrieval_set", []) or []),
        aspects=list(fixture.get("aspects", []) or []),
        commit_policy="event_gated",
    )
    diag = state.audit.get("vimarsa_diag_revision") or state.audit.get("vimarsa_diag", {})
    return state.surface or "", {
        "elapsed_s": float(time.time() - t),
        "two_pass": bool(state.audit.get("two_pass", False)),
        "revision_differs_from_draft": bool(state.audit.get("revision_differs_from_draft", False)),
        "vimarsa_event_draft": bool(state.vimarsa_event_draft),
        "vimarsa_event_revision": bool(state.audit.get("vimarsa_event_revision", False)),
        "vimarsa_brief": str(state.vimarsa_brief or ""),
        "novelty_revision": float(state.vimarsa_novelty),
        "vimarsa_diag_revision": dict(diag) if isinstance(diag, dict) else {},
        "delta_F_draft": float(state.audit.get("delta_F_draft", float("nan"))),
        "delta_F_revision": float(state.audit.get("delta_F_revision", float("nan"))),
        "delta_F": float(state.audit.get("delta_F", float("nan"))),
        "surface_draft": str(state.surface_draft or ""),
        "surface_revision": str(state.surface_revision or ""),
        "committed": str(state.committed),
        "commit_policy": str(state.commit_policy),
        "n_storehouse_patterns": int(state.audit.get("n_storehouse_patterns", 0)),
        "budget_ledger": dict(state.audit.get("budget_ledger", {}) or {}),
    }


def _aspect_max_cosine(text: str, aspects: list[str], embed: Embedder) -> float:
    if not aspects or not text.strip():
        return 0.0
    s = embed.encode(text)
    a = embed.encode(aspects)
    if a.ndim == 1:
        a = a[None, :]
    sims = a @ s
    return float(sims.max())


def _novelty(text: str, retrieval_set: list[str], embed: Embedder) -> float:
    if not retrieval_set or not text.strip():
        return 1.0
    s = embed.encode(text)
    r = embed.encode(retrieval_set)
    if r.ndim == 1:
        r = r[None, :]
    sims = r @ s
    return float(max(0.0, 1.0 - float(sims.max())))


def _count_distinct_uses(text: str) -> int:
    """Count distinct lines (AUT proxy for response variety)."""
    lines = [ln.strip(" -*0123456789.") for ln in text.splitlines() if ln.strip()]
    return len({ln.lower() for ln in lines if len(ln) > 3})


def _leakage_check(label: str, text: str) -> list[str]:
    """Return list of leak matches in ``text`` (with negation context filter)."""
    leaks: list[str] = list(_scan_leakage(text or ""))
    if leaks:
        print(f"[gate]   LEAK[{label}]: {leaks[:3]}", file=sys.stderr, flush=True)
    return leaks


def _write_arm_audit(case: str, arm: str, payload: dict[str, Any]) -> Path:
    out = AUDIT_DIR / case / arm
    out.mkdir(parents=True, exist_ok=True)
    p = out / "result.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _check_signals(
    case: str,
    fixture: dict[str, Any],
    arm_results: dict[str, dict[str, Any]],
    embed: Embedder,
    *,
    integrity_passed: bool,
    integrity_response: str,
) -> tuple[bool, list[str]]:
    """Apply v0.3 fixture's expected_signals; return (passed, failure_reasons)."""
    sig = dict(fixture.get("expected_signals", {}))
    fails: list[str] = []
    aspects = list(fixture.get("aspects", []) or [])
    retrieval = list(fixture.get("retrieval_set", []) or [])

    # ---- per-arm post-hoc embedding metrics ---------------------------
    for arm in ARMS:
        if arm not in arm_results:
            continue
        text = str(arm_results[arm].get("surface", ""))
        if not text.strip():
            fails.append(f"{arm}: empty surface")
        amx = _aspect_max_cosine(text, aspects, embed)
        nov = _novelty(text, retrieval, embed)
        arm_results[arm]["aspect_max_cosine_post"] = amx
        arm_results[arm]["novelty_post"] = nov

    # ---- v0.3 NEW: leakage regex on every Haiku surface ----------------
    if sig.get("no_leakage", True):
        for arm in ARMS:
            r = arm_results.get(arm, {})
            text = str(r.get("surface", ""))
            leaks = _leakage_check(f"{arm}.surface", text)
            r["leak_matches"] = leaks
            if leaks:
                fails.append(f"{arm}: leakage in surface: {leaks[:3]}")
        # Also probe cascade shadow draft/revision since those traversed
        # the same inner subprocess.
        for sub in ("surface_draft", "surface_revision"):
            cas = arm_results.get("haiku_cascade", {})
            text = str(cas.get(sub, ""))
            if text.strip():
                leaks = _leakage_check(f"haiku_cascade.{sub}", text)
                cas[f"leak_matches_{sub}"] = leaks
                if leaks:
                    fails.append(f"haiku_cascade.{sub}: leakage: {leaks[:3]}")

    # ---- v0.3 NEW: integrity probe must pass ---------------------------
    if sig.get("integrity_probe_must_pass", True):
        if not integrity_passed:
            fails.append(
                f"IntegrityProbe failed at boot: {integrity_response[:160]!r}"
            )

    # ---- v0.2 carry-over: post-hoc cosine / novelty floors -------------
    floor_aspect = float(sig.get("aspect_max_cosine_floor", 0.0))
    floor_nov = float(sig.get("novelty_floor", 0.0))
    if floor_aspect > 0.0 and aspects:
        any_arm_clears = any(
            float(arm_results.get(a, {}).get("aspect_max_cosine_post", 0.0)) >= floor_aspect
            for a in ARMS
        )
        if not any_arm_clears:
            fails.append(f"no arm clears aspect_max_cosine_floor={floor_aspect}")
    if floor_nov > 0.0:
        any_arm_clears_nov = any(
            float(arm_results.get(a, {}).get("novelty_post", 0.0)) >= floor_nov
            for a in ARMS
        )
        if not any_arm_clears_nov:
            fails.append(f"no arm clears novelty_floor={floor_nov}")

    # ---- v0.3 NEW: haiku_cascade-specific assertions -------------------
    cas = arm_results.get("haiku_cascade", {})

    # vimarsa_event firing requirement (per fixture)
    if sig.get("haiku_cascade_vimarsa_event_required", False):
        if not bool(cas.get("vimarsa_event_draft", False)):
            fails.append(
                "haiku_cascade.vimarsa_event_draft did not fire "
                "(fixture requires it)"
            )
    # AUT-style fixtures explicitly forbid the event from firing
    elif "haiku_cascade_vimarsa_event_required" in sig and bool(
        cas.get("vimarsa_event_draft", False)
    ):
        fails.append(
            "haiku_cascade.vimarsa_event_draft fired but fixture set "
            "haiku_cascade_vimarsa_event_required=false"
        )

    # ΔF floor: only enforce when fixture supplies aspects (otherwise
    # ΔF is structurally 0 and a floor would be meaningless).
    delta_F_floor = float(sig.get("delta_F_floor", 0.0))
    if delta_F_floor > 0.0 and aspects:
        df_d = float(cas.get("delta_F_draft", 0.0))
        if abs(df_d) < delta_F_floor:
            fails.append(
                f"haiku_cascade.delta_F_draft degenerate: |{df_d:.4f}| < "
                f"{delta_F_floor}"
            )

    # revision_differs_from_draft: only required when commit committed
    # the revision; aut_brick correctly commits draft.
    if sig.get("revision_differs_from_draft", False):
        committed = str(cas.get("committed", ""))
        if committed == "revision" and not bool(
            cas.get("revision_differs_from_draft", False)
        ):
            fails.append(
                "haiku_cascade committed revision but revision == draft "
                "(no architectural delta)"
            )

    # bare ≠ cascade textual identity check
    if sig.get("haiku_cascade_differs_from_haiku_bare", False):
        cb = str(cas.get("surface", ""))
        bb = str(arm_results.get("haiku_bare", {}).get("surface", ""))
        if cb.strip() and bb.strip() and cb.strip() == bb.strip():
            fails.append("haiku_cascade surface == haiku_bare surface")

    # AUT distinct-uses floor
    if "n_distinct_uses_floor" in sig:
        floor = int(sig["n_distinct_uses_floor"])
        for a in ("haiku_bare", "haiku_cascade"):
            text = str(arm_results.get(a, {}).get("surface", ""))
            n = _count_distinct_uses(text)
            arm_results[a]["n_distinct_uses"] = n
            if n < floor:
                fails.append(f"{a}: only {n} distinct uses (< {floor})")

    return (len(fails) == 0), fails


def _run_case(
    case: str,
    fixture: dict[str, Any],
    *,
    embed: Embedder,
    haiku_lm: HaikuLM,
    integrity_passed: bool,
    integrity_response: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    print(f"\n[gate] === {case} ===", flush=True)
    arm_results: dict[str, dict[str, Any]] = {}
    prompt = str(fixture["prompt"])

    print("[gate]   haiku_bare ...", flush=True)
    text, meta = _bare_call_haiku(haiku_lm, prompt, seed=seed)
    arm_results["haiku_bare"] = {"arm": "haiku_bare", "surface": text, **meta}
    _write_arm_audit(case, "haiku_bare", arm_results["haiku_bare"])

    print("[gate]   haiku_cascade ...", flush=True)
    text, meta = _cascade_call(haiku_lm, embed, fixture, seed=seed)
    arm_results["haiku_cascade"] = {"arm": "haiku_cascade", "surface": text, **meta}
    _write_arm_audit(case, "haiku_cascade", arm_results["haiku_cascade"])

    passed, fails = _check_signals(
        case,
        fixture,
        arm_results,
        embed,
        integrity_passed=integrity_passed,
        integrity_response=integrity_response,
    )
    summary = {
        "case": case,
        "passed": bool(passed),
        "failures": fails,
        "arm_signals": {
            a: {
                k: arm_results[a].get(k)
                for k in (
                    "elapsed_s",
                    "two_pass",
                    "revision_differs_from_draft",
                    "vimarsa_event_draft",
                    "vimarsa_event_revision",
                    "novelty_revision",
                    "novelty_post",
                    "aspect_max_cosine_post",
                    "n_distinct_uses",
                    "delta_F_draft",
                    "delta_F_revision",
                    "delta_F",
                    "committed",
                    "commit_policy",
                    "leak_matches",
                    "leak_matches_surface_draft",
                    "leak_matches_surface_revision",
                )
                if k in arm_results[a]
            }
            for a in ARMS
        },
    }
    (AUDIT_DIR / case / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[gate]   passed={passed}  failures={fails}", flush=True)
    return passed, summary


def _check_claude_cli() -> bool:
    try:
        proc = subprocess.run(["claude", "--version"], capture_output=True, timeout=10)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="return non-zero if any signal fails")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["duck_rabbit_textual", "aut_brick"],
    )
    args = parser.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if not _check_claude_cli():
        print("[gate] claude CLI not found - cannot run Haiku gates",
              file=sys.stderr)
        return 2

    embed = Embedder()
    print("[gate] starting clean-substrate HaikuLM ...", flush=True)
    haiku_lm = HaikuLM(config=HaikuConfig.from_env(), embedder=embed)

    print("[gate] running IntegrityProbe on inner subprocess ...", flush=True)
    probe = IntegrityProbe()
    probe_result = probe.run(haiku_lm)
    print(
        f"[gate]   integrity_probe.passed={probe_result.passed} "
        f"leak_matches={probe_result.leak_matches} "
        f"positive_hint={probe_result.positive_hint}",
        flush=True,
    )
    integrity_payload = {
        "passed": probe_result.passed,
        "response": probe_result.response,
        "leak_matches": probe_result.leak_matches,
        "positive_hint": probe_result.positive_hint,
        "env_hash": probe_result.env_hash,
        "flags_hash": probe_result.flags_hash,
        "probe_at_iso": probe_result.probe_at_iso,
        "cost_usd": probe_result.cost_usd,
    }
    (AUDIT_DIR / "integrity_probe.json").write_text(
        json.dumps(integrity_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_passed = True
    summaries: list[dict[str, Any]] = []
    for case in args.cases:
        fixture = _load_fixture(case)
        passed, summary = _run_case(
            case,
            fixture,
            embed=embed,
            haiku_lm=haiku_lm,
            integrity_passed=probe_result.passed,
            integrity_response=probe_result.response,
            seed=args.seed,
        )
        summaries.append(summary)
        all_passed = all_passed and passed
    cost_report = haiku_lm.report()
    overall: dict[str, Any] = {
        "passed": bool(all_passed),
        "case_summaries": summaries,
        "haiku_cost_report": cost_report,
        "integrity_probe": integrity_payload,
    }
    overall_path = AUDIT_DIR / "overall.json"
    overall_path.write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[gate] overall passed={all_passed}  haiku_cost="
          f"{float(cost_report['ledger_total_usd']):.4f} USD over "
          f"{int(cost_report['ledger_n_calls'])} calls",
          flush=True)
    if not all_passed:
        print("[gate] failures by case:", file=sys.stderr)
        for s in summaries:
            if s["failures"]:
                print(f"  - {s['case']}: {s['failures']}", file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
