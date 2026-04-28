#!/usr/bin/env python3
"""Phase 4 prove-gate: validate two cases across all four arms before benchmark.

Per the user's frozen scope ("take one case and prove/validate thoroughly,
debug, correct and then proceed for full benchmark"), the prove-gate runs
``duck_rabbit_textual`` and ``aut_brick`` across:

* ``local_bare``    - Qwen2-1.5B raw ``LM.generate``
* ``local_cascade`` - Qwen2-1.5B through ``run_cascade``
* ``haiku_bare``    - Haiku via ``HaikuLM.generate``
* ``haiku_cascade`` - Haiku through ``run_cascade``

For each (case, arm) we write the surface, draft, revision, brief, and
diagnostic to ``audit/prove_gate/<case>/<arm>/result.json`` and assert the
fixture's ``expected_signals``.

Exit codes:
* 0 - all gates passed.
* 1 - a hard gate failed (vimarsa never fires, revision == draft, etc).

Cost envelope: ~ $0.20-0.40 of Haiku (4 cascade calls + 4 bare calls,
K=3, two-pass-always means each cascade call is 2*K = 6 Haiku
generations; bare is 1 generation). Roughly 16 Haiku calls total
@ ~$0.02/call cached = ~$0.32. Well within the $15 pilot envelope.

Usage::

    uv run python scripts/prove_gate.py [--strict]

``--strict`` returns non-zero on any expected-signal failure; without
``--strict`` it logs failures but still exits 0 so the report can be
inspected.
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
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.substrate.lm_protocol import LMProtocol  # noqa: E402
from pce.types import Constraint  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
AUDIT_DIR = REPO_ROOT / "audit" / "prove_gate"
ARMS = ("local_bare", "local_cascade", "haiku_bare", "haiku_cascade")
K = 3
MAX_TOKENS = 220


def _load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _bare_call_local(lm: LocalLM, prompt: str, *, seed: int) -> tuple[str, dict[str, Any]]:
    t = time.time()
    out = lm.generate(prompt, max_tokens=MAX_TOKENS, sampler={"tau": 0.9, "top_p": 0.95}, seed=seed)
    return out.text, {"elapsed_s": float(time.time() - t)}


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
    lm: LMProtocol,
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
        lm=lm,
        embed=embed,
        K=K,
        max_tokens=MAX_TOKENS,
        base_seed=seed,
        retrieval_set=list(fixture.get("retrieval_set", []) or []),
        aspects=list(fixture.get("aspects", []) or []),
    )
    diag = state.audit.get("vimarsa_diag_revision") or state.audit.get("vimarsa_diag", {})
    return state.surface or "", {
        "elapsed_s": float(time.time() - t),
        "two_pass": bool(state.audit.get("two_pass", False)),
        "revision_differs_from_draft": bool(state.audit.get("revision_differs_from_draft", False)),
        "vimarsa_event_draft": bool(state.vimarsa_event_draft),
        "vimarsa_event_revision": bool(state.vimarsa_event),
        "vimarsa_brief": str(state.vimarsa_brief or ""),
        "novelty_revision": float(state.vimarsa_novelty),
        "vimarsa_diag_revision": dict(diag) if isinstance(diag, dict) else {},
        "delta_F_draft": float(state.audit.get("delta_F_draft", float("nan"))),
        "delta_F_revision": float(state.audit.get("delta_F_revision", float("nan"))),
        "surface_draft": str(state.surface_draft or ""),
        "surface_revision": str(state.surface_revision or ""),
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
) -> tuple[bool, list[str]]:
    """Apply the fixture's expected_signals; return (passed, failure_reasons)."""
    sig = dict(fixture.get("expected_signals", {}))
    fails: list[str] = []
    aspects = list(fixture.get("aspects", []) or [])
    retrieval = list(fixture.get("retrieval_set", []) or [])
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
    floor_aspect = float(sig.get("aspect_max_cosine_floor", 0.0))
    floor_nov = float(sig.get("novelty_floor", 0.0))
    if floor_aspect > 0.0 and aspects:
        any_arm_clears = any(
            float(arm_results.get(a, {}).get("aspect_max_cosine_post", 0.0)) >= floor_aspect
            for a in ARMS
        )
        if not any_arm_clears:
            fails.append(
                f"no arm clears aspect_max_cosine_floor={floor_aspect}"
            )
    if floor_nov > 0.0:
        any_arm_clears_nov = any(
            float(arm_results.get(a, {}).get("novelty_post", 0.0)) >= floor_nov
            for a in ARMS
        )
        if not any_arm_clears_nov:
            fails.append(f"no arm clears novelty_floor={floor_nov}")
    if sig.get("vimarsa_event_at_least_one_arm"):
        any_event = any(
            bool(arm_results.get(a, {}).get("vimarsa_event_revision", False))
            or bool(arm_results.get(a, {}).get("vimarsa_event_draft", False))
            for a in ("local_cascade", "haiku_cascade")
        )
        if not any_event:
            fails.append("vimarsa never fired across cascade arms")
    if sig.get("revision_differs_from_draft"):
        any_diff = any(
            bool(arm_results.get(a, {}).get("revision_differs_from_draft", False))
            for a in ("local_cascade", "haiku_cascade")
        )
        if not any_diff:
            fails.append("revision == draft on all cascade arms")
    if sig.get("haiku_cascade_differs_from_haiku_bare"):
        cb = str(arm_results.get("haiku_cascade", {}).get("surface", ""))
        bb = str(arm_results.get("haiku_bare", {}).get("surface", ""))
        if cb.strip() and bb.strip() and cb.strip() == bb.strip():
            fails.append("haiku_cascade surface == haiku_bare surface")
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
    local_lm: LocalLM,
    haiku_lm: HaikuLM,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    print(f"\n[gate] === {case} ===", flush=True)
    arm_results: dict[str, dict[str, Any]] = {}
    prompt = str(fixture["prompt"])
    # local_bare
    print("[gate]   local_bare ...", flush=True)
    text, meta = _bare_call_local(local_lm, prompt, seed=seed)
    arm_results["local_bare"] = {"arm": "local_bare", "surface": text, **meta}
    _write_arm_audit(case, "local_bare", arm_results["local_bare"])
    # local_cascade
    print("[gate]   local_cascade ...", flush=True)
    text, meta = _cascade_call(local_lm, embed, fixture, seed=seed)
    arm_results["local_cascade"] = {"arm": "local_cascade", "surface": text, **meta}
    _write_arm_audit(case, "local_cascade", arm_results["local_cascade"])
    # haiku_bare
    print("[gate]   haiku_bare ...", flush=True)
    text, meta = _bare_call_haiku(haiku_lm, prompt, seed=seed)
    arm_results["haiku_bare"] = {"arm": "haiku_bare", "surface": text, **meta}
    _write_arm_audit(case, "haiku_bare", arm_results["haiku_bare"])
    # haiku_cascade
    print("[gate]   haiku_cascade ...", flush=True)
    text, meta = _cascade_call(haiku_lm, embed, fixture, seed=seed)
    arm_results["haiku_cascade"] = {"arm": "haiku_cascade", "surface": text, **meta}
    _write_arm_audit(case, "haiku_cascade", arm_results["haiku_cascade"])

    passed, fails = _check_signals(case, fixture, arm_results, embed)
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
        print("[gate] claude CLI not found - haiku arms will be skipped",
              file=sys.stderr)
        return 2
    embed = Embedder()
    print("[gate] loading LocalLM...", flush=True)
    local_lm = LocalLM()
    print("[gate] starting HaikuLM...", flush=True)
    haiku_lm = HaikuLM(config=HaikuConfig.from_env(), embedder=embed)
    all_passed = True
    summaries: list[dict[str, Any]] = []
    for case in args.cases:
        fixture = _load_fixture(case)
        passed, summary = _run_case(
            case, fixture, embed=embed, local_lm=local_lm, haiku_lm=haiku_lm,
            seed=args.seed,
        )
        summaries.append(summary)
        all_passed = all_passed and passed
    cost_report = haiku_lm.report()
    overall: dict[str, Any] = {
        "passed": bool(all_passed),
        "case_summaries": summaries,
        "haiku_cost_report": cost_report,
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
