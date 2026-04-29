#!/usr/bin/env python3
"""Phase 4 gate -- causal vimarsa: event_gated commits + always-shadow revision.

Runs ``haiku_cascade`` with ``commit_policy="event_gated"`` against
``duck_rabbit_textual`` and ``aut_brick`` over multiple seeds. Asserts:

* ``revision_differs_from_draft`` is True on >=80% of paired runs (per-fixture).
* When ``vimarsa_event`` fires, ``state.committed == "revision"``;
  when it does not, ``state.committed == "draft"``. (Commit policy honor: 100%.)
* Both ``surface_draft`` and ``surface_revision`` are populated on every run
  (so H8 stays measurable regardless of commit policy).

Cost envelope: 2 fixtures * 4 seeds * 2K=8 Haiku calls = ~64 Haiku calls
@ ~$0.03/call = ~$2 of additional spend.

Writes ``audit/v0_3_phase4_gate.json`` with the per-run summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pce.cascade import run_cascade  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM  # noqa: E402
from pce.types import Constraint  # noqa: E402

FIXTURES = ("duck_rabbit_textual", "aut_brick")
DEFAULT_SEEDS = (4242, 4243, 4244, 4245)
K = 3
MAX_TOKENS = 220
REV_DIFF_FLOOR = 0.80


def _load_fixture(name: str) -> dict[str, Any]:
    path = REPO_ROOT / "tests" / "fixtures" / f"{name}.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _run_one(haiku: HaikuLM, embed: Embedder, fx: dict[str, Any], seed: int) -> dict[str, Any]:
    constraint = Constraint(
        text=str(fx["constraint_text"]),
        embedding=embed.encode(str(fx["constraint_text"])),
        must_avoid=tuple(fx.get("must_avoid", []) or ()),
    )
    state = run_cascade(
        prompt=str(fx["prompt"]),
        constraint=constraint,
        lm=haiku,
        embed=embed,
        K=K,
        max_tokens=MAX_TOKENS,
        base_seed=seed,
        retrieval_set=list(fx.get("retrieval_set", []) or []),
        aspects=list(fx.get("aspects", []) or []),
        commit_policy="event_gated",
    )
    return {
        "fixture": str(fx["id"]),
        "seed": int(seed),
        "vimarsa_event": bool(state.vimarsa_event),
        "committed": str(state.committed),
        "commit_policy": str(state.commit_policy),
        "revision_differs_from_draft": bool(
            state.audit.get("revision_differs_from_draft", False)
        ),
        "delta_F_draft": float(state.audit.get("delta_F_draft", 0.0)),
        "delta_F_revision": float(state.audit.get("delta_F_revision", 0.0)),
        "surface_draft": str(state.surface_draft or ""),
        "surface_revision": str(state.surface_revision or ""),
        "surface_committed": str(state.surface or ""),
        "draft_present": state.surface_draft is not None,
        "revision_present": state.surface_revision is not None,
    }


def _check_policy_honored(run: dict[str, Any]) -> tuple[bool, str]:
    event = bool(run["vimarsa_event"])
    committed = str(run["committed"])
    expected = "revision" if event else "draft"
    if committed != expected:
        return False, f"event={event} but committed={committed!r}, expected {expected!r}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "audit" / "v0_3_phase4_gate.json")
    parser.add_argument("--cost-cap-usd", type=float, default=10.0)
    args = parser.parse_args()

    cfg = HaikuConfig.from_env()
    cfg = HaikuConfig(
        model=cfg.model,
        cli_bin=cfg.cli_bin,
        timeout_s=cfg.timeout_s,
        use_sdk=False,
        cost_cap_usd=args.cost_cap_usd,
        cli_retry=cfg.cli_retry,
        cli_backoff_s=cfg.cli_backoff_s,
        clean_substrate=True,
        clean_home_root=cfg.clean_home_root,
        system_prompt_override=cfg.system_prompt_override,
    )
    embed = Embedder()
    haiku = HaikuLM(config=cfg, embedder=embed)

    runs: list[dict[str, Any]] = []
    per_fixture: dict[str, dict[str, Any]] = {}
    for fx_name in FIXTURES:
        fx = _load_fixture(fx_name)
        fx_runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            print(f"[phase4_gate] {fx_name} seed={seed}", flush=True)
            r = _run_one(haiku, embed, fx, seed)
            fx_runs.append(r)
            runs.append(r)
        n = len(fx_runs)
        n_diff = sum(1 for r in fx_runs if r["revision_differs_from_draft"])
        n_present_both = sum(1 for r in fx_runs if r["draft_present"] and r["revision_present"])
        policy_failures = []
        for r in fx_runs:
            ok, msg = _check_policy_honored(r)
            if not ok:
                policy_failures.append(msg)
        per_fixture[fx_name] = {
            "n": n,
            "n_revision_differs_from_draft": n_diff,
            "frac_revision_differs": n_diff / n if n else 0.0,
            "n_both_surfaces_present": n_present_both,
            "policy_failures": policy_failures,
            "runs": fx_runs,
        }

    overall = {
        "fixtures": per_fixture,
        "haiku_cost_report": haiku.report(),
        "rev_diff_floor": REV_DIFF_FLOOR,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(overall, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    fails: list[str] = []
    for name, data in per_fixture.items():
        frac = float(data["frac_revision_differs"])
        if frac < REV_DIFF_FLOOR:
            fails.append(
                f"{name}: revision_differs_from_draft = {frac:.2f} < {REV_DIFF_FLOOR}"
            )
        if data["policy_failures"]:
            fails.append(f"{name}: policy honor failures: {data['policy_failures']}")
        if data["n_both_surfaces_present"] != data["n"]:
            fails.append(
                f"{name}: only {data['n_both_surfaces_present']}/{data['n']} runs had both surfaces"
            )

    if fails:
        print("[phase4_gate] FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(
        f"[phase4_gate] OK: revision_differs >= {REV_DIFF_FLOOR} on all fixtures, "
        f"commit policy honored 100%, both surfaces present 100%. "
        f"Spend ${haiku.report()['ledger_total_usd']:.4f} USD."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
