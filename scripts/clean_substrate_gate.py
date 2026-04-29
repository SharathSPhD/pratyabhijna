#!/usr/bin/env python3
"""Phase 2 gate — clean Haiku CLI substrate end-to-end.

Spawns:

* 10 IntegrityProbe runs through fresh subprocesses (cache disabled)
* 50 short Haiku generation calls

…and checks that all 60 responses are leakage-free against
`pce.substrate.integrity.LEAKAGE_REGEX`.

Writes `audit/v0_3_phase2_gate.json` with per-call probe outcomes and the
leakage scan summary. Exits 0 only if all checks pass.

Cost guard: capped to ~$1 (60 short calls @ ~$0.001 each).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pce.substrate.haiku_lm import HaikuConfig, HaikuLM
from pce.substrate.integrity import IntegrityProbe, _scan_leakage

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "audit" / "v0_3_phase2_gate.json"

PROBE_RUNS = 10
GENERATION_CALLS = 50
GENERATION_PROMPT = (
    "Compose one short, surprising sentence about clouds at dawn. "
    "Reply with the sentence only, no preamble."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--probe-runs", type=int, default=PROBE_RUNS)
    parser.add_argument("--generation-calls", type=int, default=GENERATION_CALLS)
    parser.add_argument("--cost-cap-usd", type=float, default=2.0)
    args = parser.parse_args()

    cfg = HaikuConfig.from_env()
    cfg = HaikuConfig(
        model=cfg.model,
        cli_bin=cfg.cli_bin,
        timeout_s=cfg.timeout_s,
        use_sdk=False,
        cost_cap_usd=args.cost_cap_usd,
        cli_retry=0,  # gate must surface every flake honestly
        cli_backoff_s=cfg.cli_backoff_s,
        clean_substrate=True,
        clean_home_root=cfg.clean_home_root,
        system_prompt_override=cfg.system_prompt_override,
    )
    lm = HaikuLM(config=cfg)
    probe = IntegrityProbe()

    probe_results: list[dict[str, object]] = []
    for i in range(args.probe_runs):
        result = probe.run(lm, force=True)  # always re-probe; gate measures real freshness
        probe_results.append({
            "i": i,
            "passed": result.passed,
            "leak_matches": result.leak_matches,
            "positive_hint": result.positive_hint,
            "response_excerpt": result.response[:200],
        })

    gen_results: list[dict[str, object]] = []
    for i in range(args.generation_calls):
        cand = lm.generate(GENERATION_PROMPT, max_tokens=64, sampler={"tau": 0.9}, seed=i)
        leaks = _scan_leakage(cand.text)
        gen_results.append({
            "i": i,
            "leak_free": len(leaks) == 0,
            "leaks": leaks,
            "text_excerpt": cand.text[:200],
        })

    n_probe_pass = sum(1 for r in probe_results if r["passed"])
    n_gen_clean = sum(1 for r in gen_results if r["leak_free"])

    summary = {
        "n_probe_runs": args.probe_runs,
        "n_probe_pass": n_probe_pass,
        "n_generation_calls": args.generation_calls,
        "n_generation_clean": n_gen_clean,
        "probe_results": probe_results,
        "generation_results": gen_results,
        "lm_report": lm.report(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    pass_threshold_probe = args.probe_runs  # all 10/10
    pass_threshold_gen = args.generation_calls  # all 50/50

    failures: list[str] = []
    if n_probe_pass < pass_threshold_probe:
        failures.append(
            f"IntegrityProbe: {n_probe_pass}/{args.probe_runs} clean (need {pass_threshold_probe})"
        )
    if n_gen_clean < pass_threshold_gen:
        failures.append(
            f"Generation leakage scan: {n_gen_clean}/{args.generation_calls} clean (need {pass_threshold_gen})"
        )

    if failures:
        print("[clean_substrate_gate] FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"[clean_substrate_gate] OK: {n_probe_pass}/{args.probe_runs} probes clean, "
        f"{n_gen_clean}/{args.generation_calls} generations clean. "
        f"Total spend ${lm.report()['ledger_total_usd']:.4f} USD."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
