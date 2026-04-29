#!/usr/bin/env python3
"""v0.3 plugin smoke test: in-process MCP tool invocations.

Loads ``plugin/mcp/server.py``, then calls every MCP tool through the
FastMCP ``_tool_manager.call_tool`` async API. Records each result to
``audit/phase6_v0.3/smoke.jsonl`` and updates ``audit/phase6_v0.3/smoke.json``
with pass/fail counts.

v0.3 adds two new tools (``haiku_clean_substrate_probe`` and
``hopfield_state``) and exercises the new ``pce_cascade`` arm enum
(``haiku_bare_2K`` and ``haiku_generic_revise``) so the smoke run covers
the full v0.3 surface area of 19 tools.

Skipping live LM-touching tools is opt-in (they take ~30s to load Qwen2-1.5B).
``--with-haiku`` opts in to the Haiku-touching probes (each call costs
roughly $0.02 USD; the four-arm cascade probe set is roughly $0.30).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_MCP = REPO_ROOT / "plugin" / "mcp"
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(PLUGIN_MCP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import server  # noqa: E402

LM_TOUCHING = {"cit", "iccha", "cascade", "pce_cascade"}
HAIKU_TOUCHING = {
    "haiku_bare",
    "pce_cascade_haiku",
    "pce_cascade_haiku_bare_2k",
    "pce_cascade_haiku_generic_revise",
    "haiku_clean_substrate_probe",
}


PROBES: list[dict[str, object]] = [
    # Pure-numpy / embedding-only tools first.
    {
        "name": "report",
        "args": {},
    },
    {
        "name": "ananda",
        "args": {
            "candidate_text": "a robin sings at sunrise",
            "constraint_text": "morning birdsong",
        },
    },
    {
        "name": "apohana",
        "args": {
            "candidate_texts": [
                "a robin sings at sunrise",
                "tax brackets in OECD nations are usually piecewise-linear",
            ],
            "constraint_text": "morning birdsong",
            "must_avoid": ["a busy market square at noon"],
        },
    },
    {
        "name": "jnana",
        "args": {
            "candidate_texts": ["a", "b", "c", "d"],
            "apoha_scores": [0.2, 0.1, 0.9, 0.0],
            "ananda_scores": [0.3, 0.2, 0.95, 0.1],
            "reduction_target": "halve",
        },
    },
    {
        "name": "kriya",
        "args": {
            "selected_text": "the river of time runs both ways",
            "render_mode": "verbatim",
        },
    },
    {
        "name": "vimarsa",
        "args": {
            "prompt": "describe time",
            "surface": "the river is a clock and a clock is a river",
            "retrieval_set": ["the cat sat on the mat", "two plus two equals four"],
            "aspects": ["a flowing river that measures time", "a clock face that ripples like water"],
            "ananda_score": 0.8,
            "aspect_cosine_hit": 0.30,
        },
    },
    {
        "name": "hopfield_store",
        "args": {"text": "a small bird sings at dawn"},
    },
    {
        "name": "hopfield_recall",
        "args": {"cue_text": "morning birdsong"},
    },
    {
        "name": "consolidate_sws",
        "args": {
            "trace_texts": [
                "the cat sat on the mat",
                "the dog lay by the fire",
                "a sparrow flew to the eaves",
                "two plus two equals four",
            ],
            "n_centroids": 2,
            "n_iter": 10,
        },
    },
    {
        "name": "consolidate_rem",
        "args": {"n_steps": 8, "temperature": 1.5, "seed": 1},
    },
    {
        "name": "consolidate_cycle",
        "args": {
            "trace_texts": ["a robin", "a sparrow", "a cardinal"],
            "sws_centroids": 2,
            "rem_steps": 6,
        },
    },
    {
        "name": "reset_state",
        "args": {},
    },
    # LM-touching probes last so the server stays warm for benchmarks.
    {
        "name": "cit",
        "args": {"prompt": "Write one English sentence about rain:\n", "max_tokens": 12, "seed": 7},
    },
    {
        "name": "iccha",
        "args": {
            "prompt": "Compose a short poem.\n",
            "constraint_text": "a haiku about autumn leaves",
            "K": 3,
            "max_tokens": 20,
            "base_seed": 11,
        },
    },
    {
        "name": "cascade",
        "args": {
            "prompt": "Compose a short poem.\n",
            "constraint_text": "a haiku about autumn leaves",
            "must_avoid": ["a busy city street"],
            "aspects": ["leaves spinning in wind", "the smell of decay"],
            "retrieval_set": ["raindrops on a tin roof"],
            "K": 4,
            "max_tokens": 20,
            "base_seed": 42,
            "bypass_vimarsa": True,
        },
    },
    # v0.2 arm-switchable cascade. Bypass vimarsa to keep this fast (single
    # pass instead of two); the two-pass-always semantics are exercised by
    # the cascade test suite and the prove-gate.
    {
        "name": "pce_cascade",
        "args": {
            "prompt": "Compose a short poem.\n",
            "constraint_text": "a haiku about autumn leaves",
            "arm": "local",
            "must_avoid": ["a busy city street"],
            "aspects": ["leaves spinning in wind", "the smell of decay"],
            "retrieval_set": ["raindrops on a tin roof"],
            "K": 4,
            "max_tokens": 20,
            "base_seed": 42,
            "bypass_vimarsa": True,
        },
    },
    {
        "name": "hopfield_state",
        "args": {"last_n": 3},
    },
    # Haiku-touching probes (opt-in via --with-haiku; each call costs ~$0.02).
    {
        "name": "haiku_clean_substrate_probe",
        "args": {"force": True},
    },
    {
        "name": "haiku_bare",
        "args": {
            "prompt": "In one sentence, name two animals one might see in a duck-rabbit illusion.",
            "max_tokens": 64,
            "seed": 0,
        },
    },
    {
        "name": "pce_cascade_haiku",
        "real_name": "pce_cascade",
        "args": {
            "prompt": "In one sentence, name two animals one might see in a duck-rabbit illusion.",
            "constraint_text": "name two animals visible in an ambiguous figure",
            "arm": "haiku",
            "must_avoid": ["a single literal description of a duck"],
            "aspects": ["a duck with an upward beak", "a rabbit with backward ears"],
            "K": 3,
            "max_tokens": 80,
            "base_seed": 7,
            "commit_policy": "event_gated",
        },
    },
    # v0.3 control arm 1: best-of-K=2K bare scorer.
    {
        "name": "pce_cascade_haiku_bare_2k",
        "real_name": "pce_cascade",
        "args": {
            "prompt": "In one sentence, name two animals one might see in a duck-rabbit illusion.",
            "constraint_text": "name two animals visible in an ambiguous figure",
            "arm": "haiku_bare_2K",
            "aspects": ["a duck with an upward beak", "a rabbit with backward ears"],
            "K": 2,
            "max_tokens": 80,
            "base_seed": 7,
        },
    },
    # v0.3 control arm 2: 2-pass with generic creative-revise brief.
    {
        "name": "pce_cascade_haiku_generic_revise",
        "real_name": "pce_cascade",
        "args": {
            "prompt": "In one sentence, name two animals one might see in a duck-rabbit illusion.",
            "constraint_text": "name two animals visible in an ambiguous figure",
            "arm": "haiku_generic_revise",
            "aspects": ["a duck with an upward beak", "a rabbit with backward ears"],
            "K": 2,
            "max_tokens": 80,
            "base_seed": 7,
        },
    },
]


async def _call_one(name: str, args: dict[str, object]) -> tuple[bool, dict[str, object]]:
    t0 = time.time()
    try:
        result = await server.mcp._tool_manager.call_tool(name, args, convert_result=False)
        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif isinstance(result, dict):
            payload = result
        else:
            payload = {"value": str(result)}
        return True, {"ok": True, "elapsed_s": time.time() - t0, "result": payload}
    except Exception as e:
        return False, {
            "ok": False,
            "elapsed_s": time.time() - t0,
            "error": str(e),
            "trace": traceback.format_exc()[-2000:],
        }


async def _run(
    skip_lm: bool, with_haiku: bool, out_jsonl: Path, out_json: Path
) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pass_count = 0
    fail_count = 0
    skipped = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for probe in PROBES:
            name = str(probe["name"])
            real_name = str(probe.get("real_name") or name)
            args = probe["args"] if isinstance(probe.get("args"), dict) else {}
            assert isinstance(args, dict)
            if skip_lm and name in LM_TOUCHING:
                skipped += 1
                continue
            if not with_haiku and name in HAIKU_TOUCHING:
                skipped += 1
                continue
            print(f"[smoke] -> {name}", flush=True)
            ok, payload = await _call_one(real_name, args)
            elapsed_val = payload.get('elapsed_s', 0.0)
            elapsed = float(elapsed_val) if isinstance(elapsed_val, (int, float)) else 0.0
            print(f"  {'PASS' if ok else 'FAIL'}  ({elapsed:.2f}s)", flush=True)
            f.write(json.dumps({"tool": name, "args": args, **payload}, ensure_ascii=False) + "\n")
            if ok:
                pass_count += 1
            else:
                fail_count += 1

    summary = {
        "ok": fail_count == 0,
        "pass": pass_count,
        "fail": fail_count,
        "skipped": skipped,
        "skip_lm": skip_lm,
        "with_haiku": with_haiku,
        "expected_total": len(PROBES) - skipped,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[smoke] summary: {summary}", flush=True)
    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lm", action="store_true", help="Skip slow LM-touching tools")
    parser.add_argument(
        "--with-haiku", action="store_true",
        help="Include Haiku-touching probes (each call costs ~$0.02 USD)"
    )
    parser.add_argument(
        "--out-jsonl", type=Path, default=REPO_ROOT / "audit" / "phase6_v0.3" / "smoke.jsonl"
    )
    parser.add_argument(
        "--out-json", type=Path, default=REPO_ROOT / "audit" / "phase6_v0.3" / "smoke.json"
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.skip_lm, args.with_haiku, args.out_jsonl, args.out_json))


if __name__ == "__main__":
    sys.exit(main())
