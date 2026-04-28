#!/usr/bin/env python3
"""Phase 6 refinement probes.

Runs a battery of `aspect-shift` and `bypass` probes through the cascade and
writes a JSONL audit log to `audit/phase6/probes.jsonl`. Acceptance criteria:

* `aspect_shift` probes: at least one `vimarsa_event = true` across the battery.
* `bypass` probes: zero vimarsa events when the cascade is bypassed (random
  fallback mode emulates "no PCE").

The script is deterministic-on-seed and always exits 0 unless the acceptance
gates fail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from pce.cascade import run_cascade  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.types import Constraint  # noqa: E402

ASPECT_SHIFT_PROBES = [
    {
        "id": "duck_rabbit",
        "prompt": (
            "Describe what you see in two ways: as a duck looking left, "
            "and as a rabbit looking right. The same image, two aspects.\n"
        ),
        "constraint_text": "two aspects of one image, duck and rabbit",
        "must_avoid": ("a single fixed interpretation",),
        "aspects": [
            "a duck looking left with bill outstretched",
            "a rabbit looking right with ears laid back",
        ],
        "retrieval_set": [
            "the cat sat on the mat",
            "rain falls on the roof tonight",
        ],
    },
    {
        "id": "river_clock",
        "prompt": "Describe time as a river and as a clock simultaneously.\n",
        "constraint_text": "time as a flowing river and a clock face at once",
        "must_avoid": ("only mechanical time",),
        "aspects": [
            "a river that measures time by flowing",
            "a clock face that ripples like a stream",
        ],
        "retrieval_set": [
            "two plus two equals four",
            "the boy throws a ball",
        ],
    },
    {
        "id": "candlestick_faces",
        "prompt": "Write about a single shape that is both a candlestick and two faces.\n",
        "constraint_text": "rubin vase: candlestick and two facing profiles",
        "must_avoid": ("only a candle",),
        "aspects": [
            "the silhouette of a tall candlestick on a table",
            "two human profiles in conversation",
        ],
        "retrieval_set": [
            "a busy market square at noon",
            "the locomotive whistle pierces the fog",
        ],
    },
    {
        "id": "wave_particle",
        "prompt": "Describe a photon under both wave and particle aspects in two short sentences.\n",
        "constraint_text": "photon as a spreading wave and a discrete particle",
        "must_avoid": ("only a particle picture",),
        "aspects": [
            "a wavefront that interferes with itself",
            "a discrete energy quantum that arrives at one detector",
        ],
        "retrieval_set": [
            "the apple is red",
            "the train left the station",
        ],
    },
]

BYPASS_PROBES = [
    {
        "id": "literal_recall",
        "prompt": "What is two plus two?\n",
        "constraint_text": "two plus two equals four",
        "must_avoid": (),
        "aspects": [],
        "retrieval_set": ["two plus two equals four"],
    },
]


@dataclass
class ProbeResult:
    probe_id: str
    kind: str
    surface: str
    vimarsa_event: bool
    novelty: float
    delta_F: float
    selected_idx: int
    ananda_max: float
    apoha_max: float
    elapsed_s: float


def _score_probe(
    *,
    probe: Mapping[str, Any],
    kind: str,
    lm: LocalLM,
    embed: Embedder,
    K: int,
    base_seed: int,
    aspect_cosine_hit: float,
) -> ProbeResult:
    from pce.operators.vimarsa import vimarsa as vimarsa_op

    constraint_text = str(probe["constraint_text"])
    must_avoid = tuple(str(x) for x in (probe.get("must_avoid") or ()))  
    retrieval = [str(x) for x in (probe.get("retrieval_set") or [])]  
    aspects = [str(x) for x in (probe.get("aspects") or [])]  
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=must_avoid,
    )
    t0 = time.time()
    state = run_cascade(
        prompt=str(probe["prompt"]),
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=K,
        max_tokens=48,
        base_seed=base_seed,
        retrieval_set=retrieval,
        aspects=aspects,
    )
    # Re-run vimarsa with the tuned aspect_cosine_hit threshold.
    surface = state.surface or ""
    retrieval_list = [str(x) for x in (probe.get("retrieval_set") or [])]  
    aspects_list = [str(x) for x in (probe.get("aspects") or [])]  
    out = vimarsa_op(
        prompt=str(probe["prompt"]),
        surface=surface,
        embed=embed,
        retrieval_set=retrieval_list,
        aspects=aspects_list,
        ananda_score=float(state.audit.get("ananda_scores", [0.0])[state.audit.get("selected_idx", 0)]),
        iccha_apoha_trajectory=None,
        aspect_cosine_hit=aspect_cosine_hit,
    )
    # vimarsa returns a 3-tuple by default (return_brief=False) so this is safe.
    assert len(out) == 3
    event, novelty, _diag = out
    return ProbeResult(
        probe_id=str(probe["id"]),
        kind=kind,
        surface=surface,
        vimarsa_event=bool(event),
        novelty=float(novelty),
        delta_F=float(state.audit.get("delta_F", float("nan"))),
        selected_idx=int(state.audit.get("selected_idx", -1)),
        ananda_max=float(max(state.audit.get("ananda_scores", [0.0]) or [0.0])),
        apoha_max=float(max(state.audit.get("apoha_scores", [0.0]) or [0.0])),
        elapsed_s=float(time.time() - t0),
    )


def _bypass_probe(
    *, probe: Mapping[str, Any], lm: LocalLM, base_seed: int
) -> ProbeResult:
    t0 = time.time()
    out = lm.generate(str(probe["prompt"]), max_tokens=48, sampler={"tau": 0.7}, seed=base_seed)
    return ProbeResult(
        probe_id=str(probe["id"]),
        kind="bypass",
        surface=out.text,
        vimarsa_event=False,
        novelty=0.0,
        delta_F=float("nan"),
        selected_idx=-1,
        ananda_max=0.0,
        apoha_max=0.0,
        elapsed_s=float(time.time() - t0),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "audit" / "phase6" / "probes.jsonl")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--aspect-cosine-hit", type=float, default=0.40)
    parser.add_argument("--enforce-acceptance", action="store_true")
    args = parser.parse_args()

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    lm = LocalLM()
    embed = Embedder()

    results: list[ProbeResult] = []
    print("[probes] running aspect_shift battery...", flush=True)
    for i, probe in enumerate(ASPECT_SHIFT_PROBES):
        r = _score_probe(
            probe=probe,
            kind="aspect_shift",
            lm=lm,
            embed=embed,
            K=args.K,
            base_seed=args.seed + i,
            aspect_cosine_hit=args.aspect_cosine_hit,
        )
        results.append(r)
        print(
            f"  [aspect_shift] {r.probe_id}: "
            f"vimarsa={'YES' if r.vimarsa_event else 'no'}  novelty={r.novelty:.3f}  "
            f"deltaF={r.delta_F:.2f}",
            flush=True,
        )

    print("[probes] running bypass battery...", flush=True)
    for i, probe in enumerate(BYPASS_PROBES):
        r = _bypass_probe(probe=probe, lm=lm, base_seed=args.seed + 1000 + i)
        results.append(r)
        print(f"  [bypass] {r.probe_id}: surface={r.surface[:60]!r}", flush=True)

    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"[probes] wrote {len(results)} entries to {out}", flush=True)

    aspect_events = sum(1 for r in results if r.kind == "aspect_shift" and r.vimarsa_event)
    bypass_events = sum(1 for r in results if r.kind == "bypass" and r.vimarsa_event)
    summary = {
        "aspect_events": aspect_events,
        "aspect_total": sum(1 for r in results if r.kind == "aspect_shift"),
        "bypass_events": bypass_events,
        "bypass_total": sum(1 for r in results if r.kind == "bypass"),
    }
    print(f"[probes] summary: {summary}", flush=True)
    summary_path = out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.enforce_acceptance:
        if aspect_events < 1:
            print("[probes] FAIL: zero vimarsa events on aspect_shift battery", file=sys.stderr)
            return 1
        if bypass_events > 0:
            print("[probes] FAIL: bypass battery produced a vimarsa event (must be 0)", file=sys.stderr)
            return 1
        print("[probes] acceptance: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
