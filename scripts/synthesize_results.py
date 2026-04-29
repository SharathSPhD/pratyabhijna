#!/usr/bin/env python3
"""Synthesise realistic-looking results JSONs for pipeline testing.

Writes to a target directory (default benchmarks/_synth/) so it cannot collide
with real benchmark output. Used by Phase 9 smoke to verify the stats →
figures → autoreport pipeline end-to-end before the live benchmark finishes.

The synthesiser preserves the exact schema of `benchmarks/driver.py` outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DOMAINS_NS = {"poetry_gen": 12, "poetry_interp": 10, "aut": 8, "sci_creativity": 8}
# v0.3 4-arm matrix is primary; v0.1/v0.2 arms preserved for backward
# compatibility so older audits still parse.
ARMS = (
    "claude_haiku",
    "local_bare",
    "local_cascade",
    "haiku_bare",
    "haiku_cascade",
    "haiku_bare_2K_scorer",
    "haiku_generic_revise_2pass",
)
AXES_PER_DOMAIN = {
    "poetry_gen": ("creativity", "lexical_diversity", "idiosyncrasy",
                    "emotional_resonance", "literary_devices", "imagery"),
    "poetry_interp": ("aspect_count", "novelty", "coverage"),
    "aut": ("creativity", "lexical_diversity", "feasibility"),
    "sci_creativity": ("non_textbook_novelty", "framing_coverage", "depth", "multi_framing"),
}


def synthesize(out_dir: Path, *, seed: int = 4242,
                effect_sizes: dict[str, float] | None = None) -> None:
    rng = np.random.default_rng(seed)
    eff = effect_sizes or {"aut": 0.30, "poetry_interp": 0.20, "poetry_gen": -0.10, "sci_creativity": 0.05}
    out_dir.mkdir(parents=True, exist_ok=True)
    vimarsa_fired_rate = 0.45
    for dom, n in DOMAINS_NS.items():
        rows: dict[str, dict[str, object]] = {}
        for i in range(1, n + 1):
            item_id = {"poetry_gen": "p", "poetry_interp": "i", "aut": "a", "sci_creativity": "s"}[dom] + f"{i:02d}"
            base = float(rng.uniform(0.30, 0.65))
            arm_data: dict[str, dict[str, object]] = {}
            for arm in ARMS:
                if arm == "local_cascade":
                    shift = float(eff[dom]) + float(rng.normal(0.0, 0.05))
                elif arm == "local_bare":
                    shift = float(eff[dom]) * 0.4 + float(rng.normal(0.0, 0.07))
                elif arm == "haiku_bare":
                    # haiku_bare sits above local_bare (cloud-grade substrate)
                    shift = 0.10 + float(rng.normal(0.0, 0.04))
                elif arm == "haiku_cascade":
                    # haiku_cascade adds the cascade contribution on top of haiku_bare
                    shift = 0.10 + float(eff[dom]) + float(rng.normal(0.0, 0.04))
                elif arm == "haiku_bare_2K_scorer":
                    # extra-compute control: between haiku_bare and haiku_cascade
                    shift = 0.10 + 0.5 * float(eff[dom]) + float(rng.normal(0.0, 0.04))
                elif arm == "haiku_generic_revise_2pass":
                    # generic 2-pass: revision exists but no PCE brief content.
                    shift = 0.10 + 0.6 * float(eff[dom]) + float(rng.normal(0.0, 0.04))
                else:  # claude_haiku (v0.1 alias of haiku_bare; mirror its distribution)
                    shift = 0.10 + float(rng.normal(0.0, 0.04))
                comp = float(np.clip(base + shift, 0.0, 1.0))
                axes = {a: float(np.clip(comp + rng.normal(0, 0.05), 0.0, 1.0)) for a in AXES_PER_DOMAIN[dom]}
                n_words = int(rng.integers(20, 80))
                n_chars = int(n_words * float(rng.uniform(4.5, 6.5)))
                meta: dict[str, object] = {
                    "ok": True,
                    "elapsed_s": float(rng.uniform(2.0, 90.0)),
                    "n_chars": n_chars,
                    "n_words": n_words,
                }
                if arm in ("local_cascade", "haiku_cascade", "haiku_generic_revise_2pass"):
                    fired = bool(rng.random() < vimarsa_fired_rate)
                    committed = "revision" if fired else "draft"
                    surface_draft = f"[synthetic-draft-{arm}-{item_id}] creative content"
                    surface_revision = f"[synthetic-revision-{arm}-{item_id}] aspect-shifted content"
                    meta.update({
                        "vimarsa_event": fired,
                        "novelty": float(rng.uniform(0.4, 0.95)),
                        "delta_F": float(rng.normal(-0.1 if fired else 0.05, 0.2)),
                        "selected_idx": int(rng.integers(0, 4)),
                        "committed": committed,
                        "commit_policy": (
                            "event_gated" if arm == "haiku_cascade"
                            else "always_revise" if arm == "haiku_generic_revise_2pass"
                            else "always_revise"
                        ),
                        "surface_draft": surface_draft,
                        "surface_revision": surface_revision,
                    })
                arm_data[arm] = {
                    "text": f"[synthetic-{arm}-{item_id}] " + ("aspect-shift content" if dom == "poetry_interp" else "creative content"),
                    "axes": axes,
                    "composite": comp,
                    "meta": meta,
                }
            rows[item_id] = {
                "item": {"id": item_id, "topic": f"synthetic topic for {item_id}"},
                **arm_data,
            }
        (out_dir / f"{dom}.json").write_text(
            json.dumps({"domain": dom, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "_synth")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    synthesize(args.out_dir, seed=args.seed)
    print(f"synthetic results -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
