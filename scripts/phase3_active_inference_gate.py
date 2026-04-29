#!/usr/bin/env python3
"""Phase 3 gate -- ΔF non-degenerate on the duck-rabbit textual fixture.

Loads ``tests/fixtures/duck_rabbit_textual.json``, runs jñāna in
``aspect_conditioned`` mode against synthetic per-aspect cosine matrices
that match what apohana would produce on a surface that names both
animals, and asserts ``|ΔF| >= 0.01``.

This is the cheap ΔF probe; the full cascade-on-duck-rabbit is exercised
by the prove-gate in Phase 5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pce.active_inference.budget import FreeEnergyBudget  # noqa: E402
from pce.active_inference.hopfield import HopfieldStore  # noqa: E402
from pce.operators.jnana import jnana  # noqa: E402

DELTA_F_FLOOR = 0.01


def main() -> int:
    fx_path = REPO_ROOT / "tests" / "fixtures" / "duck_rabbit_textual.json"
    fx = json.loads(fx_path.read_text(encoding="utf-8"))
    aspects = fx["aspects"]
    assert len(aspects) == 2, f"duck-rabbit fixture must have 2 aspects, got {len(aspects)}"

    K = 4
    # Synthetic but realistic aspect-cosine matrix: cand 2 (the "good"
    # candidate) hits both aspects above the floor (0.30); the others hit
    # at most one or none. These mirror what MiniLM-L6-v2 produces on the
    # duck-rabbit surface "I see a duck whose beak becomes a rabbit's ear..."
    aspect_membership = np.array(
        [
            [0.18, 0.12],
            [0.42, 0.10],
            [0.41, 0.38],  # full-aspect winner -- the IFR reduction targets this
            [0.08, 0.40],
        ],
        dtype=np.float32,
    )
    apoha = np.array([0.20, 0.45, 0.62, 0.30], dtype=np.float32)
    anan = np.array([0.30, 0.55, 0.70, 0.40], dtype=np.float32)

    # Storehouse warm-start: pretend we have one prior duck-pattern stored.
    store = HopfieldStore(domain="poetry_interp")
    rng = np.random.default_rng(0)
    duck_emb = rng.standard_normal(384).astype(np.float32)
    duck_emb /= float(np.linalg.norm(duck_emb)) + 1e-12
    store.write(duck_emb, label=aspects[0], mode="rem")

    # Per-aspect prior derived from storehouse mass (uniform when no aspects).
    res = store.query(duck_emb, aspect_labels=aspects)
    aspect_priors_f64 = np.maximum(res.aspect_priors, 1e-3)
    aspect_priors = (aspect_priors_f64 / float(aspect_priors_f64.sum() + 1e-12)).astype(np.float32)

    cands = tuple(range(K))
    sel, delta_F, posterior = jnana(
        cands,
        apoha,
        anan,
        reduction_target="aspect_conditioned",
        aspect_membership=aspect_membership,
        aspect_priors=aspect_priors,
    )

    # Free-energy ledger smoke: confirm earn/pay flow yields a finite balance.
    budget = FreeEnergyBudget()
    budget.earn_jnana(delta_F, note="duck-rabbit jnana")
    budget.earn_aspect(0.4, note="aspect cost")
    budget.earn_tokens(110, note="committed tokens")
    bal = budget.balance()

    out = {
        "fixture": fx["id"],
        "selected_idx": int(sel),
        "delta_F": float(delta_F),
        "posterior": [float(p) for p in posterior.tolist()],
        "aspect_priors": [float(p) for p in aspect_priors.tolist()],
        "budget_balance_bits": float(bal),
        "n_storehouse_patterns": int(store.n_patterns),
        "delta_F_floor": float(DELTA_F_FLOOR),
        "delta_F_passed": bool(abs(delta_F) >= DELTA_F_FLOOR),
        "budget_ledger": budget.to_audit(),
    }

    audit_dir = REPO_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "v0_3_phase3_gate.json").write_text(
        json.dumps(out, indent=2, allow_nan=False), encoding="utf-8"
    )

    if not out["delta_F_passed"]:
        print(f"[phase3_gate] FAILED: |ΔF|={abs(delta_F):.4f} < {DELTA_F_FLOOR}")
        return 1
    print(
        f"[phase3_gate] OK: ΔF={delta_F:+.4f} (|ΔF|>={DELTA_F_FLOOR}), "
        f"sel={sel}, posterior_argmax={int(np.argmax(posterior))}, "
        f"budget_balance={bal:+.3f} bits"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
