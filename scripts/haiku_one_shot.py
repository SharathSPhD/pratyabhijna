#!/usr/bin/env python3
"""One-shot probe: call HaikuLM on a duck-rabbit interpretation prompt.

Verifies the Phase 2 adapter end-to-end:

* `claude` CLI is reachable and returns JSON.
* Cost telemetry is parsed into the ledger.
* Audit log is written.

Cost: one Haiku call (~$0.001-0.10 depending on cache state).

Run from repo root:

    uv run python scripts/haiku_one_shot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.haiku_lm import HaikuConfig, HaikuLM  # noqa: E402

PROMPT = (
    "Read the following short description aloud:\n\n"
    "  'A small bird perched on the fence; its beak swept upward into a long ear, "
    "and its eye seemed to look in two directions at once.'\n\n"
    "In one or two sentences, name the two animals one might see in this image, "
    "and describe the moment when one becomes the other."
)


def main() -> int:
    embed = Embedder()
    lm = HaikuLM(config=HaikuConfig.from_env(), embedder=embed)
    print(f"[haiku-probe] Calling {lm.name} via {lm.config.cli_bin}...", flush=True)
    cand = lm.generate(PROMPT, max_tokens=200, sampler={"tau": 0.9, "top_p": 0.95}, seed=42)
    print(f"\n--- Haiku response ---\n{cand.text}\n----------------------\n")
    print(f"[haiku-probe] embedding shape={cand.embedding.shape}")
    print(f"[haiku-probe] sampler={cand.sampler}")
    rep = lm.report()
    print(f"[haiku-probe] cost_ledger total={rep['ledger_total_usd']:.6f} USD over {rep['ledger_n_calls']} calls")
    audit_dir = REPO_ROOT / "audit" / "haiku"
    if audit_dir.exists():
        latest = sorted(audit_dir.glob("*.json"))[-1]
        rec = json.loads(latest.read_text(encoding="utf-8"))
        print(f"[haiku-probe] last audit: {latest.name} cost={rec['cost_usd']:.6f}")
    if not cand.text.strip():
        print("[haiku-probe] FAIL: empty response", file=sys.stderr)
        return 1
    print("[haiku-probe] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
