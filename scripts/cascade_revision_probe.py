#!/usr/bin/env python3
"""Phase 3 gate: real-LM cascade probe to verify the two-pass revision delta.

Runs ``run_cascade`` on the duck-rabbit textual prompt with ``LocalLM`` and
asserts:

1. Both passes complete (``state.surface_draft`` and
   ``state.surface_revision`` are populated).
2. ``state.surface == state.surface_revision`` (revision is the cascade
   output).
3. ``state.surface_revision != state.surface_draft`` (the revision actually
   differs textually).
4. ``state.audit["two_pass"] is True``.
5. ``state.vimarsa_brief`` is a non-empty string.

Cost: two K=3 cascades on Qwen2-1.5B-Instruct (CPU).
Run from repo root::

    uv run python scripts/cascade_revision_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pce.cascade import run_cascade  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.types import Constraint  # noqa: E402

DUCK_RABBIT_PROMPT = (
    "Read the following short description aloud:\n\n"
    "  'A small bird perched on the fence; its beak swept upward into a long ear, "
    "and its eye seemed to look in two directions at once.'\n\n"
    "In one or two sentences, name the two animals one might see in this image, "
    "and describe the moment when one becomes the other."
)
DUCK_RABBIT_ASPECTS = [
    "a duck with a long beak pointing upward to the right",
    "a rabbit with two long ears pointing back to the left",
]
DUCK_RABBIT_RETRIEVAL = [
    "the cat sat on the mat",
    "two plus two equals four",
]


def main() -> int:
    embed = Embedder()
    lm = LocalLM()
    constraint = Constraint(
        text="a vivid description naming two animals in one ambiguous figure",
        embedding=embed.encode(
            "a vivid description naming two animals in one ambiguous figure"
        ),
        must_avoid=("a single literal description of a duck",),
    )
    print("[probe] running cascade on duck-rabbit textual...", flush=True)
    state = run_cascade(
        prompt=DUCK_RABBIT_PROMPT,
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=3,
        max_tokens=120,
        base_seed=42,
        retrieval_set=DUCK_RABBIT_RETRIEVAL,
        aspects=DUCK_RABBIT_ASPECTS,
    )
    print("\n--- DRAFT ---")
    print((state.surface_draft or "").strip()[:600])
    print("\n--- REVISION ---")
    print((state.surface_revision or "").strip()[:600])
    print("\n--- BRIEF ---")
    print((state.vimarsa_brief or "").strip())
    print(
        f"\n[probe] two_pass={state.audit['two_pass']}  "
        f"revision_differs={state.audit['revision_differs_from_draft']}  "
        f"vimarsa_event_draft={state.vimarsa_event_draft}  "
        f"vimarsa_event={state.vimarsa_event}"
    )
    fails: list[str] = []
    if state.surface_draft is None or state.surface_revision is None:
        fails.append("draft or revision missing")
    if state.surface != state.surface_revision:
        fails.append("state.surface != state.surface_revision")
    if state.surface_revision == state.surface_draft:
        fails.append("revision == draft (no delta)")
    if not state.audit["two_pass"]:
        fails.append("two_pass=False")
    if not (state.vimarsa_brief and state.vimarsa_brief.strip()):
        fails.append("brief empty")
    if fails:
        print(f"[probe] FAIL: {', '.join(fails)}", file=sys.stderr)
        return 1
    print("[probe] OK: all Phase 3 gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
