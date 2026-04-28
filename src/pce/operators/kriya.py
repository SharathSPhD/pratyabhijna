"""`kriyā` - surface enaction.

Turns the selected `Candidate` into a final surface text. Three render modes
documented in [docs/operator-spec.md §6](../../../docs/operator-spec.md#6-kriya--surface-enaction):

* `verbatim`     - identity on `selected.text`.
* `polish`       - re-pass through the local LM with a low-temperature polish prompt.
* `claude_polish`- delegate to a caller-supplied Claude renderer.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.types import Candidate

RenderMode = Literal["verbatim", "polish", "claude_polish"]


def kriya(
    selected: Candidate,
    *,
    render_mode: RenderMode = "verbatim",
    lm: LocalLM | None = None,
    embed: Embedder | None = None,
    claude_renderer: Callable[[str], str] | None = None,
    polish_max_tokens: int = 96,
) -> str:
    text = selected.text
    if render_mode == "verbatim":
        return text
    if render_mode == "polish":
        if lm is None:
            raise ValueError("kriya: render_mode='polish' requires `lm`")
        prompt = (
            "Refine the following text, preserving its meaning, imagery, and overall length."
            "\n\nText:\n" + text + "\n\nRefined:\n"
        )
        polished = lm.generate(
            prompt,
            max_tokens=polish_max_tokens,
            sampler={"tau": 0.40, "top_p": 0.92, "top_k": 30.0},
            seed=int(selected.seed) + 9001,
        )
        out = polished.text or text
        if embed is not None:
            sim = embed.cosine(selected.embedding, embed.encode(out))
            if sim < 0.85:
                # Polish degraded semantic fidelity; fall back to the verbatim text.
                return text
        return out
    if render_mode == "claude_polish":
        if claude_renderer is None:
            raise ValueError("kriya: render_mode='claude_polish' requires `claude_renderer`")
        return str(claude_renderer(text))
    raise ValueError(f"unknown render_mode: {render_mode}")
