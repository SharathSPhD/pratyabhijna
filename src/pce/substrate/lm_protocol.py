"""Pluggable LM substrate contract.

Per [docs/adr/v0.2/ADR-004-pluggable-lm-protocol.md](../../../docs/adr/v0.2/ADR-004-pluggable-lm-protocol.md),
the cascade and operators take an `LMProtocol`. Two implementations ship:

* `LocalLM` (existing, see `lm.py`) — `Qwen/Qwen2-1.5B-Instruct` via `transformers`.
* `HaikuLM` (see `haiku_lm.py`) — Anthropic Claude Haiku via the `claude` CLI
  (and an optional Anthropic SDK code path gated by `PCE_USE_SDK=1`).

Future substrates implement this Protocol and the cascade picks them up
without further changes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pce.types import Candidate


@runtime_checkable
class LMProtocol(Protocol):
    """Substrate the cascade calls into.

    Implementations are responsible for:

    * deterministic behavior under `seed` to the extent possible (cloud LMs
      without seed control should still be deterministic in their prompt
      construction, see `HaikuLM._seed_prefix`);
    * cost telemetry via `audit/<substrate>/<ts>.json` and updates to
      `audit/cost_ledger.json` when the substrate incurs real cost;
    * graceful failure (raise a clear `RuntimeError` rather than fall back
      silently to a different substrate or a mock).
    """

    name: str
    """Stable substrate identifier, e.g. 'qwen2-1.5b' or 'claude-haiku'."""

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampler: dict[str, float],
        seed: int,
    ) -> Candidate:
        """Produce one Candidate.

        Sampler keys: ``tau`` (temperature), ``top_p``, ``top_k``. Implementations
        may ignore keys they cannot honor and record what they did honor in
        ``Candidate.sampler``.

        ``Candidate.embedding`` may be left as a sentinel (``np.zeros(0,
        dtype=np.float32)``); the cascade re-embeds via the shared ``Embedder``
        for cross-substrate consistency.
        """
        ...

    def report(self) -> dict[str, Any]:
        """Substrate diagnostic: model id, dtype, device, etc."""
        ...
