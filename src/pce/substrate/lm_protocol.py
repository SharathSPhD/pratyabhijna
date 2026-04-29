"""Pluggable LM substrate contract.

Per [docs/adr/v0.3/ADR-001-clean-haiku-cli.md](../../../docs/adr/v0.3/ADR-001-clean-haiku-cli.md)
the protocol is renamed `GeneratorProtocol` (with `LMProtocol` kept as an alias for
backward compatibility) and gains capability flags so callers can interrogate what
the substrate honestly exposes.

Two implementations ship:

* `LocalLM` (existing, see `lm.py`) — `Qwen/Qwen2-1.5B-Instruct` via `transformers`.
  Advertises `supports_logprobs=True`, `supports_score=True`, `supports_entropy=False`.
* `HaikuLM` (see `haiku_lm.py`) — Anthropic Claude Haiku via the `claude` CLI in a
  clean-substrate inner subprocess (no API key). Advertises all capability flags as
  `False`; exposes `length_proxy_logp` so callers cannot mistake length for real logprobs.

Future substrates implement this Protocol and the cascade picks them up
without further changes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pce.types import Candidate


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Substrate the cascade calls into.

    Implementations are responsible for:

    * deterministic behavior under `seed` to the extent possible (cloud LMs
      without seed control should still be deterministic in their prompt
      construction, see `HaikuLM._seed_prefix`);
    * cost telemetry via `audit/<substrate>/<ts>.json` and updates to
      `audit/cost_ledger.json` when the substrate incurs real cost;
    * graceful failure (raise a clear `RuntimeError` rather than fall back
      silently to a different substrate or a mock).

    Capability flags let callers interrogate what the substrate honestly exposes.
    The cascade and active-inference modules (`pce.active_inference.budget`,
    `pce.operators.jnana`) consult these flags so they never rely on a signal the
    substrate cannot provide. See ADR-001 (clean-haiku-cli) and ADR-005
    (free-energy-budget) for the v0.3 contract.
    """

    name: str
    """Stable substrate identifier, e.g. 'qwen2-1.5b' or 'claude-haiku'."""

    supports_logprobs: bool
    """True iff `Candidate.logp` is a real log-probability from the substrate."""

    supports_score: bool
    """True iff the substrate can score arbitrary completions out-of-band."""

    supports_entropy: bool
    """True iff the substrate exposes per-token entropy or top-k logits."""

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

    def length_proxy_logp(self, candidate: Candidate) -> float:
        """A calibrated, *honest* fallback when `supports_logprobs` is False.

        For substrates that do not expose real log-probabilities, this returns a
        deterministic monotone proxy (typically `-output_tokens * log(2)`) so that
        downstream code which only uses `logp` as a tie-breaker stays stable. Callers
        must not interpret the return value as a true log-probability.
        """
        ...

    def report(self) -> dict[str, Any]:
        """Substrate diagnostic: model id, dtype, device, etc."""
        ...


# Backward-compatible alias used throughout v0.1/v0.2 code.
LMProtocol = GeneratorProtocol
