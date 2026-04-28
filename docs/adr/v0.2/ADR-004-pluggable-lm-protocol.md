# ADR-004 (v0.2) — `LMProtocol` shape and substrate-pluggability rules

Status: Accepted (frozen during planning round 3).
Date: 2026-04-28.
Related TRIZ card: [docs/triz/C4-substrate-vs-overhead.md](../../triz/C4-substrate-vs-overhead.md).

## Context

ADR-001 introduces `HaikuLM` alongside `LocalLM`, both satisfying a new `LMProtocol`. This ADR fixes the *exact shape* of that protocol and the rules for adding new substrates in v0.3 and beyond.

## Decision

```python
# src/pce/substrate/lm_protocol.py
from typing import Any, Protocol, runtime_checkable

from pce.types import Candidate

@runtime_checkable
class LMProtocol(Protocol):
    """Substrate the cascade calls into. Pluggable per ADR-001.

    Implementations: LocalLM, HaikuLM (v0.2). Any future substrate (Sonnet,
    Opus, Llama, Mistral) implements this protocol and the cascade picks it
    up without further changes.
    """

    name: str  # e.g. "qwen2-1.5b" | "claude-haiku" | "claude-sonnet"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        sampler: dict[str, float],
        seed: int,
    ) -> Candidate:
        """Produce one Candidate. Sampler keys: tau, top_p, top_k.

        Embedding may be left as a sentinel (`np.zeros(0, dtype=np.float32)`);
        the cascade re-embeds via the shared Embedder for cross-substrate
        consistency.
        """
        ...

    def report(self) -> dict[str, Any]:
        """Substrate diagnostic: model id, dtype, device, etc."""
        ...
```

Rules for substrates:

1. Implementations live under `src/pce/substrate/<substrate>_lm.py`.
2. The substrate's `generate` is responsible for *content* only (text + token-level metadata). Embeddings are computed by the shared `Embedder` to keep the geometry comparable across substrates.
3. The substrate's `seed` honours determinism *as far as possible*; for cloud LMs without seed control (Haiku via CLI today), the seed seeds the local PRNG that randomizes prompt nonces (see `HaikuLM._seed_prefix`) and the seed is recorded in the per-call audit.
4. The substrate handles its own retry policy. The cascade does not retry.
5. Cost telemetry is the substrate's responsibility. Substrates that incur cost write to `audit/<substrate>/<ts>.json` and update `audit/cost_ledger.json`. Free substrates (LocalLM) skip the ledger.

The MCP plugin gains a new tool:

```python
@mcp.tool()
def pce_cascade(
    arm: Literal["local", "haiku"] = "local",
    prompt: str,
    constraint_text: str,
    must_avoid: list[str] | None = None,
    aspects: list[str] | None = None,
    retrieval_set: list[str] | None = None,
    K: int = 4,
    max_tokens: int = 200,
    base_seed: int = 0,
    bypass_vimarsa: bool = False,
) -> dict[str, Any]:
    lm = _get_lm() if arm == "local" else _get_haiku()
    state = run_cascade(...)
    return state.to_audit() | {"surface": state.surface, "arm": arm}
```

## Consequences

Positive:

- Substrate is swappable without touching cascade or operators.
- v0.3 adds Sonnet/Opus by writing one `*_lm.py` and registering it in the MCP tool's `arm` enum.
- Cross-substrate ablations (e.g. `local_cascade` vs `haiku_cascade` at fixed prompts/aspects) become first-class.

Negative:

- Substrates have non-uniform seed semantics. Mitigation: per-substrate `report()` documents the determinism guarantee.
- Cost telemetry duplication if multiple paid substrates are added. Mitigation: factor out `audit/cost_ledger.py` helpers in v0.3.

## Alternatives considered

- *Abstract base class instead of Protocol*: rejected to keep substrates independent of `pce.substrate.base` (Protocol allows third-party substrates with no PCE inheritance).
- *Async generate*: rejected for v0.2; sync API is simpler and the cascade adds parallelism via `asyncio.gather` over sync calls in a thread pool when parallelism is needed (Haiku arm only).

## Implementation pointers

- `src/pce/substrate/lm_protocol.py` — Protocol definition (this file is type-only at runtime; no behavior).
- `src/pce/substrate/lm.py` — `LocalLM` already shaped to satisfy the Protocol; only docstring + `name` attribute added.
- `src/pce/substrate/haiku_lm.py` — new.
- `plugin/mcp/server.py` — add `pce_cascade(arm=...)` tool.
- `tests/substrate/test_lm_protocol.py` — assert both `LocalLM` and `HaikuLM` satisfy `isinstance(x, LMProtocol)`.
