# ADR-001 (v0.2) — Pluggable Haiku substrate via `LMProtocol`

Status: Accepted (frozen during planning round 1).
Date: 2026-04-28.
Related TRIZ card: [docs/triz/C1-cost-vs-quality.md](../../triz/C1-cost-vs-quality.md).

## Context

The v0.1 benchmark compared `local_cascade` (Qwen2-1.5B + PCE) against `claude_haiku` (no PCE), confounding substrate strength with the architectural contribution of the cascade. The adversarial review identified this as the P0-1 finding.

The user's frozen scope (planning round 1) elected the *hybrid four-arm* substrate strategy: keep the v0.1 local arms as ablation, add `haiku_bare` and `haiku_cascade` as the apples-to-apples primary contrast.

## Decision

Define an `LMProtocol` (a `typing.Protocol`) in `src/pce/substrate/lm_protocol.py` that captures the minimal contract `iccha`, `cit`, and `kriya` need from a substrate:

```python
class LMProtocol(Protocol):
    name: str
    def generate(self, prompt: str, *, max_tokens: int, sampler: dict[str, float], seed: int) -> Candidate: ...
    def report(self) -> dict[str, Any]: ...
```

Implement two substrates that satisfy this protocol:

1. `LocalLM` (existing, refactored): Qwen2-1.5B-Instruct via `transformers`. Auto-detects device (CUDA, MPS, CPU) and dtype (float16, float32).
2. `HaikuLM` (new): wraps `claude -p --model haiku --output-format json` via `subprocess.run`. Optional Anthropic SDK code path gated by `PCE_USE_SDK=1` and `ANTHROPIC_API_KEY`. Per-call audit logs to `audit/haiku/<ts>.json` recording `prompt_sha`, `response`, `cost_usd`, `latency_ms`, `seed`, `sampler`.

The cascade and operators take an `LMProtocol` rather than a concrete `LocalLM`. The benchmark driver constructs the appropriate substrate per arm and threads it through `run_cascade`.

The MCP plugin gains a new tool `pce_cascade(arm: Literal["local", "haiku"] = "local", ...)` that selects the substrate at the boundary.

## Consequences

Positive:

- The v0.2 pilot can run all four arms (local_bare, local_cascade, haiku_bare, haiku_cascade) through a single driver code path.
- The primary scientific question (does PCE improve a strong substrate?) becomes testable as `haiku_cascade - haiku_bare` under H1.v2-H4.v2.
- The same-substrate ablation (`local_cascade - local_bare`) is preserved as H6.v2.
- Future v0.3 substrates (Sonnet, Opus, Llama) plug in by implementing `LMProtocol`.

Negative:

- The `HaikuLM` adapter adds a `claude` CLI dependency to the runtime path. Mitigation: the local arms still work without `claude`; `HaikuLM.generate()` raises a clear `RuntimeError` when `claude` is missing rather than silently falling back.
- Each Haiku call costs real money. Mitigation: per-call cost telemetry plus a global `audit/cost_ledger.json` that the driver checks before each new call; the driver aborts gracefully when ledger >= $18.
- The Anthropic SDK path adds an optional dependency. Mitigation: the SDK code path is import-guarded; if `anthropic` is not installed, `PCE_USE_SDK=1` raises an early actionable error.

## Alternatives considered

- *Haiku-only PCE* (drop local entirely): rejected because the existing local benchmark machinery and ablation value is preserved cheaply by keeping both substrates pluggable.
- *Local cit+iccha + Haiku kriya/judge*: rejected because the hybrid substrate within a single cascade run is harder to reason about (which substrate is responsible for which lift?) and harder to ship in this session.
- *Wrap a different Anthropic model* (Sonnet for cit, Haiku for revision): deferred to v0.3.

## Implementation pointers

- `src/pce/substrate/lm_protocol.py` — Protocol definition.
- `src/pce/substrate/lm.py` — refactor to satisfy the protocol; minimal surface change.
- `src/pce/substrate/haiku_lm.py` — new; uses `subprocess.run(["claude", "-p", "--model", model, "--output-format", "json", prompt], ...)`.
- `audit/haiku/` — per-call audit logs.
- `audit/cost_ledger.json` — running total + last-call timestamp.
- `tests/substrate/test_haiku_lm.py` — unit tests with subprocess monkeypatch + dry-run mode.
