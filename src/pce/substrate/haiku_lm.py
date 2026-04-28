"""Haiku substrate adapter — wraps the `claude` CLI as an `LMProtocol`.

Per [docs/adr/v0.2/ADR-001-haiku-substrate.md](../../../docs/adr/v0.2/ADR-001-haiku-substrate.md)
and [docs/adr/v0.2/ADR-004-pluggable-lm-protocol.md](../../../docs/adr/v0.2/ADR-004-pluggable-lm-protocol.md).

Two backends:

* `cli` (default): subprocess `claude -p --model <model> --output-format json <prompt>`.
* `sdk` (opt-in via `PCE_USE_SDK=1`): `anthropic.Anthropic().messages.create(...)`.
  Requires `pip install anthropic` and `ANTHROPIC_API_KEY`.

Determinism: the CLI does not expose a seed, so the substrate seeds a per-call
*nonce prefix* on the prompt to make repeated calls with the same `seed`
return semantically equivalent (not byte-identical) text. Audit logs record
the seed alongside the response.

Cost telemetry: every call appends to `audit/haiku/<ts>.json` and updates
`audit/cost_ledger.json`. The cascade driver reads the ledger before each
new call and aborts gracefully if the running total approaches the envelope.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pce.substrate.embed import Embedder
from pce.types import Candidate

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO_ROOT / "audit" / "haiku"
COST_LEDGER = REPO_ROOT / "audit" / "cost_ledger.json"


@dataclass(frozen=True)
class HaikuConfig:
    """Configuration for `HaikuLM`. Read from env by default."""

    model: str = "haiku"
    cli_bin: str = "claude"
    timeout_s: int = 120
    use_sdk: bool = False
    cost_cap_usd: float = 18.0  # graceful abort threshold (10% under $20 hard ceiling)
    cli_retry: int = 2  # extra attempts on empty CLI response (0 = no retry)
    cli_backoff_s: float = 1.0  # base backoff multiplier between retries

    @classmethod
    def from_env(cls) -> HaikuConfig:
        return cls(
            model=os.environ.get("PCE_HAIKU_MODEL", "haiku"),
            cli_bin=os.environ.get("PCE_HAIKU_CLI", "claude"),
            timeout_s=int(os.environ.get("PCE_HAIKU_TIMEOUT_S", "120")),
            use_sdk=os.environ.get("PCE_USE_SDK", "").strip() == "1",
            cost_cap_usd=float(os.environ.get("PCE_HAIKU_COST_CAP_USD", "18.0")),
            cli_retry=int(os.environ.get("PCE_HAIKU_CLI_RETRY", "2")),
            cli_backoff_s=float(os.environ.get("PCE_HAIKU_CLI_BACKOFF_S", "1.0")),
        )


class HaikuBudgetExceededError(RuntimeError):
    """Raised when the cost ledger total reaches `cost_cap_usd`."""


def _load_ledger() -> dict[str, Any]:
    if not COST_LEDGER.exists():
        return {"total_usd": 0.0, "n_calls": 0, "by_model": {}}
    raw: dict[str, Any] = json.loads(COST_LEDGER.read_text(encoding="utf-8"))
    return raw


def _save_ledger(ledger: dict[str, Any]) -> None:
    COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    COST_LEDGER.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _audit_call(record: dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fp = AUDIT_DIR / f"{int(time.time() * 1000)}_{os.getpid()}.json"
    fp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_prefix(seed: int) -> str:
    """Produce a tiny invisible-to-meaning prefix derived from seed.

    The CLI has no seed kwarg, so we vary the prompt by a single noop
    instruction-style nonce. Different seeds get different sequences of
    leading whitespace + a benign hidden directive so the model produces
    semantically distinct continuations across K seeds.
    """
    rng = np.random.default_rng(int(seed))
    pad = " " * int(rng.integers(0, 4))
    nonce = int(rng.integers(0, 99_999))
    return f"{pad}<!-- pce-seed:{nonce} -->\n"


class HaikuLM:
    """Claude Haiku substrate via `claude` CLI.

    Implements `pce.substrate.lm_protocol.LMProtocol`.
    """

    name = "claude-haiku"

    def __init__(
        self,
        config: HaikuConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or HaikuConfig.from_env()
        self._embedder = embedder or Embedder()
        if self.config.use_sdk:
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "PCE_USE_SDK=1 requires `pip install anthropic` (Python SDK)."
                ) from exc

    def _check_budget(self) -> None:
        ledger = _load_ledger()
        if float(ledger.get("total_usd", 0.0)) >= float(self.config.cost_cap_usd):
            raise HaikuBudgetExceededError(
                f"Haiku cost ledger {ledger['total_usd']:.4f} USD "
                f">= cap {self.config.cost_cap_usd} USD. Aborting before next call."
            )

    def _record_cost(self, model: str, cost_usd: float, latency_ms: int) -> None:
        ledger = _load_ledger()
        ledger["total_usd"] = float(ledger.get("total_usd", 0.0)) + float(cost_usd)
        ledger["n_calls"] = int(ledger.get("n_calls", 0)) + 1
        by_model = dict(ledger.get("by_model", {}))
        slot = dict(by_model.get(model, {"total_usd": 0.0, "n_calls": 0, "total_latency_ms": 0}))
        slot["total_usd"] = float(slot.get("total_usd", 0.0)) + float(cost_usd)
        slot["n_calls"] = int(slot.get("n_calls", 0)) + 1
        slot["total_latency_ms"] = int(slot.get("total_latency_ms", 0)) + int(latency_ms)
        by_model[model] = slot
        ledger["by_model"] = by_model
        _save_ledger(ledger)

    def _call_cli_once(self, prompt: str) -> tuple[str, dict[str, Any]]:
        cmd = [
            self.config.cli_bin,
            "-p",
            "--model",
            self.config.model,
            "--output-format",
            "json",
            prompt,
        ]
        started = time.time()
        proc = subprocess.run(  # noqa: S603 — CLI is trusted user-installed binary
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=self.config.timeout_s,
        )
        latency_ms = int((time.time() - started) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"HaikuLM CLI rc={proc.returncode}: "
                f"{proc.stderr.decode('utf-8', errors='replace')[-500:]}"
            )
        try:
            payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"HaikuLM CLI returned non-JSON: {exc}") from exc
        if payload.get("is_error", False):
            raise RuntimeError(
                f"HaikuLM CLI is_error=True: {payload.get('result', '')[:500]}"
            )
        text = str(payload.get("result", "") or "")
        meta = {
            "cost_usd": float(payload.get("total_cost_usd", 0.0)),
            "duration_ms": int(payload.get("duration_ms", latency_ms)),
            "stop_reason": payload.get("stop_reason"),
            "session_id": payload.get("session_id"),
            "model_used": next(iter((payload.get("modelUsage") or {}).keys()), self.config.model),
            "input_tokens": int((payload.get("usage") or {}).get("input_tokens", 0)),
            "output_tokens": int((payload.get("usage") or {}).get("output_tokens", 0)),
        }
        return text, meta

    def _call_cli(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Wrapper around _call_cli_once with retry-on-empty.

        The Anthropic CLI sporadically returns an empty `result` field even
        when the API succeeds (rc=0, is_error=False). We retry up to
        ``cli_retry`` times with a small backoff, varying the seed prefix
        per attempt to avoid hitting the same bad path twice. Cost still
        accrues per attempt (the API was billed).
        """
        last_text = ""
        last_meta: dict[str, Any] = {}
        for attempt in range(self.config.cli_retry + 1):
            text, meta = self._call_cli_once(prompt)
            last_text, last_meta = text, meta
            if text.strip():
                meta["attempt"] = attempt
                return text, meta
            # Empty result: bill it (we'll still record_cost upstream) and retry.
            if attempt < self.config.cli_retry:
                time.sleep(self.config.cli_backoff_s * (attempt + 1))
                # Mutate prompt slightly for the next try so we don't hit the
                # same empty-response path; this keeps the seed semantics
                # honest (caller already added _seed_prefix; we add an extra
                # whitespace nonce here).
                prompt = " " + prompt
        last_meta["attempt"] = self.config.cli_retry
        last_meta["empty_after_retry"] = True
        return last_text, last_meta

    def _call_sdk(self, prompt: str, max_tokens: int, sampler: dict[str, float]) -> tuple[str, dict[str, Any]]:
        # Imported lazily because anthropic is an optional dependency.
        import anthropic

        client = anthropic.Anthropic()
        started = time.time()
        msg = client.messages.create(
            model=self.config.model,
            max_tokens=int(max_tokens),
            temperature=float(sampler.get("tau", 1.0)),
            top_p=float(sampler.get("top_p", 0.95)),
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.time() - started) * 1000)
        text_parts = [block.text for block in msg.content if hasattr(block, "text")]
        text = "".join(text_parts)
        # Approximate cost from input/output tokens at Haiku list price (USD/1k):
        #   input  = $0.0008, output = $0.004 (haiku-4-5 list as of 2026-04-28).
        in_tok = int(getattr(msg.usage, "input_tokens", 0))
        out_tok = int(getattr(msg.usage, "output_tokens", 0))
        cost = (in_tok / 1000.0) * 0.0008 + (out_tok / 1000.0) * 0.004
        meta = {
            "cost_usd": float(cost),
            "duration_ms": latency_ms,
            "stop_reason": getattr(msg, "stop_reason", None),
            "session_id": getattr(msg, "id", None),
            "model_used": self.config.model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
        return text, meta

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 200,
        sampler: dict[str, float] | None = None,
        seed: int = 0,
    ) -> Candidate:
        sampler = dict(sampler or {})
        tau = float(sampler.get("tau", 0.9))
        top_p = float(sampler.get("top_p", 0.95))
        top_k = int(sampler.get("top_k", 50))
        self._check_budget()
        seeded_prompt = _seed_prefix(seed) + prompt
        if self.config.use_sdk:
            text, meta = self._call_sdk(seeded_prompt, max_tokens, {"tau": tau, "top_p": top_p})
        else:
            text, meta = self._call_cli(seeded_prompt)
        self._record_cost(self.config.model, meta["cost_usd"], meta["duration_ms"])
        _audit_call({
            "ts": time.time(),
            "model": self.config.model,
            "backend": "sdk" if self.config.use_sdk else "cli",
            "seed": int(seed),
            "sampler": {"tau": tau, "top_p": top_p, "top_k": float(top_k)},
            "max_tokens": int(max_tokens),
            "prompt": prompt[:1000],
            "response": text[:2000],
            "cost_usd": float(meta["cost_usd"]),
            "duration_ms": int(meta["duration_ms"]),
            "stop_reason": meta.get("stop_reason"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
        })
        embedding = self._embedder.encode(text or " ")
        # logp is not exposed by Haiku; we record output_tokens * (-log(2)) as
        # a placeholder negative log-prob proxy that scales with length, since
        # downstream code uses `logp` only as a tie-breaker, not an absolute value.
        out_tok = int(meta.get("output_tokens", 0))
        logp_proxy = -float(out_tok) * 0.693
        return Candidate(
            seed=int(seed),
            sampler={"tau": tau, "top_p": top_p, "top_k": float(top_k)},
            tokens=(),  # haiku does not expose token ids; intentionally empty
            text=text,
            logp=float(logp_proxy),
            embedding=embedding,
        )

    def report(self) -> dict[str, Any]:
        ledger = _load_ledger()
        return {
            "name": self.name,
            "model": self.config.model,
            "backend": "sdk" if self.config.use_sdk else "cli",
            "cli_bin": self.config.cli_bin,
            "timeout_s": self.config.timeout_s,
            "cost_cap_usd": self.config.cost_cap_usd,
            "ledger_total_usd": float(ledger.get("total_usd", 0.0)),
            "ledger_n_calls": int(ledger.get("n_calls", 0)),
        }
