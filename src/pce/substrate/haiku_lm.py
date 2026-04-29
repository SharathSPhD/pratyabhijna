"""Haiku substrate adapter — wraps the `claude` CLI as a `GeneratorProtocol`.

Per [docs/adr/v0.3/ADR-001-clean-haiku-cli.md](../../../docs/adr/v0.3/ADR-001-clean-haiku-cli.md)
and [docs/SPEC_v0.3.md §1.1](../../../docs/SPEC_v0.3.md): each `claude --print` call runs
in a *clean inner subprocess* with isolation flags and a scrubbed environment so
Claude Code system prompt, plugin context, skill context, MCP context, and project
`CLAUDE.md` never leak into the response.

The *outer host* (Python or Claude Code session) keeps its PCE plugin loaded so
the cascade is callable at all. Only the inner spawned subprocess is sanitized.
The `IntegrityProbe` ([src/pce/substrate/integrity.py](integrity.py)) attests to
the inner-subprocess view; `scripts/verify_outer_host_loads_pce.py` attests to the
outer-host view.

Two backends:

* `cli` (default, v0.3 clean substrate): subprocess `claude --print --output-format
  json --model haiku --system-prompt "..." --disable-slash-commands --strict-mcp-config
  --setting-sources "" --permission-mode bypassPermissions --no-session-persistence
  <prompt>`, invoked via `subprocess.run(env=clean_env, cwd=tmp_clean_dir)`.
* `sdk` (opt-in via `PCE_USE_SDK=1`): `anthropic.Anthropic().messages.create(...)`.
  Requires `pip install anthropic` and `ANTHROPIC_API_KEY`. NOT used in v0.3 by
  scope (per user constraint); kept importable for backward compatibility.

Determinism: the CLI does not expose a seed, so the substrate seeds a per-call
*nonce prefix* on the prompt to make repeated calls with the same `seed`
return semantically equivalent (not byte-identical) text. Audit logs record
the seed alongside the response.

Cost telemetry: every call appends to `audit/haiku/<ts>.json` and updates
`audit/cost_ledger.json`. The cascade driver reads the ledger before each
new call and aborts gracefully if the running total approaches the envelope.
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pce.substrate.embed import Embedder
from pce.substrate.errors import (
    HaikuApiError,
    HaikuCLIError,
    HaikuError,
    HaikuRateLimitError,
)
from pce.types import Candidate

__all__ = [
    "HaikuLM",
    "HaikuConfig",
    "HaikuBudgetExceededError",
    "CleanSubstrateAuthError",
    "HaikuError",
    "HaikuRateLimitError",
    "HaikuApiError",
    "HaikuCLIError",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO_ROOT / "audit" / "haiku"
COST_LEDGER = REPO_ROOT / "audit" / "cost_ledger.json"

# Frozen system prompt override. Replaces the default Claude Code system prompt
# (which carries plugin / skill / framing context). Value is the same neutral
# stub Claude.ai uses on a fresh session.
DEFAULT_SYSTEM_PROMPT_OVERRIDE: str = "You are a helpful assistant."

# Frozen CLI isolation flags. Order is stable so flags_hash is reproducible.
#
# `--tools ""` is variadic in the Claude CLI: it consumes everything up to the
# next known flag. We MUST keep `--tools` "" early in the chain so that another
# named flag (here: `--strict-mcp-config`) immediately follows and breaks out
# of variadic consumption — otherwise the user prompt gets parsed as a tool
# name and the call fails. Empirically this disables all built-in tools (Bash,
# Read, Edit, Write, etc.) and removes the bundled-skill list from the system
# prompt; without it `--disable-slash-commands` only blocks /skill-name
# invocation but the model still sees "you have 10 skills loaded: ...".
DEFAULT_ISOLATION_FLAGS: tuple[str, ...] = (
    "--tools",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--setting-sources",
    "",
    "--permission-mode",
    "bypassPermissions",
    "--no-session-persistence",
)

# Allow-list of env vars carried into the inner subprocess. Everything else is
# dropped. We never `os.environ.copy()`; clean_env is built explicitly.
ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "__CF_USER_TEXT_ENCODING",  # macOS-specific; harmless if absent on Linux
)

# Env vars that, if present in the parent, indicate the parent is itself a
# Claude Code session. We warn but do NOT mutate parent state.
CLAUDE_PARENT_ENV_PREFIXES: tuple[str, ...] = ("CLAUDE_CODE_", "CLAUDE_PROJECT_DIR")


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
    # v0.3 clean-substrate config.
    clean_substrate: bool = True
    clean_home_root: str | None = None  # None -> tempfile.gettempdir()
    system_prompt_override: str = DEFAULT_SYSTEM_PROMPT_OVERRIDE

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
            clean_substrate=os.environ.get("PCE_HAIKU_CLEAN_SUBSTRATE", "1").strip() != "0",
            clean_home_root=os.environ.get("PCE_HAIKU_CLEAN_HOME") or None,
            system_prompt_override=os.environ.get(
                "PCE_HAIKU_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT_OVERRIDE
            ),
        )


class HaikuBudgetExceededError(RuntimeError):
    """Raised when the cost ledger total reaches `cost_cap_usd`."""


class CleanSubstrateAuthError(RuntimeError):
    """Raised when the clean inner subprocess cannot authenticate.

    Most commonly: the OAuth credential is not reachable from the scrubbed
    `HOME`. On macOS the keychain symlink may have failed; on Linux the
    credentials JSON may live somewhere we did not symlink. The remedy is to
    either (a) re-run `claude /login` to refresh credentials, or (b) override
    `PCE_HAIKU_CLEAN_SUBSTRATE=0` to fall back to the v0.2 inheriting-env path
    (which leaks Claude Code context but at least authenticates).
    """


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
    """Produce a tiny invisible-to-meaning prefix derived from seed."""
    rng = np.random.default_rng(int(seed))
    pad = " " * int(rng.integers(0, 4))
    nonce = int(rng.integers(0, 99_999))
    return f"{pad}<!-- pce-seed:{nonce} -->\n"


# --- Clean substrate plumbing (v0.3) ----------------------------------------

# We track instances we've created at module level so atexit can clean them up
# even if the parent process forgets to call `HaikuLM.close()`.
_CREATED_CLEAN_HOMES: list[Path] = []


def _atexit_cleanup() -> None:
    for p in list(_CREATED_CLEAN_HOMES):
        try:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:  # pragma: no cover — best-effort cleanup
            pass


atexit.register(_atexit_cleanup)


def _setup_clean_home(pid: int, root: str | None = None) -> Path:
    """Create a scrubbed HOME for the inner subprocess.

    Layout (macOS):
      /tmp/pce_home_<pid>/
        Library/Keychains -> ~/Library/Keychains       (symlink, OAuth)

    Layout (Linux):
      /tmp/pce_home_<pid>/
        .config/claude    -> ~/.config/claude          (symlink, OAuth, if exists)

    Crucially we do NOT symlink:
      ~/.claude/                  (plugins, skills, settings, agents, sessions, CLAUDE.md auto-memory)
      ~/.config/claude/plugins/   (if claude grows a Linux plugin dir there)
      project CLAUDE.md           (avoided by also setting cwd to a temp dir outside the repo)
    """
    base = Path(root) if root else Path(tempfile.gettempdir())
    clean_home = base / f"pce_home_{pid}"
    clean_home.mkdir(parents=True, exist_ok=True)
    try:
        clean_home.chmod(0o700)
    except (OSError, PermissionError):  # pragma: no cover
        pass

    real_home_str = os.environ.get("HOME", "")
    if not real_home_str:
        real_home_str = os.path.expanduser("~")
    real_home = Path(real_home_str)

    if platform.system() == "Darwin":
        # macOS: keychain lives at ~/Library/Keychains/login.keychain-db
        kc_src = real_home / "Library" / "Keychains"
        if kc_src.exists():
            (clean_home / "Library").mkdir(exist_ok=True)
            kc_link = clean_home / "Library" / "Keychains"
            if not kc_link.exists():
                os.symlink(kc_src, kc_link)
    else:
        # Linux/other: claude credentials JSON commonly lives at ~/.config/claude/
        cfg_src = real_home / ".config" / "claude"
        if cfg_src.exists():
            (clean_home / ".config").mkdir(exist_ok=True)
            cfg_link = clean_home / ".config" / "claude"
            if not cfg_link.exists():
                os.symlink(cfg_src, cfg_link)
        # Also try ~/.claude/ as a fallback for older installs (only the file
        # that is a credential-JSON, never the whole directory which contains
        # plugins/skills/CLAUDE.md memory).
        # We prefer to be conservative here and only handle the .config path;
        # if credentials live elsewhere on a Linux box, set PCE_HAIKU_CLEAN_HOME
        # to a manually-prepared dir.

    _CREATED_CLEAN_HOMES.append(clean_home)
    return clean_home


def _setup_clean_cwd(pid: int) -> Path:
    """Create a per-process scratch cwd outside the repo.

    The `claude` CLI walks up from cwd looking for `CLAUDE.md`; running from
    a fresh empty dir under /tmp guarantees nothing is auto-discovered.
    """
    cwd = Path(tempfile.gettempdir()) / f"pce_clean_{pid}"
    cwd.mkdir(parents=True, exist_ok=True)
    try:
        cwd.chmod(0o700)
    except (OSError, PermissionError):  # pragma: no cover
        pass
    _CREATED_CLEAN_HOMES.append(cwd)
    return cwd


def _build_clean_env(home: Path) -> dict[str, str]:
    """Build a minimal env from the allow-list. NEVER `os.environ.copy()`."""
    env: dict[str, str] = {}
    parent_env = os.environ
    for key in ENV_ALLOWLIST:
        if key in parent_env:
            env[key] = parent_env[key]
    env["HOME"] = str(home)
    # Make sure PATH at least contains where `claude` likely lives.
    if "PATH" not in env:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    # Force the subprocess into a vanilla locale so JSON parsing is stable.
    env.setdefault("LANG", "C.UTF-8")
    return env


def _warn_if_parent_has_claude_env() -> None:
    leaks = [
        k for k in os.environ
        if any(k.startswith(p) for p in CLAUDE_PARENT_ENV_PREFIXES)
    ]
    if leaks:
        warnings.warn(
            f"HaikuLM: parent process holds Claude Code env vars {leaks!r}. "
            "These are NOT inherited by the inner subprocess (we build env from "
            "an explicit allow-list), but you may want to investigate why the "
            "host process has them.",
            RuntimeWarning,
            stacklevel=3,
        )


# ----------------------------------------------------------------------------


class HaikuLM:
    """Claude Haiku substrate via the `claude` CLI in a clean inner subprocess.

    Implements `pce.substrate.lm_protocol.GeneratorProtocol` (alias `LMProtocol`).
    """

    name: str = "claude-haiku"
    supports_logprobs: bool = False
    supports_score: bool = False
    supports_entropy: bool = False

    def __init__(
        self,
        config: HaikuConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or HaikuConfig.from_env()
        self._embedder = embedder or Embedder()
        self._isolation_flags: tuple[str, ...] = (
            DEFAULT_ISOLATION_FLAGS if self.config.clean_substrate else ()
        )
        self._clean_home: Path | None = None
        self._clean_cwd: Path | None = None
        if self.config.clean_substrate and not self.config.use_sdk:
            _warn_if_parent_has_claude_env()
            self._clean_home = _setup_clean_home(os.getpid(), self.config.clean_home_root)
            self._clean_cwd = _setup_clean_cwd(os.getpid())
        if self.config.use_sdk:
            try:
                import anthropic  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "PCE_USE_SDK=1 requires `pip install anthropic` (Python SDK)."
                ) from exc

    # --- protocol surface ----------------------------------------------------

    def length_proxy_logp(self, candidate: Candidate) -> float:
        """Honest length-proportional logp proxy. NOT a real log-probability.

        Returns `-output_tokens * log(2)` so longer outputs have lower (more
        negative) proxy, matching the convention real logp follows. Callers
        must check `supports_logprobs` and treat this as a tie-breaker only.
        """
        # Re-derive output token count from the candidate's text length when
        # the substrate didn't expose tokens (Haiku doesn't).
        if candidate.tokens:
            n_tok = len(candidate.tokens)
        else:
            # ~4 chars / token rule of thumb for English.
            n_tok = max(1, len(candidate.text) // 4)
        return -float(n_tok) * 0.693

    # --- internals -----------------------------------------------------------

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

    def _build_cmd(self, prompt: str) -> list[str]:
        cmd = [
            self.config.cli_bin,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.config.model,
        ]
        if self.config.clean_substrate:
            cmd += ["--system-prompt", self.config.system_prompt_override]
            cmd += list(self._isolation_flags)
        cmd.append(prompt)
        return cmd

    def _call_cli_once(self, prompt: str) -> tuple[str, dict[str, Any]]:
        cmd = self._build_cmd(prompt)
        run_kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "capture_output": True,
            "check": False,
            "timeout": self.config.timeout_s,
        }
        if self.config.clean_substrate:
            assert self._clean_home is not None and self._clean_cwd is not None
            run_kwargs["env"] = _build_clean_env(self._clean_home)
            run_kwargs["cwd"] = str(self._clean_cwd)
        started = time.time()
        proc = subprocess.run(cmd, **run_kwargs)  # noqa: S603 — CLI is trusted user-installed binary
        latency_ms = int((time.time() - started) * 1000)

        # v0.4 (ADR-006): parse stdout JSON even when rc != 0. Claude CLI returns
        # the useful error body on stdout (e.g. is_error=True, api_error_status=429
        # for quota exhaustion) while still exiting non-zero. The v0.3 path
        # discarded that body via `stderr_tail` and raised a generic RuntimeError;
        # v0.4 raises a typed HaikuRateLimitError / HaikuApiError so the driver
        # and smoke harness can distinguish externally-caused failures from
        # implementation bugs.
        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
        payload: dict[str, Any] | None
        try:
            payload = json.loads(stdout_text) if stdout_text.strip() else None
        except json.JSONDecodeError:
            payload = None

        if payload is not None and payload.get("is_error", False):
            api_status = payload.get("api_error_status")
            if api_status == 429:
                raise HaikuRateLimitError(
                    payload.get("result", "rate-limited") or "rate-limited",
                    parsed=payload,
                )
            raise HaikuApiError(
                payload.get("result", f"api error (status={api_status})") or
                f"api error (status={api_status})",
                parsed=payload,
            )
        if proc.returncode != 0:
            # Heuristic: detect 401/auth failures up front so we surface the right error.
            if (
                "401" in stderr_tail
                or "auth" in stderr_tail.lower()
                or "credential" in stderr_tail.lower()
            ):
                raise CleanSubstrateAuthError(
                    f"Clean inner subprocess failed to authenticate (rc={proc.returncode}). "
                    f"OAuth credential may not be reachable from the scrubbed HOME. "
                    f"stderr tail: {stderr_tail!r}. "
                    f"Remedy: run `claude /login` to refresh, or set "
                    f"PCE_HAIKU_CLEAN_SUBSTRATE=0 to disable isolation (leaks Claude Code context)."
                )
            raise HaikuCLIError(
                f"rc={proc.returncode}: {stderr_tail}",
                parsed={"stderr": stderr_tail, "stdout": stdout_text[:1000]},
            )
        if payload is None:
            raise HaikuCLIError(
                "CLI returned no parseable JSON body on stdout",
                parsed={"stderr": stderr_tail, "stdout": stdout_text[:1000]},
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
            "clean_substrate": bool(self.config.clean_substrate),
            "isolation_flags": list(self._isolation_flags),
        }
        return text, meta

    def _call_cli(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Wrapper around _call_cli_once with retry-on-empty.

        v0.4: ``HaikuRateLimitError`` and ``HaikuApiError`` propagate immediately
        without retry — those are externally caused and retrying inside the
        same call burns more quota / cost. Only ``HaikuCLIError`` (generic
        non-zero rc with no JSON body) and empty-but-OK responses are retried.
        """
        last_text = ""
        last_meta: dict[str, Any] = {}
        for attempt in range(self.config.cli_retry + 1):
            try:
                text, meta = self._call_cli_once(prompt)
            except (HaikuRateLimitError, HaikuApiError):
                raise
            except HaikuCLIError:
                if attempt >= self.config.cli_retry:
                    raise
                time.sleep(self.config.cli_backoff_s * (attempt + 1))
                continue
            last_text, last_meta = text, meta
            if text.strip():
                meta["attempt"] = attempt
                return text, meta
            if attempt < self.config.cli_retry:
                time.sleep(self.config.cli_backoff_s * (attempt + 1))
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
        in_tok = int(getattr(msg.usage, "input_tokens", 0))
        out_tok = int(getattr(msg.usage, "output_tokens", 0))
        cost = (in_tok / 1000.0) * 0.0008 + (out_tok / 1000.0) * 0.004
        meta: dict[str, Any] = {
            "cost_usd": float(cost),
            "duration_ms": latency_ms,
            "stop_reason": getattr(msg, "stop_reason", None),
            "session_id": getattr(msg, "id", None),
            "model_used": self.config.model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "clean_substrate": False,
            "isolation_flags": [],
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
            "clean_substrate": bool(meta.get("clean_substrate", False)),
            "isolation_flags": list(meta.get("isolation_flags", [])),
        })
        embedding = self._embedder.encode(text or " ")
        out_tok = int(meta.get("output_tokens", 0))
        logp_proxy = -float(out_tok) * 0.693 if out_tok > 0 else -1.0
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
            "clean_substrate": bool(self.config.clean_substrate),
            "isolation_flags": list(self._isolation_flags),
            "clean_home": str(self._clean_home) if self._clean_home else None,
            "clean_cwd": str(self._clean_cwd) if self._clean_cwd else None,
            "supports_logprobs": self.supports_logprobs,
            "supports_score": self.supports_score,
            "supports_entropy": self.supports_entropy,
        }
