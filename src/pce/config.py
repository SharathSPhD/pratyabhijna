"""PCE v0.4 portable configuration.

This module provides a single, IDE-agnostic configuration surface that
collapses the v0.3 ad-hoc env-var chain (``PCE_HAIKU_MODEL``,
``PCE_HAIKU_CLI``, ``PCE_HAIKU_TIMEOUT_S``, …) into one ``PCEConfig``
dataclass. ``PCEConfig.load()`` resolves a config from five layers; each
later layer overrides the earlier ones (i.e. CLI flags win over env
vars, env vars win over user TOML, etc.):

1. **Hard-coded defaults** — the dataclass field defaults
   (``cascade_model="haiku"``, ``judge_model="sonnet"``, ``cli_bin="claude"``).
2. **Repo TOML** — ``./pce.toml`` next to the working dir; lets projects
   pin their own model defaults without editing user state.
3. **User TOML** — ``~/.config/pce/config.toml`` (XDG; honours
   ``$XDG_CONFIG_HOME/pce/config.toml`` if set).
4. **Env vars** — ``PCE_*`` (e.g. ``PCE_MODEL``, ``PCE_JUDGE_MODEL``,
   ``PCE_CLI``, ``PCE_TIMEOUT_S``, ``PCE_COST_CAP_USD``); the v0.3-era
   ``PCE_HAIKU_*`` aliases are still honoured for back-compat.
5. **Explicit override** — ``overrides`` passed to
   ``PCEConfig.load(overrides=...)``; this is how the standalone
   ``pce`` CLI threads ``--model`` / ``--cli-bin`` / ``--config`` /
   ``--timeout-s`` through.

The Anthropic Python SDK code path that v0.3 carried (opt-in via
``PCE_USE_SDK=1``) is deprecated in v0.4 and removed at substrate level
([docs/adr/v0.4/ADR-007-sdk-removal.md]). Setting ``PCE_USE_SDK=1`` now
emits a ``DeprecationWarning`` and is ignored. The OAuth/CLI substrate
is the only supported path; this is what makes the plugin portable
across Cursor, Claude Code, Bedrock, Vertex, and direct API backends
without changing the cascade code itself.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib  # noqa: I001
else:  # pragma: no cover — supported python is 3.11+
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = [
    "PCEConfig",
    "MODEL_ALIASES",
    "resolve_model",
    "default_user_config_path",
    "default_repo_config_path",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


# Short model aliases that route to "current production" Anthropic models
# served via the ``claude`` CLI. Users may also pass any explicit Anthropic
# CLI model ID (e.g. ``claude-haiku-4-5-20251001`` or the full Bedrock ARN
# ``global.anthropic.claude-sonnet-4-5-20250929-v1:0``). Aliases are
# resolved at call time by ``resolve_model``.
MODEL_ALIASES: dict[str, str] = {
    # The CLI accepts both the bare alias and the full ID. We keep the bare
    # alias so that the same config file works whether the CLI is talking
    # to OAuth, Bedrock (which expects the full ID), or Vertex.
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
    # Convenience aliases that are sometimes useful in TOML.
    "claude-haiku": "haiku",
    "claude-sonnet": "sonnet",
    "claude-opus": "opus",
}


def resolve_model(name_or_id: str) -> str:
    """Map a short alias (``haiku``) to its CLI argument; pass through otherwise.

    The Anthropic CLI accepts both forms. When the user has set
    ``CLAUDE_CODE_USE_BEDROCK=1`` and ``ANTHROPIC_MODEL`` to a full Bedrock
    inference-profile ID, that env var wins via the standard CLI behaviour
    and the value we return here is used as the ``--model`` flag (which
    Bedrock interprets as a routing hint).
    """
    key = (name_or_id or "").strip().lower()
    return MODEL_ALIASES.get(key, name_or_id)


def default_user_config_path() -> Path:
    """Return the user-level TOML config path (XDG-aware)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "pce" / "config.toml"


def default_repo_config_path() -> Path:
    """Return the repo-level TOML config path (next to cwd)."""
    return Path.cwd() / "pce.toml"


@dataclass(frozen=True)
class PCEConfig:
    """Portable PCE configuration.

    Field ordering tracks the v0.4 paper's *substrate plurality* claim:
    cascade and judge are picked independently so a researcher can run
    cascade on Haiku and judge on Sonnet (the default), or e.g. cascade on
    Sonnet and judge on Opus for a stronger oracle.

    Only fields the standalone CLI / Cursor manifest / Phase 7 driver
    actually read are surfaced here. Anything substrate-internal stays in
    ``HaikuConfig``; ``HaikuConfig.from_env`` defers to ``PCEConfig`` for
    its model / cli / timeout / cost-cap fields.
    """

    cascade_model: str = "haiku"
    judge_model: str = "sonnet"
    cli_bin: str = "claude"
    timeout_s: int = 120
    cost_cap_usd: float = 18.0
    cli_retry: int = 2
    cli_backoff_s: float = 1.0
    clean_substrate: bool = True
    clean_home_root: str | None = None
    system_prompt_override: str = "You are a helpful assistant."
    extras: dict[str, Any] = field(default_factory=dict)

    # ----- loaders -----------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        user_toml: Path | None = None,
        repo_toml: Path | None = None,
        env: dict[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> PCEConfig:
        """Resolve a ``PCEConfig`` by layering five sources, later wins.

        The precedence chain is, low → high:
        ``defaults → repo TOML (./pce.toml) → user TOML
        (~/.config/pce/config.toml) → env vars (PCE_*) → explicit
        overrides``. Each later source wins over the earlier ones.

        ``user_toml`` and ``repo_toml`` may point to non-existent paths;
        missing files are skipped silently. ``env`` defaults to
        ``os.environ``; ``overrides`` defaults to ``{}``. Malformed TOML
        emits a ``RuntimeWarning`` and is skipped.
        """
        if env is None:
            env = dict(os.environ)
        if overrides is None:
            overrides = {}

        cfg = cls()

        repo_path = repo_toml if repo_toml is not None else default_repo_config_path()
        user_path = user_toml if user_toml is not None else default_user_config_path()
        for path in (repo_path, user_path):
            if not path.exists():
                continue
            try:
                with path.open("rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                warnings.warn(
                    f"PCEConfig: ignoring malformed TOML at {path}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            cfg = cfg._merge_toml(data)

        cfg = cfg._merge_env(env)

        if overrides:
            cfg = cfg._merge_overrides(overrides)

        # SDK deprecation warning — user-facing notice that v0.4 only
        # supports the OAuth/CLI substrate.
        if env.get("PCE_USE_SDK", "").strip() == "1":
            warnings.warn(
                "PCE_USE_SDK is deprecated and ignored as of v0.4. "
                "The Anthropic Python SDK code path was removed; PCE now "
                "exclusively uses the `claude` CLI substrate (which itself "
                "can be pointed at Bedrock, Vertex, or direct API via "
                "ANTHROPIC_*/CLAUDE_CODE_* env vars). See "
                "docs/adr/v0.4/ADR-007-sdk-removal.md for the rationale.",
                DeprecationWarning,
                stacklevel=2,
            )

        return cfg

    # ----- merging helpers ---------------------------------------------------

    def _merge_toml(self, data: dict[str, Any]) -> PCEConfig:
        """Merge a parsed TOML mapping. Unknown keys go into ``extras``."""
        section = data.get("pce") if isinstance(data.get("pce"), dict) else data
        if not isinstance(section, dict):
            return self
        return self._apply_kwargs(section, source="toml")

    def _merge_env(self, env: dict[str, str]) -> PCEConfig:
        """Merge the ``PCE_*`` env-var family + v0.3 ``PCE_HAIKU_*`` aliases."""
        kw: dict[str, Any] = {}

        def _put(field_name: str, env_name: str, value: str | None, cast: type | None = None) -> None:
            if value is None or value == "":
                return
            if cast is None:
                kw[field_name] = value
                return
            try:
                kw[field_name] = cast(value)
            except (TypeError, ValueError):
                warnings.warn(
                    f"PCEConfig: ignoring invalid env value for {env_name}={value!r}",
                    RuntimeWarning,
                    stacklevel=3,
                )

        def _first(names: tuple[str, ...]) -> tuple[str, str | None]:
            """Return (env_name_used, value) for first set var, or (names[0], None)."""
            for n in names:
                v = env.get(n)
                if v is not None and v != "":
                    return n, v
            return names[0], None

        n, v = _first(("PCE_MODEL", "PCE_CASCADE_MODEL", "PCE_HAIKU_MODEL"))
        _put("cascade_model", n, v)
        _put("judge_model", "PCE_JUDGE_MODEL", env.get("PCE_JUDGE_MODEL"))
        n, v = _first(("PCE_CLI", "PCE_HAIKU_CLI"))
        _put("cli_bin", n, v)
        n, v = _first(("PCE_TIMEOUT_S", "PCE_HAIKU_TIMEOUT_S"))
        _put("timeout_s", n, v, int)
        n, v = _first(("PCE_COST_CAP_USD", "PCE_HAIKU_COST_CAP_USD"))
        _put("cost_cap_usd", n, v, float)
        n, v = _first(("PCE_CLI_RETRY", "PCE_HAIKU_CLI_RETRY"))
        _put("cli_retry", n, v, int)
        n, v = _first(("PCE_CLI_BACKOFF_S", "PCE_HAIKU_CLI_BACKOFF_S"))
        _put("cli_backoff_s", n, v, float)
        v_clean = env.get("PCE_CLEAN_SUBSTRATE") or env.get("PCE_HAIKU_CLEAN_SUBSTRATE")
        if v_clean is not None:
            kw["clean_substrate"] = v_clean.strip() != "0"
        n, v = _first(("PCE_CLEAN_HOME", "PCE_HAIKU_CLEAN_HOME"))
        _put("clean_home_root", n, v)
        n, v = _first(("PCE_SYSTEM_PROMPT", "PCE_HAIKU_SYSTEM_PROMPT"))
        _put("system_prompt_override", n, v)

        if not kw:
            return self
        return replace(self, **kw)

    def _merge_overrides(self, overrides: dict[str, Any]) -> PCEConfig:
        return self._apply_kwargs(overrides, source="override")

    def _apply_kwargs(self, kwargs: dict[str, Any], *, source: str) -> PCEConfig:
        known = {f for f in self.__dataclass_fields__ if f != "extras"}
        clean: dict[str, Any] = {}
        extras: dict[str, Any] = dict(self.extras)
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in known:
                clean[k] = v
            else:
                extras[k] = v
        if not clean and not extras:
            return self
        if extras and "extras" not in clean:
            clean["extras"] = extras
        return replace(self, **clean)

    # ----- accessors used by the substrate adapters ---------------------------

    def resolved_cascade_model(self) -> str:
        return resolve_model(self.cascade_model)

    def resolved_judge_model(self) -> str:
        return resolve_model(self.judge_model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
