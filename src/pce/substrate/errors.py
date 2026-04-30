"""Typed exception hierarchy for the Haiku CLI substrate.

Per [docs/adr/v0.4/ADR-006-haiku-rate-limit-error.md](../../../docs/adr/v0.4/ADR-006-haiku-rate-limit-error.md).

The v0.3 ``HaikuLM._call_cli_once`` collapsed every non-zero ``rc`` into a
generic ``RuntimeError("HaikuLM CLI rc={rc}")`` and threw away the JSON body
on stdout. Claude CLI returns a usable JSON body on ``stdout`` even when the
exit code is 1 (e.g. quota exhaustion sets ``"is_error": true,
"api_error_status": 429`` while still exiting non-zero), so the diagnostic
information was discarded. v0.4 surfaces it through this typed hierarchy so
downstream callers can distinguish quota / rate-limit failures (externally
caused) from real implementation bugs.

The four classes form a hierarchy:

* ``HaikuError``           — base; carries an optional ``parsed`` dict.
* ``HaikuRateLimitError``  — Claude CLI returned ``api_error_status == 429``.
* ``HaikuApiError``        — Claude CLI returned a non-rate-limit API error.
* ``HaikuCLIError``        — Claude CLI exited non-zero with no parseable body.

The driver and smoke harness branch on ``HaikuRateLimitError`` to persist
state and exit cleanly when quota is exhausted, instead of silently
reporting an implementation failure.
"""
from __future__ import annotations

from typing import Any


class HaikuError(Exception):
    """Base class for HaikuLM-originated errors.

    ``parsed`` carries whatever JSON Claude CLI emitted on stdout; it may be
    empty for true CLI failures (rc != 0 with no JSON body).
    """

    def __init__(self, message: str, *, parsed: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.parsed: dict[str, Any] = dict(parsed or {})

    def api_error_status(self) -> int | None:
        status = self.parsed.get("api_error_status")
        if status is None:
            return None
        try:
            return int(status)
        except (TypeError, ValueError):
            return None

    def result(self) -> str:
        return str(self.parsed.get("result", "") or "")


class HaikuRateLimitError(HaikuError):
    """Claude CLI returned ``api_error_status == 429`` (rate limit / quota)."""


class HaikuApiError(HaikuError):
    """Claude CLI returned a non-rate-limit API error (e.g. 5xx)."""


class HaikuCLIError(HaikuError):
    """Claude CLI exited non-zero with no parseable JSON body."""
