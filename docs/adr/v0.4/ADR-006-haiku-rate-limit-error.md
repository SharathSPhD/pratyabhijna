# ADR-006 (v0.4) — `HaikuRateLimitError` typed exception

Status: Accepted (frozen at end of Phase 1).
Date: 2026-04-29.

## Context

The v0.3 adversarial review found that the fresh sample plugin smoke run encountered `429: You're out of extra usage` mid-run (later control-arm probes), but the error surface collapsed to a generic `HaikuLM CLI rc=1` with empty stderr. The downstream caller could not distinguish a real implementation bug from an externally-caused quota exhaustion. Smoke summaries cannot self-diagnose under this collapse.

The v0.3 `HaikuLM._call_cli_once` only surfaced `stderr` on non-zero exit. Claude CLI returns the useful JSON error body on `stdout` even when the exit code is 1, so the diagnostic information was discarded.

## Decision

Add a typed exception hierarchy in `src/pce/substrate/errors.py`:

```python
class HaikuError(Exception):
    """Base class for HaikuLM-originated errors."""
    def __init__(self, message: str, parsed: dict | None = None) -> None:
        super().__init__(message)
        self.parsed = parsed or {}

class HaikuRateLimitError(HaikuError):
    """Claude CLI returned `api_error_status == 429` (rate-limit / quota)."""

class HaikuApiError(HaikuError):
    """Claude CLI returned a non-rate-limit API error (e.g. 5xx)."""

class HaikuCLIError(HaikuError):
    """Claude CLI exited non-zero with no parseable JSON body."""
```

`HaikuLM._call_cli_once` rewritten to:

```python
proc = subprocess.run(...)
stdout_text = proc.stdout.decode(errors="replace")
parsed: dict | None = _try_parse_json(stdout_text)
if parsed and parsed.get("is_error"):
    api_status = parsed.get("api_error_status")
    if api_status == 429:
        raise HaikuRateLimitError(
            parsed.get("result", "rate-limited"), parsed=parsed
        )
    raise HaikuApiError(
        parsed.get("result", f"api error (status={api_status})"), parsed=parsed
    )
if proc.returncode != 0:
    raise HaikuCLIError(
        f"rc={proc.returncode}",
        parsed={"stderr": proc.stderr.decode(errors='replace'), "stdout": stdout_text},
    )
return parsed
```

The `parsed` dict is propagated into the cascade audit trace so the smoke summary can record `api_error_status` and `result` per failed call.

## Driver behaviour under rate-limit

`benchmarks/driver.py` catches `HaikuRateLimitError` at the per-item boundary:

- Log to `audit/cost_ledger_v0_4.json` with `error_type="rate_limit"` and the full parsed body.
- Persist all completed items to disk before exiting.
- Exit code 2 (distinct from 1 for normal errors) so CI / scripts can branch.

`scripts/smoke_plugin.py --with-haiku` similarly reports rate-limit failures separately from implementation failures in the summary JSON:

```json
{
  "ok": false,
  "pass": 20,
  "fail_implementation": 0,
  "fail_rate_limit": 2,
  "fail_other": 0,
  "skipped": 0
}
```

## Consequences

Positive:

- Smoke runs and pilot runs now distinguish quota exhaustion from real bugs.
- Operators can see `api_error_status: 429` in the cost ledger and know to retry on the next quota window.
- The v0.3 review's recommendation 4 ("Add a sample-smoke note that quota/rate-limit failures are externally caused") is satisfied.

Negative:

- One new module (`src/pce/substrate/errors.py`) and an updated `_call_cli_once`. Small surface; unit-tested.
- `HaikuRateLimitError` is unrecoverable mid-pilot — driver exits without retry. v0.5 may add an opt-in retry policy.

## Implementation files

- `src/pce/substrate/errors.py` — typed exception hierarchy.
- `src/pce/substrate/haiku_lm.py` — rewritten `_call_cli_once`; propagates `parsed` into audit.
- `benchmarks/driver.py` — catches `HaikuRateLimitError`, persists state, exits 2.
- `scripts/smoke_plugin.py` — splits `fail_*` counters; surfaces `api_error_status`.
- `tests/test_haiku_rate_limit_error.py` — synthetic 429 stdout fixture raises `HaikuRateLimitError`; synthetic non-429 API error raises `HaikuApiError`; non-JSON rc=1 raises `HaikuCLIError`.

## Acceptance gate (Phase 2)

- `tests/test_haiku_rate_limit_error.py` passes (all three branches).
- Synthetic 429 fixture is logged with `api_error_status=429` in the audit trace.
- `scripts/smoke_plugin.py --with-haiku` summary distinguishes `fail_implementation`, `fail_rate_limit`, `fail_other`.
