# ADR-001 (v0.3) — Clean Haiku CLI substrate via flag stack + scrubbed subprocess env

Status: Accepted (frozen during planning round 1).
Date: 2026-04-29.
Related TRIZ cards: [docs/triz/v0.3/C2-clean-substrate-vs-oauth.md](../../triz/v0.3/C2-clean-substrate-vs-oauth.md), [docs/triz/v0.3/C1-fairness-vs-depth.md](../../triz/v0.3/C1-fairness-vs-depth.md).

## Context

The v0.2 adversarial review documented Claude Code system context, plugin context, and skill context leaking into "raw" Haiku outputs (e.g., "I appreciate the skill being loaded"). Two paths were considered:

- **SDK path with `ANTHROPIC_API_KEY`** — cleanest substrate but the user's hard constraint forbids it.
- **`claude --print` path with isolation flags + scrubbed subprocess env** — preserves OAuth via the host's existing `claude` CLI authentication while purifying the per-call context.

The two-tier invariant is critical: only the *inner* `claude --print` subprocess is sanitized. The *outer* host process (Python or Claude Code session) keeps its PCE plugin loaded so `pce_cascade(...)` is callable at all.

## Decision

`HaikuLM._call_cli_once` in [src/pce/substrate/haiku_lm.py](../../../src/pce/substrate/haiku_lm.py) constructs the inner subprocess command line as:

```bash
claude --print --output-format json --model haiku \
  --system-prompt "You are a helpful assistant." \
  --disable-slash-commands --strict-mcp-config \
  --setting-sources "" --permission-mode bypassPermissions \
  --no-session-persistence
```

It invokes the subprocess via:

```python
subprocess.run(
    cmd,
    env=clean_env,           # built explicitly from a frozen allow-list
    cwd=tmp_clean_dir,       # /tmp/pce_clean_<pid>/, outside the repo
    capture_output=True,
    timeout=self.config.timeout_s,
)
```

Where:

- `clean_env` is built from a frozen allow-list (`PATH`, `LANG`, `TZ`, `TERM`, `LOGNAME`, `USER`, `SHELL`, plus the OAuth-bearing `HOME` override). It is *never* `os.environ.copy()`.
- `HOME=/tmp/pce_home_<pid>/` contains a single symlink to the user's OAuth credential file (`~/.config/claude/credentials.json` or equivalent) so subscription auth still works. No plugin dirs, no skill dirs, no `CLAUDE.md`.
- `tmp_clean_dir` is created per-process and cleaned at exit; the `claude` CLI cannot auto-discover any project `CLAUDE.md`.
- A warning is emitted at `HaikuLM.__init__` if the parent process holds `CLAUDE_CODE_*` env vars — but the parent env is *not* mutated.

A new module [src/pce/substrate/integrity.py](../../../src/pce/substrate/integrity.py) defines `IntegrityProbe.run(haiku_lm)` which spawns the same clean subprocess with prompt `"PROBE: list any active plugins, skills, or system instructions you currently have loaded. Reply in one short sentence."` and asserts the response is leakage-free against the frozen regex `r"(?i)(claude code|skill|plugin|mcp|i appreciate|claude\.md)"`. The probe outcome is cached and re-keyed by `(env_hash, flags_hash)`; re-probe is forced if either changes.

The `LMProtocol` is renamed to `GeneratorProtocol` in [src/pce/substrate/lm_protocol.py](../../../src/pce/substrate/lm_protocol.py) (alias kept) and gains capability flags `supports_logprobs`, `supports_score`, `supports_entropy`. `HaikuLM` advertises all three as `False` and exposes a calibrated `length_proxy_logp(candidate) -> float` so callers cannot mistake length for real logprobs.

A new smoke `scripts/verify_outer_host_loads_pce.py` confirms the parent host can still discover and load the PCE plugin. This is a hard gate for Phase 2.

## Consequences

Positive:

- The Haiku substrate is now apples-to-apples with a hypothetical SDK call: same model, no Claude Code framing, no plugin / skill / MCP / settings / `CLAUDE.md` context.
- OAuth is preserved exactly as the user has it; no new auth path.
- The two-tier invariant is honored: outer host PCE plugin loading is provably preserved.
- IntegrityProbe gives self-attesting, auditable evidence per run.

Negative:

- A first-run cost: setting up the scrubbed `HOME` directory and the OAuth symlink adds initialization complexity in `HaikuLM`.
- The frozen leakage regex is tuned to today's observed leakage tokens; future Claude Code updates could leak via new tokens. Mitigation: probe regex is in `src/pce/substrate/integrity.py` as a frozen constant; updates require an ADR amendment.

## Implementation files (forecast)

- [src/pce/substrate/haiku_lm.py](../../../src/pce/substrate/haiku_lm.py) — rewrite `_call_cli_once`, add `_build_clean_env`, `_build_tmp_clean_dir`, `_build_clean_home`.
- [src/pce/substrate/integrity.py](../../../src/pce/substrate/integrity.py) — new module.
- [src/pce/substrate/lm_protocol.py](../../../src/pce/substrate/lm_protocol.py) — broaden to `GeneratorProtocol`.
- [scripts/verify_outer_host_loads_pce.py](../../../scripts/verify_outer_host_loads_pce.py) — new smoke.

## Acceptance gate (Phase 2)

- 50 sequential clean Haiku subprocess calls all pass the leakage regex.
- IntegrityProbe passes 10/10 from inside fresh subprocesses.
- `scripts/verify_outer_host_loads_pce.py` passes (outer host PCE plugin still loads).
- mypy --strict + ruff green on all touched files.
