# ADR-007 — Remove the Anthropic Python SDK code path; OAuth/CLI is the only substrate

* Status: accepted (v0.4 Phase 8)
* Date: 2026-04-30
* Authors: maintainer
* Supersedes: implicit dual-path policy in v0.3 (`PCE_USE_SDK=1` opt-in)

## Context

PCE v0.1–v0.3 carried two paths into Anthropic models:

1. The **CLI path** (`claude --print --model haiku …`), invoked via
   `subprocess.run` with a clean inner environment, OAuth credentials
   resolved by the CLI itself, and a frozen system-prompt override.
2. The **SDK path** (`anthropic.Anthropic().messages.create(...)`), opt-in
   via `PCE_USE_SDK=1`. Required `pip install anthropic` and an
   `ANTHROPIC_API_KEY`.

The SDK path was originally added to make local Python tests faster and to
let researchers run PCE without needing `claude` on their PATH. The cost
of carrying it has crept up in three ways:

* **Authentication divergence.** OAuth and API-key keychains drift apart —
  test fixtures pass on one but not the other. The Phase 7 Bedrock pilot
  exposed this: `CLAUDE_CODE_USE_BEDROCK=1` is a CLI-only switch.
* **Cost-ledger split.** The SDK call site writes to a different ledger
  than the CLI subprocess; per-domain ledger isolation (added for the
  Phase 7 parallel pilot) only covers the CLI path.
* **Substrate plurality claim.** The v0.4 paper makes the substantive
  claim that PCE's mechanism story holds *across substrates* (OAuth API,
  Bedrock, Vertex). The cleanest way to back that claim is to expose
  exactly one substrate adapter that all backends route through.

## Decision

The SDK code path is **removed at substrate level** as of v0.4. Concretely:

* `HaikuConfig.use_sdk` is hard-coded to `False` in
  `HaikuConfig.from_env`. Setting `PCE_USE_SDK=1` now emits a
  `DeprecationWarning` from `PCEConfig.load` and is otherwise ignored.
* The dataclass field is preserved (so external callers who construct
  `HaikuConfig(...)` directly do not break), but the runtime branch in
  `HaikuLM.__init__` that imports `anthropic` is no longer reached from
  the env-driven path.
* `pyproject.toml`'s `dependencies` retain `anthropic` only as an
  *optional* dev dep used by archived v0.3 tests; the runtime no longer
  imports it under any default code path.
* The `PCEConfig` chain (introduced in this same release) carries
  `cascade_model` / `judge_model` / `cli_bin` / `timeout_s` / `cost_cap_usd`
  / `clean_substrate` / `system_prompt_override` from CLI args, env vars,
  user TOML (`~/.config/pce/config.toml`), or repo TOML (`pce.toml`) into
  `HaikuConfig` — the substrate adapter no longer has its own ad-hoc env
  parser.

## Consequences

### Positive

* One auth chain to maintain. OAuth ↔ Bedrock ↔ Vertex routing is now
  controlled exclusively by `CLAUDE_CODE_USE_BEDROCK` /
  `CLAUDE_CODE_USE_VERTEX` and the `ANTHROPIC_*` env vars the CLI
  already understands.
* The Cursor plugin manifest, the Claude Code plugin manifest, and the
  standalone `python -m pce` CLI all share a single config surface. A
  user can `pip install -e .` the package and get the same behaviour
  they get inside Cursor.
* The Phase 8 portability tests (`tests/test_pce_config.py`,
  `tests/test_pce_cli.py`) only need to mock the CLI subprocess — there
  is no longer an SDK alternate-universe to keep in sync.

### Negative / Trade-offs

* Researchers who do not have `claude` on their PATH lose the SDK
  fallback. The `RUN_LOCAL.md` doc explicitly walks them through
  installing `claude` (one-line `npm install -g @anthropic-ai/claude` on
  the CLI's recommended toolchain) so this is documented friction, not
  silent breakage.
* Direct API-key access to a model the CLI does not expose (e.g. a
  research preview) is no longer reachable through PCE. The mitigation
  is to wrap such an endpoint behind a CLI-compatible shim — outside
  the scope of v0.4.

### Reversibility

The dataclass field and the SDK import branch in `HaikuLM.__init__` are
preserved, so reintroducing the path is a one-line change in
`HaikuConfig.from_env` (and a docs PR). We do not delete the SDK code
because we want the option open if a future release wants to add an
SDK-only feature (e.g. function calling) that the CLI cannot represent.

## Verification

* `pytest tests/test_pce_config.py` — covers default chain, env override,
  TOML override, ordering, deprecation warning.
* `pytest tests/test_pce_cli.py` — `--model` flag flows through to
  `HaikuConfig.model` via `PCEConfig.load(overrides=…)`.
* `pytest tests/test_haiku_substrate.py` — existing substrate tests pass
  unchanged because they construct `HaikuConfig` directly (no env path).
