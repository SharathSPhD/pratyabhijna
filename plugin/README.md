# Pratyabhijna Creative Engine — `plugin/`

Cursor + Claude Code plugin wrapping the Pratyabhijñā × Active Inference engine. Both hosts speak MCP, so the same FastMCP server (`plugin/mcp/server.py`) backs the Cursor and Claude Code manifests; the only differences are the manifest paths.

## Components

| Component       | Count | Path                                  |
|-----------------|-------|---------------------------------------|
| MCP tools       | 20    | `mcp/server.py` (FastMCP)             |
| Skills          | 5     | `skills/<name>/SKILL.md`              |
| Agents          | 5     | `agents/<name>.md`                    |
| Slash commands  | 5     | `commands/<name>.md`                  |
| Hooks           | 3     | `hooks/hooks.json` + `hooks/*.sh`     |

Tool count is the runtime truth (`mcp.list_tools()`), asserted by `tests/test_plugin_manifests.py`. Tools mirror the seven cascade operators (`cit`, `iccha`, `apohana`, `ananda`, `jnana`, `kriya`, `vimarsa`), the cascade orchestrator (`pce_cascade`), four consolidation / Hopfield-storehouse routines (`store.add`, `store.recall`, `store.consolidate_sws`, `store.consolidate_rem`, `hopfield_state`), the bare/clean substrate probes (`haiku_bare`, `haiku_clean_substrate_probe`), the embedder / lm helpers (`embed`, `lm.generate`, `lm.entropy`), plus `report` and `reset_state` introspection.

## Portability contract

The plugin is *not* a self-contained MCPB — it is a FastMCP server that imports the in-tree `pce` package. To run it, all of the following must be present:

1. **Repo on disk.** A full clone of `pratyabhijna` (the manifest references files under `${CLAUDE_PLUGIN_ROOT}/plugin/...` and the server imports from `${CLAUDE_PLUGIN_ROOT}/src/pce/`).
2. **`uv` on PATH.** The MCP launcher (`plugin/.mcp.json`) shells out to `uv run`. Install via [astral-sh/uv](https://github.com/astral-sh/uv) — `pipx install uv` or the platform installers.
3. **`claude` CLI on PATH, signed in.** The cascade calls Anthropic models exclusively through `claude --print`. The Anthropic Python SDK code path was removed in ADR-007; setting `PCE_USE_SDK=1` emits a deprecation warning and is ignored.
4. **`${CLAUDE_PLUGIN_ROOT}` resolves.** Claude Code sets this when it loads a plugin; `cursor` does the same via its own loader. Hand-rolled MCP launches (e.g. dropping the JSON snippet below into `~/.cursor/mcp.json`) substitute an absolute path for the variable.

The plugin does NOT attempt to install dependencies on first launch. If `mcp`, `numpy`, `sentence-transformers`, etc. are missing, `uv run` will create the venv and resolve them from `pyproject.toml`. If you want a snappier first launch, run `uv pip install -e .` once at the repo root before invoking the plugin.

## Installation

### Claude Code

From the repo root:

```bash
claude plugin install ./plugin
```

The MCP server runs as `pratyabhijna` and exposes tools under `pratyabhijna_mcp__<name>`.

### Cursor (MCP-only fallback)

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "pratyabhijna": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/pratyabhijna",
        "run",
        "python",
        "/absolute/path/to/pratyabhijna/plugin/mcp/server.py"
      ]
    }
  }
}
```

## Usage

```bash
# Compose through the cascade
/pce-compose topic: rain at dusk --form haiku

# Interpret an ambiguous text
/pce-interpret "The river is a clock and a clock is a river."

# Alternative Uses Task
/pce-aut brick --K 8

# Audit the most recent cascade
/pce-trace
```

## Configuration

The plugin reads its configuration through `pce.config.PCEConfig`, which layers defaults → repo `pce.toml` → user `~/.config/pce/config.toml` → env vars → CLI overrides. See [`docs/RUN_LOCAL.md`](../docs/RUN_LOCAL.md) for the full chain. Plugin-relevant env vars:

| Env var                              | Default                       | Purpose |
|--------------------------------------|-------------------------------|---------|
| `PCE_LM_MODEL`                       | `Qwen/Qwen2-1.5B-Instruct`    | HF model id for the local cit substrate. |
| `PCE_LM_DTYPE`                       | `float32`                     | `float16` / `bfloat16` / `float32`. |
| `PCE_LM_DEVICE`                      | autodetect                    | `cpu` / `cuda` / `mps`. |
| `PCE_CASCADE_MODEL` / `PCE_HAIKU_MODEL` | `haiku`                    | Anthropic model alias for the cascade arm. |
| `PCE_CLI` / `PCE_HAIKU_CLI`          | `claude`                      | Path to the OAuth-bound Claude CLI binary. |
| `PCE_COST_CAP_USD` / `PCE_HAIKU_COST_CAP_USD` | `18.0`               | Per-process Bedrock cost ceiling. |

## Audit trail

Every MCP tool call appends a one-line JSONL record to `audit/phase8/mcp_calls.jsonl`. Hook events go to `audit/phase8/hook_events.jsonl`. The cascade also writes per-call metadata to `audit/haiku/<ts>.json` and updates `audit/cost_ledger.json`.
