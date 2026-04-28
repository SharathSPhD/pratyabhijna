# Pratyabhijna Creative Engine - plugin/

Claude Code plugin wrapping the Pratyabhijna x Active Inference engine.

## Components

| Component | Count | Path |
|-----------|-------|------|
| MCP tools | 15    | `mcp/server.py` (FastMCP) |
| Skills    | 5     | `skills/<name>/SKILL.md` |
| Agents    | 5     | `agents/<name>.md` |
| Slash commands | 5 | `commands/<name>.md` |
| Hooks     | 3     | `hooks/hooks.json` + `hooks/*.sh` |

The 15 MCP tools mirror the seven cascade operators (`cit`, `iccha`, `apohana`, `ananda`, `jnana`, `kriya`, `vimarsa`), the cascade orchestrator, four consolidation / Hopfield-storehouse routines, plus a `report` and `reset_state` introspection pair.

## Installation

### Claude Code

From the repo root:

```bash
claude plugin install ./plugin
```

The MCP server runs as `pratyabhijna` and exposes tools under `pratyabhijna_mcp__<name>`.

### Cursor (MCP-only)

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
/pce-poem topic: rain at dusk --form haiku

# Interpret an ambiguous text
/pce-interpret """The river is a clock and a clock is a river."""

# Alternative Uses Task
/pce-aut brick --K 8

# Audit the most recent cascade
/pce-audit
```

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `PCE_LM_MODEL` | `Qwen/Qwen2-1.5B-Instruct` | HF model id for the cit substrate. |
| `PCE_LM_DTYPE` | `float32` | One of `float16`, `bfloat16`, `float32`. |
| `PCE_LM_DEVICE` | `cpu` | One of `cpu`, `cuda`, `mps`. |

## Audit trail

Every MCP tool call appends a one-line JSONL record to `audit/phase8/mcp_calls.jsonl`. Hook events go to `audit/phase8/hook_events.jsonl`.
