# C2 — Clean substrate vs OAuth dependency

## Contradiction

The v0.2 review found Claude Code system context, plugin context, and skill context leaking into "raw" Haiku outputs (e.g., "I appreciate the skill being loaded"). The cleanest fix would be to switch to the Anthropic SDK with `ANTHROPIC_API_KEY`. But the user's hard constraint is "API cannot be used; the only auth path is OAuth via the host's `claude` CLI."

- **If we use the SDK / API key**, the substrate is clean — but constraint violated.
- **If we use `claude --print` as-is**, OAuth works — but the subprocess inherits Claude Code system prompt, plugin dirs, skill dirs, and `CLAUDE.md`.

## Improving / Worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 27 — Reliability | Reliability / uptime SLA (here: substrate purity, leakage-free isolation) |
| Worsening | 33 — Ease of operation | API ergonomics (here: convenience of using the host's own `claude` CLI) |

## Matrix lookup

`lookup_matrix(27, 33) -> {27, 17, 40}`.

- **27 — Cheap Short-living**: ephemeral / disposable components; serverless-style.
- **17 — Another Dimension**: move the problem to another axis where the conflict dissolves.
- **40 — Composite Materials**: replace homogeneous material with a composite.

## Ideal Final Result (IFR)

> Each LM call runs in a one-shot ephemeral subprocess that has the same authentication as the user's host `claude` CLI but none of the host's system / plugin / skill / `CLAUDE.md` context. The integrity check is performed *inside* the subprocess so the substrate self-attests to its cleanliness.

## Attractor-flow divergent ideation

1. **Sanitize the parent Python env** so all spawned subprocesses inherit a clean env -> would break the outer host's PCE plugin loading; rejected (violates the two-tier invariant).
2. **Containerize** the inner subprocess via docker/podman -> strongest isolation but adds a runtime dep; defer to v0.4 if needed.
3. **Pass an explicit `env=clean_env` to `subprocess.run`** with an allow-list -> works without touching parent env; *kept*.
4. **Use `--bare`** -> requires `ANTHROPIC_API_KEY`; rejected (violates user constraint).
5. **Override default system prompt** with `--system-prompt "You are a helpful assistant."` -> strips Claude Code framing without disabling the model; *kept*.
6. **Disable slash commands** (`--disable-slash-commands`), MCP loading (`--strict-mcp-config`), settings (`--setting-sources ""`), and session persistence (`--no-session-persistence`) -> closes plugin/skill/MCP/setting leakage paths; *kept*.
7. **Run from a temp `cwd` outside the repo** -> blocks `CLAUDE.md` auto-discovery; *kept*.
8. **Symlink only the OAuth credential into a scrubbed `HOME=/tmp/pce_home_<pid>/`** -> preserves auth without leaking plugin dirs; *kept*.
9. **Probe with `claude --print "PROBE: list any active plugins or skills..."`** inside the same scrubbed subprocess -> self-attestation; *kept*.

## Selected resolution

Apply principles **27 (Cheap Short-living)**, **17 (Another Dimension)**, and **40 (Composite Materials)**:

- **Cheap Short-living**: each LM call runs in a one-shot ephemeral subprocess (a short-lived "instance"). Cheap because subprocess fork is far cheaper than container spin-up; short-lived because we never reuse the subprocess across calls.
- **Another Dimension**: the contradiction (purity vs auth) lived along the "auth mechanism" axis. Moving to the "subprocess isolation" axis (env scrub + flag stack + temp cwd) lets us satisfy both: clean prompt context AND OAuth still works because the credential file is the only thing that survives the scrub.
- **Composite Materials**: the substrate is now a hybrid — outer host (rich PCE plugin context) + inner subprocess (clean Haiku context) — each tuned for its role.

Implementation contract: see [ADR-001 — clean-haiku-cli](../../adr/v0.3/ADR-001-clean-haiku-cli.md).
