---
name: pce-cascade
description: Run the full Pratyabhijna cascade on the user's prompt. Args: prompt, constraint, K, aspects (comma-separated), must_avoid (comma-separated).
---

Use the `pratyabhijna_mcp__cascade` tool. Parse the user's free-text args:

* `prompt`: the first line.
* `constraint`: the part after `||` (else default to "free composition").
* `K`: integer between 4 and 12 (default 6).
* `aspects`: split on `;`.
* `must_avoid`: split on `;`.

Return the full audit dict pretty-printed in a code block, then the surface text.
