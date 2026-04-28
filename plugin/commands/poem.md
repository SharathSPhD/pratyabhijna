---
name: pce-poem
description: Compose a poem through the cascade. Args: a single line topic + optional `--form haiku|sonnet|free` and `--avoid <cliché>`.
---

Delegate to the `pratyabhijna-poet` agent. Pass:

* the topic line as `prompt`,
* `constraint_text` derived from `--form`,
* `must_avoid` from `--avoid`.

Return only the surface text plus an optional one-line italicised note when a vimarsa event fired.
