---
name: pce-interpret
description: Interpret a poem or ambiguous text. Args: the text in triple-quotes; optional `--aspect a; --aspect b`.
---

Delegate to the `pratyabhijna-interpreter` agent. Extract:

* the triple-quoted block as `surface`,
* `aspects` from `--aspect` flags (≥ 2 required; if not given, propose 2 from the text yourself),
* `retrieval_set` from sentences in the surface itself plus any "obvious reading" the user mentions.

Always report novelty, aspect_count, and a synthesis paragraph.
