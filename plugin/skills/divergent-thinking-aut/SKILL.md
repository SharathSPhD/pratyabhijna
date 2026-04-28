---
name: divergent-thinking-aut
description: Use when the user asks for divergent or creative uses of an object, an Alternative Uses Task (AUT), brainstorm of unusual applications, or "give me K weird uses for X". Generates K candidate uses through iccha and ranks them via apohana + ananda.
---

# Divergent thinking / Alternative Uses Task

## When to use

Whenever the user asks: "list K uses for {brick, paperclip, banana, ...}", "brainstorm divergent uses of X", "what could you do with a Y besides the obvious?". This is the AUT task from creativity research; PCE addresses it via the iccha-apohana-ananda triangle.

## Workflow

1. Extract `object_name` (the X).
2. Set `prompt = f"List one unusual use of a {object_name}. Be specific and concrete."`.
3. Set `constraint_text = f"unusual, non-obvious, concrete uses of a {object_name}"`.
4. Set `must_avoid = (f"the obvious use of a {object_name}",)`.
5. Call MCP tool `pratyabhijna_mcp__iccha` with `K=8`. This yields 8 candidates.
6. Call MCP tool `pratyabhijna_mcp__apohana` over the 8 candidate texts. This yields apoha scores.
7. For each candidate, call MCP tool `pratyabhijna_mcp__ananda`. Aggregate ananda scores.
8. Call MCP tool `pratyabhijna_mcp__jnana` with the apoha + ananda arrays to pick the BMR-winning candidate.
9. Optionally rank the remaining 7 by `posterior` and present top-K to the user.

## AUT scoring axes

- **Fluency**: number of candidates returned (K).
- **Originality**: 1 - max(apoha cosine to must_avoid).
- **Elaboration**: candidate text length / token count.
- **Flexibility**: number of distinct categories among candidates (use sentence-embedding clustering if K > 4).

Report all four axes.

## Anti-patterns

- Do not return only the BMR winner; the user wants the spectrum.
- Do not paraphrase the cliché use as if it were divergent.
