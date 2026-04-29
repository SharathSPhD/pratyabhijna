# C4 — Vimarsa as event vs vimarsa as guarantee

## Contradiction

For vimarsa to be a true causal control mechanism it must *gate* the commit (event-gated revision). For H8 (revision-vs-draft delta) to be measurable on every item, every item must have a scored revision regardless of the gate. The two semantics fight:

- **Pure event-gated**: only event-fired items have a revision; H8 measurable only on a subset; can't compare to draft on non-fired items.
- **Always-revise**: H8 measurable everywhere, but vimarsa is no longer a control — it becomes a scaffold, which is exactly what the v0.2 review criticized.

## Improving / Worsening parameters

| | TRIZ parameter | Software equivalent |
|--|----------------|----------------------|
| Improving | 27 — Reliability | Reliability / uptime SLA (here: vimarsa's reliability as a causal control) |
| Worsening | 39 — Productivity | Throughput / output per unit cost (here: the wasted compute of always-revising) |

## Matrix lookup

`lookup_matrix(27, 39) -> {1, 35, 29, 38}`.

- **1 — Segmentation**: divide into independent parts so each segment can be optimized.
- **35 — Parameter Changes**: shift the system into a more favorable regime.
- **29 — Pneumatics and Hydraulics** (software analogy: streams / flow): use continuous flowing media instead of rigid switches.
- **38 — Strong Oxidants** (software analogy: stronger optimization environment): JIT / specialization.

## Ideal Final Result (IFR)

> Vimarsa is event-gated for *commit* (only when evidence justifies revision) AND always runs as a *shadow* pass for measurement. The committed surface depends on the event; the audit row always contains both `surface_draft` and `surface_revision`, so H8 is always measurable at zero extra committed-output cost.

## Attractor-flow divergent ideation

1. **Pure event-gated commit** -> H8 broken. Rejected.
2. **Pure always-revise** -> vimarsa is scaffold. Rejected.
3. **Event-gated commit + always-shadow revision (always scored)** -> the cascade always pays for both passes, but the committed output respects the event. H8 is computed on items where event committed revision (`H8.v3`); the cascade is causally controlled because if event=False, the user gets the draft. *Kept*.
4. **Doubled cost** is the price -> mitigated by the free-energy budget (C3) which can abort the shadow pass when underwater; abort logged on the row.
5. **Segmenting by event** lets us compute *separate* statistics for fired vs not-fired items -> useful side benefit; *kept*.

## Selected resolution

Apply principles **1 (Segmentation)** and **35 (Parameter Changes)**:

- **Segmentation**: the cascade is segmented into draft pass and shadow revision pass. They run independently; the commit decision is a separate logical segment.
- **Parameter Changes**: introduce `commit_policy: Literal["event_gated", "always_revise", "always_draft"]` as a parameter so the same cascade entry point serves the architectural arm (event_gated), the matched-revision control (always_revise with generic brief), and an internal sanity arm (always_draft).
- **Streams / Flow** (29): every item flows through draft -> shadow_revision -> commit, but the commit valve opens or closes based on event. The free-energy budget (C3) acts as a backpressure regulator on the shadow pass.

Implementation contract: see [ADR-002 — event-gated-shadow-revision](../../adr/v0.3/ADR-002-event-gated-shadow-revision.md).
