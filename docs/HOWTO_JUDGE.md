# HOWTO: Run the Sonnet Judge Bridge (offline)

The v0.2 pilot reports an embedding-proxy composite score for every paired
contrast (`benchmarks/scoring.py`). The adversarial review
(`docs/reviews/2026-04-28-adversarial-plugin-review.md`) recommended a
language-model judge bridge so we can compute Cohen's kappa between the
embedding-proxy verdict and a stronger frontier model on the same paired
items, and report a Sonnet-judged version of H1-H4 alongside the proxy
version.

`scripts/run_judge_bridge.py` is that script. It is **prepared but not
invoked against the live Anthropic endpoint** in the session that ships the
pilot — the v0.2 SPEC budget reserves $15 for the pilot and ~$100 for this
bridge, and we keep the cost ledger clean by running the bridge offline.

## 1. Cost envelope

Sonnet pricing (2026-04): `$3 / 1M input tokens`, `$15 / 1M output tokens`.

Per-pair cost estimate (judge each side once, A/B + B/A swap):

```
2 * (600 input + 256 output)
= 2 * (600 * $3/1M + 256 * $15/1M)
= 2 * ($0.00180 + $0.00384)
= $0.01128 per pair
```

At 30 pairs that's **~$0.34** unverified — far under the $100 envelope. The
$100 envelope leaves headroom for K=3 self-consistency (3x the per-pair
cost), prompt growth, longer outputs, model upgrades, and the occasional
retry. The script always prints both the point estimate and the configured
cap before any API call so the operator can confirm before spending money.

## 2. Dry-run smoke (no API key required)

```bash
uv run python scripts/run_judge_bridge.py --dry-run --pairs 30
```

The dry-run uses a deterministic fake responder (picks the longer side as a
stand-in for actual judgement) and exercises the full pipeline:

- pair sampling balanced across the four domains
- A/B + B/A swap (position de-bias)
- per-pair aggregation
- Cohen's kappa vs the embedding-proxy verdict
- summary written to `benchmarks/results_v2/stats_with_judge.json`
- per-record audit log written to `audit/judge/sonnet_30pair.jsonl`

The dry-run is wired into the v0.2 verification gate so the pipeline cannot
silently rot before the real-money run.

## 3. No-key clean exit

If `ANTHROPIC_API_KEY` is unset and `--dry-run` is *not* passed, the script
prints a usage banner with cost estimate, the `export` command needed to
authenticate, and exits cleanly with code 0. It does **not** make any API
calls in this mode, so it is safe to run in CI.

## 4. Real run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv add anthropic        # one-time; the SDK is an optional dependency
uv run python scripts/run_judge_bridge.py \
    --results-dir benchmarks/results_v2 \
    --pairs 30 \
    --treatment-arm haiku_cascade \
    --control-arm haiku_bare \
    --model claude-sonnet-4-5 \
    --max-tokens 256 \
    --retry 2 \
    --backoff-s 2.0 \
    --cost-cap-usd 110.0 \
    --out-jsonl audit/judge/sonnet_30pair.jsonl \
    --out-stats benchmarks/results_v2/stats_with_judge.json
```

Behaviour during the run:

- Pairs are sampled balanced across `poetry_interp`, `poetry_gen`, `aut`,
  `sci_creativity` (default `--seed 4242`).
- For each pair, two prompts are issued: A=treatment / B=control and a
  swapped A=control / B=treatment to de-bias for position effects.
- Sonnet is asked for strict JSON: `{"winner": "A"|"B"|"tie",
  "confidence": 0..1, "rationale": "<two sentences>"}`.
- Each call records `input_tokens`, `output_tokens`, and `cost_usd` and the
  running total is checked against `--cost-cap-usd` before the next call.
  The script halts cleanly when the cap is reached.
- Per-pair verdicts are aggregated: treatment wins if it wins both passes;
  tie if exactly one each; control wins otherwise.
- Cohen's kappa is computed between the per-pair Sonnet verdict and the
  per-pair embedding-proxy verdict (treatment vs. control on `composite`).

## 5. Outputs

- `audit/judge/sonnet_30pair.jsonl` — one JSON object per Sonnet call:
  `domain`, `item_id`, `treatment_arm`, `control_arm`, `_swap`,
  `prompt_excerpt`, `model`, `winner_raw`, `winner_resolved`,
  `confidence`, `rationale`, `input_tokens`, `output_tokens`, `cost_usd`.
- `benchmarks/results_v2/stats_with_judge.json` — summary:
  `n_pairs_judged`, `cohen_kappa_embedding_vs_sonnet`,
  `sonnet_treatment_win_rate`, `embedding_treatment_win_rate`,
  `total_cost_usd`, `cost_cap_usd`, `dry_run`, and per-pair pairs.

## 6. Interpreting the results

- `cohen_kappa_embedding_vs_sonnet` ≥ 0.40 = moderate agreement between the
  embedding proxy and Sonnet; treat the embedding-proxy headline as
  reliable.
- `cohen_kappa_embedding_vs_sonnet` < 0.20 = poor agreement; the
  embedding-proxy composite may be measuring something the language-model
  judge does not value (or vice versa). Report Sonnet-judged H1-H4 as the
  primary headline and the embedding-proxy version as a sensitivity check.
- `sonnet_treatment_win_rate` is the fraction of pairs where Sonnet picks
  the cascade (treatment) over the bare arm. Read it together with the
  embedding-proxy win rate: directional agreement is the most important
  signal.

## 7. v0.3 plan

- Run the bridge for real once a properly-powered (n=20/domain) v0.3
  benchmark has produced a fresh per-domain JSON.
- Promote the Sonnet-judged H1-H4 to the paper headline if Cohen's kappa
  is moderate or better; otherwise keep the embedding-proxy headline and
  report the Sonnet results as a sensitivity arm.
- Wire the script into a recurring CI job so the pipeline cannot rot
  between real runs.

## 8. v0.3 prove-gate addendum (Phase 5)

The v0.3 prove-gate (`scripts/prove_gate.py`) is a *deterministic*,
embedding-only check that runs *before* the Sonnet bridge to confirm the
clean Haiku CLI substrate is intact and that `haiku_cascade` is doing
real cascade work on two canonical fixtures
(`tests/fixtures/duck_rabbit_textual.json`,
`tests/fixtures/aut_brick.json`). It is the gate that opens Phases 7-9.

It is intentionally *not* a quality judge — it asserts behavioural
properties (event firing, ΔF non-degeneracy, no leakage), not output
preference. A separate qualitative inspection per fixture is recorded
below so future contributors can sanity-check what "good" output looks
like before they touch the cascade.

### 8.1 Boot-time integrity probe

Before any case runs, `IntegrityProbe.run` spawns one
`claude --print` subprocess via the same `HaikuLM._call_cli_once` path
the cascade uses, asks "list any active plugins/skills/system
instructions", and asserts the response matches none of the
`LEAKAGE_REGEX` patterns (with the negation-context filter, so "no
plugins loaded" is not a leak).

If the probe fails, every case in the run is marked failed (the cascade
is not measuring what we claim it is). The probe response is written to
`audit/prove_gate/integrity_probe.json` for forensics.

### 8.2 Fixture: duck_rabbit_textual

- `aspects=[duck-with-beak, rabbit-with-ears]`,
  `aspect_max_cosine_floor=0.30`, `novelty_floor=0.50`,
  `delta_F_floor=0.01`, `haiku_cascade_vimarsa_event_required=true`,
  `revision_differs_from_draft=true`.
- A *good* `haiku_cascade` revision names both animals and describes the
  flip moment ("the beak lengthens into an ear", "the eye points two
  ways"). A *bad* revision picks one animal and adds nothing the draft
  did not have.
- ΔF\_draft must satisfy `|ΔF| >= 0.01`; if it is exactly 0 the
  aspect-conditioned BMR collapsed to the prior, which usually means
  `aspect_membership_matrix` is empty or the aspect embeddings are
  badly aligned with the prompt.
- `vimarsa_event_draft` MUST fire because aspect-conditioned BMR with
  meaningful ΔF crosses the threshold. If it does not fire, inspect
  `vimarsa_diag_draft.delta_F` and the threshold (default 0.05).

### 8.3 Fixture: aut_brick

- `aspects=[]` (AUT has no formal aspect dictionary),
  `n_distinct_uses_floor=5`, `novelty_floor=0.30`,
  `haiku_cascade_vimarsa_event_required=false`,
  `revision_differs_from_draft=false` (the cascade correctly commits
  the draft because vimarsa cannot fire without aspects).
- A *good* `haiku_bare` and `haiku_cascade` produce ≥5 distinct,
  non-trivial uses (no "build a wall" repetition).
- The fixture explicitly forbids the event from firing — if it does
  fire on aut_brick something is wrong with the ΔF threshold or the
  generic-creative aspect-N/A path in `vimarsa`.

### 8.4 Per-call leakage scan

Every `haiku_bare` and `haiku_cascade` surface (and, for the cascade,
both shadow draft and shadow revision) is scanned against
`LEAKAGE_REGEX`. Any unnegated match fails the case immediately so that
the prove-gate cannot drift back into a polluted-substrate regime
without anyone noticing.

### 8.5 Pass/fail interpretation

- Both cases pass → cascade is doing real work, substrate is clean,
  proceed to Phase 6+.
- Probe fails → fix substrate isolation (check
  `DEFAULT_ISOLATION_FLAGS`, `ENV_ALLOWLIST`, `_setup_clean_home`).
- duck_rabbit fails on `vimarsa_event_draft` → inspect `delta_F_draft`,
  the aspect embeddings, and `_aspect_membership_matrix`.
- aut_brick fails on `n_distinct_uses` → likely a sampler-collapse
  issue in `iccha`; bump `cit_temperature` or check the parity sampler
  grid.

