"""Sonnet judge bridge for the v0.2 pilot.

Why this exists.
----------------
The v0.2 pilot uses an embedding-proxy composite score
(``benchmarks/scoring.py``) for every contrast. That keeps the run cheap and
deterministic but it does not exercise a *language-model* judge. The
adversarial review (``docs/reviews/2026-04-28-adversarial-plugin-review.md``)
asked for a sample of pairs to be re-scored by a stronger frontier model so
that we can:

1. compute Cohen's kappa between the embedding-proxy verdict and the Sonnet
   verdict on the same paired items, and
2. report a Sonnet-judged version of the H1-H4 paired contrasts alongside
   the embedding-proxy version.

We deliberately *do not run* the script against the live Sonnet endpoint in
the same session that ships the pilot --- the v0.2 SPEC budget is ~$15 for
the pilot and ~$100 for this judge bridge, and the bridge is best run
offline with a refreshed cost ledger. This script is therefore prepared,
documented, and dry-run-tested in this repo. To execute it for real:

    export ANTHROPIC_API_KEY=sk-ant-...
    uv run python scripts/run_judge_bridge.py \
        --results-dir benchmarks/results_v2 \
        --pairs 30 \
        --out-jsonl audit/judge/sonnet_30pair.jsonl \
        --out-stats benchmarks/results_v2/stats_with_judge.json \
        --cost-cap-usd 110.0

Without the env var the script prints a usage banner + cost estimate and
exits cleanly. With ``--dry-run`` the script substitutes a deterministic
fake responder so the full pipeline (sampling, scoring, kappa, paired
stats) can be exercised in CI and in this session.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Sonnet pricing as of 2026-04: input $3 / 1M tok, output $15 / 1M tok. The
# bridge prompt is ~600 input tokens and we cap output at 256 tokens, so a
# single judged pair costs roughly:
#   2 * (600 * $3 + 256 * $15) / 1e6 = 2 * (1800 + 3840) / 1e6
#                                    = $0.01128 per pair (judge each side once)
# At 30 pairs that's $0.34 unverified --- vastly under the $100 envelope.
# The $100 envelope leaves headroom for: per-pair K=3 self-consistency, prompt
# growth, longer outputs, model upgrades, and the occasional retry. We surface
# the per-pair point estimate and the upper-bound at K=3 so the operator sees
# both before spending real money.
SONNET_INPUT_USD_PER_1K = 3.0 / 1000.0
SONNET_OUTPUT_USD_PER_1K = 15.0 / 1000.0
DEFAULT_INPUT_TOKENS_ESTIMATE = 600
DEFAULT_OUTPUT_TOKENS_ESTIMATE = 256


# --------------------------------------------------------------------------- #
# Pair sampling                                                               #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class JudgePair:
    domain: str
    item_id: str
    prompt: str
    treatment_arm: str
    treatment_text: str
    control_arm: str
    control_text: str
    treatment_composite: float
    control_composite: float


def _load_domain(results_dir: Path, domain: str) -> dict[str, Any]:
    path = results_dir / f"{domain}.json"
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _item_prompt(item: dict[str, Any], domain: str) -> str:
    """Synthesise a single-string prompt from the per-domain item record."""
    if domain == "poetry_interp":
        return f"Interpret this line of poetry, surfacing as many distinct readings as possible: {item.get('line', '')!r}"
    if domain == "poetry_gen":
        return f"Write a {item.get('form', 'poem')} on {item.get('topic', 'a subject of your choice')!r} avoiding cliched imagery."
    if domain == "aut":
        return f"List as many alternative uses as you can for: {item.get('object', '')!r}."
    if domain == "sci_creativity":
        return str(item.get("prompt") or f"Solve this scientific creativity probe: {item.get('id', '')!r}")
    return str(item.get("prompt", ""))


def sample_pairs(
    results_dir: Path,
    *,
    n_pairs: int,
    treatment_arm: str,
    control_arm: str,
    seed: int,
) -> list[JudgePair]:
    """Sample n_pairs paired items, balanced across the four domains."""
    rng = random.Random(seed)
    domains = ["poetry_interp", "poetry_gen", "aut", "sci_creativity"]
    per_domain = max(1, n_pairs // len(domains))
    pool: list[JudgePair] = []
    for domain in domains:
        data = _load_domain(results_dir, domain)
        rows = data.get("rows", {})
        candidate_ids = []
        for item_id, row in rows.items():
            t = row.get(treatment_arm)
            c = row.get(control_arm)
            if not t or not c or not t.get("text") or not c.get("text"):
                continue
            candidate_ids.append(item_id)
        rng.shuffle(candidate_ids)
        for item_id in candidate_ids[:per_domain]:
            row = rows[item_id]
            pair = JudgePair(
                domain=domain,
                item_id=item_id,
                prompt=_item_prompt(row.get("item", {}), domain),
                treatment_arm=treatment_arm,
                treatment_text=row[treatment_arm]["text"],
                control_arm=control_arm,
                control_text=row[control_arm]["text"],
                treatment_composite=float(row[treatment_arm].get("composite", 0.0) or 0.0),
                control_composite=float(row[control_arm].get("composite", 0.0) or 0.0),
            )
            pool.append(pair)
    return pool[:n_pairs]


# --------------------------------------------------------------------------- #
# Sonnet judge call                                                           #
# --------------------------------------------------------------------------- #


JUDGE_PROMPT_TEMPLATE = """\
You are a careful literary / scientific creativity judge. Read the user
prompt and the two candidate responses A and B. Decide which response is
more *creative* in the sense the prompt rewards: for poetry-interpretation
prompts that means more distinct simultaneously-coexisting readings; for
poetry-generation prompts that means stronger non-cliched imagery and
formal craft; for AUT prompts that means more genuinely-distinct,
plausible uses; for scientific-creativity prompts that means a deeper,
more non-obvious framing.

Output STRICT JSON of the form:
{{"winner": "A" | "B" | "tie", "confidence": 0..1,
 "rationale": "<two sentences>"}}

Prompt:
{prompt}

Response A (arm: {arm_a}):
{text_a}

Response B (arm: {arm_b}):
{text_b}
"""


def _build_judge_prompt(pair: JudgePair, *, swap: bool) -> str:
    """Build the judge prompt; swap=True puts control as A (de-biases position)."""
    if swap:
        arm_a, text_a = pair.control_arm, pair.control_text
        arm_b, text_b = pair.treatment_arm, pair.treatment_text
    else:
        arm_a, text_a = pair.treatment_arm, pair.treatment_text
        arm_b, text_b = pair.control_arm, pair.control_text
    return JUDGE_PROMPT_TEMPLATE.format(
        prompt=pair.prompt,
        arm_a=arm_a,
        text_a=text_a[:1500],
        arm_b=arm_b,
        text_b=text_b[:1500],
    )


def _fake_responder(prompt: str) -> dict[str, Any]:
    """Deterministic dry-run responder that picks the longer side."""
    a_marker = "Response A"
    b_marker = "Response B"
    a_idx = prompt.find(a_marker)
    b_idx = prompt.find(b_marker)
    a_block = prompt[a_idx:b_idx]
    b_block = prompt[b_idx:]
    winner = "A" if len(a_block) > len(b_block) else "B"
    return {
        "winner": winner,
        "confidence": 0.75,
        "rationale": "[dry-run] Pick the longer block as a deterministic stand-in for actual judgement.",
        "_usage_input_tokens": DEFAULT_INPUT_TOKENS_ESTIMATE,
        "_usage_output_tokens": DEFAULT_OUTPUT_TOKENS_ESTIMATE,
    }


def _call_sonnet(prompt: str, *, model: str, max_tokens: int, retry: int, backoff_s: float) -> dict[str, Any]:
    """Call the Sonnet endpoint via the Anthropic SDK; raises on hard errors."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("anthropic SDK not installed; run uv add anthropic") from exc

    client = anthropic.Anthropic()
    last_err: Exception | None = None
    for attempt in range(retry + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            try:
                parsed_raw = json.loads(text.strip())
            except json.JSONDecodeError as je:
                raise RuntimeError(f"Sonnet returned non-JSON: {text!r}") from je
            parsed: dict[str, Any] = dict(parsed_raw)
            usage = getattr(resp, "usage", None)
            parsed["_usage_input_tokens"] = int(getattr(usage, "input_tokens", DEFAULT_INPUT_TOKENS_ESTIMATE) or 0)
            parsed["_usage_output_tokens"] = int(getattr(usage, "output_tokens", DEFAULT_OUTPUT_TOKENS_ESTIMATE) or 0)
            return parsed
        except Exception as exc:  # noqa: BLE001 - we retry every transient
            last_err = exc
            if attempt >= retry:
                break
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"Sonnet judge failed after {retry + 1} attempts: {last_err!r}") from last_err


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def _embedding_verdict(pair: JudgePair) -> str:
    """Embedding-proxy verdict: A = treatment, B = control."""
    if pair.treatment_composite > pair.control_composite:
        return "A_treatment"
    if pair.control_composite > pair.treatment_composite:
        return "B_control"
    return "tie"


def _judge_verdict(record: dict[str, Any]) -> str:
    """Map the Sonnet record back into A_treatment / B_control / tie."""
    raw = record.get("winner", "tie").upper()
    swap = bool(record.get("_swap"))
    if raw == "TIE":
        return "tie"
    # If swap=True, A was control and B was treatment, so flip the label.
    if not swap:
        return "A_treatment" if raw == "A" else "B_control"
    return "B_control" if raw == "A" else "A_treatment"


def _cohen_kappa(observed: list[tuple[str, str]]) -> float:
    """Compute Cohen's kappa over a list of (rater_a, rater_b) labels."""
    if not observed:
        return 0.0
    categories = sorted({lbl for pair in observed for lbl in pair})
    n = len(observed)
    agree = sum(1 for a, b in observed if a == b) / n
    a_marg = {c: sum(1 for a, _ in observed if a == c) / n for c in categories}
    b_marg = {c: sum(1 for _, b in observed if b == c) / n for c in categories}
    chance = sum(a_marg[c] * b_marg[c] for c in categories)
    if chance >= 1.0:
        return 1.0
    return (agree - chance) / (1.0 - chance)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _print_cost_banner(n_pairs: int, k_self_consistency: int, cap: float) -> None:
    point = (
        2 * n_pairs * k_self_consistency
        * (DEFAULT_INPUT_TOKENS_ESTIMATE * SONNET_INPUT_USD_PER_1K / 1000.0
           + DEFAULT_OUTPUT_TOKENS_ESTIMATE * SONNET_OUTPUT_USD_PER_1K / 1000.0)
    )
    print(f"sonnet judge bridge cost estimate (point):  ${point:.2f}", flush=True)
    print(f"sonnet judge bridge cost cap:               ${cap:.2f}", flush=True)
    print(f"  pairs={n_pairs}  k_self_consistency={k_self_consistency}  prompts/pair=2 (A/B + B/A swap)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("benchmarks/results_v2"))
    ap.add_argument("--pairs", type=int, default=30, help="Total paired items to judge (balanced across 4 domains).")
    ap.add_argument("--treatment-arm", default="haiku_cascade")
    ap.add_argument("--control-arm", default="haiku_bare")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS_ESTIMATE)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--retry", type=int, default=2)
    ap.add_argument("--backoff-s", type=float, default=2.0)
    ap.add_argument("--cost-cap-usd", type=float, default=110.0)
    ap.add_argument("--out-jsonl", type=Path, default=Path("audit/judge/sonnet_30pair.jsonl"))
    ap.add_argument("--out-stats", type=Path, default=Path("benchmarks/results_v2/stats_with_judge.json"))
    ap.add_argument("--dry-run", action="store_true", help="Use a deterministic fake responder (no API key needed).")
    args = ap.parse_args()

    pairs = sample_pairs(
        args.results_dir,
        n_pairs=args.pairs,
        treatment_arm=args.treatment_arm,
        control_arm=args.control_arm,
        seed=args.seed,
    )
    if not pairs:
        print(f"no eligible pairs under {args.results_dir}; run the pilot first", file=sys.stderr)
        return 2

    _print_cost_banner(len(pairs), k_self_consistency=1, cap=args.cost_cap_usd)

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not args.dry_run and not has_key:
        print("\nANTHROPIC_API_KEY not set. To run for real:", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  uv run python scripts/run_judge_bridge.py ...", file=sys.stderr)
        print("\nFor a dry-run smoke test (no API call):", file=sys.stderr)
        print("  uv run python scripts/run_judge_bridge.py --dry-run", file=sys.stderr)
        return 0

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    cost_running = 0.0
    with args.out_jsonl.open("w") as fh:
        for pi, pair in enumerate(pairs):
            for swap in (False, True):
                prompt = _build_judge_prompt(pair, swap=swap)
                if args.dry_run:
                    out = _fake_responder(prompt)
                else:
                    if cost_running >= args.cost_cap_usd:
                        print(f"  cost cap reached (${cost_running:.2f} >= ${args.cost_cap_usd}); stopping", flush=True)
                        break
                    out = _call_sonnet(
                        prompt,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        retry=args.retry,
                        backoff_s=args.backoff_s,
                    )
                in_tok = int(out.get("_usage_input_tokens", DEFAULT_INPUT_TOKENS_ESTIMATE))
                out_tok = int(out.get("_usage_output_tokens", DEFAULT_OUTPUT_TOKENS_ESTIMATE))
                cost = in_tok * SONNET_INPUT_USD_PER_1K / 1000.0 + out_tok * SONNET_OUTPUT_USD_PER_1K / 1000.0
                cost_running += cost
                rec = {
                    "domain": pair.domain,
                    "item_id": pair.item_id,
                    "treatment_arm": pair.treatment_arm,
                    "control_arm": pair.control_arm,
                    "_swap": swap,
                    "prompt_excerpt": prompt[:200],
                    "model": args.model if not args.dry_run else "fake-responder",
                    "winner_raw": out.get("winner"),
                    "confidence": float(out.get("confidence", 0.0) or 0.0),
                    "rationale": out.get("rationale", ""),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                }
                rec["winner_resolved"] = _judge_verdict(rec | {"winner": rec["winner_raw"]})
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"  pair {pi + 1}/{len(pairs)} {pair.domain}/{pair.item_id} done; cost so far ${cost_running:.2f}", flush=True)

    # Aggregate per-pair (combine A/B + B/A): treatment wins if it wins both
    # passes; tie if exactly one each; otherwise control wins.
    by_pair: dict[str, list[str]] = {}
    for rec in records:
        key = f"{rec['domain']}/{rec['item_id']}"
        by_pair.setdefault(key, []).append(rec["winner_resolved"])

    judge_verdicts: list[tuple[str, str]] = []
    for pair in pairs:
        key = f"{pair.domain}/{pair.item_id}"
        verdicts = by_pair.get(key, [])
        if not verdicts:
            continue
        n_treat = sum(1 for v in verdicts if v == "A_treatment")
        n_ctrl = sum(1 for v in verdicts if v == "B_control")
        if n_treat > n_ctrl:
            sonnet = "A_treatment"
        elif n_ctrl > n_treat:
            sonnet = "B_control"
        else:
            sonnet = "tie"
        emb = _embedding_verdict(pair)
        judge_verdicts.append((emb, sonnet))

    kappa = _cohen_kappa(judge_verdicts)
    treatment_win_rate = sum(1 for _, s in judge_verdicts if s == "A_treatment") / max(1, len(judge_verdicts))

    summary: dict[str, Any] = {
        "n_pairs_total": len(pairs),
        "n_pairs_judged": len(judge_verdicts),
        "treatment_arm": args.treatment_arm,
        "control_arm": args.control_arm,
        "model": args.model if not args.dry_run else "fake-responder",
        "cohen_kappa_embedding_vs_sonnet": kappa,
        "sonnet_treatment_win_rate": treatment_win_rate,
        "embedding_treatment_win_rate": sum(1 for e, _ in judge_verdicts if e == "A_treatment") / max(1, len(judge_verdicts)),
        "total_cost_usd": cost_running,
        "cost_cap_usd": args.cost_cap_usd,
        "dry_run": bool(args.dry_run),
        "per_pair": [
            {"domain": p.domain, "item_id": p.item_id, "embedding_verdict": e, "sonnet_verdict": s}
            for (p, (e, s)) in zip(pairs[: len(judge_verdicts)], judge_verdicts, strict=False)
        ],
    }

    args.out_stats.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {len(records)} judge records to {args.out_jsonl}", flush=True)
    print(f"wrote summary to {args.out_stats}", flush=True)
    print(
        f"\ncohen kappa (embedding vs sonnet): {kappa:.3f}  "
        f"sonnet treatment-win-rate: {treatment_win_rate:.2%}  "
        f"embedding treatment-win-rate: {summary['embedding_treatment_win_rate']:.2%}",
        flush=True,
    )
    print(f"total cost: ${cost_running:.4f} (cap ${args.cost_cap_usd})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
