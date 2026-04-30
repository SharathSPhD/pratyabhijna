#!/usr/bin/env python3
"""v0.4 Sonnet judge bridge — stratified subset, OAuth-only (ADR-004).

Reads ``benchmarks/results_v0.4/<domain>.json``, draws a stratified
subset (by default 8 items per domain × 4 domains = 32 items) over the
quartiles of the proxy delta ``score(haiku_cascade) - score(haiku_bare)``,
and asks Anthropic Sonnet for a single A/B/tie verdict per pair via
``claude --print --model sonnet``. The same OAuth-only substrate as the
Haiku cascade — no API key path is added.

Outputs:

* ``benchmarks/results_v0.4/judge.jsonl``      one row per judge call.
* ``benchmarks/results_v0.4/judge_agreement.json`` aggregate H9.v4
  metrics (sign-agreement, Spearman ρ, position-bias check, prompt
  sha256, stratification audit).

Frozen prompt: ``scripts/judge_prompt_v0_4.txt``. Its sha256 is recorded
on every judge row and in ``judge_agreement.json`` so the v0.4 pilot can
be exactly reproduced post-hoc (ADR-004 § Acceptance gate).

Cost discipline (ADR-004 + TRIZ C4):

* A 4-item dry-run is required before the full subset (``--dry-run``
  uses a deterministic fake responder).
* Projected cost on the full subset must be ≤ $5 (default cap;
  override via ``--cost-cap-usd``).
* On 429 / out-of-quota the bridge logs and bails out gracefully — a
  partial ``judge.jsonl`` is still valid for H9.v4.

CLI::

  uv run python scripts/judge_subset.py --dry-run            # 4-item smoke
  uv run python scripts/judge_subset.py                       # full Sonnet
  uv run python scripts/judge_subset.py --n-per-domain 4      # 16-item subset
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for _p in (str(SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PROMPT_PATH = REPO_ROOT / "scripts" / "judge_prompt_v0_4.txt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results_v0.4"
DEFAULT_OUT_JSONL = DEFAULT_RESULTS_DIR / "judge.jsonl"
DEFAULT_OUT_AGREEMENT = DEFAULT_RESULTS_DIR / "judge_agreement.json"
DOMAINS: tuple[str, ...] = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")

# Sonnet pricing (2026-04): input $3/MTok, output $15/MTok. The judge
# prompt is ~700 input tokens including both responses (capped at 1500
# chars each) plus a ~120-token JSON output. So a single judged pair
# costs roughly 700 * 3/1e6 + 120 * 15/1e6 = $0.0021 + $0.0018 = $0.004.
# 32 items × 1 call/item = ~$0.13 — well under the $5 cap.
SONNET_INPUT_USD_PER_TOK = 3.0 / 1_000_000.0
SONNET_OUTPUT_USD_PER_TOK = 15.0 / 1_000_000.0
INPUT_TOKEN_ESTIMATE = 700
OUTPUT_TOKEN_ESTIMATE = 120

JudgeWinner = Literal["A", "B", "tie"]


@dataclasses.dataclass
class JudgePair:
    """One judged pair = one item + the two surfaces being compared."""

    domain: str
    item_id: str
    item_prompt: str
    treatment_arm: str
    control_arm: str
    treatment_text: str
    control_text: str
    treatment_composite: float
    control_composite: float
    proxy_delta: float
    quartile: int  # 0..3, the stratum the pair was drawn from


@dataclasses.dataclass
class JudgeRow:
    """One JSONL row written to ``judge.jsonl``."""

    domain: str
    item_id: str
    treatment_arm: str
    control_arm: str
    position_swap: bool
    proxy_delta: float
    judge_delta: float  # + treatment, - control, 0 tie
    winner_raw: JudgeWinner
    winner_resolved: Literal["treatment", "control", "tie"]
    confidence: float
    rationale: str
    quartile: int
    prompt_sha256: str
    # v0.4.1 review fix #D: also record the sha256 of the *formatted*
    # prompt — i.e. template + this pair's two surfaces, in their
    # actual A/B positions for this row's `position_swap`. The legacy
    # `prompt_sha256` only hashes the template, so the audit trail
    # cannot detect substitution drift in the cascade outputs that
    # were fed to the judge. The new field closes that gap.
    formatted_prompt_sha256: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_s: float
    api_error_status: int | None = None


def _item_prompt(item: dict[str, Any], domain: str) -> str:
    """Reconstruct the user-facing prompt from the per-domain item record.

    Mirrors the prompt the cascade saw (see ``scripts/run_judge_bridge.py``).
    """
    if domain == "poetry_interp":
        return (
            "Interpret this line of poetry, surfacing as many distinct readings "
            f"as possible: {item.get('line', '')!r}"
        )
    if domain == "poetry_gen":
        return (
            f"Write a {item.get('form', 'poem')} on "
            f"{item.get('topic', 'a subject of your choice')!r} avoiding cliched imagery."
        )
    if domain == "aut":
        return (
            "List as many alternative uses as you can for: "
            f"{item.get('object', '')!r}."
        )
    if domain == "sci_creativity":
        return str(
            item.get("prompt") or f"Solve this scientific creativity probe: {item.get('id', '')!r}"
        )
    return str(item.get("prompt", ""))


def _load_prompt_template(path: Path) -> tuple[str, str]:
    """Load the frozen judge prompt and return ``(text, sha256)``.

    The sha256 is over the raw bytes so the hash is stable independent
    of the local newline rendering.
    """
    raw = path.read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _stratify_pairs(
    pairs: list[JudgePair],
    *,
    n_per_domain: int,
    rng: random.Random,
) -> list[JudgePair]:
    """Quartile-stratified sample of ``n_per_domain`` items per domain.

    Items are bucketed into proxy-delta quartiles within each domain and
    we draw an equal number per quartile (rounded). When a quartile is
    short, we backfill from neighbours so the per-domain count is exact.
    """
    selected: list[JudgePair] = []
    by_domain: dict[str, list[JudgePair]] = {}
    for p in pairs:
        by_domain.setdefault(p.domain, []).append(p)

    for _domain, dom_pairs in by_domain.items():
        if not dom_pairs:
            continue
        dom_pairs = sorted(dom_pairs, key=lambda x: x.proxy_delta)
        n = len(dom_pairs)
        # Assign quartiles by order index.
        for i, p in enumerate(dom_pairs):
            quartile = min(3, (i * 4) // max(n, 1))
            object.__setattr__(p, "quartile", quartile)
        # Per-quartile target count.
        per_q = max(1, n_per_domain // 4)
        chosen: list[JudgePair] = []
        by_q: dict[int, list[JudgePair]] = {q: [] for q in range(4)}
        for p in dom_pairs:
            by_q[p.quartile].append(p)
        for q in range(4):
            avail = list(by_q[q])
            rng.shuffle(avail)
            chosen.extend(avail[:per_q])
        # Backfill if short.
        if len(chosen) < n_per_domain:
            leftovers = [p for p in dom_pairs if p not in chosen]
            rng.shuffle(leftovers)
            chosen.extend(leftovers[: n_per_domain - len(chosen)])
        selected.extend(chosen[:n_per_domain])
    return selected


def _load_pairs(
    results_dir: Path,
    *,
    treatment_arm: str,
    control_arm: str,
) -> list[JudgePair]:
    """Build the eligible pair pool from per-domain result files.

    A pair is *eligible* iff both arms emitted text and a finite composite.
    """
    pool: list[JudgePair] = []
    for domain in DOMAINS:
        path = results_dir / f"{domain}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows", {})
        for item_id, row in rows.items():
            t = row.get(treatment_arm)
            c = row.get(control_arm)
            if not isinstance(t, dict) or not isinstance(c, dict):
                continue
            t_text = str(t.get("text", "")).strip()
            c_text = str(c.get("text", "")).strip()
            if not t_text or not c_text:
                continue
            try:
                t_comp = float(t.get("composite") or 0.0)
                c_comp = float(c.get("composite") or 0.0)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(t_comp) and math.isfinite(c_comp)):
                continue
            pool.append(
                JudgePair(
                    domain=domain,
                    item_id=str(item_id),
                    item_prompt=_item_prompt(row.get("item", {}), domain),
                    treatment_arm=treatment_arm,
                    control_arm=control_arm,
                    treatment_text=t_text,
                    control_text=c_text,
                    treatment_composite=t_comp,
                    control_composite=c_comp,
                    proxy_delta=t_comp - c_comp,
                    quartile=-1,
                )
            )
    return pool


def _format_judge_prompt(
    template: str,
    *,
    pair: JudgePair,
    swap: bool,
) -> str:
    """Format the frozen prompt template with the pair's surfaces.

    Position randomisation: when ``swap=True``, control occupies slot A
    and treatment occupies slot B. The mapping is recorded on the
    judged row so the analysis can correct for any residual position
    bias.
    """
    if swap:
        text_a, text_b = pair.control_text, pair.treatment_text
    else:
        text_a, text_b = pair.treatment_text, pair.control_text
    # Use explicit replace (not str.format) so the literal JSON example
    # in the frozen prompt does not collide with format placeholders.
    return (
        template
        .replace("__PROMPT__", pair.item_prompt)
        .replace("__TEXT_A__", text_a[:1500])
        .replace("__TEXT_B__", text_b[:1500])
    )


def _fake_responder(prompt: str) -> dict[str, Any]:
    """Deterministic dry-run responder.

    Picks the longer ``<A>...</A>`` vs ``<B>...</B>`` block as the
    winner — useful for verifying the bridge plumbing without spending
    Sonnet quota.
    """
    # Use the *last* `<A>` / `<B>` tag so the synthetic responder finds the
    # actual response blocks rather than the explanatory text in the frozen
    # prompt template (which mentions "<A>...</A>" and "<B>...</B>").
    a_idx = prompt.rfind("<A>")
    b_idx = prompt.rfind("<B>")
    if a_idx < 0 or b_idx < 0:
        return {
            "winner": "tie",
            "confidence": 0.5,
            "rationale": "[dry-run] could not locate A/B blocks; defaulting to tie.",
            "_input_tokens": INPUT_TOKEN_ESTIMATE,
            "_output_tokens": OUTPUT_TOKEN_ESTIMATE,
        }
    a_block = prompt[a_idx : prompt.find("</A>", a_idx)]
    b_block = prompt[b_idx : prompt.find("</B>", b_idx)]
    if len(a_block) > len(b_block):
        winner = "A"
    elif len(b_block) > len(a_block):
        winner = "B"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "confidence": 0.6,
        "rationale": "[dry-run] deterministic stand-in: prefer the longer candidate block.",
        "_input_tokens": INPUT_TOKEN_ESTIMATE,
        "_output_tokens": OUTPUT_TOKEN_ESTIMATE,
    }


def _call_sonnet_cli(
    prompt: str,
    *,
    model: str,
    timeout_s: int,
    cli_bin: str,
) -> dict[str, Any]:
    """Invoke ``claude --print --model sonnet`` (OAuth-only, ADR-004).

    Returns the parsed JSON verdict. Raises :class:`RuntimeError` on
    non-JSON output or non-zero exit. Token usage is estimated from
    string length when ``claude --print`` does not surface it.
    """
    cmd = [
        cli_bin,
        "--print",
        "--model",
        model,
        "--output-format",
        "json",
        prompt,
    ]
    started = time.time()
    proc = subprocess.run(  # noqa: S603 — local CLI, controlled args
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.time() - started
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0 and not stdout:
        raise RuntimeError(
            f"sonnet CLI failed rc={proc.returncode} stderr={stderr[:240]!r}"
        )
    # Strip the outer Claude --output-format=json envelope (which wraps
    # the actual model text in a "result" field).
    text = stdout
    try:
        envelope = json.loads(stdout)
        if isinstance(envelope, dict) and "result" in envelope:
            text = str(envelope.get("result") or "").strip()
            usage = envelope.get("usage") or {}
            in_tok = int(usage.get("input_tokens", INPUT_TOKEN_ESTIMATE) or 0)
            out_tok = int(usage.get("output_tokens", OUTPUT_TOKEN_ESTIMATE) or 0)
        else:
            in_tok, out_tok = INPUT_TOKEN_ESTIMATE, OUTPUT_TOKEN_ESTIMATE
    except json.JSONDecodeError:
        in_tok, out_tok = INPUT_TOKEN_ESTIMATE, OUTPUT_TOKEN_ESTIMATE
    # Defensive: model sometimes wraps its JSON in markdown fences.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) >= 3 else text.strip("`")
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"sonnet returned non-JSON verdict: {text[:200]!r}") from exc
    if not isinstance(verdict, dict):
        raise RuntimeError(f"sonnet verdict is not a JSON object: {verdict!r}")
    verdict["_input_tokens"] = in_tok
    verdict["_output_tokens"] = out_tok
    verdict["_elapsed_s"] = elapsed
    return verdict


def _resolve_winner(
    *,
    raw: str,
    swap: bool,
) -> Literal["treatment", "control", "tie"]:
    """Map the swap-aware A/B/tie answer back to treatment / control / tie."""
    raw_norm = raw.upper().strip()
    if raw_norm == "TIE":
        return "tie"
    if raw_norm not in ("A", "B"):
        return "tie"
    if not swap:
        return "treatment" if raw_norm == "A" else "control"
    return "control" if raw_norm == "A" else "treatment"


def _sign(x: float) -> int:
    if x > 1e-12:
        return 1
    if x < -1e-12:
        return -1
    return 0


def _aggregate_h9(rows: list[JudgeRow]) -> dict[str, Any]:
    """Compute the H9.v4 aggregate metrics from a list of judged rows.

    Returns sign-agreement, Spearman ρ between proxy and judge deltas,
    position-bias means, and per-quartile breakdown.
    """
    if not rows:
        return {
            "name": "H9.v4",
            "status": "empty",
            "n": 0,
            "supported": False,
            "note": "no judge rows produced",
        }
    proxy_deltas = [r.proxy_delta for r in rows]
    judge_deltas = [r.judge_delta for r in rows]
    proxy_signs = [_sign(d) for d in proxy_deltas]
    judge_signs = [_sign(d) for d in judge_deltas]
    n_total = len(rows)
    # Sign agreement (excluding judge ties to avoid trivial zeros dominating).
    eligible = [
        (ps, js)
        for ps, js in zip(proxy_signs, judge_signs, strict=True)
        if js != 0 and ps != 0
    ]
    sign_rate = (
        sum(1 for ps, js in eligible if ps == js) / len(eligible)
        if eligible
        else float("nan")
    )

    # Spearman rho (use scipy if present; otherwise compute by hand).
    rho = float("nan")
    rho_p = float("nan")
    try:
        from scipy import stats

        if n_total >= 3:
            res = stats.spearmanr(proxy_deltas, judge_deltas)
            rho = float(res.statistic)
            rho_p = float(res.pvalue)
    except Exception:  # noqa: BLE001 — scipy missing or insufficient data
        pass

    # Position-bias check: judge_delta given each swap state.
    no_swap = [r.judge_delta for r in rows if not r.position_swap]
    swapped = [r.judge_delta for r in rows if r.position_swap]
    pos_bias = {
        "no_swap_mean": float(sum(no_swap) / len(no_swap)) if no_swap else None,
        "swapped_mean": float(sum(swapped) / len(swapped)) if swapped else None,
        "n_no_swap": len(no_swap),
        "n_swapped": len(swapped),
    }
    by_quartile: dict[str, dict[str, Any]] = {}
    for q in range(4):
        sub = [r for r in rows if r.quartile == q]
        if not sub:
            by_quartile[str(q)] = {"n": 0}
            continue
        by_quartile[str(q)] = {
            "n": len(sub),
            "mean_proxy_delta": float(sum(r.proxy_delta for r in sub) / len(sub)),
            "mean_judge_delta": float(sum(r.judge_delta for r in sub) / len(sub)),
            "judge_treatment_wins": sum(
                1 for r in sub if r.winner_resolved == "treatment"
            ),
            "judge_control_wins": sum(
                1 for r in sub if r.winner_resolved == "control"
            ),
            "judge_ties": sum(1 for r in sub if r.winner_resolved == "tie"),
        }
    total_cost = float(sum(r.cost_usd for r in rows))
    return {
        "name": "H9.v4",
        "status": "ok",
        "n": n_total,
        "n_eligible_for_sign_agreement": len(eligible),
        "sign_agreement_rate": sign_rate if math.isfinite(sign_rate) else None,
        "spearman_rho": rho if math.isfinite(rho) else None,
        "spearman_p": rho_p if math.isfinite(rho_p) else None,
        "supported": bool(
            math.isfinite(sign_rate) and sign_rate > 0.5
            and math.isfinite(rho_p) and rho_p < 0.05
        ),
        "position_bias": pos_bias,
        "by_quartile": by_quartile,
        "total_cost_usd": total_cost,
        "treatment_arm": rows[0].treatment_arm,
        "control_arm": rows[0].control_arm,
        "model": rows[0].model,
        "prompt_sha256": rows[0].prompt_sha256,
    }


def _project_cost(n_pairs: int) -> float:
    return float(
        n_pairs
        * (
            INPUT_TOKEN_ESTIMATE * SONNET_INPUT_USD_PER_TOK
            + OUTPUT_TOKEN_ESTIMATE * SONNET_OUTPUT_USD_PER_TOK
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument(
        "--out-agreement", type=Path, default=DEFAULT_OUT_AGREEMENT
    )
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--treatment-arm", default="haiku_cascade")
    parser.add_argument("--control-arm", default="haiku_bare")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--cli-bin", default="claude")
    parser.add_argument(
        "--n-per-domain", type=int, default=8,
        help="Items per domain (default 8 -> n=32). 1 -> 4 items total (dry-run smoke).",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--cost-cap-usd", type=float, default=5.0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use a deterministic fake responder (no Sonnet call).",
    )
    parser.add_argument(
        "--swap-rate", type=float, default=0.5,
        help="Fraction of pairs that get position-swapped A<->B for de-biasing (default 0.5).",
    )
    args = parser.parse_args()

    template, prompt_sha = _load_prompt_template(args.prompt_path)
    pool = _load_pairs(
        args.results_dir,
        treatment_arm=args.treatment_arm,
        control_arm=args.control_arm,
    )
    if not pool:
        print(
            f"[judge] no eligible pairs under {args.results_dir}; "
            f"populate pilot results first.",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    subset = _stratify_pairs(pool, n_per_domain=args.n_per_domain, rng=rng)
    proj_cost = _project_cost(len(subset))
    print(
        f"[judge] subset n={len(subset)} (per-domain={args.n_per_domain})  "
        f"projected cost ${proj_cost:.2f}  cap ${args.cost_cap_usd:.2f}  "
        f"prompt_sha256={prompt_sha[:12]}...",
        flush=True,
    )
    if proj_cost > args.cost_cap_usd:
        print(
            f"[judge] projected cost ${proj_cost:.2f} exceeds cap ${args.cost_cap_usd:.2f}; "
            "shrink --n-per-domain or raise --cost-cap-usd.",
            file=sys.stderr,
        )
        return 3

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[JudgeRow] = []
    cost_running = 0.0
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for i, pair in enumerate(subset):
            swap = rng.random() < args.swap_rate
            prompt = _format_judge_prompt(template, pair=pair, swap=swap)
            formatted_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            api_error: int | None = None
            try:
                if args.dry_run:
                    verdict = _fake_responder(prompt)
                else:
                    verdict = _call_sonnet_cli(
                        prompt,
                        model=args.model,
                        timeout_s=args.timeout_s,
                        cli_bin=args.cli_bin,
                    )
            except RuntimeError as exc:
                # Surface the error but continue so partial judge.jsonl is salvageable.
                msg = str(exc)
                if "429" in msg or "rate" in msg.lower() or "out of" in msg.lower():
                    api_error = 429
                else:
                    api_error = -1
                verdict = {
                    "winner": "tie",
                    "confidence": 0.0,
                    "rationale": f"[error] {msg[:200]}",
                    "_input_tokens": INPUT_TOKEN_ESTIMATE,
                    "_output_tokens": OUTPUT_TOKEN_ESTIMATE,
                    "_elapsed_s": 0.0,
                }
                if api_error == 429:
                    print(
                        "[judge] 429/quota detected; aborting subset early "
                        "(partial judge.jsonl preserved).",
                        file=sys.stderr,
                    )
            in_tok = int(verdict.get("_input_tokens", INPUT_TOKEN_ESTIMATE))
            out_tok = int(verdict.get("_output_tokens", OUTPUT_TOKEN_ESTIMATE))
            cost = (
                in_tok * SONNET_INPUT_USD_PER_TOK
                + out_tok * SONNET_OUTPUT_USD_PER_TOK
            )
            cost_running += cost
            raw = str(verdict.get("winner", "tie"))
            resolved = _resolve_winner(raw=raw, swap=swap)
            judge_delta = (
                +1.0 if resolved == "treatment"
                else -1.0 if resolved == "control"
                else 0.0
            )
            row = JudgeRow(
                domain=pair.domain,
                item_id=pair.item_id,
                treatment_arm=pair.treatment_arm,
                control_arm=pair.control_arm,
                position_swap=bool(swap),
                proxy_delta=float(pair.proxy_delta),
                judge_delta=judge_delta,
                winner_raw=raw if raw in ("A", "B", "tie") else "tie",  # type: ignore[arg-type]
                winner_resolved=resolved,
                confidence=float(verdict.get("confidence") or 0.0),
                rationale=str(verdict.get("rationale", ""))[:500],
                quartile=pair.quartile,
                prompt_sha256=prompt_sha,
                formatted_prompt_sha256=formatted_sha,
                model=args.model if not args.dry_run else "fake-responder",
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                elapsed_s=float(verdict.get("_elapsed_s", 0.0) or 0.0),
                api_error_status=api_error,
            )
            rows.append(row)
            fh.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"  [{i + 1}/{len(subset)}] {pair.domain}/{pair.item_id} "
                f"q={pair.quartile} swap={swap} -> {resolved} (raw={raw}) "
                f"cost=${cost_running:.4f}",
                flush=True,
            )
            if cost_running >= args.cost_cap_usd:
                print(
                    f"[judge] cost cap reached (${cost_running:.4f}); stopping.",
                    flush=True,
                )
                break

    agreement = _aggregate_h9(rows)
    args.out_agreement.parent.mkdir(parents=True, exist_ok=True)
    args.out_agreement.write_text(
        json.dumps(agreement, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(
        f"\n[judge] wrote {len(rows)} rows -> {args.out_jsonl}",
        flush=True,
    )
    print(f"[judge] wrote H9.v4 aggregate -> {args.out_agreement}", flush=True)
    sa = agreement.get("sign_agreement_rate")
    rho = agreement.get("spearman_rho")
    print(
        f"[judge] sign-agreement={sa}  Spearman ρ={rho}  "
        f"total cost=${cost_running:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
