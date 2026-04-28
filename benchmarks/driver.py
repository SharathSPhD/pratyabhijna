#!/usr/bin/env python3
"""Phase 9 benchmark driver.

For each item across the four task domains, run three arms:

* `claude_haiku`  - Claude Haiku via `claude -p --model haiku`. (no PCE)
* `local_bare`    - Qwen2-1.5B-Instruct via raw LM.generate(). (no PCE)
* `local_cascade` - Qwen2-1.5B-Instruct through run_cascade(). (with PCE)

Each call's response is scored locally with `benchmarks.scoring.*` and the raw
text + axis dict + composite score is appended to a per-domain JSON file.

Robustness:
* Per-call timeout. If `claude` fails (rate limit, network), the row is
  recorded with `error` set and `composite=NaN`.
* Resumable: if `--out-dir` already contains a file for a domain, skip items
  whose `id` is already present.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from benchmarks import items as bench_items  # noqa: E402
from benchmarks import scoring as bench_scoring  # noqa: E402
from pce.cascade import run_cascade  # noqa: E402
from pce.substrate.embed import Embedder  # noqa: E402
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.types import Constraint  # noqa: E402

ARMS = ("claude_haiku", "local_bare", "local_cascade")

DEFAULT_DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")


def _build_prompt(domain: str, item: dict[str, Any]) -> tuple[str, str, list[str], list[str], list[str]]:
    """Return (user_prompt, constraint_text, must_avoid, aspects, retrieval_set)."""
    if domain == "poetry_gen":
        prompt = (
            f"Compose a {item['form']} about: {item['topic']}.\n"
            f"Avoid: {', '.join(item['must_avoid'])}.\n"
            f"Output only the poem.\n"
        )
        constraint = f"a {item['form']} about {item['topic']}"
        must_avoid = list(item["must_avoid"])
        aspects: list[str] = []
        retrieval: list[str] = []
        return prompt, constraint, must_avoid, aspects, retrieval
    if domain == "poetry_interp":
        prompt = (
            f"Interpret this line in two short paragraphs, naming each reading:\n\n"
            f"\"{item['surface']}\"\n\n"
            f"Reading A:\n"
        )
        constraint = f"two readings of: {item['surface']}"
        return prompt, constraint, [], list(item["aspects"]), list(item["retrieval_set"])
    if domain == "aut":
        prompt = (
            f"List 8 unusual, non-obvious uses of a {item['object']}. "
            f"Be concrete and specific. Avoid the standard everyday use. "
            f"Format: one use per line.\n"
        )
        constraint = f"unusual, non-obvious uses of a {item['object']}"
        return prompt, constraint, [f"the standard everyday use of a {item['object']}"], [], []
    if domain == "sci_creativity":
        prompt = (
            f"{item['question']} Give a non-obvious explanation in 4-6 sentences, "
            f"naming at least two different framings. Avoid the textbook one-liner.\n"
        )
        constraint = f"non-obvious explanation of: {item['question']}"
        return prompt, constraint, [
            f"the standard textbook explanation of {item['question']}",
        ], list(item.get("framings", [])), []
    raise ValueError(f"unknown domain: {domain}")


def _call_claude_haiku(prompt: str, *, timeout_s: int = 120) -> tuple[str, dict[str, Any]]:
    cmd = ["claude", "-p", "--model", "haiku", "--output-format", "json", prompt]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return "", {"ok": False, "error": "timeout", "elapsed_s": time.time() - started}
    elapsed = time.time() - started
    if proc.returncode != 0:
        return "", {
            "ok": False,
            "error": f"rc={proc.returncode}",
            "stderr": proc.stderr.decode("utf-8", errors="replace")[-1000:],
            "elapsed_s": elapsed,
        }
    try:
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        return "", {"ok": False, "error": f"json: {e}", "elapsed_s": elapsed}
    text: str = str(payload.get("result", "")) or ""
    cost = float(payload.get("total_cost_usd", 0.0))
    return text, {
        "ok": not payload.get("is_error", False),
        "elapsed_s": elapsed,
        "cost_usd": cost,
        "duration_ms": int(payload.get("duration_ms", 0)),
        "stop_reason": payload.get("stop_reason"),
    }


def _call_local_bare(
    prompt: str, *, lm: LocalLM, max_tokens: int, seed: int
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    out = lm.generate(prompt, max_tokens=max_tokens, sampler={"tau": 0.9, "top_p": 0.95}, seed=seed)
    return out.text, {"ok": True, "elapsed_s": time.time() - started}


def _call_local_cascade(
    prompt: str, *, lm: LocalLM, embed: Embedder,
    constraint_text: str, must_avoid: list[str], aspects: list[str], retrieval_set: list[str],
    K: int, max_tokens: int, seed: int,
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid),
    )
    state = run_cascade(
        prompt=prompt,
        constraint=constraint,
        lm=lm,
        embed=embed,
        K=K,
        max_tokens=max_tokens,
        base_seed=seed,
        retrieval_set=retrieval_set,
        aspects=aspects,
    )
    return state.surface or "", {
        "ok": True,
        "elapsed_s": time.time() - started,
        "vimarsa_event": bool(state.vimarsa_event),
        "novelty": float(state.vimarsa_novelty),
        "delta_F": float(state.audit.get("delta_F", float("nan"))),
        "selected_idx": int(state.audit.get("selected_idx", -1)),
    }


SCORERS = {
    "poetry_gen": bench_scoring.score_poetry_gen,
    "poetry_interp": bench_scoring.score_poetry_interp,
    "aut": bench_scoring.score_aut,
    "sci_creativity": bench_scoring.score_sci_creativity,
}


def _domain_items(domain: str, n: int | None = None) -> list[dict[str, Any]]:
    if domain == "poetry_gen":
        out = list(bench_items.POETRY_GEN)
    elif domain == "poetry_interp":
        out = list(bench_items.POETRY_INTERP)
    elif domain == "aut":
        out = list(bench_items.AUT)
    elif domain == "sci_creativity":
        out = list(bench_items.SCI_CREATIVITY)
    else:
        raise ValueError(domain)
    if n is not None:
        out = out[:n]
    return out


def _load_existing(out_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not out_path.exists():
        return {}
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return data.get("rows", {})


def _save(out_path: Path, rows: dict[str, dict[str, dict[str, Any]]], domain: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"domain": domain, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def run_domain(
    *,
    domain: str,
    arms: tuple[str, ...],
    out_path: Path,
    n: int | None,
    skip_claude: bool,
    K: int,
    max_tokens: int,
    seed: int,
    embed: Embedder,
    lm: LocalLM | None,
) -> int:
    items = _domain_items(domain, n=n)
    rows = _load_existing(out_path)
    scorer = SCORERS[domain]
    print(f"[bench] domain={domain}  items={len(items)}  arms={list(arms)}", flush=True)
    for i, item in enumerate(items):
        item_id = item["id"]
        item_rows = rows.setdefault(item_id, {"item": item})
        prompt, constraint_text, must_avoid, aspects, retrieval = _build_prompt(domain, item)
        for arm in arms:
            if arm in item_rows:
                continue
            print(f"  [{domain}] {item_id} :: {arm} ...", flush=True)
            text = ""
            meta: dict[str, Any] = {}
            if arm == "claude_haiku":
                if skip_claude:
                    item_rows[arm] = {"skipped": True}
                    continue
                text, meta = _call_claude_haiku(prompt)
            elif arm == "local_bare":
                assert lm is not None
                text, meta = _call_local_bare(
                    prompt, lm=lm, max_tokens=max_tokens, seed=seed + i
                )
            elif arm == "local_cascade":
                assert lm is not None
                text, meta = _call_local_cascade(
                    prompt,
                    lm=lm,
                    embed=embed,
                    constraint_text=constraint_text,
                    must_avoid=must_avoid,
                    aspects=aspects,
                    retrieval_set=retrieval,
                    K=K,
                    max_tokens=max_tokens,
                    seed=seed + i,
                )
            else:
                raise ValueError(arm)
            score = scorer(text, item=item, embed=embed)
            item_rows[arm] = {
                "text": text,
                "axes": score.axes,
                "composite": float(score.composite) if not np.isnan(score.composite) else None,
                "meta": meta,
            }
            _save(out_path, rows, domain)  # checkpoint after every call
    print(f"[bench] domain={domain} complete -> {out_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains", nargs="+", default=list(DEFAULT_DOMAINS),
        help="One or more domain ids",
    )
    parser.add_argument("--n-poetry-gen", type=int, default=15)
    parser.add_argument("--n-poetry-interp", type=int, default=15)
    parser.add_argument("--n-aut", type=int, default=10)
    parser.add_argument("--n-sci-creativity", type=int, default=10)
    parser.add_argument(
        "--arms", nargs="+", default=list(ARMS),
        help="Subset of: claude_haiku local_bare local_cascade",
    )
    parser.add_argument("--skip-claude", action="store_true")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results"
    )
    args = parser.parse_args()

    embed = Embedder()
    need_lm = "local_bare" in args.arms or "local_cascade" in args.arms
    lm = LocalLM() if need_lm else None

    n_map = {
        "poetry_gen": args.n_poetry_gen,
        "poetry_interp": args.n_poetry_interp,
        "aut": args.n_aut,
        "sci_creativity": args.n_sci_creativity,
    }
    for domain in args.domains:
        out_path = args.out_dir / f"{domain}.json"
        run_domain(
            domain=domain,
            arms=tuple(args.arms),
            out_path=out_path,
            n=n_map.get(domain),
            skip_claude=args.skip_claude,
            K=args.K,
            max_tokens=args.max_tokens,
            seed=args.seed,
            embed=embed,
            lm=lm,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
