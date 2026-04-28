#!/usr/bin/env python3
"""PCE v0.2 benchmark driver.

For each item across the four task domains, run up to four arms:

* ``local_bare``     - Qwen2-1.5B-Instruct via raw LM.generate().              (no PCE)
* ``local_cascade``  - Qwen2-1.5B-Instruct through run_cascade().              (with PCE)
* ``haiku_bare``     - Claude Haiku via HaikuLM.generate() (single call).      (no PCE)
* ``haiku_cascade``  - Claude Haiku through run_cascade(lm=HaikuLM).           (with PCE)

The v0.1 ``claude_haiku`` arm is retained as a backward-compat alias for
``haiku_bare`` so the v0.1 benchmark script paths still resolve.

Each call's response is scored locally with ``benchmarks.scoring.*`` and the raw
text + axis dict + composite score is appended to a per-domain JSON file.

Cost telemetry: HaikuLM owns a shared cost ledger that is snapshotted to
``audit/cost_ledger.json`` after every Haiku-touching call so the run can be
budget-capped.

Robustness:
* Per-call timeout. If a call fails (rate limit, network), the row is recorded
  with ``error`` set and ``composite=NaN``.
* Resumable: if ``--out-dir`` already contains a file for a domain, skip items
  whose ``id`` is already present.
"""
from __future__ import annotations

import argparse
import json
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
from pce.substrate.haiku_lm import HaikuBudgetExceededError, HaikuLM  # noqa: E402
from pce.substrate.lm import LocalLM  # noqa: E402
from pce.substrate.lm_protocol import LMProtocol  # noqa: E402
from pce.types import Constraint  # noqa: E402

# v0.2 arms (default).  ``claude_haiku`` is a v0.1 alias.
ARMS_V2 = ("local_bare", "local_cascade", "haiku_bare", "haiku_cascade")
ARMS_V1_ALIASES = {"claude_haiku": "haiku_bare"}

DEFAULT_DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")
# NB: HaikuLM's *persistent* cost ledger is `audit/cost_ledger.json` and is
# read+written by HaikuLM itself; we must NOT overwrite it here. The driver
# snapshot below (written for the benchmark report) lives at a different path.
COST_SNAPSHOT_PATH = REPO_ROOT / "audit" / "cost_snapshot.json"

# Parity sampler -- matches operators.iccha.PARITY_SAMPLER so haiku_bare and
# haiku_cascade share the same sampler distribution.
PARITY_SAMPLER: dict[str, float] = {"tau": 0.9, "top_p": 0.95, "top_k": 50.0}


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


def _snapshot_cost_ledger(haiku_lm: HaikuLM | None) -> None:
    """Write a JSON snapshot of HaikuLM's cost ledger after every Haiku-touching call."""
    if haiku_lm is None:
        return
    rep = haiku_lm.report()
    COST_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_SNAPSHOT_PATH.write_text(
        json.dumps({"haiku": rep, "ts": time.time()}, indent=2),
        encoding="utf-8",
    )


def _call_local_bare(
    prompt: str, *, lm: LocalLM, max_tokens: int, seed: int
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    out = lm.generate(prompt, max_tokens=max_tokens, sampler=PARITY_SAMPLER, seed=seed)
    return out.text, {"ok": True, "elapsed_s": time.time() - started}


def _call_haiku_bare(
    prompt: str, *, haiku_lm: HaikuLM, max_tokens: int, seed: int,
) -> tuple[str, dict[str, Any]]:
    started = time.time()
    try:
        out = haiku_lm.generate(prompt, max_tokens=max_tokens, sampler=PARITY_SAMPLER, seed=seed)
    except HaikuBudgetExceededError as e:
        return "", {"ok": False, "error": f"budget: {e}", "elapsed_s": time.time() - started}
    except Exception as e:  # noqa: BLE001
        return "", {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.time() - started}
    return out.text, {
        "ok": True,
        "elapsed_s": time.time() - started,
        "haiku_total_usd": float(haiku_lm.report()["ledger_total_usd"]),
        "haiku_n_calls": int(haiku_lm.report()["ledger_n_calls"]),
    }


def _call_cascade(
    prompt: str,
    *,
    lm: LMProtocol,
    embed: Embedder,
    constraint_text: str,
    must_avoid: list[str],
    aspects: list[str],
    retrieval_set: list[str],
    K: int,
    max_tokens: int,
    seed: int,
    haiku_lm: HaikuLM | None,
) -> tuple[str, dict[str, Any]]:
    """Run two-pass-always cascade through ``lm`` (LocalLM or HaikuLM)."""
    started = time.time()
    constraint = Constraint(
        text=constraint_text,
        embedding=embed.encode(constraint_text),
        must_avoid=tuple(must_avoid),
    )
    try:
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
    except HaikuBudgetExceededError as e:
        return "", {"ok": False, "error": f"budget: {e}", "elapsed_s": time.time() - started}
    except Exception as e:  # noqa: BLE001
        return "", {"ok": False, "error": f"{type(e).__name__}: {e}", "elapsed_s": time.time() - started}
    meta: dict[str, Any] = {
        "ok": True,
        "elapsed_s": time.time() - started,
        "vimarsa_event": bool(state.vimarsa_event),
        "novelty": float(state.vimarsa_novelty),
        "delta_F": float(state.audit.get("delta_F", float("nan"))),
        "selected_idx": int(state.audit.get("selected_idx", -1)),
        "two_pass": bool(state.audit.get("two_pass", False)),
        "revision_differs_from_draft": bool(state.audit.get("revision_differs_from_draft", False)),
    }
    if haiku_lm is not None:
        rep = haiku_lm.report()
        meta["haiku_total_usd"] = float(rep["ledger_total_usd"])
        meta["haiku_n_calls"] = int(rep["ledger_n_calls"])
    return state.surface or "", meta


SCORERS = {
    "poetry_gen": bench_scoring.score_poetry_gen,
    "poetry_interp": bench_scoring.score_poetry_interp,
    "aut": bench_scoring.score_aut,
    "sci_creativity": bench_scoring.score_sci_creativity,
}


def _domain_items(domain: str, n: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]]
    if domain == "poetry_gen":
        out = [dict(x) for x in bench_items.POETRY_GEN]
    elif domain == "poetry_interp":
        out = [dict(x) for x in bench_items.POETRY_INTERP]
    elif domain == "aut":
        out = [dict(x) for x in bench_items.AUT]
    elif domain == "sci_creativity":
        out = [dict(x) for x in bench_items.SCI_CREATIVITY]
    else:
        raise ValueError(domain)
    if n is not None:
        out = out[:n]
    return out


def _load_existing(out_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not out_path.exists():
        return {}
    data = json.loads(out_path.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, dict[str, Any]]] = data.get("rows", {})
    return rows


def _save(out_path: Path, rows: dict[str, dict[str, dict[str, Any]]], domain: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"domain": domain, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalise_arm(arm: str) -> str:
    return ARMS_V1_ALIASES.get(arm, arm)


def run_domain(
    *,
    domain: str,
    arms: tuple[str, ...],
    out_path: Path,
    n: int | None,
    K: int,
    max_tokens: int,
    seed: int,
    embed: Embedder,
    lm: LocalLM | None,
    haiku_lm: HaikuLM | None,
    cost_cap_usd: float | None,
) -> int:
    items = _domain_items(domain, n=n)
    rows = _load_existing(out_path)
    scorer = SCORERS[domain]
    print(f"[bench] domain={domain}  items={len(items)}  arms={list(arms)}", flush=True)
    for i, item in enumerate(items):
        item_id = item["id"]
        item_rows = rows.setdefault(item_id, {"item": item})
        prompt, constraint_text, must_avoid, aspects, retrieval = _build_prompt(domain, item)
        for raw_arm in arms:
            arm = _normalise_arm(raw_arm)
            if arm in item_rows:
                continue
            if cost_cap_usd is not None and haiku_lm is not None and arm.startswith("haiku"):
                if float(haiku_lm.report()["ledger_total_usd"]) >= cost_cap_usd:
                    print(f"  [{domain}] {item_id} :: {arm} SKIPPED (cost cap reached)", flush=True)
                    item_rows[arm] = {"skipped": True, "reason": "cost_cap"}
                    _save(out_path, rows, domain)
                    continue
            print(f"  [{domain}] {item_id} :: {arm} ...", flush=True)
            text = ""
            meta: dict[str, Any] = {}
            if arm == "local_bare":
                assert lm is not None
                text, meta = _call_local_bare(
                    prompt, lm=lm, max_tokens=max_tokens, seed=seed + i
                )
            elif arm == "local_cascade":
                assert lm is not None
                text, meta = _call_cascade(
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
                    haiku_lm=None,
                )
            elif arm == "haiku_bare":
                assert haiku_lm is not None
                text, meta = _call_haiku_bare(
                    prompt, haiku_lm=haiku_lm, max_tokens=max_tokens, seed=seed + i,
                )
                _snapshot_cost_ledger(haiku_lm)
            elif arm == "haiku_cascade":
                assert haiku_lm is not None
                text, meta = _call_cascade(
                    prompt,
                    lm=haiku_lm,
                    embed=embed,
                    constraint_text=constraint_text,
                    must_avoid=must_avoid,
                    aspects=aspects,
                    retrieval_set=retrieval,
                    K=K,
                    max_tokens=max_tokens,
                    seed=seed + i,
                    haiku_lm=haiku_lm,
                )
                _snapshot_cost_ledger(haiku_lm)
            else:
                raise ValueError(arm)
            if text:
                score = scorer(text, item=item, embed=embed)
                composite = float(score.composite) if not np.isnan(score.composite) else None
                axes = score.axes
            else:
                composite = None
                axes = {}
            item_rows[arm] = {
                "text": text,
                "axes": axes,
                "composite": composite,
                "meta": meta,
            }
            _save(out_path, rows, domain)
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
        "--arms", nargs="+", default=list(ARMS_V2),
        help=f"Subset of: {ARMS_V2 + tuple(ARMS_V1_ALIASES.keys())}",
    )
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results_v2"
    )
    parser.add_argument(
        "--cost-cap-usd", type=float, default=20.0,
        help="Hard stop on Haiku-arms when ledger exceeds this. 0 disables.",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot preset: n=8 poetry, 6 aut/sci, K=4, max_tokens=200, all four arms.",
    )
    args = parser.parse_args()

    arms = tuple(_normalise_arm(a) for a in args.arms)
    n_map = {
        "poetry_gen": args.n_poetry_gen,
        "poetry_interp": args.n_poetry_interp,
        "aut": args.n_aut,
        "sci_creativity": args.n_sci_creativity,
    }
    if args.pilot:
        # Pilot scope: n=20 paired total (5 per domain), K=3, max_tokens=150.
        # This sits inside the SPEC_v0.2 "n=20-30" envelope and finishes
        # within the ~$15 / ~75-min budget on a laptop CPU/MPS substrate.
        n_map = {"poetry_gen": 5, "poetry_interp": 5, "aut": 5, "sci_creativity": 5}
        arms = ARMS_V2

    embed = Embedder()
    need_local = any(a in arms for a in ("local_bare", "local_cascade"))
    need_haiku = any(a in arms for a in ("haiku_bare", "haiku_cascade"))
    lm = LocalLM() if need_local else None
    haiku_lm = HaikuLM() if need_haiku else None
    cost_cap = args.cost_cap_usd if args.cost_cap_usd > 0 else None

    for domain in args.domains:
        out_path = args.out_dir / f"{domain}.json"
        run_domain(
            domain=domain,
            arms=arms,
            out_path=out_path,
            n=n_map.get(domain),
            K=args.K,
            max_tokens=args.max_tokens,
            seed=args.seed,
            embed=embed,
            lm=lm,
            haiku_lm=haiku_lm,
            cost_cap_usd=cost_cap,
        )
    if haiku_lm is not None:
        _snapshot_cost_ledger(haiku_lm)
        rep = haiku_lm.report()
        print(f"[bench] haiku cost: ${float(rep['ledger_total_usd']):.4f} over {int(rep['ledger_n_calls'])} calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
