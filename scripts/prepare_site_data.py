#!/usr/bin/env python3
"""Materialise the v0.4 data assets the Astro site needs into docs/site/public/data/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "benchmarks" / "results_v0.4"
SHOWCASE = REPO_ROOT / "benchmarks" / "showcase_v0.4"
AUDIT = REPO_ROOT / "audit" / "v0.4"
SITE_DATA = REPO_ROOT / "docs" / "site" / "public" / "data"
SITE_FIGS = REPO_ROOT / "docs" / "site" / "public" / "figures" / "v0.4"
SITE_BENCH = REPO_ROOT / "docs" / "site" / "public" / "benchmarks"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_stats() -> None:
    stats_path = RESULTS / "stats.json"
    if not stats_path.exists():
        # v0.4.1 review fix #8: emit a structurally valid empty bundle
        # so the Astro site builds even when the stats.json is missing
        # (e.g. on a fresh clone before the pilot has been replayed).
        # The Zod schema in docs/site/src/lib/stats.ts will surface this
        # as a missing-data state, not a crash.
        empty = {
            "config": {"version": "v0.4", "framework": "PCE"},
            "primary": {},
            "fixed_effects": {},
            "shadow_revision": {"g": 0.0, "n": 0, "p": 1.0, "ci": [0.0, 0.0], "supported": False},
            "gate_calibration": {
                "event_gated": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
                "learned_gate": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0},
                "supported": False,
            },
            "commit_policy": {"leader_board": [], "winner": "—"},
        }
        write_json(SITE_DATA / "stats_v0.4.json", empty)
        print(f"WARN: {stats_path} missing; wrote empty stats bundle.")
        return
    raw = json.loads(stats_path.read_text(encoding="utf-8"))

    primary = {}
    for h in ("H1", "H2", "H3", "H4"):
        row = raw["primary"][h]
        primary[h] = {
            "hypothesis": h,
            "domain": row["domain"],
            "n": row["n"],
            "g": row["hedges_g"],
            "ci": row["bca_ci_95"],
            "p": row["permutation_p_one_sided"],
            "supported": row["supported"],
        }

    h5 = raw["H5"]
    h8a = raw["H8a_v4_shadow_revision_vs_draft"]
    h8b = raw["H8b_v4_gate_calibration"]
    h8c = raw["H8c_v4_commit_policy_comparison"]

    leader = []
    for row in h8c["leader_board"]:
        leader.append({
            "policy": row["policy"].replace("haiku_cascade_", ""),
            "g": row["hedges_g"],
            "ci": row["bca_ci_95"],
            "vs_bare": row["estimate"],
            "p": row["permutation_p_one_sided"],
            "n": row["n"],
        })
    winner = leader[0]["policy"] if leader else "—"

    bundle = {
        "config": raw["config"],
        "primary": primary,
        "fixed_effects": {
            "H5": {
                "pooled_g": h5["pooled_g"],
                "ci": h5["ci_95"],
                "method": h5["method"],
                "supported": h5["supported"],
                "per_domain_g": h5["per_domain_g"],
                "per_domain_n": h5["per_domain_n"],
            }
        },
        "shadow_revision": {
            "g": h8a["hedges_g"],
            "n": h8a["n"],
            "p": h8a["permutation_p_one_sided"],
            "ci": h8a["bca_ci_95"],
            "supported": h8a["supported"],
        },
        "gate_calibration": {
            "event_gated": h8b["event_gated"],
            "learned_gate": h8b["learned_gate"],
            "supported": h8b["supported"],
        },
        "commit_policy": {
            "leader_board": leader,
            "winner": winner,
            "pairwise_p": h8c.get("pairwise_p", {}),
            "supported": h8c["supported"],
        },
    }
    write_json(SITE_DATA / "stats_v0.4.json", bundle)


def prepare_judge_agreement() -> None:
    src = RESULTS / "judge_agreement.json"
    if not src.exists():
        write_json(SITE_DATA / "judge_agreement_v0.4.json", {
            "rho": 0.0, "sign_agreement": 0.0, "n": 0, "pairs": [],
        })
        return
    raw = json.loads(src.read_text(encoding="utf-8"))
    pairs_path = RESULTS / "judge.jsonl"
    pairs = []
    if pairs_path.exists():
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append({
                "item_id": row.get("item_id", ""),
                "domain": row.get("domain", ""),
                "proxy_delta": row.get("proxy_delta"),
                "judge_delta": row.get("judge_delta"),
                "agree": (row.get("proxy_delta") or 0) * (row.get("judge_delta") or 0) >= 0,
            })
    write_json(SITE_DATA / "judge_agreement_v0.4.json", {
        "rho": raw.get("spearman_rho", raw.get("rho", 0.0)),
        "sign_agreement": raw.get("sign_agreement_rate", raw.get("sign_agreement", 0.0)),
        "n": raw.get("n", len(pairs)),
        "pairs": pairs,
    })


def prepare_cost_ledger() -> None:
    """Write cost_ledger_v0.4.json with the v0.4.1 split: Haiku cascade
    ledger, Sonnet judge ledger, and a derived combined total."""
    src = AUDIT / "cost_ledger_merged.json"
    judge_src = RESULTS / "judge_agreement.json"
    judge = json.loads(judge_src.read_text(encoding="utf-8")) if judge_src.exists() else {}
    judge_usd = float(judge.get("total_cost_usd") or 0.0)
    judge_calls = int(judge.get("n") or 0)

    if not src.exists():
        write_json(SITE_DATA / "cost_ledger_v0.4.json", {
            "total_usd": 0.0, "n_calls": 0, "per_domain": {},
            "judge_usd": judge_usd, "judge_calls": judge_calls,
            "combined_usd": judge_usd, "combined_calls": judge_calls,
        })
        return
    raw = json.loads(src.read_text(encoding="utf-8"))
    per_domain: dict[str, dict[str, float]] = {}
    for domain in ("poetry_gen", "poetry_interp", "aut", "sci_creativity"):
        d_path = AUDIT / f"cost_ledger_{domain}.json"
        if d_path.exists():
            d = json.loads(d_path.read_text(encoding="utf-8"))
            per_domain[domain] = {
                "calls": int(d.get("n_calls", d.get("total_calls", 0))),
                "cost_usd": float(d.get("total_usd", d.get("total_cost_usd", 0.0))),
            }
    cascade_usd = float(raw.get("total_usd", raw.get("total_cost_usd", 0.0)))
    cascade_calls = int(raw.get("n_calls", raw.get("total_calls", 0)))
    write_json(SITE_DATA / "cost_ledger_v0.4.json", {
        "total_usd": cascade_usd,
        "n_calls": cascade_calls,
        "per_domain": per_domain,
        "judge_usd": judge_usd,
        "judge_calls": judge_calls,
        "combined_usd": cascade_usd + judge_usd,
        "combined_calls": cascade_calls + judge_calls,
    })


CATEGORY_MAP = {
    "sanskrit_anustubh": "sanskrit",
    "sanskrit_gayatri": "sanskrit",
    "sanskrit_indravajra": "sanskrit",
    "english_dickinson_slant": "english",
    "english_imagist_haiku": "english",
    "english_pastoral_traditional": "english",
    "science_galaxy_arms": "science",
    "science_ice_geometry": "science",
    "science_unreasonable_effectiveness": "science",
}


def prepare_showcase() -> None:
    if not SHOWCASE.exists():
        write_json(SITE_DATA / "showcase_index_v0.4.json", {
            "generated_at": "n/a", "total": 0,
            "per_category": {"sanskrit": 0, "english": 0, "science": 0},
            "demos": [],
        })
        return

    demos: list[dict[str, Any]] = []
    for slug in sorted(p.name for p in SHOWCASE.iterdir() if p.is_dir()):
        d = SHOWCASE / slug
        prompt_path = d / "prompt.json"
        trace_path = d / "trace.json"
        validator_path = d / "validator.json"
        scoring_path = d / "scoring.json"
        if not prompt_path.exists() or not trace_path.exists():
            continue
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        validator = json.loads(validator_path.read_text(encoding="utf-8")) if validator_path.exists() else {}
        scoring = json.loads(scoring_path.read_text(encoding="utf-8")) if scoring_path.exists() else {}

        category = CATEGORY_MAP.get(slug, "science")
        validator_status = "pass" if validator.get("ok") else ("review" if validator else "review")

        demos.append({
            "slug": slug,
            "category": category,
            "title": prompt.get("title", slug),
            "prompt_summary": (prompt.get("prompt") or "")[:240],
            "seed": prompt.get("seed", 4242),
            "validator_status": validator_status,
            "has_revision": bool(trace.get("shadow_revision") or trace.get("revised")),
            "has_judge": bool(scoring.get("judge")),
        })

        # Assemble a per-demo bundle the Astro site can fetch
        bundle = {
            "slug": slug,
            "category": category,
            "prompt": prompt,
            "trace": trace,
            "validator": validator,
            "scoring": scoring,
        }
        for body_file in ("draft.txt", "shadow_revision.txt", "revised.txt", "committed.txt"):
            p = d / body_file
            if p.exists():
                bundle[body_file.replace(".txt", "")] = p.read_text(encoding="utf-8")
        write_json(SITE_DATA / "showcase" / f"{slug}.json", bundle)

    counts = {"sanskrit": 0, "english": 0, "science": 0}
    for entry in demos:
        counts[entry["category"]] = counts.get(entry["category"], 0) + 1

    write_json(SITE_DATA / "showcase_index_v0.4.json", {
        "generated_at": "v0.4-phase8",
        "total": len(demos),
        "per_category": counts,
        "demos": demos,
    })


def copy_static_benchmarks() -> None:
    SITE_BENCH.mkdir(parents=True, exist_ok=True)
    if RESULTS.exists():
        dst = SITE_BENCH / "results_v0.4"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(RESULTS, dst)
    if SHOWCASE.exists():
        dst = SITE_BENCH / "showcase_v0.4"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(SHOWCASE, dst)


def copy_paper_pdf() -> None:
    """Stage paper/main.pdf into docs/site/public/paper/main.pdf.

    v0.4.1 review fix #8: paper/main.pdf is gitignored (it's a build
    artefact) but paper/v0.4/main.pdf is the frozen snapshot we ship.
    Try the freshly built one first, then fall back to the snapshot;
    that keeps the live site working even if the LaTeX build hasn't
    been re-run on this machine.
    """
    candidates = [
        REPO_ROOT / "paper" / "main.pdf",
        REPO_ROOT / "paper" / "v0.4" / "main.pdf",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("WARN: no paper PDF found at paper/main.pdf or paper/v0.4/main.pdf; "
              "the site will 404 on the Paper PDF link.")
        return
    dst = REPO_ROOT / "docs" / "site" / "public" / "paper" / "main.pdf"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_figures() -> None:
    src = REPO_ROOT / "paper" / "figures" / "v0.4"
    if not src.exists():
        return
    SITE_FIGS.mkdir(parents=True, exist_ok=True)
    for png in src.glob("*.png"):
        shutil.copy2(png, SITE_FIGS / png.name)


def main() -> int:
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    prepare_stats()
    prepare_judge_agreement()
    prepare_cost_ledger()
    prepare_showcase()
    copy_figures()
    copy_static_benchmarks()
    copy_paper_pdf()
    print(f"Site data prepared under {SITE_DATA.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
