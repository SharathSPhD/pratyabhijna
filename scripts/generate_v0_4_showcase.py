#!/usr/bin/env python3
"""Generate the nine v0.4 showcase outputs from ``scripts/showcase_specs.toml``.

The generator has two operating modes:

1. **Curate from Phase 7** (default). For ``source = "phase7_cascade"``
   entries it pulls the cascade output directly out of
   ``benchmarks/results_v0.4/<domain>.json``. No API calls are made; the
   ledger is untouched. This is the path used to produce the v0.4.0
   release showcase.

2. **Curated reference**. For ``source = "curated_reference"`` entries
   (the three Sanskrit chandas) it embeds the verse text from the spec
   file and runs the appropriate validator. This is also offline.

3. **Live regeneration** (``--mode=live --slug=<slug>``). Future-facing:
   re-runs the cascade against the spec's prompt + constraint via the
   ``HaikuLM`` substrate. Costs OAuth quota; gated behind an explicit
   ``--mode=live`` flag so accidental invocations cannot drain a budget.

Output layout per slug, under ``benchmarks/showcase_v0.4/<slug>/``:

  prompt.json     — {prompt, constraint, source, spec_metadata}
  trace.json      — full cascade trace (composite, draft, revision,
                    vimarsa_event, score_draft/revision, K_effective,
                    elapsed_s, validator output)
  draft.txt       — human-readable draft surface
  revised.txt     — human-readable revised surface (or '<no revision>')
  scoring.json    — per-axis scoring breakdown
  validator.json  — output of the appropriate tools/* validator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import english_meter, sanskrit_chandas, scientific_lint  # noqa: E402

DEFAULT_SHOWCASE_ROOT = REPO_ROOT / "benchmarks" / "showcase_v0.4"


def _load_specs(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    specs = data.get("showcase", [])
    if not isinstance(specs, list):
        raise SystemExit(f"showcase_specs.toml: 'showcase' must be a list, got {type(specs)}")
    return specs


def _validate_surface(spec: dict[str, Any], text: str) -> dict[str, Any]:
    domain = spec.get("domain")
    if domain == "sanskrit_chandas":
        chandas_name = spec.get("chandas")
        if not chandas_name:
            return {"validator": "none", "ok": False, "note": "spec missing 'chandas'"}
        v = sanskrit_chandas.validate(text, chandas_name)
        out = v.to_dict()
        out["validator"] = "tools.sanskrit_chandas"
        return out
    if domain == "poetry_gen":
        out: dict[str, Any] = {"validator": "tools.english_meter"}
        out["syllables_per_line"] = english_meter.syllable_count_per_line(text)
        out["meter_pattern_per_line"] = english_meter.meter_pattern_per_line(text)
        out["imagism"] = english_meter.imagism_density(text)
        if (spec.get("style") == "imagist" or spec.get("style") == "pastoral"):
            out["haiku_5_7_5"] = english_meter.haiku_5_7_5_ok(text)
        return out
    if domain == "sci_creativity":
        out = scientific_lint.lint_summary(text)
        out["validator"] = "tools.scientific_lint"
        return out
    return {"validator": "none", "ok": True, "note": f"no validator for domain={domain}"}


def _curate_from_phase7(spec: dict[str, Any]) -> dict[str, Any]:
    rf = spec.get("result_file")
    iid = spec.get("item_id")
    if not rf or not iid:
        raise SystemExit(f"spec {spec.get('slug')!r} needs result_file and item_id")
    payload = json.loads((REPO_ROOT / rf).read_text(encoding="utf-8"))
    rows = payload.get("rows", {})
    if iid not in rows:
        raise SystemExit(f"item_id {iid!r} not found in {rf}")
    row = rows[iid]
    cascade = row.get("haiku_cascade", {})
    bare = row.get("haiku_bare", {})
    meta = cascade.get("meta", {}) if isinstance(cascade.get("meta"), dict) else {}
    item = row.get("item", {}) if isinstance(row.get("item"), dict) else {}
    # The bare-vs-cascade contrast is what the showcase displays.
    # Pull the always-revise multiplexer arm's score_draft/score_revision
    # so the showcase can show the *honest* shadow-revision score even when
    # the event_gated policy chose to commit the draft.
    ar_meta: dict[str, Any] = {}
    ar_row = row.get("haiku_cascade_always_revise", {})
    if isinstance(ar_row, dict) and isinstance(ar_row.get("meta"), dict):
        ar_meta = ar_row["meta"]
    return {
        "domain": payload.get("domain"),
        "item_id": iid,
        "model": "claude-haiku via Bedrock (Phase 7)",
        "composite_cascade": cascade.get("composite"),
        "composite_bare": bare.get("composite"),
        # The cascade ALWAYS runs both passes. We surface them both so
        # the showcase trace viewer can animate "draft → vimarśa-event →
        # shadow revision → commit decision".
        "draft": meta.get("surface_draft", ""),
        "shadow_revision": meta.get("surface_revision", ""),
        "committed": cascade.get("text", ""),
        "committed_choice": (
            "revision" if meta.get("revision_differs_from_draft") and
            cascade.get("text") == meta.get("surface_revision")
            else "draft"
        ),
        "score_draft": ar_meta.get("score_draft"),
        "score_revision": ar_meta.get("score_revision"),
        "vimarsa_event": meta.get("vimarsa_event"),
        "delta_F_draft": meta.get("delta_F_draft"),
        "delta_F_revision": meta.get("delta_F_revision"),
        "K_effective": meta.get("K_effective"),
        "elapsed_s": meta.get("elapsed_s"),
        "haiku_n_calls": meta.get("haiku_n_calls"),
        "haiku_total_usd": meta.get("haiku_total_usd"),
        "axes_cascade": cascade.get("axes", {}),
        "axes_bare": bare.get("axes", {}),
        "item_prompt": item.get("prompt"),
        "item_metadata": {k: v for k, v in item.items() if k != "prompt"},
        "source": "phase7_cascade",
        "result_file": rf,
    }


def _curated_reference(spec: dict[str, Any]) -> dict[str, Any]:
    text = spec.get("curated_text", "").strip()
    if not text:
        raise SystemExit(f"spec {spec.get('slug')!r} has source=curated_reference but no curated_text")
    return {
        "domain": spec.get("domain"),
        "item_id": spec.get("slug"),
        "model": "n/a (curated reference; awaiting v0.5 chandas-aware cascade)",
        "draft": "",
        "revised": text,
        "vimarsa_event": None,
        "source": "curated_reference",
        "curated_origin": spec.get("curated_origin", ""),
    }


def generate_one(spec: dict[str, Any], showcase_root: Path) -> dict[str, Any]:
    slug = spec.get("slug")
    if not slug:
        raise SystemExit(f"spec missing slug: {spec}")
    src = spec.get("source", "phase7_cascade")

    if src == "phase7_cascade":
        trace = _curate_from_phase7(spec)
    elif src == "curated_reference":
        trace = _curated_reference(spec)
    elif src == "live":
        raise SystemExit(
            f"spec {slug!r} source='live' is not implemented in this build; "
            "use --mode live with the upcoming live regeneration runner once "
            "OAuth quota is restored."
        )
    else:
        raise SystemExit(f"unknown source for {slug!r}: {src}")

    # Validators run on the committed surface (what the user actually gets).
    surface_text = (
        trace.get("committed") or trace.get("revised")
        or trace.get("shadow_revision") or trace.get("draft") or ""
    )
    validator = _validate_surface(spec, surface_text)

    out_dir = showcase_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_doc = {
        "slug": slug,
        "title": spec.get("title"),
        "domain": spec.get("domain"),
        "prompt": spec.get("prompt") or spec.get("prompt_summary"),
        "constraint": spec.get("constraint"),
        "style": spec.get("style"),
        "chandas": spec.get("chandas"),
        "source": src,
        "notes": spec.get("notes"),
    }
    (out_dir / "prompt.json").write_text(
        json.dumps(prompt_doc, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (out_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (out_dir / "draft.txt").write_text(trace.get("draft") or "", encoding="utf-8")
    (out_dir / "shadow_revision.txt").write_text(
        trace.get("shadow_revision") or trace.get("revised") or "", encoding="utf-8",
    )
    (out_dir / "committed.txt").write_text(
        trace.get("committed") or trace.get("revised") or "", encoding="utf-8",
    )
    # Back-compat alias for tooling that already references revised.txt
    (out_dir / "revised.txt").write_text(
        trace.get("committed") or trace.get("revised") or trace.get("shadow_revision") or "",
        encoding="utf-8",
    )
    (out_dir / "scoring.json").write_text(
        json.dumps(
            {
                "axes_cascade": trace.get("axes_cascade", {}),
                "axes_bare": trace.get("axes_bare", {}),
                "composite_cascade": trace.get("composite_cascade"),
                "composite_bare": trace.get("composite_bare"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "validator.json").write_text(
        json.dumps(validator, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    try:
        rel = str(out_dir.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(out_dir)
    return {"slug": slug, "out_dir": rel}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--specs", type=Path,
                   default=REPO_ROOT / "scripts" / "showcase_specs.toml")
    p.add_argument("--showcase-root", type=Path, default=DEFAULT_SHOWCASE_ROOT)
    p.add_argument("--slug", default=None,
                   help="generate just this slug (default: all)")
    p.add_argument("--mode", choices=("curate", "live"), default="curate",
                   help="curate: pull from Phase 7 / spec (default); live: re-run cascade (NOT YET IMPLEMENTED)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    specs = _load_specs(args.specs)
    if args.slug:
        specs = [s for s in specs if s.get("slug") == args.slug]
        if not specs:
            print(f"no spec matches slug={args.slug!r}", file=sys.stderr)
            return 2

    if args.mode == "live":
        print("live regeneration not implemented in this build", file=sys.stderr)
        return 3

    args.showcase_root.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        if args.dry_run:
            written.append({"slug": spec.get("slug"), "dry_run": True})
            continue
        written.append(generate_one(spec, args.showcase_root))
    print(json.dumps({"written": written, "showcase_root": str(args.showcase_root)},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
