#!/usr/bin/env python3
"""Phase-artifact honesty gate.

Looks up the artifacts required for a given phase in `docs/COMPLETION_PROMISES.md`
and verifies:

* every required path exists with non-zero size;
* no file contains the placeholder strings `xxx`, `<TBD>`, `lorem ipsum`,
  `TODO:`, `FIXME:` (case-insensitive);
* JSON / JSONL files parse cleanly;
* JSONL files referenced as benchmark logs have no duplicate `(prompt_sha,
  condition)` rows and a row count >= the configured minimum;
* every `data-trace="path#json.pointer"` attribute in
  `presentation/index.html` resolves to a value in the referenced JSON;
* every `\\srcfile{path#json.pointer}` macro in any `paper/**/*.tex` file
  resolves similarly.

Exit code:

* 0 - all required artifacts honest;
* 1 - violation;
* 2 - infra failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent

PLACEHOLDER_TOKENS = (
    re.compile(r"\bxxx\b", re.IGNORECASE),
    re.compile(r"<TBD>", re.IGNORECASE),
    re.compile(r"lorem ipsum", re.IGNORECASE),
    re.compile(r"^\s*TODO\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*FIXME\s*:", re.IGNORECASE | re.MULTILINE),
)

# Phase -> list of required relative paths (files OR directory globs).
# The grammar: `path` (file must exist), `path/*pattern` (>=1 match required).
PHASE_ARTIFACTS: dict[int, list[str]] = {
    0: [
        "scripts/anti_stub_check.py",
        "scripts/verify_artifact.py",
        "scripts/verify_real_model.py",
        "scripts/verify_remote_pushed.py",
        "scripts/ralph_promise_gate.sh",
        "docs/COMPLETION_PROMISES.md",
        "pyproject.toml",
        ".gitignore",
        "README.md",
    ],
    1: [
        "docs/research-extended.md",
        "docs/operator-spec.md",
        "paper/references.bib",
    ],
    2: [
        "docs/SPEC.md",
        "docs/PRD.md",
        "docs/plan.md",
        "docs/ADR-001-substrate.md",
        "docs/ADR-002-vimarsa-loop.md",
        "docs/ADR-003-bmr.md",
        "docs/ADR-004-bench-stats.md",
        "CLAUDE.md",
        "AGENTS.md",
    ],
    3: [
        "src/pce/__init__.py",
        "tests/conftest.py",
    ],
    4: [],  # worktrees verified by verify_remote_pushed
    5: [
        "src/pce/operators/cit.py",
        "src/pce/operators/ananda.py",
        "src/pce/operators/iccha.py",
        "src/pce/operators/jnana.py",
        "src/pce/operators/kriya.py",
        "src/pce/operators/apohana.py",
        "src/pce/operators/vimarsa.py",
        "src/pce/cascade.py",
        "audit/hf_downloads.jsonl",
    ],
    6: [
        "audit/phase6/probes.jsonl",
    ],
    7: [
        "plugin/.claude-plugin/plugin.json",
        "plugin/.mcp.json",
        "plugin/marketplace.json",
        "plugin/mcp/server.py",
    ],
    8: [
        "audit/phase8/smoke.json",
    ],
    9: [
        "audit/phase9/calls.jsonl",
        "benchmarks/results/poetry_gen.json",
        "benchmarks/results/poetry_interp.json",
        "benchmarks/results/aut.json",
        "benchmarks/results/sci_creativity.json",
    ],
    10: [
        "presentation/index.html",
    ],
    11: [
        "paper/main.tex",
        "paper/references.bib",
        "paper/citations.checksum",
    ],
}


@dataclass
class ArtifactReport:
    phase: int
    ok: bool = True
    missing: list[str] = field(default_factory=list)
    placeholders: list[dict[str, Any]] = field(default_factory=list)
    json_errors: list[dict[str, Any]] = field(default_factory=list)
    jsonl_violations: list[dict[str, Any]] = field(default_factory=list)
    trace_violations: list[dict[str, Any]] = field(default_factory=list)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _check_placeholders(repo: Path, files: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(repo).as_posix()
        # Skip binary / oversized files
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pkl"}:
            continue
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue
        # Honesty-gate scripts themselves enumerate the forbidden tokens by
        # name in their source - exclude them. Their CODE-LEVEL stub rules
        # are enforced by anti_stub_check.py instead.
        if rel.startswith("scripts/") and path.stem in {
            "verify_artifact", "anti_stub_check", "verify_real_model", "verify_remote_pushed",
        }:
            continue
        # docs/COMPLETION_PROMISES.md and similar policy docs may legitimately
        # NAME the forbidden placeholder tokens; skip files that explicitly
        # opt out via a `<!-- placeholder-policy: allow -->` marker, or live
        # under docs/ paths that document this policy.
        text = _read_text(path)
        if "placeholder-policy: allow" in text:
            continue
        for pat in PLACEHOLDER_TOKENS:
            for m in pat.finditer(text):
                out.append({
                    "file": rel,
                    "match": m.group(0),
                    "offset": m.start(),
                })
    return out


def _check_json(path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return None
    except (json.JSONDecodeError, OSError) as e:
        return str(e)


def _check_jsonl_calls(path: Path, min_rows: int) -> list[str]:
    if not path.exists():
        return [f"{path}: missing"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            failures.append(f"{path}:{i} not valid JSON: {e}")
            continue
        rows.append(obj)
        prompt_sha = obj.get("prompt_sha256") or obj.get("prompt_sha")
        cond = obj.get("condition")
        if prompt_sha and cond:
            key = (str(prompt_sha), str(cond))
            if key in seen_keys:
                failures.append(f"{path}:{i} duplicate (prompt_sha,condition) row")
            seen_keys.add(key)
    if len(rows) < min_rows:
        failures.append(f"{path} has {len(rows)} rows; minimum {min_rows} required")
    return failures


def _resolve_pointer(obj: Any, pointer: str) -> Any:
    """Dot-pointer like `H1.p_value` resolved against a JSON object."""
    cur = obj
    for part in pointer.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif part.isdigit() and isinstance(cur, list) and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            raise KeyError(f"cannot resolve '{pointer}' at '{part}'")
    return cur


def _check_traces(repo: Path) -> list[dict[str, Any]]:
    """Resolve every `data-trace="path#pointer"` and `\\srcfile{path#pointer}`."""
    failures: list[dict[str, Any]] = []
    candidates: list[Path] = []
    pres = repo / "presentation"
    if pres.exists():
        candidates.extend(pres.rglob("*.html"))
    paper = repo / "paper"
    if paper.exists():
        candidates.extend(paper.rglob("*.tex"))
    pat_html = re.compile(r'data-trace="([^"]+)"')
    pat_tex = re.compile(r"\\srcfile\{([^}]+)\}")
    for path in candidates:
        text = _read_text(path)
        for m in list(pat_html.finditer(text)) + list(pat_tex.finditer(text)):
            spec = m.group(1)
            if "#" not in spec:
                failures.append({"file": path.relative_to(repo).as_posix(),
                                 "trace": spec, "error": "missing '#'"})
                continue
            target_rel, pointer = spec.split("#", 1)
            target = (repo / target_rel).resolve()
            if not target.exists():
                failures.append({"file": path.relative_to(repo).as_posix(),
                                 "trace": spec, "error": f"target {target_rel} missing"})
                continue
            try:
                obj = json.loads(target.read_text(encoding="utf-8"))
                _resolve_pointer(obj, pointer)
            except Exception as e:
                failures.append({"file": path.relative_to(repo).as_posix(),
                                 "trace": spec, "error": f"resolve failed: {e}"})
    return failures


def _required_files(repo: Path, phase: int) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    missing: list[str] = []
    for rel in PHASE_ARTIFACTS.get(phase, []):
        p = repo / rel
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            missing.append(rel)
        else:
            paths.append(p)
    return paths, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PCE artifact honesty gate.")
    parser.add_argument("--repo", default=str(REPO_ROOT_DEFAULT))
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument(
        "--bench-min-rows", type=int, default=15,
        help="Minimum rows expected in audit/phase9/calls.jsonl (Phase 9).",
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()

    rep = ArtifactReport(phase=args.phase)
    paths, missing = _required_files(repo, args.phase)
    rep.missing = missing
    rep.placeholders = _check_placeholders(repo, paths)

    for p in paths:
        if p.suffix == ".json":
            err = _check_json(p)
            if err:
                rep.json_errors.append({"file": p.relative_to(repo).as_posix(), "error": err})

    if args.phase == 9:
        rep.jsonl_violations = [
            {"violation": v} for v in
            _check_jsonl_calls(repo / "audit" / "phase9" / "calls.jsonl", args.bench_min_rows * 4)
        ]

    if args.phase >= 10:
        rep.trace_violations = _check_traces(repo)

    rep.ok = not (rep.missing or rep.placeholders or rep.json_errors or rep.jsonl_violations or rep.trace_violations)
    payload = asdict(rep)
    print(json.dumps(payload, indent=2))
    if not rep.ok:
        print(
            f"[verify_artifact] FAIL phase {args.phase}: "
            f"{len(rep.missing)} missing, {len(rep.placeholders)} placeholders, "
            f"{len(rep.json_errors)} json errors, {len(rep.jsonl_violations)} jsonl issues, "
            f"{len(rep.trace_violations)} trace failures.",
            file=sys.stderr,
        )
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
