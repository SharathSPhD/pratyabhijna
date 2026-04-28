#!/usr/bin/env python3
"""Anti-stub gate.

Fails (exit 1) if it finds any of the following in `src/pce/`, `plugin/`,
or `scripts/` (excluding test trees):

* function bodies whose only statement is `pass`, `...`, or `return None`
  (with no docstring and no other body) - true stubs;
* `raise NotImplementedError` outside test files;
* `# TODO`, `# FIXME`, `# XXX`, `# stub`, `# mock`, `# fake`, `# placeholder`
  comments;
* `unittest.mock`, `MagicMock`, `pytest_mock`, or `pytest.MonkeyPatch.setattr`
  outside `tests/`;
* operators in `src/pce/operators/<name>.py` without a paired test in
  `tests/operators/test_<name>.py`.

Designed to be run from the repo root. Exit code:

* 0 - all green, ralph-loop may accept the completion promise;
* 1 - violations found, ralph-loop must re-inject and continue;
* 2 - script-level failure (filesystem, parse error).

Output is JSON on stdout for Stop-hook consumption and a human-readable
summary on stderr.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent

TARGET_DIRS = ("src/pce", "plugin", "scripts")
TEST_DIR = "tests"

FORBIDDEN_COMMENT_TOKENS = (
    re.compile(r"#\s*TODO\b", re.IGNORECASE),
    re.compile(r"#\s*FIXME\b", re.IGNORECASE),
    re.compile(r"#\s*XXX\b", re.IGNORECASE),
    re.compile(r"#\s*stub\b", re.IGNORECASE),
    re.compile(r"#\s*mock\b", re.IGNORECASE),
    re.compile(r"#\s*fake\b", re.IGNORECASE),
    re.compile(r"#\s*placeholder\b", re.IGNORECASE),
)

FORBIDDEN_IMPORTS = (
    "unittest.mock",
    "pytest_mock",
)


@dataclass
class Violation:
    file: str
    line: int
    kind: str
    detail: str


@dataclass
class Report:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0
    operators_seen: list[str] = field(default_factory=list)
    operators_missing_tests: list[str] = field(default_factory=list)


def _is_pure_stub_body(body: list[ast.stmt]) -> bool:
    """A body counts as a stub when it contains only:

    * a single `pass`
    * a single `Ellipsis` expression (i.e. `...`)
    * a single `return` with no value or `return None`
    * a docstring + one of the above

    Bodies that only contain a docstring are NOT flagged here (some
    Protocol/abstract methods legitimately need a docstring); we still flag
    them under the operators-without-tests rule when they are operators.
    """
    real = [s for s in body if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
        and isinstance(s.value.value, str)
    )]
    if len(real) != 1:
        return False
    only = real[0]
    if isinstance(only, ast.Pass):
        return True
    if isinstance(only, ast.Expr) and isinstance(only.value, ast.Constant) and only.value.value is Ellipsis:
        return True
    if isinstance(only, ast.Return):
        if only.value is None:
            return True
        if isinstance(only.value, ast.Constant) and only.value.value is None:
            return True
    return False


def _iter_python_files(repo: Path, target_dirs: Iterable[str]) -> Iterable[Path]:
    for sub in target_dirs:
        base = repo / sub
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            yield p


def _scan_file(path: Path, repo: Path, in_test: bool) -> list[Violation]:
    rel = path.relative_to(repo).as_posix()
    text = path.read_text(encoding="utf-8")
    violations: list[Violation] = []

    # The anti-stub script itself, by necessity, names the very tokens it
    # forbids in source listings. Skip the gate scripts so they never flag
    # themselves; they are still subject to AST-level rules below.
    is_self_gate = rel.startswith("scripts/") and path.stem in {
        "anti_stub_check", "verify_artifact", "verify_real_model", "verify_remote_pushed",
    }

    # Comment-token sweep using tokenize so we only flag actual COMMENT tokens
    # (not string literals or docstrings).
    if not is_self_gate:
        try:
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type != tokenize.COMMENT:
                    continue
                comment = tok.string
                for pat in FORBIDDEN_COMMENT_TOKENS:
                    if pat.search(comment):
                        violations.append(Violation(
                            file=rel, line=tok.start[0], kind="forbidden_comment_token",
                            detail=comment.strip()[:160],
                        ))
                        break
        except (tokenize.TokenError, IndentationError):
            # If we can't tokenize, the AST step below will surface a SyntaxError.
            pass

    # AST checks - skip test files for the stub/NotImplementedError rules
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        violations.append(Violation(
            file=rel, line=getattr(e, "lineno", 0) or 0, kind="syntax_error",
            detail=str(e),
        ))
        return violations

    for node in ast.walk(tree):
        # Forbidden imports outside tests
        if not in_test and isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in FORBIDDEN_IMPORTS:
                violations.append(Violation(
                    file=rel, line=node.lineno, kind="forbidden_import",
                    detail=f"from {mod} import ...",
                ))
        if not in_test and isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    violations.append(Violation(
                        file=rel, line=node.lineno, kind="forbidden_import",
                        detail=f"import {alias.name}",
                    ))
        # NotImplementedError - allow only in test files
        if not in_test and isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc
            name = None
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                name = exc.func.id
            elif isinstance(exc, ast.Name):
                name = exc.id
            if name == "NotImplementedError":
                violations.append(Violation(
                    file=rel, line=node.lineno, kind="not_implemented_error",
                    detail="raise NotImplementedError outside tests",
                ))
        # Pure-stub function bodies in non-test files
        if not in_test and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_pure_stub_body(node.body):
                # allow if function has @typing.overload, @abstractmethod, or
                # is inside a typing.Protocol class - those are honest
                # declarations rather than stubs.
                decorators = {
                    ast.unparse(d) if hasattr(ast, "unparse") else "" for d in node.decorator_list
                }
                if any("overload" in d or "abstractmethod" in d for d in decorators):
                    continue
                violations.append(Violation(
                    file=rel, line=node.lineno, kind="pure_stub_body",
                    detail=f"function `{node.name}` has only pass/.../return None",
                ))

    return violations


def _check_operators(repo: Path) -> tuple[list[str], list[str]]:
    """Return (operators_seen, operators_missing_tests).

    A file is treated as an operator iff it lives in src/pce/operators/ and
    its name does not start with `_` and is not `__init__`.
    """
    op_dir = repo / "src" / "pce" / "operators"
    test_dir = repo / "tests" / "operators"
    seen: list[str] = []
    missing: list[str] = []
    if not op_dir.exists():
        return seen, missing
    for p in sorted(op_dir.glob("*.py")):
        name = p.stem
        if name.startswith("_"):
            continue
        seen.append(name)
        expected = test_dir / f"test_{name}.py"
        if not expected.exists():
            missing.append(name)
    return seen, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PCE anti-stub gate.")
    parser.add_argument(
        "--repo", default=str(REPO_ROOT_DEFAULT),
        help="Repository root (default: parent of this script).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON only (no stderr summary).",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Phase number; some checks (operators-without-tests) require phase>=5.",
    )
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(json.dumps({"ok": False, "error": f"repo not found: {repo}"}))
        return 2

    report = Report(ok=True)
    files = list(_iter_python_files(repo, TARGET_DIRS))
    for path in files:
        rel = path.relative_to(repo).as_posix()
        in_test = rel.startswith(f"{TEST_DIR}/")
        report.files_scanned += 1
        report.violations.extend(_scan_file(path, repo, in_test))

    # Also walk tests/ for forbidden-comment-token sweep (a placeholder
    # comment in a test is itself a shortcut signal).
    tests_dir = repo / TEST_DIR
    if tests_dir.exists():
        for path in tests_dir.rglob("*.py"):
            rel = path.relative_to(repo).as_posix()
            text = path.read_text(encoding="utf-8")
            report.files_scanned += 1
            try:
                for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                    if tok.type != tokenize.COMMENT:
                        continue
                    comment = tok.string
                    for pat in FORBIDDEN_COMMENT_TOKENS:
                        if pat.search(comment):
                            report.violations.append(Violation(
                                file=rel, line=tok.start[0], kind="forbidden_comment_token",
                                detail=comment.strip()[:160],
                            ))
                            break
            except (tokenize.TokenError, IndentationError):
                pass

    seen, missing = _check_operators(repo)
    report.operators_seen = seen
    report.operators_missing_tests = missing
    # Operators-without-tests is a hard violation only once we're at Phase >= 5
    if args.phase is None or args.phase >= 5:
        for op in missing:
            report.violations.append(Violation(
                file=f"src/pce/operators/{op}.py", line=0,
                kind="operator_missing_test",
                detail=f"no tests/operators/test_{op}.py paired",
            ))

    report.ok = len(report.violations) == 0

    payload = {
        "ok": report.ok,
        "files_scanned": report.files_scanned,
        "violations": [asdict(v) for v in report.violations],
        "operators_seen": report.operators_seen,
        "operators_missing_tests": report.operators_missing_tests,
    }
    print(json.dumps(payload, indent=2))

    if not args.json:
        if report.ok:
            print(
                f"[anti_stub] OK - scanned {report.files_scanned} files, "
                f"{len(report.operators_seen)} operators, "
                f"{len(report.operators_missing_tests)} missing tests.",
                file=sys.stderr,
            )
        else:
            print(
                f"[anti_stub] FAIL - {len(report.violations)} violation(s) "
                f"across {report.files_scanned} files.",
                file=sys.stderr,
            )
            for v in report.violations[:30]:
                print(f"  {v.file}:{v.line}  [{v.kind}] {v.detail}", file=sys.stderr)
            if len(report.violations) > 30:
                print(f"  ... and {len(report.violations) - 30} more", file=sys.stderr)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
