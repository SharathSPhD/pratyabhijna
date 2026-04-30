"""v0.4 Phase 7 powered-pilot orchestrator (AWS Bedrock backend).

Runs the v0.4 powered pilot end-to-end on a Bedrock-connected Claude Code
host. Designed to be invoked from a fresh clone on a fresh machine where
the user's Claude Code Desktop session is already configured against
Anthropic's models on AWS Bedrock.

Pipeline (all paths relative to repo root)::

  1. Pre-flight checks
       * `claude` CLI is on PATH and `--version` succeeds.
       * `CLAUDE_CODE_USE_BEDROCK=1` (warn-only if not set; user may have
         configured the desktop's CLI symlink to default to Bedrock).
       * Bedrock model IDs resolved from --haiku-model / --sonnet-model
         flags, falling back to BEDROCK_HAIKU_MODEL / BEDROCK_SONNET_MODEL
         env vars, finally to the global cross-region inference profiles
         (`global.anthropic.claude-haiku-4-5-20251001-v1:0` /
         `global.anthropic.claude-sonnet-4-5-20250929-v1:0`).
       * AWS region resolved from AWS_REGION / AWS_DEFAULT_REGION
         (defaults to `us-east-1` with a warning).

  2. Phase 7-A: parallel pilot
       Spawns one `python -m benchmarks.driver` subprocess PER DOMAIN with:
         * `--domains <domain>` (single-domain slice)
         * `--out-dir benchmarks/results_v0.4`
         * `--cost-cap-usd 0` (uncapped; Bedrock is paid via subscription)
         * `--retry-failed` (so partial poetry_gen.json from the macOS run
           is filled in rather than re-done)
         * `--n-poetry-gen 20 --n-poetry-interp 20 --n-aut 20 --n-sci-creativity 20`
       Each subprocess writes to its own non-overlapping audit sinks via
       env vars, so the four workers never race on shared files:
         PCE_HAIKU_COST_LEDGER       = audit/v0.4/cost_ledger_<domain>.json
         PCE_COST_SNAPSHOT_PATH      = audit/v0.4/cost_snapshot_<domain>.json
         PCE_INTEGRITY_LOG_PATH      = audit/v0.4/integrity_probes_<domain>.jsonl
         PCE_HAIKU_AUDIT_DIR         = audit/haiku/<domain>
         PCE_HAIKU_MODEL             = <bedrock haiku model id>
         CLAUDE_CODE_USE_BEDROCK     = 1
         (all AWS_* env vars carried through)
       Per-worker stdout streams to logs/v0_4_pilot.<domain>.bedrock.log.
       The orchestrator polls every 30 s, prints a STATUS line, and writes
       benchmarks/results_v0.4/STATUS.md so a watching agent can tail it.

  3. Merge
       * Sums per-domain ledgers into audit/v0.4/cost_ledger_merged.json.
       * Concatenates per-domain integrity logs into
         audit/v0.4/integrity_probes_merged.jsonl.

  4. Phase 7-B: judge subset (sequential, single Sonnet stream)
       `python scripts/judge_subset.py --model <bedrock sonnet model id>
        --n-per-domain 8` — emits benchmarks/results_v0.4/judge.jsonl
       and benchmarks/results_v0.4/judge_agreement.json.

  5. Phase 7-C: stats
       `python -m benchmarks.stats --version v0.4
        --results-dir benchmarks/results_v0.4` — emits
       benchmarks/results_v0.4/stats.json with H1.v4..H9.v4.

  6. Final STATUS.md + git checkpoint
       Writes a final benchmarks/results_v0.4/STATUS.md summarizing
       coverage, cost, and pass/fail by hypothesis. Optionally commits +
       pushes the result tree if --git-push is set.

Exit codes:
  0   pipeline completed (results_v0.4/stats.json written).
  1   pre-flight failure.
  2   one or more domain workers exited non-zero AND no rows landed.
  3   judge subset failed.
  4   stats failed.
  5   git push failed (only when --git-push set).

Usage::

  uv run python scripts/run_v0_4_bedrock.py --git-push
  uv run python scripts/run_v0_4_bedrock.py --skip-judge --skip-stats   # pilot only
  uv run python scripts/run_v0_4_bedrock.py --max-parallel 2            # throttle
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAINS = ("poetry_gen", "poetry_interp", "aut", "sci_creativity")

# Global cross-region inference profile IDs (Anthropic's recommended
# defaults on Bedrock as of 2026). Override via --haiku-model /
# --sonnet-model flags or BEDROCK_HAIKU_MODEL / BEDROCK_SONNET_MODEL env.
DEFAULT_HAIKU_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_SONNET_MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


@dataclass
class DomainWorker:
    domain: str
    proc: subprocess.Popen[bytes]
    log_path: Path
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None
    halted_reason: str | None = field(default=None)


def _ts() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _log(msg: str) -> None:
    print(f"[orch {_ts()}] {msg}", flush=True)


def _preflight(args: argparse.Namespace) -> tuple[str, str]:
    """Validate environment, return (haiku_model, sonnet_model)."""
    if shutil.which(args.cli_bin) is None:
        _log(f"FATAL: `{args.cli_bin}` not found on PATH.")
        sys.exit(1)
    try:
        proc = subprocess.run(  # noqa: S603
            [args.cli_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _log(f"claude --version: {(proc.stdout or proc.stderr).strip()[:200]}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        _log(f"FATAL: `{args.cli_bin} --version` failed: {exc}")
        sys.exit(1)

    use_bedrock = os.environ.get("CLAUDE_CODE_USE_BEDROCK", "")
    if use_bedrock != "1":
        _log(
            "WARN: CLAUDE_CODE_USE_BEDROCK is not set to '1' in the parent env. "
            "If your `claude` CLI defaults to Bedrock via desktop config that's fine; "
            "otherwise rate-limited OAuth quota will be used and the pilot will halt."
        )

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        _log("WARN: AWS_REGION / AWS_DEFAULT_REGION unset; Bedrock SDK may default to us-east-1.")

    haiku_model = (
        args.haiku_model
        or os.environ.get("BEDROCK_HAIKU_MODEL")
        or os.environ.get("ANTHROPIC_SMALL_FAST_MODEL")
        or DEFAULT_HAIKU_MODEL
    )
    sonnet_model = (
        args.sonnet_model
        or os.environ.get("BEDROCK_SONNET_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or DEFAULT_SONNET_MODEL
    )
    _log(f"haiku model = {haiku_model}")
    _log(f"sonnet model = {sonnet_model}")
    return haiku_model, sonnet_model


def _build_worker_env(*, base_env: dict[str, str], domain: str, haiku_model: str) -> dict[str, str]:
    env = dict(base_env)
    audit_root = REPO_ROOT / "audit" / "v0.4"
    audit_root.mkdir(parents=True, exist_ok=True)
    env["PCE_HAIKU_MODEL"] = haiku_model
    env["PCE_HAIKU_COST_LEDGER"] = str(audit_root / f"cost_ledger_{domain}.json")
    env["PCE_COST_SNAPSHOT_PATH"] = str(audit_root / f"cost_snapshot_{domain}.json")
    env["PCE_INTEGRITY_LOG_PATH"] = str(audit_root / f"integrity_probes_{domain}.jsonl")
    env["PCE_HAIKU_AUDIT_DIR"] = str(REPO_ROOT / "audit" / "haiku" / domain)
    # Loosen the per-call cost cap: the in-process budget guard is OAuth-era
    # belt-and-braces; Bedrock has no per-call cap surface here.
    env["PCE_HAIKU_COST_CAP_USD"] = "1000000"
    return env


def _spawn_worker(
    *,
    domain: str,
    out_dir: Path,
    haiku_model: str,
    n_per_domain: int,
    K: int,
    seed: int,
    retry_failed: bool,
    no_integrity_probe: bool,
    log_dir: Path,
) -> DomainWorker:
    log_path = log_dir / f"v0_4_pilot.{domain}.bedrock.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.driver",
        "--domains",
        domain,
        "--out-dir",
        str(out_dir),
        "--cost-cap-usd",
        "0",
        "--K",
        str(K),
        "--seed",
        str(seed),
        f"--n-{domain.replace('_', '-')}",
        str(n_per_domain),
    ]
    if retry_failed:
        cmd.append("--retry-failed")
    if no_integrity_probe:
        cmd.append("--no-integrity-probe")
    env = _build_worker_env(
        base_env=dict(os.environ), domain=domain, haiku_model=haiku_model
    )
    fh = log_path.open("wb")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    _log(f"started [{domain}] pid={proc.pid} log={log_path.name}")
    return DomainWorker(
        domain=domain, proc=proc, log_path=log_path, started_at=time.time()
    )


def _domain_progress(out_dir: Path, domain: str) -> dict[str, Any]:
    fp = out_dir / f"{domain}.json"
    if not fp.exists():
        return {"items_with_rows": 0, "expected_arms": 4, "complete_items": 0}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"items_with_rows": 0, "expected_arms": 4, "complete_items": 0}
    n_items = 0
    n_complete = 0
    base_arms = {"haiku_bare", "haiku_cascade", "haiku_bare_2K_scorer", "haiku_generic_revise_2pass"}
    for item_id, item_rows in data.items():
        if not isinstance(item_rows, dict):
            continue
        if item_id == "_integrity_probes":
            continue
        n_items += 1
        present = {k for k in item_rows if k in base_arms}
        if present == base_arms:
            n_complete += 1
    return {"items_with_rows": n_items, "expected_arms": 4, "complete_items": n_complete}


def _ledger_total(ledger_path: Path) -> tuple[float, int]:
    if not ledger_path.exists():
        return 0.0, 0
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0, 0
    return float(data.get("total_usd", 0.0)), int(data.get("n_calls", 0))


def _write_status_md(
    *,
    workers: list[DomainWorker],
    out_dir: Path,
    audit_root: Path,
    started_at: float,
    haiku_model: str,
    sonnet_model: str,
    phase: str,
    extras: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    elapsed = time.time() - started_at
    lines.append(f"# v0.4 Phase 7 (Bedrock pilot) — {phase}")
    lines.append("")
    lines.append(f"- Updated: {_ts()}")
    lines.append(f"- Elapsed: {elapsed/60.0:.1f} min")
    lines.append(f"- Haiku model: `{haiku_model}`")
    lines.append(f"- Sonnet model: `{sonnet_model}`")
    lines.append("")
    lines.append("## Per-domain progress")
    lines.append("")
    lines.append("| domain | items_with_rows | complete_items (4/4 arms) | ledger_$ | n_calls | proc | elapsed (min) |")
    lines.append("|---|---|---|---|---|---|---|")
    total_calls = 0
    total_cost = 0.0
    for w in workers:
        prog = _domain_progress(out_dir, w.domain)
        ledger_path = audit_root / f"cost_ledger_{w.domain}.json"
        cost, n_calls = _ledger_total(ledger_path)
        total_cost += cost
        total_calls += n_calls
        wall = (
            (w.finished_at - w.started_at) if w.finished_at else (time.time() - w.started_at)
        )
        if w.exit_code is None:
            status = "RUNNING"
        elif w.exit_code == 0:
            status = "DONE"
        else:
            status = f"EXIT={w.exit_code}"
            if w.halted_reason:
                status += f" ({w.halted_reason})"
        lines.append(
            f"| {w.domain} | {prog['items_with_rows']} | {prog['complete_items']} | "
            f"{cost:.3f} | {n_calls} | {status} | {wall/60.0:.1f} |"
        )
    lines.append("")
    lines.append(f"**Pilot totals:** ${total_cost:.3f} over {total_calls} Bedrock calls.")
    lines.append("")
    if extras:
        lines.append("## Phase outputs")
        lines.append("")
        for k, v in extras.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wait_for_workers(
    *,
    workers: list[DomainWorker],
    out_dir: Path,
    audit_root: Path,
    started_at: float,
    haiku_model: str,
    sonnet_model: str,
    poll_s: float,
) -> None:
    while any(w.exit_code is None for w in workers):
        for w in workers:
            if w.exit_code is not None:
                continue
            rc = w.proc.poll()
            if rc is not None:
                w.exit_code = int(rc)
                w.finished_at = time.time()
                tail = ""
                try:
                    tail = w.log_path.read_text(encoding="utf-8")[-1200:]
                except OSError:
                    pass
                if "HALT: HaikuRateLimitError" in tail:
                    w.halted_reason = "HaikuRateLimitError"
                _log(
                    f"worker [{w.domain}] exited rc={w.exit_code} "
                    f"after {(w.finished_at - w.started_at)/60.0:.1f} min"
                    f"{' [' + w.halted_reason + ']' if w.halted_reason else ''}"
                )
        _write_status_md(
            workers=workers,
            out_dir=out_dir,
            audit_root=audit_root,
            started_at=started_at,
            haiku_model=haiku_model,
            sonnet_model=sonnet_model,
            phase="Phase 7-A pilot (parallel domains)",
        )
        if any(w.exit_code is None for w in workers):
            time.sleep(poll_s)


def _merge_ledgers(audit_root: Path, domains: tuple[str, ...]) -> Path:
    merged_path = audit_root / "cost_ledger_merged.json"
    total_usd = 0.0
    n_calls = 0
    by_model: dict[str, dict[str, float]] = {}
    for domain in domains:
        ledger_path = audit_root / f"cost_ledger_{domain}.json"
        if not ledger_path.exists():
            continue
        try:
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total_usd += float(data.get("total_usd", 0.0))
        n_calls += int(data.get("n_calls", 0))
        for model, slot in (data.get("by_model") or {}).items():
            agg = by_model.setdefault(
                model, {"total_usd": 0.0, "n_calls": 0, "total_latency_ms": 0}
            )
            agg["total_usd"] += float(slot.get("total_usd", 0.0))
            agg["n_calls"] += int(slot.get("n_calls", 0))
            agg["total_latency_ms"] += int(slot.get("total_latency_ms", 0))
    merged = {"total_usd": total_usd, "n_calls": n_calls, "by_model": by_model}
    merged_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged_path


def _merge_integrity_logs(audit_root: Path, domains: tuple[str, ...]) -> Path:
    merged_path = audit_root / "integrity_probes_merged.jsonl"
    with merged_path.open("w", encoding="utf-8") as out:
        for domain in domains:
            src = audit_root / f"integrity_probes_{domain}.jsonl"
            if not src.exists():
                continue
            with src.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
    return merged_path


def _run_judge(
    *,
    out_dir: Path,
    sonnet_model: str,
    n_per_domain: int,
    cli_bin: str,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "judge_subset.py"),
        "--results-dir",
        str(out_dir),
        "--out-jsonl",
        str(out_dir / "judge.jsonl"),
        "--n-per-domain",
        str(n_per_domain),
        "--model",
        sonnet_model,
        "--cli-bin",
        cli_bin,
        "--cost-cap-usd",
        "0",
    ]
    _log(f"judge: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))  # noqa: S603
    return int(proc.returncode)


def _run_stats(*, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.stats",
        "--version",
        "v0.4",
        "--results-dir",
        str(out_dir),
        "--out",
        str(out_dir / "stats.json"),
    ]
    _log(f"stats: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))  # noqa: S603
    return int(proc.returncode)


def _git_push(branch: str) -> int:
    for cmd in (
        ["git", "add", "benchmarks/results_v0.4", "audit/v0.4"],
        [
            "git",
            "-c",
            "user.email=pce-bedrock@local",
            "-c",
            "user.name=pce-bedrock-pilot",
            "commit",
            "-m",
            f"v0.4 phase 7: Bedrock pilot results ({_ts()})",
            "--allow-empty",
        ],
        ["git", "push", "origin", branch],
    ):
        _log(f"git: {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode  # noqa: S603
        if rc != 0:
            return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "results_v0.4")
    parser.add_argument("--audit-root", type=Path, default=REPO_ROOT / "audit" / "v0.4")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--n-per-domain", type=int, default=20)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--haiku-model", default=None)
    parser.add_argument("--sonnet-model", default=None)
    parser.add_argument("--cli-bin", default=os.environ.get("PCE_HAIKU_CLI", "claude"))
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum simultaneous domain workers. Default 4 (one per domain).",
    )
    parser.add_argument("--poll-s", type=float, default=30.0)
    parser.add_argument("--no-integrity-probe", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=True,
        help="Pass --retry-failed to each worker (default true so partial macOS state is filled in).",
    )
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument(
        "--git-push",
        action="store_true",
        help="Commit results_v0.4/ + audit/v0.4/ to current branch and push to origin.",
    )
    parser.add_argument(
        "--branch",
        default="v0.4-mechanism-study",
        help="Branch name for --git-push.",
    )
    parser.add_argument(
        "--judge-n-per-domain",
        type=int,
        default=8,
        help="Number of items per domain for the Sonnet judge (default 8 = 32 total).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audit_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    haiku_model, sonnet_model = _preflight(args)
    started_at = time.time()

    workers: list[DomainWorker] = []
    if not args.skip_pilot:
        # Honor --max-parallel by chunking the domain list.
        pending = list(args.domains)
        while pending or any(w.exit_code is None for w in workers):
            running = [w for w in workers if w.exit_code is None]
            while pending and len(running) < args.max_parallel:
                domain = pending.pop(0)
                w = _spawn_worker(
                    domain=domain,
                    out_dir=args.out_dir,
                    haiku_model=haiku_model,
                    n_per_domain=args.n_per_domain,
                    K=args.K,
                    seed=args.seed,
                    retry_failed=args.retry_failed,
                    no_integrity_probe=args.no_integrity_probe,
                    log_dir=args.log_dir,
                )
                workers.append(w)
                running.append(w)
            if running:
                _wait_for_workers(
                    workers=workers,
                    out_dir=args.out_dir,
                    audit_root=args.audit_root,
                    started_at=started_at,
                    haiku_model=haiku_model,
                    sonnet_model=sonnet_model,
                    poll_s=args.poll_s,
                )

        any_landed = any(
            (args.out_dir / f"{d}.json").exists()
            and json.loads((args.out_dir / f"{d}.json").read_text(encoding="utf-8"))
            for d in args.domains
        )
        if not any_landed:
            _log("FATAL: pilot finished but produced no result rows.")
            return 2

    merged_ledger = _merge_ledgers(args.audit_root, tuple(args.domains))
    merged_probes = _merge_integrity_logs(args.audit_root, tuple(args.domains))
    _log(f"merged ledger -> {merged_ledger}")
    _log(f"merged integrity log -> {merged_probes}")

    judge_rc = 0
    if not args.skip_judge:
        judge_rc = _run_judge(
            out_dir=args.out_dir,
            sonnet_model=sonnet_model,
            n_per_domain=args.judge_n_per_domain,
            cli_bin=args.cli_bin,
        )
        if judge_rc != 0:
            _log(f"judge subset failed (rc={judge_rc}); continuing to stats.")

    stats_rc = 0
    if not args.skip_stats:
        stats_rc = _run_stats(out_dir=args.out_dir)
        if stats_rc != 0:
            _log(f"stats failed (rc={stats_rc}).")

    extras = {
        "merged_ledger": str(merged_ledger.relative_to(REPO_ROOT)),
        "merged_integrity_log": str(merged_probes.relative_to(REPO_ROOT)),
    }
    judge_path = args.out_dir / "judge.jsonl"
    if judge_path.exists():
        extras["judge.jsonl"] = str(judge_path.relative_to(REPO_ROOT))
    judge_agg = args.out_dir / "judge_agreement.json"
    if judge_agg.exists():
        extras["judge_agreement.json"] = str(judge_agg.relative_to(REPO_ROOT))
    stats_path = args.out_dir / "stats.json"
    if stats_path.exists():
        extras["stats.json"] = str(stats_path.relative_to(REPO_ROOT))
    extras["judge_rc"] = str(judge_rc)
    extras["stats_rc"] = str(stats_rc)
    _write_status_md(
        workers=workers,
        out_dir=args.out_dir,
        audit_root=args.audit_root,
        started_at=started_at,
        haiku_model=haiku_model,
        sonnet_model=sonnet_model,
        phase="DONE",
        extras=extras,
    )

    if args.git_push:
        push_rc = _git_push(args.branch)
        if push_rc != 0:
            return 5

    if judge_rc != 0:
        return 3
    if stats_rc != 0:
        return 4
    _log(f"v0.4 Phase 7 pipeline complete in {(time.time() - started_at)/60.0:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
