"""Standalone PCE command-line interface (Phase 8 portability).

This module is the single source of truth for invoking PCE *without* an IDE.
It is wired to the ``pce`` console script via ``pyproject.toml``::

    pip install -e .
    pce config show
    pce smoke
    pce cascade --prompt "Write a haiku about rain on a tin roof" \\
                --constraint "imagistic specificity"
    pce judge-pair --domain poetry_gen --item p07 \\
                   --treatment-text path/A.txt --control-text path/B.txt
    pce showcase --regenerate sanskrit_anustubh

The CLI never depends on Cursor, Claude Code, or the plugin manifest. It
talks to the same ``HaikuLM`` substrate the plugin uses, but reads model
selection through ``PCEConfig`` (TOML + env + ``--model`` override). When
``claude`` is not on PATH the CLI prints actionable instructions instead
of crashing with a generic CLI-not-found error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pce.config import PCEConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _emit(text: str, *, file: Any = None) -> None:
    print(text, file=file or sys.stdout, flush=True)


def _err(text: str) -> None:
    _emit(text, file=sys.stderr)


def _check_cli_present(cli_bin: str) -> str | None:
    """Return None if the CLI is on PATH, otherwise an actionable hint."""
    if shutil.which(cli_bin):
        return None
    return textwrap.dedent(f"""
        ERROR: cannot find the `{cli_bin}` binary on PATH.

        PCE v0.4 talks to Anthropic models exclusively through the `claude`
        CLI (the OAuth/CLI substrate documented in
        docs/adr/v0.4/ADR-007-sdk-removal.md). Install it via the official
        instructions at https://docs.claude.com/en/docs/claude-code/quickstart
        and re-run.

        If the binary is installed under a different name, point PCE at it
        with one of:
          export PCE_CLI=/path/to/claude
          export PCE_HAIKU_CLI=/path/to/claude   # back-compat alias
          pce --cli-bin /path/to/claude <subcommand> ...
    """).strip()


def _load_config(args: argparse.Namespace) -> PCEConfig:
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["cascade_model"] = args.model
    if args.judge_model:
        overrides["judge_model"] = args.judge_model
    if args.cli_bin:
        overrides["cli_bin"] = args.cli_bin
    if args.timeout_s is not None:
        overrides["timeout_s"] = args.timeout_s
    user_toml = Path(args.config) if args.config else None
    return PCEConfig.load(user_toml=user_toml, overrides=overrides)


def _build_lm(cfg: PCEConfig):  # type: ignore[no-untyped-def]
    """Construct a ``HaikuLM`` from ``cfg``. Imported lazily so ``pce config``
    and ``pce --help`` work without importing the substrate (which loads
    sentence-transformers and is slow)."""
    from pce.substrate.haiku_lm import HaikuConfig, HaikuLM

    hc = HaikuConfig(
        model=cfg.resolved_cascade_model(),
        cli_bin=cfg.cli_bin,
        timeout_s=cfg.timeout_s,
        use_sdk=False,
        cost_cap_usd=cfg.cost_cap_usd,
        cli_retry=cfg.cli_retry,
        cli_backoff_s=cfg.cli_backoff_s,
        clean_substrate=cfg.clean_substrate,
        clean_home_root=cfg.clean_home_root,
        system_prompt_override=cfg.system_prompt_override,
    )
    return HaikuLM(config=hc)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    """`pce config show` — print the resolved config + sources."""
    cfg = _load_config(args)
    payload: dict[str, Any] = {
        "resolved": cfg.to_dict(),
        "resolved_cascade_model": cfg.resolved_cascade_model(),
        "resolved_judge_model": cfg.resolved_judge_model(),
        "user_toml": str(args.config) if args.config else None,
        "cli_bin_present_on_path": shutil.which(cfg.cli_bin) is not None,
    }
    _emit(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """`pce smoke` — call the substrate once with a trivial prompt to verify
    the OAuth/CLI chain is reachable. Does NOT exercise the full cascade.
    """
    cfg = _load_config(args)
    if (msg := _check_cli_present(cfg.cli_bin)) is not None:
        _err(msg)
        return 2
    if args.dry_run:
        _emit(json.dumps({
            "dry_run": True,
            "model": cfg.resolved_cascade_model(),
            "cli_bin": cfg.cli_bin,
            "timeout_s": cfg.timeout_s,
        }, indent=2))
        return 0
    lm = _build_lm(cfg)
    result = lm.generate(
        "Reply with exactly the four characters: PONG",
        n=1, max_tokens=8, seed=4242,
    )
    cand = result[0]
    out = {
        "model": cfg.resolved_cascade_model(),
        "text": cand.text.strip(),
        "ok": "PONG" in cand.text.upper(),
        "tokens_estimate": getattr(cand, "n_tokens", None),
    }
    _emit(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


def cmd_cascade(args: argparse.Namespace) -> int:
    """`pce cascade --prompt … --constraint …` — run one cascade pass and
    print a JSON summary (composite score, draft, revision, vimarsa event,
    cost). Writes the full trace to ``--out`` if supplied.
    """
    cfg = _load_config(args)
    if (msg := _check_cli_present(cfg.cli_bin)) is not None:
        _err(msg)
        return 2
    if not args.prompt:
        _err("--prompt is required for `pce cascade`")
        return 2
    if args.dry_run:
        _emit(json.dumps({
            "dry_run": True,
            "model": cfg.resolved_cascade_model(),
            "prompt": args.prompt,
            "constraint": args.constraint,
            "K": args.k,
            "cit_temperature": args.cit_temperature,
        }, indent=2))
        return 0

    from pce.cascade import run_cascade
    from pce.substrate.embed import Embedder
    from pce.types import Constraint

    embedder = Embedder()
    constraint_text = args.constraint or "well-crafted, original, on-topic"
    c = Constraint(text=constraint_text, embedding=embedder.embed(constraint_text))
    lm = _build_lm(cfg)
    state = run_cascade(
        prompt=args.prompt,
        constraint=c,
        lm=lm, embed=embedder,
        K=args.k,
        cit_temperature=args.cit_temperature,
        max_tokens=args.max_tokens,
        base_seed=args.seed,
    )
    summary = {
        "model": cfg.resolved_cascade_model(),
        "prompt": args.prompt,
        "constraint": constraint_text,
        "committed_text": state.surface,
        "draft_text": getattr(state, "surface_draft", None),
        "revised_differs_from_draft": getattr(state, "revision_differs_from_draft", None),
        "vimarsa_event": getattr(state, "vimarsa_event", None),
        "delta_F_draft": getattr(state, "delta_F_draft", None),
        "delta_F_revision": getattr(state, "delta_F_revision", None),
        "K_effective": getattr(state, "K_effective", None),
        "elapsed_s": getattr(state, "elapsed_s", None),
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_state_dump(state, summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary["trace_written_to"] = str(out_path)
    _emit(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _state_dump(state: Any, summary: dict[str, Any]) -> dict[str, Any]:
    """Serialise a CascadeState into a JSON-safe trace dict."""
    out: dict[str, Any] = dict(summary)
    out["__cascade_state_repr__"] = type(state).__name__
    for attr in (
        "surface", "surface_draft", "surface_revision",
        "vimarsa_event", "delta_F_draft", "delta_F_revision",
        "K_effective", "elapsed_s",
    ):
        if hasattr(state, attr):
            v = getattr(state, attr)
            try:
                json.dumps(v)
                out[attr] = v
            except (TypeError, ValueError):
                out[attr] = repr(v)
    return out


def cmd_judge_pair(args: argparse.Namespace) -> int:
    """`pce judge-pair` — invoke the v0.4 Sonnet judge on a treatment/control
    text pair and print {winner, confidence, rationale}. Reads the same
    judge prompt the benchmark harness uses (frozen sha pinned in
    ``benchmarks/judge.py``)."""
    cfg = _load_config(args)
    if (msg := _check_cli_present(cfg.cli_bin)) is not None:
        _err(msg)
        return 2
    treatment = Path(args.treatment_text).read_text(encoding="utf-8")
    control = Path(args.control_text).read_text(encoding="utf-8")
    if args.dry_run:
        _emit(json.dumps({
            "dry_run": True,
            "judge_model": cfg.resolved_judge_model(),
            "domain": args.domain,
            "item_id": args.item_id,
            "len_treatment": len(treatment),
            "len_control": len(control),
        }, indent=2))
        return 0

    try:
        from benchmarks import judge as bench_judge
    except ImportError:
        _err("ERROR: benchmarks.judge unavailable; install dev deps or run from repo root")
        return 3
    if not hasattr(bench_judge, "judge_pair"):
        _err("ERROR: benchmarks.judge does not expose judge_pair(); cannot proceed")
        return 3
    out = bench_judge.judge_pair(
        treatment=treatment,
        control=control,
        domain=args.domain,
        item_id=args.item_id,
        judge_model=cfg.resolved_judge_model(),
        cli_bin=cfg.cli_bin,
    )
    _emit(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_showcase(args: argparse.Namespace) -> int:
    """`pce showcase` — list / regenerate showcase outputs. By default
    shows the on-disk index of ``benchmarks/showcase_v0.4/``. With
    ``--regenerate <slug|all>`` it shells out to the shared generator
    script."""
    showcase_root = REPO_ROOT / "benchmarks" / "showcase_v0.4"
    if args.regenerate:
        gen_script = REPO_ROOT / "scripts" / "generate_v0_4_showcase.py"
        if not gen_script.exists():
            _err(f"ERROR: generator script not found at {gen_script}")
            return 4
        cmd = [sys.executable, str(gen_script), "--slug", args.regenerate]
        if args.model:
            cmd += ["--model", args.model]
        _emit(f"$ {' '.join(cmd)}")
        return os.spawnvp(os.P_WAIT, cmd[0], cmd)
    if not showcase_root.exists():
        _emit(json.dumps({"showcase_root": str(showcase_root), "items": []}))
        return 0
    items = []
    for p in sorted(showcase_root.glob("*/trace.json")):
        try:
            tr = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        items.append({
            "slug": p.parent.name,
            "domain": tr.get("domain"),
            "model": tr.get("model"),
            "composite": tr.get("composite"),
            "trace": str(p.relative_to(REPO_ROOT)),
        })
    _emit(json.dumps({"showcase_root": str(showcase_root), "items": items}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _add_global_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="cascade model alias (haiku/sonnet/opus) or full Anthropic CLI model ID")
    p.add_argument("--judge-model", help="judge model alias or full ID (used by judge-pair)")
    p.add_argument("--cli-bin", help="path to the `claude` CLI binary")
    p.add_argument("--timeout-s", type=int, default=None, help="per-call CLI timeout (seconds)")
    p.add_argument("--config", default=None,
                   help="explicit user TOML; otherwise read from ~/.config/pce/config.toml")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved invocation without calling the CLI")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pce",
        description="Pratyabhijna Creative Engine — standalone CLI (v0.4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              pce config show
              pce smoke --dry-run
              pce cascade --prompt "Haiku about rain" --constraint "imagism" --k 3
              pce judge-pair --domain poetry_gen --item p07 \\
                  --treatment-text out/treatment.txt --control-text out/control.txt
              pce showcase
              pce showcase --regenerate sanskrit_anustubh
        """).strip(),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # config
    p_cfg = sub.add_parser("config", help="show resolved PCE config")
    p_cfg.add_argument("action", nargs="?", default="show", choices=("show",))
    _add_global_flags(p_cfg)
    p_cfg.set_defaults(func=cmd_config)

    # smoke
    p_smk = sub.add_parser("smoke", help="single-call OAuth/CLI smoke test")
    _add_global_flags(p_smk)
    p_smk.set_defaults(func=cmd_smoke)

    # cascade
    p_csc = sub.add_parser("cascade", help="run one cascade pass on a prompt")
    p_csc.add_argument("--prompt", required=False, help="user prompt (required unless --dry-run)")
    p_csc.add_argument("--constraint", default=None,
                       help="natural-language constraint pulling generation toward an axis")
    p_csc.add_argument("--k", type=int, default=4, help="best-of-K candidates per pass")
    p_csc.add_argument("--cit-temperature", type=float, default=1.0,
                       help="best-of-K candidate width (>1 widens, <1 narrows)")
    p_csc.add_argument("--max-tokens", type=int, default=300)
    p_csc.add_argument("--seed", type=int, default=4242)
    p_csc.add_argument("--out", default=None,
                       help="optional path to write the full JSON trace")
    _add_global_flags(p_csc)
    p_csc.set_defaults(func=cmd_cascade)

    # judge-pair
    p_jp = sub.add_parser("judge-pair", help="invoke Sonnet judge on a treatment/control pair")
    p_jp.add_argument("--domain", required=True,
                      choices=("poetry_gen", "poetry_interp", "aut", "sci_creativity"))
    p_jp.add_argument("--item-id", required=True, dest="item_id",
                      help="item slug (used in the audit trail)")
    p_jp.add_argument("--treatment-text", required=True, dest="treatment_text",
                      help="path to file containing the treatment-arm output")
    p_jp.add_argument("--control-text", required=True, dest="control_text",
                      help="path to file containing the control-arm output")
    _add_global_flags(p_jp)
    p_jp.set_defaults(func=cmd_judge_pair)

    # showcase
    p_sc = sub.add_parser("showcase", help="list or regenerate showcase outputs")
    p_sc.add_argument("--regenerate", default=None, metavar="SLUG_OR_ALL",
                      help="regenerate one slug (e.g. sanskrit_anustubh) or 'all'")
    _add_global_flags(p_sc)
    p_sc.set_defaults(func=cmd_showcase)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
