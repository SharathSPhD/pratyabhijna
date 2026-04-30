"""Tests for ``src/pce/cli.py`` — standalone PCE CLI.

Coverage:
  * ``--help`` renders for each subcommand
  * ``pce config show`` returns the resolved config
  * ``--model`` flag overrides cascade_model
  * ``pce smoke --dry-run`` does not call the substrate
  * ``pce cascade --dry-run`` does not call the substrate
  * Missing CLI binary produces an actionable error and exit 2
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from pce import cli as pce_cli


def _capture(argv: list[str]) -> tuple[int, str, str]:
    out = StringIO()
    err = StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        rc = pce_cli.main(argv)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return rc, out.getvalue(), err.getvalue()


def test_help_works() -> None:
    with pytest.raises(SystemExit) as ei:
        _capture(["--help"])
    assert ei.value.code == 0


def test_config_show(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("PCE_MODEL", raising=False)
    monkeypatch.delenv("PCE_HAIKU_MODEL", raising=False)
    rc, out, _ = _capture(["config", "show"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["resolved"]["cascade_model"] == "haiku"
    assert payload["resolved"]["judge_model"] == "sonnet"
    assert "resolved_cascade_model" in payload


def test_model_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("PCE_MODEL", raising=False)
    monkeypatch.delenv("PCE_HAIKU_MODEL", raising=False)
    rc, out, _ = _capture(["--", "config", "show", "--model", "opus"])
    # `pce config show --model opus` not `pce -- config …` — argparse
    # treats `--model` as a flag of the subcommand. So the safer form:
    rc, out, _ = _capture(["config", "show", "--model", "opus"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["resolved"]["cascade_model"] == "opus"


def test_smoke_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli.shutil, "which", lambda _b: "/fake/claude")
    rc, out, _ = _capture(["smoke", "--dry-run"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["model"] == "haiku"


def test_cascade_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli.shutil, "which", lambda _b: "/fake/claude")
    rc, out, _ = _capture(["cascade", "--dry-run", "--prompt", "hello world", "--constraint", "imagism", "--k", "3"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["prompt"] == "hello world"
    assert payload["K"] == 3


def test_cascade_requires_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli.shutil, "which", lambda _b: "/fake/claude")
    rc, _out, err = _capture(["cascade"])
    assert rc == 2
    assert "--prompt is required" in err


def test_missing_cli_bin_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli.shutil, "which", lambda _b: None)
    rc, _out, err = _capture(["smoke"])
    assert rc == 2
    assert "PCE_CLI" in err
    assert "claude" in err


def test_showcase_empty_when_root_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli, "REPO_ROOT", tmp_path)
    rc, out, _ = _capture(["showcase"])
    assert rc == 0
    payload = json.loads(out)
    assert payload["items"] == []


def test_showcase_index_when_traces_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(pce_cli, "REPO_ROOT", tmp_path)
    showcase = tmp_path / "benchmarks" / "showcase_v0.4" / "sanskrit_anustubh"
    showcase.mkdir(parents=True)
    (showcase / "trace.json").write_text(
        json.dumps({"domain": "poetry_gen", "model": "haiku", "composite": 0.71}),
        encoding="utf-8",
    )
    rc, out, _ = _capture(["showcase"])
    assert rc == 0
    payload = json.loads(out)
    assert len(payload["items"]) == 1
    assert payload["items"][0]["slug"] == "sanskrit_anustubh"
    assert payload["items"][0]["composite"] == 0.71
