"""v0.4.1 review fix #6: assert PCEConfig.load layering matches the docstring.

The chain is defaults < repo TOML < user TOML < env < overrides. Every
later layer must win over the earlier ones for the same field.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pce.config import PCEConfig


@pytest.fixture
def repo_toml(tmp_path: Path) -> Path:
    p = tmp_path / "pce.toml"
    p.write_text(
        '[pce]\ncascade_model = "from_repo"\njudge_model = "from_repo"\n'
        'cli_bin = "from_repo"\ntimeout_s = 11\n',
        encoding="utf-8",
    )
    return p


@pytest.fixture
def user_toml(tmp_path: Path) -> Path:
    p = tmp_path / "user.toml"
    p.write_text(
        '[pce]\ncascade_model = "from_user"\ntimeout_s = 22\n',
        encoding="utf-8",
    )
    return p


def test_defaults_only() -> None:
    cfg = PCEConfig.load(user_toml=Path("/no/such/user.toml"),
                         repo_toml=Path("/no/such/repo.toml"),
                         env={}, overrides=None)
    assert cfg.cascade_model == "haiku"
    assert cfg.judge_model == "sonnet"
    assert cfg.cli_bin == "claude"


def test_repo_toml_wins_over_defaults(repo_toml: Path) -> None:
    cfg = PCEConfig.load(repo_toml=repo_toml, user_toml=Path("/no/such"),
                         env={}, overrides=None)
    assert cfg.cascade_model == "from_repo"
    assert cfg.judge_model == "from_repo"
    assert cfg.timeout_s == 11


def test_user_toml_wins_over_repo_toml(repo_toml: Path, user_toml: Path) -> None:
    cfg = PCEConfig.load(repo_toml=repo_toml, user_toml=user_toml,
                         env={}, overrides=None)
    assert cfg.cascade_model == "from_user", "user TOML must override repo TOML"
    assert cfg.timeout_s == 22, "user TOML must override repo TOML"
    assert cfg.judge_model == "from_repo", "user TOML didn't set judge_model; repo wins"
    assert cfg.cli_bin == "from_repo"


def test_env_wins_over_user_toml(repo_toml: Path, user_toml: Path) -> None:
    cfg = PCEConfig.load(
        repo_toml=repo_toml, user_toml=user_toml,
        env={"PCE_MODEL": "from_env", "PCE_TIMEOUT_S": "33"},
        overrides=None,
    )
    assert cfg.cascade_model == "from_env"
    assert cfg.timeout_s == 33


def test_legacy_haiku_env_aliases(repo_toml: Path) -> None:
    cfg = PCEConfig.load(
        repo_toml=repo_toml, user_toml=Path("/no/such"),
        env={"PCE_HAIKU_MODEL": "legacy_alias"},
        overrides=None,
    )
    assert cfg.cascade_model == "legacy_alias"


def test_overrides_win_over_env(repo_toml: Path, user_toml: Path) -> None:
    cfg = PCEConfig.load(
        repo_toml=repo_toml, user_toml=user_toml,
        env={"PCE_MODEL": "from_env"},
        overrides={"cascade_model": "from_override", "timeout_s": 44},
    )
    assert cfg.cascade_model == "from_override"
    assert cfg.timeout_s == 44


def test_use_sdk_emits_deprecation(repo_toml: Path) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PCEConfig.load(
            repo_toml=repo_toml, user_toml=Path("/no/such"),
            env={"PCE_USE_SDK": "1"},
            overrides=None,
        )
    msgs = [str(x.message) for x in w if issubclass(x.category, DeprecationWarning)]
    assert any("PCE_USE_SDK" in m for m in msgs)
