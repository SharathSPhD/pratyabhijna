"""Tests for ``src/pce/config.py`` — PCEConfig resolution chain.

Covers:
  * defaults
  * env overrides (PCE_* primary + PCE_HAIKU_* back-compat aliases)
  * TOML overrides (user + repo)
  * resolution order: defaults < repo TOML < user TOML < env < explicit overrides
  * model alias resolution
  * SDK deprecation warning
  * ``HaikuConfig.from_env`` defers to PCEConfig
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from pce.config import (
    MODEL_ALIASES,
    PCEConfig,
    default_repo_config_path,
    default_user_config_path,
    resolve_model,
)


# ---- defaults --------------------------------------------------------------


def test_defaults_when_no_env_no_toml(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.cascade_model == "haiku"
    assert cfg.judge_model == "sonnet"
    assert cfg.cli_bin == "claude"
    assert cfg.timeout_s == 120
    assert cfg.cost_cap_usd == 18.0
    assert cfg.clean_substrate is True


def test_resolved_model_uses_alias() -> None:
    cfg = PCEConfig(cascade_model="claude-haiku")
    assert cfg.resolved_cascade_model() == "haiku"
    cfg2 = PCEConfig(cascade_model="opus")
    assert cfg2.resolved_cascade_model() == "opus"
    cfg3 = PCEConfig(cascade_model="global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    assert cfg3.resolved_cascade_model() == "global.anthropic.claude-sonnet-4-5-20250929-v1:0"


def test_resolve_model_helper_passes_through_unknown() -> None:
    assert resolve_model("custom-fine-tune-2026-04") == "custom-fine-tune-2026-04"
    assert resolve_model("HAIKU") == "haiku"
    for k in MODEL_ALIASES:
        assert resolve_model(k) in {"haiku", "sonnet", "opus"}


# ---- env -------------------------------------------------------------------


def test_env_pce_model_wins_over_default(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_MODEL": "sonnet"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.cascade_model == "sonnet"
    assert cfg.judge_model == "sonnet"  # default


def test_env_pce_haiku_model_back_compat(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_HAIKU_MODEL": "opus"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.cascade_model == "opus"


def test_env_judge_model(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_JUDGE_MODEL": "opus"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.judge_model == "opus"


def test_env_clean_substrate_can_be_disabled(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_HAIKU_CLEAN_SUBSTRATE": "0"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.clean_substrate is False


def test_env_pce_cli_overrides_pce_haiku_cli(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_CLI": "/usr/local/bin/claude", "PCE_HAIKU_CLI": "claude-old"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
    )
    assert cfg.cli_bin == "/usr/local/bin/claude"


def test_env_invalid_int_warns_and_falls_back(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = PCEConfig.load(
            env={"PCE_TIMEOUT_S": "not-a-number"},
            user_toml=tmp_path / "user.toml",
            repo_toml=tmp_path / "repo.toml",
        )
    assert cfg.timeout_s == 120
    assert any("PCE_TIMEOUT_S" in str(w.message) for w in caught)


# ---- TOML ------------------------------------------------------------------


def test_user_toml_overrides_default(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text(
        '[pce]\ncascade_model = "sonnet"\ntimeout_s = 60\n', encoding="utf-8",
    )
    cfg = PCEConfig.load(env={}, user_toml=user, repo_toml=tmp_path / "repo.toml")
    assert cfg.cascade_model == "sonnet"
    assert cfg.timeout_s == 60


def test_repo_toml_overridden_by_user_toml(tmp_path: Path) -> None:
    repo = tmp_path / "repo.toml"
    repo.write_text('[pce]\ncascade_model = "opus"\n', encoding="utf-8")
    user = tmp_path / "user.toml"
    user.write_text('[pce]\ncascade_model = "sonnet"\n', encoding="utf-8")
    cfg = PCEConfig.load(env={}, user_toml=user, repo_toml=repo)
    assert cfg.cascade_model == "sonnet"


def test_env_overrides_user_toml(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text('[pce]\ncascade_model = "sonnet"\n', encoding="utf-8")
    cfg = PCEConfig.load(env={"PCE_MODEL": "haiku"}, user_toml=user, repo_toml=tmp_path / "repo.toml")
    assert cfg.cascade_model == "haiku"


def test_explicit_overrides_win_over_env(tmp_path: Path) -> None:
    cfg = PCEConfig.load(
        env={"PCE_MODEL": "sonnet"},
        user_toml=tmp_path / "user.toml",
        repo_toml=tmp_path / "repo.toml",
        overrides={"cascade_model": "opus"},
    )
    assert cfg.cascade_model == "opus"


def test_unknown_toml_keys_go_into_extras(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text(
        '[pce]\ncascade_model = "haiku"\nrandom_research_flag = "ablation_42"\n',
        encoding="utf-8",
    )
    cfg = PCEConfig.load(env={}, user_toml=user, repo_toml=tmp_path / "repo.toml")
    assert cfg.cascade_model == "haiku"
    assert cfg.extras.get("random_research_flag") == "ablation_42"


def test_malformed_toml_warns_does_not_crash(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text("not = valid = toml", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = PCEConfig.load(env={}, user_toml=user, repo_toml=tmp_path / "repo.toml")
    assert cfg.cascade_model == "haiku"
    assert any("malformed TOML" in str(w.message) for w in caught)


# ---- deprecation -----------------------------------------------------------


def test_pce_use_sdk_emits_deprecation_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PCEConfig.load(
            env={"PCE_USE_SDK": "1"},
            user_toml=tmp_path / "user.toml",
            repo_toml=tmp_path / "repo.toml",
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("PCE_USE_SDK" in m for m in msgs)


# ---- HaikuConfig integration ----------------------------------------------


def test_haiku_config_from_env_defers_to_pce(monkeypatch: pytest.MonkeyPatch) -> None:
    from pce.substrate.haiku_lm import HaikuConfig

    monkeypatch.setenv("PCE_MODEL", "sonnet")
    monkeypatch.setenv("PCE_TIMEOUT_S", "47")
    monkeypatch.setenv("PCE_HAIKU_COST_CAP_USD", "9.99")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/nonexistent-xdg")  # avoid user toml leak
    cfg = HaikuConfig.from_env()
    assert cfg.model == "sonnet"
    assert cfg.timeout_s == 47
    assert cfg.cost_cap_usd == 9.99
    assert cfg.use_sdk is False  # always False post-ADR-007


# ---- defaults sanity ------------------------------------------------------


def test_default_paths_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    p = default_user_config_path()
    assert p == tmp_path / "xdg" / "pce" / "config.toml"
    monkeypatch.chdir(tmp_path)
    assert default_repo_config_path() == tmp_path / "pce.toml"
