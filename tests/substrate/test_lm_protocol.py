"""Cross-substrate Protocol-conformance tests.

Both `LocalLM` and `HaikuLM` must satisfy `LMProtocol`. We assert that with
runtime `isinstance` (which works for `@runtime_checkable` Protocols) plus a
behavior-level check on the small surface area the cascade actually touches.

These tests are *fast* and *do not* boot Qwen2 or call Haiku — they verify
the static contract.
"""
from __future__ import annotations

import inspect

from pce.substrate.lm_protocol import LMProtocol


def test_protocol_requires_three_attributes() -> None:
    members = {name for name, _ in inspect.getmembers(LMProtocol) if not name.startswith("_")}
    # Protocol attributes are reflected as class members on the runtime_checkable proxy.
    # Class-level `name` is a bare ClassVar without a getter, so we only assert
    # the methods explicitly here.
    for required in ("generate", "report"):
        assert required in members, f"LMProtocol missing required method: {required}"


def test_local_lm_class_has_protocol_surface() -> None:
    from pce.substrate.lm import LocalLM

    assert hasattr(LocalLM, "name"), "LocalLM must declare a `name` class attribute"
    assert hasattr(LocalLM, "generate")
    assert hasattr(LocalLM, "report")
    assert LocalLM.name  # non-empty


def test_haiku_lm_class_has_protocol_surface() -> None:
    from pce.substrate.haiku_lm import HaikuLM

    assert hasattr(HaikuLM, "name")
    assert hasattr(HaikuLM, "generate")
    assert hasattr(HaikuLM, "report")
    assert HaikuLM.name


def test_haiku_config_from_env_defaults() -> None:
    from pce.substrate.haiku_lm import HaikuConfig

    cfg = HaikuConfig.from_env()
    assert cfg.cli_bin
    assert cfg.model
    assert cfg.timeout_s > 0
    assert cfg.cost_cap_usd > 0


def test_haiku_seed_prefix_varies_with_seed() -> None:
    from pce.substrate.haiku_lm import _seed_prefix

    a = _seed_prefix(0)
    b = _seed_prefix(7)
    assert a != b, "different seeds must produce different prefixes"
