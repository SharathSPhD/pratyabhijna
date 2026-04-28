"""iccha: K-candidate sampler grid."""
from __future__ import annotations

import pytest

from pce.operators.iccha import PARITY_SAMPLER, iccha
from pce.substrate.embed import Embedder
from pce.substrate.lm import LocalLM
from pce.types import Constraint


@pytest.fixture(scope="module")
def lm() -> LocalLM:
    return LocalLM()


@pytest.fixture(scope="module")
def embed() -> Embedder:
    return Embedder()


@pytest.mark.real_model
@pytest.mark.slow
def test_iccha_grid_mode_returns_distinct_samplers(
    lm: LocalLM, embed: Embedder
) -> None:
    """In grid mode K samplers span an explore-exploit ladder (v0.1 behavior)."""
    q = embed.encode("a haiku about autumn leaves")
    constraint = Constraint(text="a haiku about autumn leaves", embedding=q)
    cands = iccha(
        "Compose a short poem.\n",
        constraint,
        lm=lm,
        K=4,
        max_tokens=24,
        base_seed=11,
        sampler_grid_mode="grid",
    )
    assert len(cands) == 4
    taus = {c.sampler["tau"] for c in cands}
    assert len(taus) >= 3


@pytest.mark.real_model
@pytest.mark.slow
def test_iccha_parity_mode_returns_identical_samplers(
    lm: LocalLM, embed: Embedder
) -> None:
    """In parity mode (v0.2 default) all K samplers equal the bare baseline."""
    q = embed.encode("a haiku about autumn leaves")
    constraint = Constraint(text="a haiku about autumn leaves", embedding=q)
    cands = iccha(
        "Compose a short poem.\n",
        constraint,
        lm=lm,
        K=4,
        max_tokens=24,
        base_seed=11,
    )
    assert len(cands) == 4
    taus = {c.sampler["tau"] for c in cands}
    assert taus == {PARITY_SAMPLER["tau"]}
    seeds = {c.seed for c in cands}
    assert len(seeds) == 4  # diversity comes from per-K seed


def test_iccha_empty_prompt_raises() -> None:
    import numpy as np
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    constraint = Constraint(text="x", embedding=q)

    class FakeLM:
        def generate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

    try:
        iccha("   ", constraint, lm=FakeLM(), K=2)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty prompt")


def test_iccha_verbatim_mode_returns_prompt_unchanged() -> None:
    """v0.2 ADR-005: verbatim prompt_mode does not append the constraint suffix."""
    import numpy as np

    from pce.operators.iccha import _build_prompt
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    constraint = Constraint(text="a haiku about autumn leaves", embedding=q)
    out = _build_prompt("Compose a poem.", constraint, prompt_mode="verbatim")
    assert out == "Compose a poem."


def test_iccha_constraint_suffix_mode_appends_constraint() -> None:
    """v0.1 legacy mode kept for backward compatibility."""
    import numpy as np

    from pce.operators.iccha import _build_prompt
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    constraint = Constraint(text="a haiku about autumn leaves", embedding=q)
    out = _build_prompt("Compose a poem.", constraint, prompt_mode="constraint_suffix")
    assert "Write a response that is a haiku about autumn leaves" in out


def test_iccha_grid_too_small_raises() -> None:
    import numpy as np
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    constraint = Constraint(text="x", embedding=q)

    class FakeLM:
        def generate(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("should not be called")

    try:
        iccha(
            "hello",
            constraint,
            lm=FakeLM(),  # type: ignore[arg-type]
            K=10,
            sampler_grid=({"tau": 1.0, "top_p": 0.95, "top_k": 50.0},),
            sampler_grid_mode="grid",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for grid < K")
