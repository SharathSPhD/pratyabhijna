"""Phase 3 smoke imports.

Verifies the project's runtime dependencies actually resolve and import. This
is the bare minimum for Phase 3's `pyproject resolved` claim - if any of these
fail at the gate, `verify_artifact.py` flags Phase 3 as red.
"""
from __future__ import annotations

import importlib

REQUIRED_MODULES = (
    "pce",
    "numpy",
    "scipy",
    "sentence_transformers",
    "transformers",
    "torch",
    "huggingface_hub",
    "pymdp",
    "mcp",
    "pydantic",
    "statsmodels",
    "pandas",
)


def test_required_modules_import() -> None:
    """Every dependency declared in pyproject.toml must import cleanly."""
    failures: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised via assertion
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "module imports failed:\n  " + "\n  ".join(failures)


def test_pce_has_version() -> None:
    import pce

    assert hasattr(pce, "__version__")
    assert isinstance(pce.__version__, str)
    assert pce.__version__.count(".") == 2


def test_numpy_scipy_sane() -> None:
    """Cheap numerical sanity check that the toolchain isn't broken."""
    import numpy as np
    from scipy.special import gammaln

    x = np.array([1.0, 2.0, 3.0])
    res = gammaln(x)
    assert res.shape == (3,)
    assert np.isfinite(res).all()
