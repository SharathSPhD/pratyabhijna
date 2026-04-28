"""Shared pytest fixtures for PCE.

Phase 3 ships the file so the import graph and the discovery path are anchored;
Phase 5 expands with substrate fixtures (real Phi-3-mini-4k weights and real
sentence-transformers MiniLM-L6-v2 embeddings).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture(scope="session", autouse=True)
def _deterministic_random_seed() -> None:
    """Lock random seeds so tests are reproducible session-wide.

    PCE's own operator tests pass explicit seeds, but third-party libraries
    (sentence-transformers, transformers) read from the global random state.
    """
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
