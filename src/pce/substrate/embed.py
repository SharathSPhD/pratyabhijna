"""Embedding substrate: thin deterministic wrapper around sentence-transformers.

The embedding model is loaded once per process (`functools.lru_cache`) so that
operators can call `Embedder.encode(...)` without paying the load cost.
"""
from __future__ import annotations

from functools import lru_cache
from typing import overload

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_id: str) -> SentenceTransformer:
    model: SentenceTransformer = SentenceTransformer(model_id)
    return model


class Embedder:
    """Deterministic encoder. Embeddings are L2-normalized."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self._model = _load_sentence_transformer(model_id)
        # The transformer's preferred accessor was renamed to
        # `get_embedding_dimension` in recent versions; we accept either.
        getter = getattr(
            self._model, "get_embedding_dimension", None
        ) or getattr(self._model, "get_sentence_embedding_dimension", None)
        self.dim: int = int(getter() or 0) if getter is not None else 0

    @overload
    def encode(self, text: str) -> npt.NDArray[np.float32]: ...
    @overload
    def encode(self, text: list[str]) -> npt.NDArray[np.float32]: ...

    def encode(
        self, text: str | list[str]
    ) -> npt.NDArray[np.float32]:
        if isinstance(text, str):
            items: list[str] = [text]
            single = True
        else:
            items = list(text)
            single = False
        if not items or any(not isinstance(t, str) for t in items):
            raise ValueError("Embedder.encode requires a string or list[str]")
        out = self._model.encode(items, normalize_embeddings=True, convert_to_numpy=True)
        arr: npt.NDArray[np.float32] = np.asarray(out, dtype=np.float32)
        if single:
            row: npt.NDArray[np.float32] = arr[0]
            return row
        return arr

    def cosine(
        self,
        a: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
    ) -> float:
        """Cosine on already-L2-normalized vectors collapses to a dot product."""
        if a.shape != b.shape:
            raise ValueError(f"Embedder.cosine shape mismatch: {a.shape} vs {b.shape}")
        return float(np.dot(a, b))
