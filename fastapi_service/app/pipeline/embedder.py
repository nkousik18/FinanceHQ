"""
Embedder — MiniLM-L6-v2 singleton for chunk and query embedding.

Single model instance shared across the process.
Used by both the indexer (chunk embedding) and retriever (query embedding).

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - Fast on CPU
  - Strong semantic similarity for retrieval tasks
  - Free, local, no API call
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from app.core.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self) -> None:
        logger.info("embedder_loading", model=MODEL_NAME)
        self._model = SentenceTransformer(MODEL_NAME)
        logger.info("embedder_ready", model=MODEL_NAME, dimensions=384)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.
        Returns float32 array of shape (len(texts), 384), L2-normalised.
        Normalised vectors → inner product == cosine similarity.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        logger.debug("embedding_texts", count=len(texts))
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalise → cosine via inner product
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (384,)."""
        return self.embed([text])[0]


@lru_cache()
def get_embedder() -> Embedder:
    """Singleton — loaded once, reused across all requests."""
    return Embedder()
