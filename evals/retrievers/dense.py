"""
R-dense (scope v2 §7): production's retrieval — FAISS IndexFlatIP over L2-normalized MiniLM
vectors, query embedded the same way, top k by inner product (= cosine). No score threshold
(production has none). Mirrors fastapi_service/app/retrieval/retriever.py `retrieve`.

A SessionIndex is the concatenation of a question view's document indexes (one FAISS index per
session, like production).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from evals.chunking.base import Chunk


def _faiss():
    """faiss, imported after torch. On macOS faiss and torch each ship an OpenMP runtime; if faiss
    loads first, the next MiniLM encode segfaults (checked 2026-11-22). Torch first is safe."""
    import torch  # noqa: F401
    import faiss
    return faiss


@dataclass
class SessionIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    faiss_index: object              # faiss.Index
    extras: dict = field(default_factory=dict)   # per-session side indexes (BM25), built lazily

    @classmethod
    def build(cls, parts: list[tuple[list[Chunk], np.ndarray]]) -> "SessionIndex":
        chunks = [c for cs, _ in parts for c in cs]
        vecs = [v for _, v in parts if len(v)]
        x = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 384), np.float32)
        if len(x) != len(chunks):
            raise ValueError(f"{len(x)} vectors for {len(chunks)} chunks")
        index = _faiss().IndexFlatIP(x.shape[1])
        index.add(x)
        return cls(chunks, x, index)


@dataclass
class Hits:
    ids: list[int]           # positions in SessionIndex.chunks, rank order
    scores: list[float]
    timings: dict[str, float]  # seconds per component


class DenseRetriever:
    retriever_id = "R-dense"
    params: dict = {"index": "IndexFlatIP", "metric": "inner_product", "normalized": True}
    needs_vector = True

    def search(self, index: SessionIndex, query_vec: np.ndarray, k: int, query: str = "") -> Hits:
        k = min(k, index.faiss_index.ntotal)
        if k == 0:
            return Hits([], [], {"vector_search": 0.0})
        t0 = time.perf_counter()
        scores, ids = index.faiss_index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        dt = time.perf_counter() - t0
        keep = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        return Hits([i for i, _ in keep], [s for _, s in keep], {"vector_search": dt})

