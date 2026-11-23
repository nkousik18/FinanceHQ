"""
R-hybrid (scope v2 §7): dense top-20 plus BM25 top-20, merged with reciprocal rank fusion (RRF, k = 60).

RRF score of a chunk = sum over the lists it appears in of 1 / (60 + rank), rank starting at 1.
Raw dense and BM25 scores are on different scales, so only ranks are used. Ties (same fused score)
-> better best-rank, then dense before BM25 order, then earlier chunk. Each list is max(20, k) deep
so a run asking for more than 20 chunks still gets them (S1/S2 use top_k 20: exactly the §7 setting).
"""
from __future__ import annotations

import time

from evals.retrievers.bm25 import bm25_index, bm25_rank
from evals.retrievers.dense import DenseRetriever, Hits, SessionIndex

RRF_K = 60
LIST_DEPTH = 20


def rrf(lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked id lists (best first). Returns (id, score), best first."""
    score: dict[int, float] = {}
    best: dict[int, tuple[int, int]] = {}         # id -> (best rank, list index of that rank)
    for li, ids in enumerate(lists):
        for rank, i in enumerate(ids, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
            if i not in best or (rank, li) < best[i]:
                best[i] = (rank, li)
    return sorted(score.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


class HybridRetriever:
    retriever_id = "R-hybrid"
    params: dict = {"fusion": "rrf", "rrf_k": RRF_K, "list_depth": LIST_DEPTH,
                    "dense": DenseRetriever.params, "bm25": "R-bm25"}
    needs_vector = True

    def __init__(self):
        self.dense = DenseRetriever()

    def prepare(self, index: SessionIndex) -> None:
        bm25_index(index)

    def search(self, index: SessionIndex, query_vec, k: int, query: str = "") -> Hits:
        depth = max(LIST_DEPTH, k)
        dense = self.dense.search(index, query_vec, depth)
        bm25_index(index)                       # no-op after prepare()
        t0 = time.perf_counter()
        bm_ids, _ = bm25_rank(index, query, depth)
        t1 = time.perf_counter()
        fused = rrf([dense.ids, bm_ids])[:k]
        t2 = time.perf_counter()
        return Hits([i for i, _ in fused], [s for _, s in fused],
                    {**dense.timings, "bm25": t1 - t0, "fusion": t2 - t1}, info=dict(dense.info))
