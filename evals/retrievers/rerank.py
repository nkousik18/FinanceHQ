"""
+rerank (scope v2 §7): cross-encoder reranking on top of any retriever.

Take the base retriever's top 20 candidates (max(20, k) if a run asks for more), score each
(question, chunk text) pair with `cross-encoder/ms-marco-MiniLM-L-6-v2`, keep the top k by that
score. Ties -> the base retriever's order. CPU, like production's embedder (the rollout target is a
small CPU host); batch = all candidates of one question.

The cross-encoder reads at most 512 tokens per pair ([CLS] question [SEP] chunk [SEP]); longer pairs
are cut from the chunk's end (sentence-transformers `longest_first`). Per question the number of
cut candidates is reported (`Hits.info["rerank_truncated"]`) — mostly R-pc parents (~800 tokens) and
C1 chunks.
"""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Callable

import numpy as np

from evals.retrievers.dense import Hits, SessionIndex

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 20
MAX_LENGTH = 512

Scorer = Callable[[str, list[str]], np.ndarray]
PairLength = Callable[[str, str], int]


@lru_cache(maxsize=1)
def _model():
    import torch  # noqa: F401  (torch before faiss, see dense._faiss)
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=MAX_LENGTH, device="cpu")


def cross_encoder_scores(query: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros(0)
    return np.asarray(_model().predict([(query, t) for t in texts], batch_size=len(texts),
                                       show_progress_bar=False), dtype=np.float64)


def pair_tokens(query: str, text: str) -> int:
    """Tokens of the (query, text) pair with special tokens, untruncated."""
    return len(_model().tokenizer(query, text, truncation=False)["input_ids"])


class Reranked:
    """Wraps a base retriever; `+rerank` in run names."""
    params_extra = {"model": RERANK_MODEL, "depth": RERANK_DEPTH, "max_length": MAX_LENGTH, "device": "cpu"}

    def __init__(self, base, scorer: Scorer | None = None, pair_len: PairLength | None = None):
        self.base = base
        self.scorer = scorer or (lambda q, t: cross_encoder_scores(q, t))     # looked up at call time
        self.pair_len = pair_len or (lambda q, t: pair_tokens(q, t))
        self.retriever_id = f"{base.retriever_id}+rerank"
        self.params = {**base.params, "rerank": self.params_extra}
        self.needs_vector = base.needs_vector
        self.needs_parents = getattr(base, "needs_parents", False)

    def search(self, index: SessionIndex, query_vec, k: int, query: str = "") -> Hits:
        hits = self.base.search(index, query_vec, max(RERANK_DEPTH, k), query=query)
        cands = hits.ranked(index)
        t0 = time.perf_counter()
        scores = self.scorer(query, [c.text for c in cands])
        t_rerank = time.perf_counter() - t0
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))[:k]
        truncated = sum(self.pair_len(query, c.text) > MAX_LENGTH for c in cands)
        info = {**hits.info, "rerank_candidates": len(cands), "rerank_truncated": truncated}
        timings = {**hits.timings, "rerank": t_rerank}
        out_scores = [float(scores[i]) for i in order]
        if hits.chunks is not None:
            return Hits([], out_scores, timings, chunks=[cands[i] for i in order], info=info)
        return Hits([hits.ids[i] for i in order], out_scores, timings, info=info)
