"""
R-bm25 (scope v2 §7): in-process BM25 over the same chunks as R-dense. A lower bound and a diagnostic.

Library: bm25s (pip, numpy only, in-process; D10, Decision Log 2026-11-22). Default parameters
(Lucene BM25, k1 = 1.5, b = 0.75).

Tokenizer (D10): lowercase; tokens are words ([a-z]+) and numbers; English stopwords dropped (bm25s
list); **no stemming** — exact identifiers (rates, amounts, section numbers, defined terms) are what
BM25 is here for (H2). Numbers are normalized so the same value matches however it is written:
thousands separators dropped ("62,500" -> "62500"), trailing decimal zeros dropped ("62500.00" ->
"62500", "3.50" -> "3.5"); "$" and "%" are not tokens. Section numbers ("2.11") stay one token.

Only chunks sharing at least one token with the query are returned (score > 0): a zero-score chunk
has no BM25 evidence, so its rank would be arbitrary. A query can therefore return fewer than k.
"""
from __future__ import annotations

import re
import time

import numpy as np

from evals.retrievers.dense import Hits, SessionIndex

_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _stopwords() -> frozenset[str]:
    from bm25s.stopwords import STOPWORDS_EN
    return frozenset(STOPWORDS_EN)


STOPWORDS = _stopwords()


def normalize_number(tok: str) -> str:
    if _THOUSANDS.match(tok):
        tok = tok.replace(",", "")
    if re.fullmatch(r"\d+\.\d+", tok):
        tok = tok.rstrip("0").rstrip(".")
    return tok


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t[0].isdigit():
            # a comma that isn't a thousands separator splits ("1,2" -> "1", "2")
            parts = [t] if "," not in t or _THOUSANDS.match(t) else t.split(",")
            out.extend(normalize_number(p) for p in parts if p)
        elif t not in STOPWORDS:
            out.append(t)
    return out


def bm25_index(index: SessionIndex):
    """The session's BM25 index, built once and kept on the SessionIndex (offline, like FAISS)."""
    if "bm25" not in index.extras:
        import bm25s
        model = bm25s.BM25()
        corpus = [tokenize(c.text) for c in index.chunks]
        if index.chunks:
            model.index(corpus, show_progress=False)
        index.extras["bm25"] = model
    return index.extras["bm25"]


def bm25_rank(index: SessionIndex, query: str, k: int) -> tuple[list[int], list[float]]:
    """Top k chunk positions with score > 0, best first; ties -> earlier chunk."""
    if not index.chunks:
        return [], []
    model = bm25_index(index)
    q = tokenize(query)
    scores = np.asarray(model.get_scores(q), dtype=np.float64) if q else np.zeros(len(index.chunks))
    order = np.lexsort((np.arange(len(scores)), -scores))      # score desc, then position asc
    keep = [int(i) for i in order[:k] if scores[i] > 0]
    return keep, [float(scores[i]) for i in keep]


class BM25Retriever:
    retriever_id = "R-bm25"
    params: dict = {"library": "bm25s", "method": "lucene", "k1": 1.5, "b": 0.75,
                    "tokenizer": "lower+numnorm+stopwords_en,no_stem", "min_score": ">0"}
    needs_vector = False

    def search(self, index: SessionIndex, query_vec, k: int, query: str = "") -> Hits:
        bm25_index(index)                       # build outside the timer
        t0 = time.perf_counter()
        ids, scores = bm25_rank(index, query, k)
        return Hits(ids, scores, {"bm25": time.perf_counter() - t0})
