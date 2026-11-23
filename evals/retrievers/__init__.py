"""Retrievers (scope v2 §7). Each has `retriever_id`, `params`, `needs_vector` (does it use the query
embedding?), optional `needs_parents` (R-pc: the run's `parent_chunker` sections) optional `prepare(index)` (index-time work, called when the session index is built) and
`search(index, query_vec, k, query=...) -> Hits`.
`+rerank` wraps any of them (`make_retriever(name, rerank=True)`)."""
from evals.retrievers.bm25 import BM25Retriever
from evals.retrievers.dense import DenseRetriever, Hits, SessionIndex
from evals.retrievers.hybrid import HybridRetriever
from evals.retrievers.parent_child import ParentChildRetriever

from evals.retrievers.rerank import Reranked

RETRIEVERS = {"R-dense": DenseRetriever, "R-bm25": BM25Retriever, "R-hybrid": HybridRetriever,
              "R-pc": ParentChildRetriever}



def make_retriever(name: str, rerank: bool = False):
    base = RETRIEVERS[name]()
    return Reranked(base) if rerank else base


__all__ = ["RETRIEVERS", "make_retriever", "Reranked", "Hits", "SessionIndex", "BM25Retriever", "DenseRetriever", "HybridRetriever", "ParentChildRetriever"]
