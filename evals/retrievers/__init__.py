"""Retrievers (scope v2 §7). Each has `retriever_id`, `params`, `needs_vector` (does it use the query
embedding?) and `search(index, query_vec, k, query=...) -> Hits`."""
from evals.retrievers.bm25 import BM25Retriever
from evals.retrievers.dense import DenseRetriever, Hits, SessionIndex
from evals.retrievers.hybrid import HybridRetriever

RETRIEVERS = {"R-dense": DenseRetriever, "R-bm25": BM25Retriever, "R-hybrid": HybridRetriever}

__all__ = ["RETRIEVERS", "Hits", "SessionIndex", "BM25Retriever", "DenseRetriever", "HybridRetriever"]
