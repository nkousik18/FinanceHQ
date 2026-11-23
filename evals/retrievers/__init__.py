"""Retrievers (scope v2 §7). Each has `retriever_id`, `params`, `needs_vector` (does it use the query
embedding?), optional `needs_parents` (R-pc: the run's `parent_chunker` sections) and `search(index, query_vec, k, query=...) -> Hits`."""
from evals.retrievers.bm25 import BM25Retriever
from evals.retrievers.dense import DenseRetriever, Hits, SessionIndex
from evals.retrievers.hybrid import HybridRetriever
from evals.retrievers.parent_child import ParentChildRetriever

RETRIEVERS = {"R-dense": DenseRetriever, "R-bm25": BM25Retriever, "R-hybrid": HybridRetriever,
              "R-pc": ParentChildRetriever}

__all__ = ["RETRIEVERS", "Hits", "SessionIndex", "BM25Retriever", "DenseRetriever", "HybridRetriever", "ParentChildRetriever"]
