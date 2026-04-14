"""
Retriever — loads FAISS index from S3, embeds query, returns top-k chunks.

In-memory cache: index + chunks are loaded once per session and cached.
Cache is cleared when a new document is uploaded for the same session.

Flow:
    query string
        → embed (MiniLM)
        → FAISS inner product search (cosine similarity)
        → top-k chunk IDs
        → return Chunk objects with scores
"""
from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass

import numpy as np
import faiss

from app.pipeline.chunker import Chunk
from app.pipeline.embedder import get_embedder
from app.storage.s3_client import get_s3_client, S3Error
from app.storage.keys import S3Keys
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 5
_SESSION_CACHE: dict[str, tuple[faiss.IndexFlatIP, list[Chunk]]] = {}


class RetrieverError(Exception):
    """Raised when retrieval fails."""


# ------------------------------------------------------------------
# Cache management
# ------------------------------------------------------------------

def _load_session(session_id: str) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """Load FAISS index + chunks from S3 into memory. Cached per session."""
    if session_id in _SESSION_CACHE:
        logger.debug("retriever_cache_hit", session_id=session_id)
        return _SESSION_CACHE[session_id]

    s3 = get_s3_client()

    # Load chunks
    try:
        chunks_json = s3.download_text(S3Keys.chunks(session_id))
    except S3Error as exc:
        raise RetrieverError(
            f"Chunks not found for session={session_id}. Run indexing first."
        ) from exc

    raw_chunks = json.loads(chunks_json)
    chunks = [Chunk(**c) for c in raw_chunks]
    logger.info("retriever_chunks_loaded", session_id=session_id, count=len(chunks))

    # Load FAISS index
    try:
        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
            tmp_path = tmp.name
        s3.download_to_file(S3Keys.faiss_index(session_id), tmp_path)
    except S3Error as exc:
        raise RetrieverError(
            f"FAISS index not found for session={session_id}. Run indexing first."
        ) from exc

    index = faiss.read_index(tmp_path)
    logger.info(
        "retriever_index_loaded",
        session_id=session_id,
        total_vectors=index.ntotal,
    )

    _SESSION_CACHE[session_id] = (index, chunks)
    return index, chunks


def evict_session(session_id: str) -> None:
    """Remove a session from the in-memory cache (call after re-indexing)."""
    if session_id in _SESSION_CACHE:
        del _SESSION_CACHE[session_id]
        logger.info("retriever_cache_evicted", session_id=session_id)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float          # cosine similarity 0–1
    rank: int             # 1 = most relevant


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def retrieve(
    session_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Embed query and return top-k most relevant chunks for the session.

    Args:
        session_id: document session to search against
        query:      natural language question
        top_k:      number of chunks to return

    Returns:
        list of RetrievedChunk sorted by score descending (rank 1 = best)
    """
    if not query.strip():
        raise RetrieverError("Query cannot be empty.")

    embedder = get_embedder()
    index, chunks = _load_session(session_id)

    # Embed query — shape (1, 384)
    query_vec = embedder.embed_one(query).reshape(1, -1)

    # Search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results: list[RetrievedChunk] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:   # FAISS returns -1 for padding
            continue
        results.append(RetrievedChunk(
            chunk=chunks[idx],
            score=float(score),
            rank=rank,
        ))

    logger.info(
        "retrieval_complete",
        session_id=session_id,
        query_preview=query[:60],
        top_k=k,
        top_score=round(results[0].score, 3) if results else 0,
    )

    return results
