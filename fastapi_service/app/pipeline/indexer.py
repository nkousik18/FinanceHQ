"""
Indexer — embeds chunks and builds a FAISS index, persists everything to S3.

FAISS index type: IndexFlatIP (inner product)
  - Works as cosine similarity because embeddings are L2-normalised
  - Exact search (no approximation) — fine for document-level chunks (< 1000)
  - Fast enough on CPU for this use case

S3 artifacts written:
  - chunks/{session_id}/chunks.json      ← chunk texts + metadata
  - chunks/{session_id}/embeddings.npy   ← raw float32 embeddings
  - chunks/{session_id}/faiss.index      ← FAISS binary index
"""
from __future__ import annotations

import io
import json
import os
import tempfile

import numpy as np
import faiss

from app.pipeline.chunker import Chunk
from app.pipeline.embedder import get_embedder
from app.storage.s3_client import get_s3_client
from app.storage.keys import S3Keys
from app.core.logging import get_logger

logger = get_logger(__name__)


class IndexingError(Exception):
    """Raised when FAISS index build or S3 persistence fails."""


def build_and_save_index(session_id: str, chunks: list[Chunk]) -> int:
    """
    Embed chunks, build FAISS index, save all artifacts to S3.
    Returns the number of chunks indexed.

    Raises IndexingError on failure.
    """
    if not chunks:
        raise IndexingError(f"No chunks to index for session={session_id}")

    embedder = get_embedder()
    s3 = get_s3_client()

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------
    texts = [c.text for c in chunks]
    logger.info("indexing_embed_start", session_id=session_id, chunk_count=len(texts))
    embeddings = embedder.embed(texts)   # (N, 384) float32 normalised
    logger.info("indexing_embed_done", session_id=session_id, shape=list(embeddings.shape))

    # ------------------------------------------------------------------
    # Build FAISS index
    # ------------------------------------------------------------------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("faiss_index_built", session_id=session_id, total_vectors=index.ntotal)

    # ------------------------------------------------------------------
    # Persist chunks.json
    # ------------------------------------------------------------------
    chunks_payload = json.dumps([c.to_dict() for c in chunks], ensure_ascii=False)
    s3.upload_json(S3Keys.chunks(session_id), chunks_payload)
    logger.info("chunks_saved", session_id=session_id, key=S3Keys.chunks(session_id))

    # ------------------------------------------------------------------
    # Persist embeddings.npy
    # ------------------------------------------------------------------
    npy_buffer = io.BytesIO()
    np.save(npy_buffer, embeddings)
    s3.upload_bytes(S3Keys.embeddings(session_id), npy_buffer.getvalue())
    logger.info("embeddings_saved", session_id=session_id, key=S3Keys.embeddings(session_id))

    # ------------------------------------------------------------------
    # Persist FAISS index
    # ------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        tmp_path = tmp.name

    faiss.write_index(index, tmp_path)
    s3.upload_file(S3Keys.faiss_index(session_id), tmp_path)
    logger.info("faiss_index_saved", session_id=session_id, key=S3Keys.faiss_index(session_id))

    return index.ntotal


def merge_into_session_index(session_id: str, new_chunks: list[Chunk]) -> int:
    """
    Embed new_chunks and merge them into the session-level FAISS index.

    If a session index already exists in S3, the new chunks are appended to
    the existing chunks + embeddings and the index is rebuilt from scratch.
    If no index exists yet, this behaves identically to build_and_save_index.

    Returns total number of chunks in the merged index.

    NOTE: Not safe for concurrent writes to the same session_id.
    Sequential uploads only — add a distributed lock (Redis etc.) for concurrency.
    """
    if not new_chunks:
        raise IndexingError(f"No chunks to merge for session={session_id}")

    embedder = get_embedder()
    s3 = get_s3_client()

    # ------------------------------------------------------------------
    # Embed new chunks
    # ------------------------------------------------------------------
    new_texts = [c.text for c in new_chunks]
    logger.info("merge_embed_start", session_id=session_id, new_chunks=len(new_texts))
    new_embeddings = embedder.embed(new_texts)   # (N, 384) float32 normalised

    # ------------------------------------------------------------------
    # Load existing session artifacts (if any)
    # ------------------------------------------------------------------
    existing_chunks: list[Chunk] = []
    existing_embeddings: np.ndarray | None = None

    if s3.exists(S3Keys.chunks(session_id)):
        raw = s3.download_text(S3Keys.chunks(session_id))
        existing_chunks = [Chunk(**d) for d in json.loads(raw)]
        logger.info("merge_loaded_existing", session_id=session_id, existing=len(existing_chunks))

    if s3.exists(S3Keys.embeddings(session_id)):
        npy_bytes = s3.download_bytes(S3Keys.embeddings(session_id))
        existing_embeddings = np.load(io.BytesIO(npy_bytes))

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    all_chunks = existing_chunks + new_chunks

    if existing_embeddings is not None:
        all_embeddings = np.vstack([existing_embeddings, new_embeddings]).astype(np.float32)
    else:
        all_embeddings = new_embeddings

    # ------------------------------------------------------------------
    # Rebuild FAISS index from merged embeddings
    # ------------------------------------------------------------------
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings)
    logger.info(
        "merge_index_rebuilt",
        session_id=session_id,
        total_vectors=index.ntotal,
        new=len(new_chunks),
        existing=len(existing_chunks),
    )

    # ------------------------------------------------------------------
    # Persist merged artifacts (overwrite session-level keys)
    # ------------------------------------------------------------------
    chunks_payload = json.dumps([c.to_dict() for c in all_chunks], ensure_ascii=False)
    s3.upload_json(S3Keys.chunks(session_id), chunks_payload)

    npy_buffer = io.BytesIO()
    np.save(npy_buffer, all_embeddings)
    s3.upload_bytes(S3Keys.embeddings(session_id), npy_buffer.getvalue())

    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        tmp_path = tmp.name
    faiss.write_index(index, tmp_path)
    s3.upload_file(S3Keys.faiss_index(session_id), tmp_path)

    logger.info("merge_complete", session_id=session_id, total_chunks=index.ntotal)
    return index.ntotal
