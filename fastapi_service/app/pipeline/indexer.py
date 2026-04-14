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
