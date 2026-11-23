"""
Chunk/embedding cache (scope v2 §9.2): one entry per (document view, chunker, embedder).

Key = sha256 of {chunker_id, chunker_params, parser_cache_key, cleaning_version, embedder,
drop_pages, format}. A document view is a document with some pages dropped (R4: a simulated scan
replaces its source page, see views.py); the plain document has no dropped pages.

Entry on disk (data/cache/chunks/, gitignored):
  <key>.json.gz   doc_id, key fields, cleaned page text, chunks (Chunk.to_dict)
  <key>.npy       float32 chunk vectors (L2-normalized MiniLM, production embedder settings)

Page text is stored so a cached run scores coverage without re-parsing or re-cleaning.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from evals.chunking.base import Chunk, Chunker, SourceDoc, Span
from evals.chunking.source import CLEANING_VERSION
from evals.chunking.tokens import MINILM_NAME

REPO = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO / "data" / "cache" / "chunks"
CACHE_FORMAT = 1
EMBEDDER_ID = f"{MINILM_NAME}|normalized|float32"
log = logging.getLogger(__name__)

Embed = Callable[[list[str]], np.ndarray]
LoadSource = Callable[[str, frozenset[int]], tuple[SourceDoc, str]]   # -> (doc, parser_cache_key)


@dataclass
class DocIndex:
    """One document view's chunks and vectors (rows aligned)."""
    source: SourceDoc            # pages only (enough for coverage)
    chunks: list[Chunk]
    vectors: np.ndarray
    key: str
    from_cache: bool = False
    seconds: float = 0.0         # chunk + embed time when built


def cache_key(chunker: Chunker, parser_cache_key: str, drop_pages: frozenset[int] = frozenset(),
              cleaning_version: str = CLEANING_VERSION, embedder: str = EMBEDDER_ID) -> str:
    payload = {"chunker_id": chunker.chunker_id, "chunker_params": chunker.params,
               "parser_cache_key": parser_cache_key, "cleaning_version": cleaning_version,
               "embedder": embedder, "drop_pages": sorted(drop_pages), "format": CACHE_FORMAT}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(d["chunk_id"], d["doc_id"], d["index"], d["text"],
                 [Span(**s) for s in d["spans"]], d.get("section", ""), d.get("meta", {}))


def _write(root: Path, key: str, doc: SourceDoc, chunks: list[Chunk], vecs: np.ndarray, meta: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    body = {"doc_id": doc.doc_id, **meta, "pages": {str(p): t for p, t in doc.pages.items()},
            "chunks": [c.to_dict() for c in chunks]}
    tmp = root / f"{key}.json.gz.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(body, f)
    np.save(root / f"{key}.npy", vecs)
    tmp.replace(root / f"{key}.json.gz")          # json last: an entry exists only when complete


def _read(root: Path, key: str) -> tuple[SourceDoc, list[Chunk], np.ndarray] | None:
    jpath, vpath = root / f"{key}.json.gz", root / f"{key}.npy"
    if not (jpath.exists() and vpath.exists()):
        return None
    with gzip.open(jpath, "rt", encoding="utf-8") as f:
        body = json.load(f)
    chunks = [_chunk_from_dict(d) for d in body["chunks"]]
    vecs = np.load(vpath)
    if len(vecs) != len(chunks):
        log.warning("cache entry %s: %d vectors for %d chunks; rebuilding", key[:12], len(vecs), len(chunks))
        return None
    doc = SourceDoc(body["doc_id"], {int(p): t for p, t in body["pages"].items()})
    return doc, chunks, vecs


class IndexCache:
    """Builds or loads DocIndex entries. `load_source(doc_id, drop_pages)` gives the SourceDoc and its
    parser cache key; it is only called when the parser key isn't known yet or an entry is missing."""

    def __init__(self, load_source: LoadSource, embed: Embed, parser_key: Callable[[str], str],
                 root: Path = CACHE_ROOT, embedder_id: str = EMBEDDER_ID):
        self.load_source, self.embed, self.parser_key = load_source, embed, parser_key
        self.root, self.embedder_id = root, embedder_id
        self._mem: dict[str, DocIndex] = {}
        self.built = self.loaded = 0

    def get(self, chunker: Chunker, doc_id: str, drop_pages: frozenset[int] = frozenset()) -> DocIndex:
        key = cache_key(chunker, self.parser_key(doc_id), drop_pages, embedder=self.embedder_id)
        if key in self._mem:
            return self._mem[key]
        hit = _read(self.root, key)
        if hit is not None:
            doc, chunks, vecs = hit
            di = DocIndex(doc, chunks, vecs, key, from_cache=True)
            self.loaded += 1
        else:
            t0 = time.perf_counter()
            src, pkey = self.load_source(doc_id, drop_pages)
            chunks = chunker.chunk(src)
            vecs = self.embed([c.text for c in chunks]) if chunks else np.zeros((0, 384), np.float32)
            secs = time.perf_counter() - t0
            _write(self.root, key, src, chunks, vecs,
                   {"chunker_id": chunker.chunker_id, "chunker_params": chunker.params,
                    "parser_cache_key": pkey, "cleaning_version": CLEANING_VERSION,
                    "embedder": self.embedder_id, "drop_pages": sorted(drop_pages)})
            di = DocIndex(SourceDoc(src.doc_id, dict(src.pages)), chunks, vecs, key, seconds=secs)
            self.built += 1
            log.info("indexed %s %s drop=%s: %d chunks in %.1fs", chunker.chunker_id, doc_id,
                     sorted(drop_pages), len(chunks), secs)
        self._mem[key] = di
        return di
