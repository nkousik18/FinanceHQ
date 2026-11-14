"""
C5 — semantic chunker (scope v2 §6.1): split where the embedding similarity between adjacent
sentences drops, i.e. where their cosine distance is above the document's 95th-percentile
adjacent distance. Same token bounds as C2.

Rules (Decision Log 2026-11-13):
- Units = C2's units: syntok sentences per line of the page stream (line and page breaks are
  sentence boundaries), a sentence over `max_tokens` cut into C3-style word-snapped pieces.
- Each unit is embedded alone with MiniLM (the production embedder, normalized). distance(k) =
  1 - cos(unit k-1, unit k). threshold = np.percentile(distances, 95) over the document;
  boundary at k iff distance(k) > threshold.
- Max (C2's 200 content tokens): a segment over the max is split at its largest internal
  distance, recursively, until every piece fits (a unit always fits).
- Min (50 tokens; C2 has no min, so C4's): a segment under the min is merged with the neighbour
  across its lower-distance boundary (the more similar one), else the other, if the sum fits the
  max; merges are done lowest-distance first. A segment that can't merge stays small.
- No overlap: chunks tile the stream (boundaries are meant to be topic shifts).

Chunk text is a slice of the page stream, so spans are exact (same as C2/C3), and the token
count comes from one tokenization of the stream (`meta.minilm_tokens`).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from evals.chunking.base import Chunk, SourceDoc
from evals.chunking.c2_sentence_window import _units
from evals.chunking.stream import build_stream
from evals.chunking.tokens import MINILM_NAME, minilm_offsets

log = logging.getLogger(__name__)

Embedder = Callable[[list[str]], np.ndarray]     # texts -> (n, d) L2-normalized rows


@lru_cache(maxsize=1)
def _model(device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MINILM_NAME, device=device)


def minilm_embedder(device: str = "cpu", batch_size: int = 64) -> Embedder:
    def embed(texts: list[str]) -> np.ndarray:
        return _model(device).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
    return embed


def adjacent_distances(emb: np.ndarray) -> np.ndarray:
    """distance[k-1] = 1 - cos(emb[k-1], emb[k]) for k = 1..n-1 (rows already normalized)."""
    if len(emb) < 2:
        return np.zeros(0)
    return 1.0 - np.einsum("ij,ij->i", emb[:-1], emb[1:])


def _split_over_max(lo: int, hi: int, tok: np.ndarray, dist: np.ndarray, max_tokens: int) -> list[int]:
    """Cut points (unit indices) inside units [lo, hi) so every piece is <= max_tokens.
    tok[k] = token start of unit k (tok[n] = end); dist[k-1] = distance before unit k."""
    if tok[hi] - tok[lo] <= max_tokens or hi - lo < 2:
        return []
    k = lo + 1 + int(np.argmax(dist[lo:hi - 1]))           # largest distance among cuts lo+1..hi-1
    return _split_over_max(lo, k, tok, dist, max_tokens) + [k] + _split_over_max(k, hi, tok, dist, max_tokens)


def _merge_small(cuts: list[int], n: int, tok: np.ndarray, dist: np.ndarray,
                 min_tokens: int, max_tokens: int) -> list[int]:
    """Drop cuts to merge segments under min_tokens into a neighbour, lowest distance first."""
    cuts = list(cuts)
    while True:
        edges = [0] + cuts + [n]
        size = [tok[edges[i + 1]] - tok[edges[i]] for i in range(len(edges) - 1)]
        best = None
        for c_i, c in enumerate(cuts):                      # cut c joins segments c_i and c_i+1
            left, right = size[c_i], size[c_i + 1]
            if min(left, right) < min_tokens and left + right <= max_tokens:
                if best is None or dist[c - 1] < dist[cuts[best] - 1]:
                    best = c_i
        if best is None:
            return cuts
        cuts.pop(best)


def semantic_cuts(dist: np.ndarray, tok: np.ndarray, percentile: float,
                  min_tokens: int, max_tokens: int) -> tuple[list[int], float | None]:
    """Unit indices where chunks start (excluding 0), and the document threshold."""
    n = len(tok) - 1
    if n < 2:
        return [], None
    thr = float(np.percentile(dist, percentile))
    cuts = [k for k in range(1, n) if dist[k - 1] > thr]
    edges = [0] + cuts + [n]
    full: list[int] = []
    for a, b in zip(edges, edges[1:]):
        full += _split_over_max(a, b, tok, dist, max_tokens) + ([b] if b < n else [])
    return _merge_small(full, n, tok, dist, min_tokens, max_tokens), thr


class C5Semantic:
    chunker_id = "C5"

    def __init__(self, percentile: float = 95.0, max_tokens: int = 200, min_tokens: int = 50,
                 embedder: Embedder | None = None, device: str = "cpu"):
        if not 0 < min_tokens < max_tokens:
            raise ValueError("need 0 < min_tokens < max_tokens")
        self.params = {"percentile": percentile, "max_tokens": max_tokens, "min_tokens": min_tokens,
                       "overlap": 0, "sentence_embedder": "all-MiniLM-L6-v2 (normalized, per unit)",
                       "splitter": "syntok 1.4.4", "tokenizer": "all-MiniLM-L6-v2"}
        self._embed = embedder or minilm_embedder(device)

    def chunk(self, doc: SourceDoc) -> list[Chunk]:
        if doc.form_fields:
            log.warning("%s: C5 chunks page text only; %d form fields ignored", doc.doc_id,
                        len(doc.form_fields))
        p = self.params
        stream = build_stream(doc)
        offsets, word_ids = minilm_offsets(stream.text)
        units = _units(stream.text, offsets, word_ids, p["max_tokens"], 0)
        if not units:
            return []
        # units tile tokens only up to whitespace/blank tokens between them; segment k..m spans
        # tokens [units[k].ti, units[m-1].tj), so tok[] holds unit starts plus the last end.
        tok = np.array([u.ti for u in units] + [units[-1].tj])
        dist = adjacent_distances(self._embed([stream.text[u.a:u.b] for u in units]))
        cuts, thr = semantic_cuts(dist, tok, p["percentile"], p["min_tokens"], p["max_tokens"])
        thr_cuts = {k for k in range(1, len(units)) if thr is not None and dist[k - 1] > thr}
        chunks: list[Chunk] = []
        edges = [0] + cuts + [len(units)]
        for a, b in zip(edges, edges[1:]):
            first, last = units[a], units[b - 1]
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-C5-{idx}", doc_id=doc.doc_id, index=idx,
                text=stream.text[first.a:last.b], spans=stream.spans(first.a, last.b),
                meta={"minilm_tokens": int(last.tj - first.ti), "stream_range": [first.a, last.b],
                      "sentences": b - a, "split_sentence": any(u.split for u in units[a:b]),
                      "boundary": ("start" if a == 0 else "semantic" if a in thr_cuts else "size"),
                      "boundary_distance": None if a == 0 else round(float(dist[a - 1]), 4),
                      "doc_threshold": None if thr is None else round(thr, 4)},
            ))
        return chunks
