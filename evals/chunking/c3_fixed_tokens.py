"""
C3 — fixed-size MiniLM token windows (scope v2 §6.1): 256 tokens, 32 overlap, no sentence awareness.

"256 tokens" = the embedder's whole input, [CLS] and [SEP] included (Decision Log 2026-11-12),
so a window holds 254 content tokens and is never truncated. Pages are one stream
(`evals.chunking.stream`); windows run across page breaks.

Windows never start or end inside a word (a WordPiece "##" continuation), so re-tokenizing a
chunk's text gives the same tokens. Word = the BERT pre-tokenizer's word (split on whitespace
and punctuation), so this is a token boundary rule, not a sentence rule.
"""
from __future__ import annotations

import logging

from evals.chunking.base import Chunk, SourceDoc
from evals.chunking.stream import build_stream
from evals.chunking.tokens import MAX_SEQ_LENGTH, SPECIAL_TOKENS, minilm_offsets

log = logging.getLogger(__name__)


def _mid_word(word_ids: list[int | None], k: int) -> bool:
    """True if token k continues the word of token k-1 (a boundary at k would split a word)."""
    return 0 < k < len(word_ids) and word_ids[k] is not None and word_ids[k] == word_ids[k - 1]


def token_windows(word_ids: list[int | None], lo: int, hi: int, size: int,
                  overlap: int) -> list[tuple[int, int]]:
    """
    Windows [i, j) over tokens lo..hi of at most `size` tokens, overlapping by about `overlap`
    tokens (snapped back to a word start, so >= overlap unless that would stall).
    """
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    out: list[tuple[int, int]] = []
    i = lo
    while i < hi:
        j = min(i + size, hi)
        if j < hi:
            k = j
            while k > i + 1 and _mid_word(word_ids, k):
                k -= 1
            j = k if not _mid_word(word_ids, k) else j     # a single word longer than size: cut it
        out.append((i, j))
        if j >= hi:
            break
        nxt = max(j - overlap, lo)
        while nxt > lo and _mid_word(word_ids, nxt):
            nxt -= 1
        i = nxt if nxt > i else j
    return out


class C3FixedTokens:
    chunker_id = "C3"

    def __init__(self, size: int = MAX_SEQ_LENGTH, overlap: int = 32):
        self.params = {"size": size, "overlap": overlap, "special_tokens": SPECIAL_TOKENS,
                       "content_tokens": size - SPECIAL_TOKENS, "tokenizer": "all-MiniLM-L6-v2"}

    def chunk(self, doc: SourceDoc) -> list[Chunk]:
        if doc.form_fields:
            log.warning("%s: C3 chunks page text only; %d form fields ignored", doc.doc_id,
                        len(doc.form_fields))
        stream = build_stream(doc)
        offsets, word_ids = minilm_offsets(stream.text)
        chunks: list[Chunk] = []
        for i, j in token_windows(word_ids, 0, len(offsets), self.params["content_tokens"],
                                  self.params["overlap"]):
            a, b = offsets[i][0], offsets[j - 1][1]
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-C3-{idx}", doc_id=doc.doc_id, index=idx,
                text=stream.text[a:b], spans=stream.spans(a, b),
                meta={"minilm_tokens": j - i, "token_range": [i, j], "stream_range": [a, b]},
            ))
        return chunks
