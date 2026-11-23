"""
C2 — sentence window, small (scope v2 §6.1): whole sentences packed up to ~200 MiniLM tokens,
~40 tokens of overlap. Fits inside MiniLM's 254 content tokens, so no chunk is truncated.

Rules (Decision Log 2026-11-12):
- Sentences: syntok, run per line of the page text. A line break is always a sentence boundary
  (Docling writes each heading, paragraph, list item and table row as its own line), and page
  breaks too. Sentences tile each line, so no text is dropped.
- Packing: add sentences while the chunk stays <= `max_tokens` (200 content tokens).
- Overlap: the next chunk starts with the longest run of the previous chunk's last sentences
  totalling <= `overlap` (40) tokens; none if the last sentence alone is longer. Overlap
  sentences are dropped from the front if they'd push the chunk over the max.
- A sentence over `max_tokens` is split into token windows (C3's rule: word boundaries,
  `overlap` tokens of overlap); those pieces are the only mid-sentence boundaries C2 makes.

Token counts come from one MiniLM tokenization of the whole stream, so a chunk's count is
exact (`meta.minilm_tokens`). Pages are one stream; chunks run across page breaks.
"""
from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass

import syntok.segmenter as syntok

from evals.chunking.base import Chunk, SourceDoc
from evals.chunking.c3_fixed_tokens import token_windows
from evals.chunking.stream import build_stream
from evals.chunking.tokens import minilm_offsets

log = logging.getLogger(__name__)
_LINE = re.compile(r"[^\n]+")


@dataclass
class _Unit:
    a: int            # stream char range
    b: int
    ti: int           # token range [ti, tj)
    tj: int
    split: bool = False


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    """Char ranges of sentences in `text`; line breaks are hard boundaries; ranges trimmed."""
    out: list[tuple[int, int]] = []
    for m in _LINE.finditer(text):
        line = m.group()
        if not line.strip():
            continue
        starts = [s[0].offset for para in syntok.analyze(line) for s in para if s]
        end = len(line.rstrip())
        for k, s in enumerate(starts):
            e = starts[k + 1] if k + 1 < len(starts) else end
            while e > s and line[e - 1].isspace():
                e -= 1
            if e > s:
                out.append((m.start() + s, m.start() + e))
    return out


def _units(text: str, offsets, word_ids, max_tokens: int, overlap: int) -> list[_Unit]:
    starts = [o[0] for o in offsets]
    units: list[_Unit] = []
    for a, b in sentence_ranges(text):
        ti, tj = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
        if tj == ti:
            continue                      # nothing the tokenizer keeps (e.g. control chars)
        if tj - ti <= max_tokens:
            units.append(_Unit(a, b, ti, tj))
            continue
        for i, j in token_windows(word_ids, ti, tj, max_tokens, overlap):
            units.append(_Unit(offsets[i][0], offsets[j - 1][1], i, j, split=True))
    return units


def _tokens(units: list[_Unit]) -> int:
    return units[-1].tj - units[0].ti if units else 0


def pack(units: list[_Unit], max_tokens: int, overlap: int) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    cur: list[_Unit] = []
    for u in units:
        if cur and u.tj - cur[0].ti > max_tokens:
            groups.append(cur)
            tail: list[_Unit] = []
            for prev in reversed(cur):
                if _tokens([prev] + tail) > overlap:
                    break
                tail.insert(0, prev)
            while tail and u.tj - tail[0].ti > max_tokens:
                tail.pop(0)
            cur = tail
        cur.append(u)
    if cur:
        groups.append(cur)
    return groups


@dataclass
class Window:
    a: int                # stream char range
    b: int
    tokens: int           # MiniLM content tokens of stream.text[a:b]
    sentences: int
    split: bool


def sentence_windows(text: str, max_tokens: int, overlap: int) -> list[Window]:
    """C2's windowing over any text (C4 reuses it inside a section)."""
    offsets, word_ids = minilm_offsets(text)
    return [Window(g[0].a, g[-1].b, _tokens(g), len(g), any(u.split for u in g))
            for g in pack(_units(text, offsets, word_ids, max_tokens, overlap), max_tokens, overlap)]


class C2SentenceWindow:
    chunker_id = "C2"

    def __init__(self, max_tokens: int = 200, overlap: int = 40):
        self.params = {"max_tokens": max_tokens, "overlap": overlap, "splitter": "syntok 1.4.4",
                       "tokenizer": "all-MiniLM-L6-v2"}

    def chunk(self, doc: SourceDoc) -> list[Chunk]:
        if doc.form_fields:
            log.warning("%s: C2 chunks page text only; %d form fields ignored", doc.doc_id,
                        len(doc.form_fields))
        stream = build_stream(doc)
        chunks: list[Chunk] = []
        for w in sentence_windows(stream.text, self.params["max_tokens"], self.params["overlap"]):
            idx = len(chunks)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-C2-{idx}", doc_id=doc.doc_id, index=idx,
                text=stream.text[w.a:w.b], spans=stream.spans(w.a, w.b),
                meta={"minilm_tokens": w.tokens, "stream_range": [w.a, w.b], "sentences": w.sentences,
                      "split_sentence": w.split},
            ))
        return chunks
