"""
§8.2 chunk metrics (scope v2), computed per document from a SourceDoc and its chunks, no queries.

All position-based metrics work on chunk **spans** (provenance into the cleaned page text), so
one definition scores every chunker. Text a chunker adds (C1 page headings and table pipes, C4
heading prefixes) has no span: it counts in size and truncation, not in coverage or overlap.

- Size: MiniLM content tokens, tiktoken cl100k_base tokens, words (whitespace split).
- Truncation (★): a chunk is truncated if it has > 254 MiniLM content tokens. Reported per chunk
  copy (dropped tokens / all tokens) and per document (page words that no chunk embeds: inside
  no chunk, or only inside the dropped tail of truncated chunks — R0's "document words no chunk
  embeds", now via spans).
- Table-break: a table (header-row span + >= 1 data-row span) is **split** if no single chunk's
  spans contain all its row spans; it is **broken** if split and some chunk touching a data row
  doesn't contain the header row. Rate = broken / tables.
- Sentence-break: reference sentences = syntok per line of the page text (C2's splitter; so C2
  and C5 only break inside sentences they split for size, by construction). Every chunk start
  and end (its first / last span edge in reading order) except the document's own start and end
  is a boundary; it is mid-sentence if it is not a sentence start / end.
- Evidence fragmentation (★, QA): % of evidence spans no single chunk covers, with §8.3's rule
  (same doc, page in the chunk's page range, >= 50% of the span's non-blank chars inside the
  chunk's spans). Needs the QA set (R3).
- Overlap redundancy: page text tokenized with MiniLM; a token is indexed once per chunk whose
  spans contain its first char. Redundancy = duplicate copies / all indexed copies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from evals.chunking.base import Chunk, SourceDoc, Span, has_text, is_blank
from evals.chunking.c1_line_window import _merge
from evals.chunking.c2_sentence_window import sentence_ranges
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_offsets, minilm_tokenizer, tiktoken_count

_WORD = re.compile(r"\S+")
_ANCHOR_WORDS = 8
_GAP = r"(?:[\s|]|-{3,})+"          # between words in C1 text: blanks, pipes, "| --- |" rows


# ---- helpers -----------------------------------------------------------------------------------
def _blank_mask(text: str) -> np.ndarray:
    return np.fromiter((is_blank(c) for c in text), bool, len(text))


def _chunk_mask(src: SourceDoc, chunk: Chunk) -> dict[int, np.ndarray]:
    """Per page, the chars inside this chunk's spans."""
    out: dict[int, np.ndarray] = {}
    for s in chunk.spans:
        m = out.setdefault(s.page, np.zeros(len(src.pages[s.page]), bool))
        m[s.start:s.end] = True
    return out


def _contains(merged: list[Span], s: Span) -> bool:
    return any(m.page == s.page and m.start <= s.start and s.end <= m.end for m in merged)


def _touches(merged: list[Span], s: Span) -> bool:
    return any(m.page == s.page and m.start < s.end and s.start < m.end for m in merged)


# ---- size --------------------------------------------------------------------------------------
@dataclass
class Sizes:
    minilm: list[int]
    tiktoken: list[int]
    words: list[int]


def sizes(chunks: list[Chunk]) -> Sizes:
    texts = [c.text for c in chunks]
    mini = [len(i) for i in minilm_tokenizer()(texts, add_special_tokens=False)["input_ids"]] if texts else []
    return Sizes(mini, [tiktoken_count(t) for t in texts], [len(t.split()) for t in texts])


# ---- truncation --------------------------------------------------------------------------------
def embedded_cut(text: str, limit: int = MAX_CONTENT_TOKENS) -> int:
    """Char offset in `text` where the embedder stops (len(text) if nothing is dropped)."""
    offsets, _ = minilm_offsets(text)
    return len(text) if len(offsets) <= limit else offsets[limit - 1][1]


def _locate_words(text: str, words: list[tuple[int, int, str]]) -> list[int]:
    """Position of each span word in the chunk text (-1 if not found). Anchored on the span's
    first words with any whitespace, table pipes or a markdown separator row between them, then
    found in order."""
    if not words:
        return []
    k = min(_ANCHOR_WORDS, len(words))
    m = re.search(_GAP.join(re.escape(w) for _, _, w in words[:k]), text)
    if m is None:
        return [-1] * len(words)
    out, cur = [], m.start()
    for _, _, w in words:
        at = text.find(w, cur)
        out.append(at)
        if at >= 0:
            cur = at + len(w)
    return out


@dataclass
class Truncation:
    chunks: int
    truncated: int
    tokens: int
    tokens_dropped: int
    page_words: int
    page_words_never_embedded: int
    unaligned_words: int        # span words of truncated chunks not found in the chunk text


def truncation(src: SourceDoc, chunks: list[Chunk], minilm: list[int]) -> Truncation:
    emb = {p: np.zeros(len(t), bool) for p, t in src.pages.items()}
    unaligned = 0
    for c, n in zip(chunks, minilm):
        if n <= MAX_CONTENT_TOKENS:
            for s in c.spans:
                emb[s.page][s.start:s.end] = True
            continue
        cut = embedded_cut(c.text)
        for s in c.spans:
            page = src.pages[s.page]
            words = [(s.start + m.start(), s.start + m.end(), m.group()) for m in _WORD.finditer(page[s.start:s.end])]
            for (a, b, w), at in zip(words, _locate_words(c.text, words)):
                if at < 0:
                    unaligned += 1
                elif at + len(w) <= cut:
                    emb[s.page][a:b] = True
    total = never = 0
    for p, t in src.pages.items():
        for m in _WORD.finditer(t):
            if not has_text(m.group()):          # e.g. a lone U+FEFF: not a word
                continue
            total += 1
            never += not emb[p][m.start():m.end()].any()
    return Truncation(len(chunks), sum(n > MAX_CONTENT_TOKENS for n in minilm), sum(minilm),
                      sum(max(0, n - MAX_CONTENT_TOKENS) for n in minilm), total, never, unaligned)


# ---- tables ------------------------------------------------------------------------------------
@dataclass
class TableBreaks:
    tables: int                 # tables with a header-row span and >= 1 data-row span
    split: int
    broken: int                 # split and some piece lacks the header row


def table_breaks(src: SourceDoc, chunks: list[Chunk]) -> TableBreaks:
    merged = [_merge(c.spans) for c in chunks]
    n = split = broken = 0
    for t in src.tables:
        head = t.row_spans[0] if t.row_spans else None
        data = [s for s in t.row_spans[1:] if s]
        if head is None or not data:
            continue
        n += 1
        if any(all(_contains(m, s) for s in [head] + data) for m in merged):
            continue
        split += 1
        if any(any(_touches(m, s) for s in data) and not _contains(m, head) for m in merged):
            broken += 1
    return TableBreaks(n, split, broken)


# ---- sentences ---------------------------------------------------------------------------------
def sentence_edges(text: str) -> tuple[set[int], set[int]]:
    """(sentence starts, sentence ends) in `text`, blank-trimmed like spans."""
    starts, ends = set(), set()
    for a, b in sentence_ranges(text):
        while a < b and is_blank(text[a]):
            a += 1
        while b > a and is_blank(text[b - 1]):
            b -= 1
        if a < b:
            starts.add(a)
            ends.add(b)
    return starts, ends


@dataclass
class SentenceBreaks:
    boundaries: int
    mid_sentence: int


def sentence_breaks(src: SourceDoc, chunks: list[Chunk]) -> SentenceBreaks:
    edges = {p: sentence_edges(t) for p, t in src.pages.items()}
    pages = sorted(src.pages)
    first_page, last_page = pages[0], pages[-1]
    doc_start = min(edges[first_page][0], default=None)
    doc_end = max(edges[last_page][1], default=None)
    n = mid = 0
    for c in chunks:
        if not c.spans:
            continue
        first = min(c.spans, key=lambda s: (s.page, s.start))
        last = max(c.spans, key=lambda s: (s.page, s.end))
        if not (first.page == first_page and first.start == doc_start):
            n += 1
            mid += first.start not in edges[first.page][0]
        if not (last.page == last_page and last.end == doc_end):
            n += 1
            mid += last.end not in edges[last.page][1]
    return SentenceBreaks(n, mid)


# ---- overlap redundancy ------------------------------------------------------------------------
@dataclass
class Redundancy:
    indexed_tokens: int         # token copies across chunks
    duplicate_tokens: int       # copies beyond the first


def redundancy(src: SourceDoc, chunks: list[Chunk]) -> Redundancy:
    count = {p: np.zeros(len(t), np.int32) for p, t in src.pages.items()}
    for c in chunks:
        for p, m in _chunk_mask(src, c).items():
            count[p] += m
    indexed = dup = 0
    for p, t in src.pages.items():
        if not t:
            continue
        offsets, _ = minilm_offsets(t)
        starts = np.fromiter((a for a, _ in offsets), np.int64, len(offsets))
        k = count[p][starts]
        indexed += int(k.sum())
        dup += int(np.maximum(k - 1, 0).sum())
    return Redundancy(indexed, dup)


# ---- evidence fragmentation --------------------------------------------------------------------
@dataclass
class Evidence:
    doc_id: str
    page: int
    span_text: str


def locate_evidence(src: SourceDoc, ev: Evidence) -> Span | None:
    """The evidence span in the cleaned page text: exact match, else whitespace-insensitive."""
    page = src.pages.get(ev.page)
    words = ev.span_text.split()
    if page is None or not words:
        return None
    at = page.find(ev.span_text.strip())
    if at >= 0:
        return Span(ev.page, at, at + len(ev.span_text.strip()))
    m = re.search(r"\s+".join(re.escape(w) for w in words), page)
    return Span(ev.page, m.start(), m.end()) if m else None


def covers(src: SourceDoc, chunk: Chunk, span: Span, min_frac: float = 0.5) -> bool:
    """§8.3: same doc, span page in the chunk's page range, >= min_frac of the span's non-blank
    characters inside the chunk's spans."""
    if chunk.doc_id != src.doc_id or chunk.page_start is None:
        return False
    if not (chunk.page_start <= span.page <= chunk.page_end):
        return False
    text = src.pages[span.page]
    visible = ~_blank_mask(text[span.start:span.end])
    if not visible.any():
        return False
    inside = np.zeros(span.end - span.start, bool)
    for s in chunk.spans:
        if s.page == span.page:
            a, b = max(s.start, span.start), min(s.end, span.end)
            if a < b:
                inside[a - span.start:b - span.start] = True
    return (inside & visible).sum() >= min_frac * visible.sum()


@dataclass
class Fragmentation:
    spans: int
    located: int
    fragmented: int             # located, and no single chunk covers it


def fragmentation(src: SourceDoc, chunks: list[Chunk], evidence: list[Evidence]) -> Fragmentation:
    n = located = frag = 0
    for ev in evidence:
        if ev.doc_id != src.doc_id:
            continue
        n += 1
        span = locate_evidence(src, ev)
        if span is None:
            continue
        located += 1
        frag += not any(covers(src, c, span) for c in chunks)
    return Fragmentation(n, located, frag)
