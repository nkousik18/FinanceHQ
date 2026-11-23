"""
R-pc, parent–child retrieval (scope v2 §7): search small child chunks, return each hit's parent.

S2 setting (S-2026-11-22-32): children = C2 chunks (the run's chunker, dense search, as R-dense);
parents = C4 sections (the run's `parent_chunker`).

**Parent of a child:**
- A C4 *section* = a run of consecutive C4 chunks with the same heading path (`Chunk.section`).
  Its text is rebuilt from the cleaned page text under the union of those chunks' spans (C4's own
  chunks overlap and repeat the heading prefix), prefixed with the heading path; so parent text
  and parent spans match exactly (F1 provenance).
- The child's section = the one whose spans share the most characters with the child's spans
  (ties -> earlier section).
- **Window cap (D10, S-2026-11-22-41): a section over 800 MiniLM content tokens -> an 800-token
  window of it centred on the child** (clipped at the section's ends, so still 800 when possible).
- **No C4 section available** (the child overlaps no C4 chunk, or only chunks with an empty heading
  path) -> the same 800-token window over the whole document (pages joined as in C2/C3).

**Ranking:** children by dense score (all of them, best first); each child maps to its parent;
a child is **skipped when its parent is already returned, or when >= 50% of the child's characters
are already inside a returned parent** (the §8.3 coverage rule) — overlapping windows otherwise
repeat text in the 1,200-word budget. Parent score = its best child's score. Stops at k parents.

Rank metrics' ideal (n_relevant for ndcg/map) stays the children count; recall@budget and entity
recall — the decision metrics — don't use it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from evals.chunking.base import Chunk, SourceDoc, Span
from evals.chunking.stream import DocStream, build_stream, stream_from_pieces
from evals.retrievers.dense import DenseRetriever, Hits, SessionIndex

WINDOW_TOKENS = 800
SECTION_SEP = "\n"
COVERED_FRAC = 0.5


def _overlap(a: list[Span], b: list[Span]) -> int:
    n = 0
    for x in a:
        for y in b:
            if x.page == y.page:
                n += max(0, min(x.end, y.end) - max(x.start, y.start))
    return n


def _chars(spans: list[Span]) -> int:
    return sum(s.end - s.start for s in spans)


def merged_spans(spans: list[Span]) -> list[Span]:
    """Union of spans, sorted by (page, start); overlapping or touching spans joined."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (s.page, s.start, s.end)):
        if out and out[-1].page == s.page and s.start <= out[-1].end:
            out[-1] = Span(s.page, out[-1].start, max(out[-1].end, s.end))
        else:
            out.append(s)
    return out


@dataclass
class _Region:
    """A section (or a whole document) as one stream, with lazily computed token offsets."""
    rid: str
    doc_id: str
    prefix: str                    # heading path + separator ("" for a document)
    stream: DocStream
    section: str = ""
    _offsets: list[tuple[int, int]] | None = field(default=None, repr=False)

    def offsets(self) -> list[tuple[int, int]]:
        if self._offsets is None:
            from evals.chunking.tokens import minilm_offsets
            self._offsets = [o for o in minilm_offsets(self.stream.text)[0] if o[1] > o[0]]
        return self._offsets

    def position(self, spans: list[Span]) -> int | None:
        """Stream offset of the middle of the child's characters inside this region."""
        pts = []
        for sp in spans:
            for page, ps, pe, off in self.stream.pages:
                if page != sp.page:
                    continue
                a, b = max(sp.start, off), min(sp.end, off + (pe - ps))
                if a < b:
                    pts += [ps + a - off, ps + b - off]
        return (min(pts) + max(pts)) // 2 if pts else None


def window(region: _Region, center: int | None, size: int = WINDOW_TOKENS) -> tuple[str, int, int]:
    """(parent id suffix, stream start, stream end): the whole region if <= size tokens, else
    `size` tokens centred on `center` (stream offset), clipped to the region."""
    offs = region.offsets()
    if len(offs) <= size:
        return "all", 0, len(region.stream.text)
    c = 0 if center is None else int(np.searchsorted([o[0] for o in offs], center, side="right")) - 1
    a = max(0, c - size // 2)
    b = min(len(offs), a + size)
    a = max(0, b - size)
    return f"t{a}", offs[a][0], offs[b - 1][1]


class ParentMap:
    """Per session: the C4 sections and document streams that children map to."""

    def __init__(self, sources: dict[str, SourceDoc], parent_chunks: dict[str, list[Chunk]],
                 size: int = WINDOW_TOKENS):
        self.sources, self.size = sources, size
        self.sections: dict[str, list[tuple[_Region, list[Span]]]] = {}
        self._docs: dict[str, _Region] = {}
        self._cache: dict[tuple[str, str], Chunk] = {}
        for doc_id, chunks in parent_chunks.items():
            src, groups = sources[doc_id], []
            for c in chunks:
                if groups and groups[-1][0] == c.section:
                    groups[-1][1].extend(c.spans)
                else:
                    groups.append((c.section, list(c.spans)))
            regs = []
            for gi, (sec, spans) in enumerate(groups):
                pieces = merged_spans(spans)
                if not sec or not pieces:
                    continue
                reg = _Region(f"{doc_id}-PC-s{gi}", doc_id, sec + SECTION_SEP,
                              stream_from_pieces(src, pieces, SECTION_SEP), sec)
                regs.append((reg, pieces))
            self.sections[doc_id] = regs

    def _doc_region(self, doc_id: str) -> _Region:
        if doc_id not in self._docs:
            self._docs[doc_id] = _Region(f"{doc_id}-PC-doc", doc_id, "", build_stream(self.sources[doc_id]))
        return self._docs[doc_id]

    def region_of(self, child: Chunk) -> _Region:
        best, best_n = None, 0
        for reg, pieces in self.sections.get(child.doc_id, []):
            n = _overlap(child.spans, pieces)
            if n > best_n:
                best, best_n = reg, n
        return best if best is not None else self._doc_region(child.doc_id)

    def parent(self, child: Chunk) -> Chunk:
        reg = self.region_of(child)
        suffix, a, b = window(reg, reg.position(child.spans), self.size)
        key = (reg.rid, suffix)
        if key not in self._cache:
            body = reg.stream.text[a:b]
            self._cache[key] = Chunk(
                chunk_id=f"{reg.rid}-{suffix}", doc_id=reg.doc_id, index=-1, text=reg.prefix + body,
                spans=reg.stream.spans(a, b),     # one per piece: the separators between pieces have no span
                section=reg.section,
                meta={"kind": "parent", "window": suffix != "all", "section_parent": bool(reg.section)})
        return self._cache[key]


class ParentChildRetriever:
    retriever_id = "R-pc"
    params: dict = {"children": "run chunker, dense (IndexFlatIP)", "parents": "parent_chunker sections",
                    "window_tokens": WINDOW_TOKENS, "fallback": "document window",
                    "skip_child_covered_frac": COVERED_FRAC}
    needs_vector = True
    needs_parents = True

    def __init__(self):
        self.dense = DenseRetriever()

    def search(self, index: SessionIndex, query_vec, k: int, query: str = "") -> Hits:
        pmap: ParentMap = index.extras["parents"]
        children = self.dense.search(index, query_vec, len(index.chunks))
        t0 = time.perf_counter()
        out: list[Chunk] = []
        scores: list[float] = []
        seen: set[str] = set()
        for i, s in zip(children.ids, children.scores):
            if len(out) >= k:
                break
            child = index.chunks[i]
            total = _chars(child.spans)
            if total and any(p.doc_id == child.doc_id and _overlap(child.spans, p.spans) >= COVERED_FRAC * total
                             for p in out):
                continue
            par = pmap.parent(child)
            if par.chunk_id in seen:
                continue
            seen.add(par.chunk_id)
            out.append(par)
            scores.append(s)
        return Hits([], scores, {**children.timings, "parent_map": time.perf_counter() - t0}, chunks=out)
