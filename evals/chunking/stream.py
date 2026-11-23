"""
A document's cleaned pages as one string, with a map back to page offsets.

C2 and C3 chunk this stream: chunk text is always a slice `stream.text[a:b]`, and its
provenance is that slice mapped back onto the pages (`DocStream.spans`). Pages are joined with
a blank line, which has no span. So every span's text is an exact substring of its chunk's text.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.chunking.base import SourceDoc, Span, is_blank

PAGE_SEP = "\n\n"


@dataclass
class DocStream:
    text: str
    pages: list[tuple[int, int, int, int]]     # (page number, start in stream, end in stream, start in page)

    def spans(self, a: int, b: int) -> list[Span]:
        """Spans for stream[a:b], one per segment touched, trimmed of blanks (`is_blank`); empty
        ones dropped. Adjacent pieces of one page stay separate spans (callers may `_merge`)."""
        out = []
        for page, ps, pe, off in self.pages:
            s, e = max(a, ps), min(b, pe)
            while s < e and is_blank(self.text[s]):
                s += 1
            while e > s and is_blank(self.text[e - 1]):
                e -= 1
            if s < e:
                out.append(Span(page, s - ps + off, e - ps + off))
        return out


def stream_from_pieces(doc: SourceDoc, pieces: list[Span], sep: str) -> DocStream:
    """Join page-text pieces (in the given order) with `sep`; each piece keeps its page offsets."""
    parts, segs, pos = [], [], 0
    for sp in pieces:
        if parts:
            parts.append(sep)
            pos += len(sep)
        t = doc.text_at(sp)
        segs.append((sp.page, pos, pos + len(t), sp.start))
        parts.append(t)
        pos += len(t)
    return DocStream("".join(parts), segs)


def build_stream(doc: SourceDoc) -> DocStream:
    return stream_from_pieces(doc, [Span(n, 0, len(doc.pages[n])) for n in sorted(doc.pages)], PAGE_SEP)
