"""
C1 — line window, the production chunker (scope v2 §6.1), with provenance.

Mirrors fastapi_service/app/pipeline/chunker.py `chunk_document` line for line, but every
line carries the spans it came from. The N2 test (evals/tests/test_chunking_c1.py) checks that
chunk ids, text, page and section are identical to production's output, so this is the
production behaviour, not a re-interpretation. Production itself is not modified.

Line assembly (as production): per page "### Page N", "", the page text lines, then per
table "\\n[TABLE]" + its markdown lines, then ""; form fields as a "### Form Fields" section.
Window: 400 "tokens" (words / 0.75 -> 300 words), overlap 60 (45 words), see `_slide`.

Provenance of a long line that production splits into word groups: page-text lines map
each piece to its own words; a table markdown line keeps the whole row's span on every piece
(markdown pipes have no position in the page text).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from evals.chunking.base import Chunk, SourceDoc, Span, has_text
from evals.parsers.production_adapter import _SERVICE  # noqa: F401  (puts fastapi_service on sys.path)

from app.pipeline import chunker as P  # noqa: E402

_WORD = re.compile(r"\S+")


@dataclass
class _Line:
    text: str
    spans: list[Span] = field(default_factory=list)


def _page_lines(page: int, text: str) -> list[_Line]:
    """text.splitlines() with each line's span in the page text (none for a line of blanks)."""
    out, pos = [], 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n\x0b\x0c\x1c\x1d\x1e\x85  ")
        out.append(_Line(line, [Span(page, pos, pos + len(line))] if has_text(line) else []))
        pos += len(raw)
    return out


def assemble_lines(doc: SourceDoc) -> list[_Line]:
    cleaned = doc.cleaned
    tables_by_page: dict[int, list] = {}
    for t in doc.tables:
        tables_by_page.setdefault(t.page, []).append(t)
    lines: list[_Line] = []
    for page in cleaned.pages:
        n = page.page_number
        lines += [_Line(f"### Page {n}"), _Line("")]
        lines += _page_lines(n, doc.pages[n])
        for t in tables_by_page.get(n, []):
            lines.append(_Line("\n[TABLE]"))
            md = t.markdown.splitlines()
            # markdown: header row (table row 0), separator, data rows 1..
            row_of_line = [0, None] + list(range(1, len(md) - 1))
            for line, r in zip(md, row_of_line):
                span = t.row_spans[r] if r is not None and r < len(t.row_spans) else None
                lines.append(_Line(line, [span] if span else []))
        lines.append(_Line(""))
    if cleaned.form_fields:
        lines += [_Line("### Form Fields"), _Line("")]
        lines += [_Line(f"{f.key}: {f.value}") for f in cleaned.form_fields if f.value]
        lines.append(_Line(""))
    return lines


def _split_long(line: _Line, size: int) -> list[_Line]:
    """Production: words[i:i+size] joined by single spaces."""
    words = line.text.split()
    pieces = [" ".join(words[i: i + size]) for i in range(0, len(words), size)]
    if len(line.spans) != 1 or line.text.split() != [m.group() for m in _WORD.finditer(line.text)]:
        return [_Line(p, list(line.spans)) for p in pieces]
    base = line.spans[0]
    pos = [m.span() for m in _WORD.finditer(line.text)]
    if base.end - base.start != len(line.text):          # table row span: not a 1:1 text map
        return [_Line(p, [base]) for p in pieces]
    out = []
    for i in range(0, len(words), size):
        a, b = pos[i][0], pos[min(i + size, len(words)) - 1][1]
        out.append(_Line(" ".join(words[i: i + size]), [Span(base.page, base.start + a, base.start + b)]))
    return out


def _overlap(group: list[_Line], overlap_words: int) -> tuple[list[_Line], int]:
    out, count = [], 0
    for line in reversed(group):
        w = P._word_count(line.text)
        if count + w > overlap_words and out:
            break
        out.insert(0, line)
        count += w
    return out, count


def _slide(lines: list[_Line], size: int, overlap_words: int) -> list[list[_Line]]:
    groups: list[list[_Line]] = []
    cur: list[_Line] = []
    words = 0
    for line in lines:
        lw = P._word_count(line.text)
        if lw > size:
            for sub in _split_long(line, size):
                cur.append(sub)
                words += P._word_count(sub.text)
                if words >= size:
                    groups.append(cur[:])
                    cur, words = _overlap(cur, overlap_words)
            continue
        cur.append(line)
        words += lw
        if words >= size:
            groups.append(cur[:])
            cur, words = _overlap(cur, overlap_words)
    if cur:
        groups.append(cur)
    return groups


def _merge(spans: list[Span]) -> list[Span]:
    """Sort and merge touching/overlapping spans on the same page."""
    out: list[Span] = []
    for s in sorted(set(spans), key=lambda s: (s.page, s.start, s.end)):
        if out and out[-1].page == s.page and s.start <= out[-1].end + 1:
            if s.end > out[-1].end:
                out[-1] = Span(s.page, out[-1].start, s.end)
        else:
            out.append(s)
    return out


class C1LineWindow:
    chunker_id = "C1"

    def __init__(self, chunk_size: int = P.DEFAULT_CHUNK_SIZE, overlap: int = P.DEFAULT_CHUNK_OVERLAP):
        self.params = {"chunk_size": chunk_size, "overlap": overlap, "words_per_token": P.WORDS_PER_TOKEN}

    def chunk(self, doc: SourceDoc) -> list[Chunk]:
        size = int(self.params["chunk_size"] * P.WORDS_PER_TOKEN)
        overlap_words = int(self.params["overlap"] * P.WORDS_PER_TOKEN)
        groups = _slide(assemble_lines(doc), size, overlap_words)

        chunks: list[Chunk] = []
        section, page = "Document", None
        for idx, group in enumerate(groups):
            for line in group:                               # production's section/page tracking
                heading = P._extract_section(line.text)
                if heading:
                    n = P._extract_page_from_heading(heading)
                    if n is not None:
                        page = n
                    else:
                        section = heading
            text = "\n".join(line.text for line in group).strip()
            if not text:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-{idx}", doc_id=doc.doc_id, index=idx, text=text,
                spans=[t for s in _merge([s for line in group for s in line.spans]) if (t := doc.trim(s))],
                section=section,
                meta={"production_page": page, "word_count": P._word_count(text),
                      "token_estimate": P._estimate_tokens(P._word_count(text))},
            ))
        return chunks
