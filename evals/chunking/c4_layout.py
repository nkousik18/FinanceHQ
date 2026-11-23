"""
C4 — layout-aware chunker (scope v2 §6.1), built on Docling's layout elements.

Scope rules and how they are implemented (choices: Decision Log 2026-11-12):
- **Section boundaries = Docling `section_header` / `title` elements.** Docling 2.147 gives every
  header level 1, so the level is read from the heading's numbering (`heading_level`):
  ARTICLE / EXHIBIT / SCHEDULE / ANNEX / unnumbered = 1, "3." = 2, "2.11" / "Section 2.11" = 3,
  "2.11.1" = 4, "A." = 5, "(a)" = 6. An unnumbered heading that directly follows another heading
  is joined to it ("Article II" + "CONDITIONS PRECEDENT"). A heading text on >= 3 pages is
  demoted to body text: it can't reset the path ("Execution Version" on 102 pages of deal_08_ca)
  but its text stays in the chunks (repeated exhibit titles, signature-page markers).
- **Run-in headings** (Kousik's decision 2026-11-13, a deviation from "Docling elements only"):
  a `text`/`list` element opening with "2.11 Title." / "Section 2.11. Title." (dotted number,
  <= 15 title-case words, ending in ".") is split into a heading and its body (`split_runin`).
  Off with `runin_headings=False` (strict C4).
- A heading that gets no body before the next heading at its level ("2.03 [Reserved.]") is
  emitted as a heading-only piece, which the min-size merge then joins to a neighbour.
- **Heading-path prefix** "A > B > C: " is part of the chunk text and its token budget; capped at
  64 MiniLM tokens by dropping outer headings first. Each heading's span goes to the first chunk
  whose text contains it.
- **Tables whole as markdown** (CleanedTable.to_markdown format). Over the max, split by rows
  with the header row (+ separator) repeated in every piece; a single row over the room left is
  cut into word-snapped token windows. If the header leaves < 32 tokens of room it is not
  repeated (`meta.header_repeated` False).
- **Form key-value pairs** grouped into "Form Fields" chunks (Docling returns none on this corpus).
- **Max 254 content tokens** (= 256 with [CLS]/[SEP], as C3) **, min 50.** Prose inside a section
  is windowed with C2's sentence rule (overlap 40) under the room the prefix leaves; a prose
  chunk under 50 tokens is merged into the next (else previous) prose chunk if the sum fits.
- Page headers, footers and page numbers are left out of chunks.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from evals.chunking.base import Chunk, Element, SourceDoc, Span, strip_blank
from evals.chunking.c1_line_window import _merge
from evals.chunking.c2_sentence_window import sentence_windows
from evals.chunking.c3_fixed_tokens import token_windows
from evals.chunking.stream import stream_from_pieces
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_count, minilm_counts, minilm_offsets

log = logging.getLogger(__name__)

FURNITURE = {"page_header", "page_footer", "page_number"}
HEADING = {"section_header", "title"}
REPEATED_HEADING_PAGES = 3
MIN_ROW_ROOM = 32

_ARTICLE = re.compile(r"^(?:ARTICLE|Article)\s+[IVXLC\d]+\b")
_PART = re.compile(r"^(?:EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex|APPENDIX|Appendix)\b")
_NUM = re.compile(r"^(?:(?:SECTION|Section|§+)\s*)?(\d{1,3}(?:\.\d{1,3})*)\.?(?=[\s(]|$)")
_CAP = re.compile(r"^[A-Z]\.\s")
_PAREN = re.compile(r"^\((?:[a-z]{1,4}|\d{1,2})\)")


def heading_level(text: str) -> tuple[int, bool]:
    """(level, numbered) from the heading's own numbering."""
    t = text.strip()
    if _ARTICLE.match(t) or _PART.match(t):
        return 1, True
    m = _NUM.match(t)
    if m:
        return 2 + m.group(1).count("."), True
    if _CAP.match(t):
        return 5, True
    if _PAREN.match(t):
        return 6, True
    return 1, False


# Run-in headings (Kousik, 2026-11-13, S-2026-11-13-10): Docling labels most numbered sections
# ("Section 2.11 Prepayments. (a) The Borrower may ...") as body text, not section headers.
RUNIN_CLASSES = {"text", "list"}
_RUNIN = re.compile(r"^(?:(?:SECTION|Section|§+)\s*)?\d{1,3}(?:\.\d{1,3})+\.?\s+(?:[=|—–-]\s+)?"
                    r"(?P<title>[^.\n]{1,150}?)\.(?=\s|$)")
_SMALL = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "or", "the",
          "to", "under", "upon", "with", "without", "vs", "per", "etc", "&", "-", "–", "—"}
_TOC_TAIL = re.compile(r"^[\d\s.ivxlcIVXLC]*$")
RUNIN_MAX_TITLE_WORDS = 15


def _title_like(title: str) -> bool:
    words = title.split()
    if not words or len(words) > RUNIN_MAX_TITLE_WORDS:
        return False
    if not (words[0][0].isupper() or words[0][0].isdigit() or words[0].startswith("[")):
        return False
    return all(w[0].isupper() or w[0].isdigit() or w[0] in "([\"'“‘" or w.lower().strip(",;:") in _SMALL
               for w in words)


def split_runin(e: Element) -> tuple[Element, Element | None] | None:
    """A body `text`/`list` element that opens with a numbered run-in heading -> (heading, body).
    Heading = dotted section number + a title of <= 15 title-case words ending in "."; body = the
    rest (None if empty). Not a heading: lowercase titles (garbled text, cross-references) and
    table-of-contents lines (rest is only a page number or dot leaders)."""
    if e.cls not in RUNIN_CLASSES or e.span is None:
        return None
    m = _RUNIN.match(e.text)
    if not m or not _title_like(m.group("title")):
        return None
    cut = m.end()
    rest = e.text[cut:]
    if rest.strip() and _TOC_TAIL.match(rest):
        return None
    lead = len(rest) - len(rest.lstrip())
    head = Element(e.page, "section_header", e.text[:cut], Span(e.span.page, e.span.start, e.span.start + cut))
    if not rest.strip():
        return head, None
    b0 = e.span.start + cut + lead
    body = Element(e.page, e.cls, strip_blank(e.text[cut + lead:]), Span(e.span.page, b0, e.span.end))
    return head, body


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class _Head:
    level: int
    text: str
    spans: list[Span] = field(default_factory=list)
    attached: bool = False


@dataclass
class _Piece:
    text: str
    spans: list[Span]
    tokens: int
    kind: str                    # prose | table | form
    section: str
    meta: dict = field(default_factory=dict)


def _sub_span(doc: SourceDoc, row: Span | None, part: str, cursor: int) -> tuple[Span | None, int]:
    """Span of a cut piece of a table row: its words located in order inside the row's page
    text, starting at `cursor`. Falls back to the whole row span if a word isn't found."""
    if row is None:
        return None, cursor
    page = doc.pages[row.page]
    words = part.replace("|", " ").split()
    pos, first = cursor, None
    for w in words:
        at = page.find(w, pos, row.end)
        if at < 0:
            return row, cursor
        first = at if first is None else first
        pos = at + len(w)
    if first is None:
        return None, cursor
    return Span(row.page, first, pos), pos


def _fmt_row(row: list[str]) -> str:
    return "| " + " | ".join(c.replace("\n", " ") for c in row) + " |"


class C4LayoutAware:
    chunker_id = "C4"

    def __init__(self, max_tokens: int = MAX_CONTENT_TOKENS, min_tokens: int = 50, overlap: int = 40,
                 prefix_max: int = 64, runin_headings: bool = True):
        self.params = {"max_tokens": max_tokens, "min_tokens": min_tokens, "overlap": overlap,
                       "prefix_max": prefix_max, "runin_headings": runin_headings,
                       "tokenizer": "all-MiniLM-L6-v2"}
        self._prefix_cache: dict[tuple[str, ...], tuple[str, int, bool]] = {}

    # ---- prefix -------------------------------------------------------------------------
    def _prefix(self, path: list[_Head]) -> tuple[str, int, bool]:
        comps = tuple(h.text for h in path)
        if comps in self._prefix_cache:
            return self._prefix_cache[comps]
        cap, out = self.params["prefix_max"], ("", 0, False)
        if comps:
            parts, cut = list(comps), False
            text = " > ".join(parts) + ": "
            n = minilm_count(text)
            while n > cap and len(parts) > 1:
                parts, cut = parts[1:], True
                text = " > ".join(parts) + ": "
                n = minilm_count(text)
            words = parts[0].split()
            while n > cap and len(words) > 1:
                words, cut = words[: max(1, len(words) * 3 // 4)], True
                text = " ".join(words) + ": "
                n = minilm_count(text)
            out = (text, n, cut)
        self._prefix_cache[comps] = out
        return out

    def _attach_headings(self, piece: _Piece, path: list[_Head]) -> None:
        for h in path:
            if not h.attached and h.text in piece.text:
                piece.spans = _merge(piece.spans + h.spans)
                h.attached = True

    def _heading_only(self, path: list[_Head]) -> _Piece:
        """A section with no body: the heading path alone (no trailing ": ")."""
        prefix, _, cut = self._prefix(path)
        text = prefix[:-2]
        return _Piece(text, [], minilm_count(text), "prose", " > ".join(h.text for h in path),
                      {"prefix_tokens": 0, "prefix_cut": cut, "heading_only": True})

    # ---- prose ---------------------------------------------------------------------------
    def _prose(self, doc: SourceDoc, run: list[Element], path: list[_Head]) -> list[_Piece]:
        spans = [e.span for e in run if e.span]
        if not spans:
            return []
        prefix, ptok, cut = self._prefix(path)
        stream = stream_from_pieces(doc, spans, "\n")
        section = " > ".join(h.text for h in path)
        out = []
        for w in sentence_windows(stream.text, self.params["max_tokens"] - ptok, self.params["overlap"]):
            out.append(_Piece(prefix + stream.text[w.a:w.b], _merge(stream.spans(w.a, w.b)),
                              ptok + w.tokens, "prose", section,
                              {"prefix_tokens": ptok, "prefix_cut": cut, "sentences": w.sentences,
                               "split_sentence": w.split}))
        return out

    # ---- tables --------------------------------------------------------------------------
    def _table(self, doc: SourceDoc, t_idx: int, path: list[_Head]) -> list[_Piece]:
        ts = doc.tables[t_idx]
        rows = ts.rows
        if not rows or not any(c.strip() for r in rows for c in r):
            return []
        prefix, ptok, cut = self._prefix(path)
        section = " > ".join(h.text for h in path)
        row_span = lambda i: ts.row_spans[i] if i < len(ts.row_spans) else None   # noqa: E731
        head = [_fmt_row(rows[0]), _fmt_row(["---"] * len(rows[0]))]
        head_spans = [s for s in [row_span(0)] if s]
        body = [(_fmt_row(r), row_span(i), i) for i, r in enumerate(rows[1:], start=1)]
        counts = minilm_counts(head + [b[0] for b in body])
        htok, mx = counts[0] + counts[1], self.params["max_tokens"]
        meta = {"table": t_idx, "prefix_tokens": ptok, "prefix_cut": cut}

        if ptok + sum(counts) <= mx:                                    # whole table
            spans = _merge(head_spans + [s for _, s, _ in body if s])
            return [_Piece(prefix + "\n".join(head + [b[0] for b in body]), spans, ptok + sum(counts),
                           "table", section, {**meta, "rows": [1, len(rows) - 1], "pieces": 1,
                                              "header_repeated": True})]

        repeat = mx - ptok - htok >= MIN_ROW_ROOM
        items = [(line, s, i, n) for (line, s, i), n in zip(body, counts[2:])]
        if not repeat:                                                  # header becomes a row
            items = [(head[0], row_span(0), 0, counts[0]), (head[1], None, 0, counts[1])] + items
            head, head_spans, htok = [], [], 0
        room = mx - ptok - htok
        groups: list[list[tuple]] = []
        cur: list[tuple] = []
        cur_tok = 0
        for line, s, i, n in items:
            if n > room:                                                # one row too long: cut it
                if cur:
                    groups.append(cur)
                    cur, cur_tok = [], 0
                offsets, word_ids = minilm_offsets(line)
                cursor = s.start if s else 0
                for a, b in token_windows(word_ids, 0, len(offsets), room, 0):
                    part = line[offsets[a][0]:offsets[b - 1][1]]
                    sub, cursor = _sub_span(doc, s, part, cursor)
                    groups.append([(part, sub, i, b - a)])
                continue
            if cur and cur_tok + n > room:
                groups.append(cur)
                cur, cur_tok = [], 0
            cur.append((line, s, i, n))
            cur_tok += n
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            spans = _merge(head_spans + [s for _, s, _, _ in g if s])
            out.append(_Piece(prefix + "\n".join(head + [x[0] for x in g]), spans,
                              ptok + htok + sum(x[3] for x in g), "table", section,
                              {**meta, "rows": [g[0][2], g[-1][2]], "pieces": len(groups),
                               "header_repeated": repeat}))
        return out

    # ---- form fields ---------------------------------------------------------------------
    def _form(self, doc: SourceDoc) -> list[_Piece]:
        lines = [f"{k}: {v}" for k, v in doc.form_fields if v]
        if not lines:
            return []
        prefix = "Form Fields: "
        ptok = minilm_count(prefix)
        out, cur, cur_tok = [], [], 0
        for line, n in zip(lines, minilm_counts(lines)):
            if cur and ptok + cur_tok + n > self.params["max_tokens"]:
                out.append(_Piece(prefix + "\n".join(cur), [], ptok + cur_tok, "form", "Form Fields"))
                cur, cur_tok = [], 0
            cur.append(line)
            cur_tok += n
        if cur:
            out.append(_Piece(prefix + "\n".join(cur), [], ptok + cur_tok, "form", "Form Fields"))
        return out

    # ---- min-size merge ------------------------------------------------------------------
    def _merge_small(self, pieces: list[_Piece]) -> list[_Piece]:
        mn, mx = self.params["min_tokens"], self.params["max_tokens"]

        def join(a: _Piece, b: _Piece) -> _Piece:
            return _Piece(a.text + "\n" + b.text, _merge(a.spans + b.spans), a.tokens + b.tokens, "prose",
                          b.section if b.tokens >= a.tokens else a.section,
                          {**(b.meta if b.tokens >= a.tokens else a.meta),
                           "merged": a.meta.get("merged", 1) + b.meta.get("merged", 1)})

        out, i = list(pieces), 0
        while i < len(out):
            p = out[i]
            if p.kind == "prose" and p.tokens < mn:
                if i + 1 < len(out) and out[i + 1].kind == "prose" and p.tokens + out[i + 1].tokens <= mx:
                    out[i + 1] = join(p, out[i + 1])
                    del out[i]
                    continue
                if i > 0 and out[i - 1].kind == "prose" and out[i - 1].tokens + p.tokens <= mx:
                    out[i - 1] = join(out[i - 1], p)
                    del out[i]
                    continue
            i += 1
        return out

    # ---- main ----------------------------------------------------------------------------
    def chunk(self, doc: SourceDoc) -> list[Chunk]:
        if not doc.elements and doc.pages:
            raise ValueError(f"{doc.doc_id}: C4 needs layout elements (use load_source)")
        heads = [e for e in doc.elements if e.cls in HEADING]
        pages_of: dict[str, set[int]] = {}
        for e in heads:
            pages_of.setdefault(_norm(e.text), set()).add(e.page)
        running = {t for t, ps in pages_of.items() if len(ps) >= REPEATED_HEADING_PAGES}

        pieces: list[_Piece] = []
        path: list[_Head] = []
        run: list[Element] = []
        after_heading = False

        def emit(new: list[_Piece]) -> None:
            for p in new:
                self._attach_headings(p, path)
            pieces.extend(new)

        for e in doc.elements:
            if e.cls in FURNITURE:
                continue
            runin = split_runin(e) if self.params["runin_headings"] else None
            if (e.cls in HEADING and _norm(e.text) not in running) or runin:
                head, body = runin if runin else (e, None)
                emit(self._prose(doc, run, path))
                run = []
                level, numbered = heading_level(head.text)
                spans = [head.span] if head.span else []
                if after_heading and not numbered and path:
                    last = path[-1]
                    path[-1] = _Head(last.level, f"{last.text} {head.text}", last.spans + spans)
                else:
                    while path and path[-1].level >= level:
                        if not path[-1].attached and path[-1].spans:
                            emit([self._heading_only(path)])
                        path.pop()
                    path.append(_Head(level, head.text, spans))
                after_heading = body is None
                if body is not None:
                    run.append(body)
                continue
            after_heading = False
            if e.table is not None:
                emit(self._prose(doc, run, path))
                run = []
                emit(self._table(doc, e.table, path))
            else:
                run.append(e)
        emit(self._prose(doc, run, path))
        while path:
            if not path[-1].attached and path[-1].spans:
                emit([self._heading_only(path)])
            path.pop()
        pieces += self._form(doc)
        pieces = self._merge_small(pieces)

        return [Chunk(chunk_id=f"{doc.doc_id}-C4-{i}", doc_id=doc.doc_id, index=i, text=p.text,
                      spans=p.spans, section=p.section,
                      meta={"kind": p.kind, "minilm_tokens": p.tokens, **p.meta})
                for i, p in enumerate(pieces)]
