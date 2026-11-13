"""
Chunker interface with provenance (scope v2 §6.1, F1).

Every chunker turns a SourceDoc into Chunks. Each Chunk records where its text came from as
spans into the **cleaned page text** (cleaner v2, S-2026-11-12-12): `Span(page, start, end)`
means `source.pages[page][start:end]`. Retrieval metrics match QA evidence spans against these.

Text a chunker adds that isn't in the page text (page headings, "[TABLE]" markers, markdown
table pipes, C4 heading prefixes) has no span. Table markdown rows point at the table row's
text inside the page text (Docling's page text holds every table row as one line).
"""
from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Protocol


def is_blank(ch: str) -> bool:
    """Whitespace or an invisible format character (e.g. U+FEFF BOM, which Docling text keeps
    and the MiniLM tokenizer drops). Spans never start or end on one, and never consist of them."""
    return ch.isspace() or unicodedata.category(ch) == "Cf"


def strip_blank(text: str) -> str:
    """str.strip() that also removes invisible format characters."""
    a, b = 0, len(text)
    while a < b and is_blank(text[a]):
        a += 1
    while b > a and is_blank(text[b - 1]):
        b -= 1
    return text[a:b]


def has_text(text: str) -> bool:
    return any(not is_blank(c) for c in text)


@dataclass(frozen=True)
class Span:
    page: int
    start: int
    end: int


@dataclass
class TableSource:
    page: int
    index: int                         # order of the table in the document
    markdown: str                      # CleanedTable.to_markdown(): header row, separator, data rows
    rows: list[list[str]]              # cleaned cell text, [row][col]
    row_spans: list[Span | None]       # each row's text in the page text (None: row has no text)


@dataclass
class Element:
    """One Docling layout element in reading order (C4). `span`: its text in the cleaned page
    text (None if the cleaner removed it or it has no text). `table`: index into SourceDoc.tables."""
    page: int
    cls: str                           # golden-set classes: section_header, text, list, table, ...
    text: str
    span: Span | None = None
    table: int | None = None


@dataclass
class SourceDoc:
    doc_id: str
    pages: dict[int, str]              # page number -> cleaned page text (pages with text only)
    tables: list[TableSource] = field(default_factory=list)
    form_fields: list[tuple[str, str]] = field(default_factory=list)   # (key, value), cleaned
    cleaned: object | None = None      # the CleanedResult, for chunkers that mirror production
    elements: list[Element] = field(default_factory=list)   # layout elements (load_source only)

    def text_at(self, span: Span) -> str:
        return self.pages[span.page][span.start:span.end]

    def trim(self, span: Span) -> Span | None:
        """The span without blank characters at its edges; None if nothing visible is left."""
        text, s, e = self.pages[span.page], span.start, span.end
        while s < e and is_blank(text[s]):
            s += 1
        while e > s and is_blank(text[e - 1]):
            e -= 1
        return Span(span.page, s, e) if s < e else None


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str                          # what gets embedded
    spans: list[Span]                  # provenance; may be empty for purely synthetic text
    section: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def page_start(self) -> int | None:
        return min((s.page for s in self.spans), default=None)

    @property
    def page_end(self) -> int | None:
        return max((s.page for s in self.spans), default=None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["page_start"], d["page_end"] = self.page_start, self.page_end
        return d


class Chunker(Protocol):
    chunker_id: str
    params: dict

    def chunk(self, doc: SourceDoc) -> list[Chunk]: ...
