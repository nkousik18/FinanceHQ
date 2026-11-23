"""C1 = production chunker (N2) and provenance checks (F1)."""
import logging

import pytest
import structlog

from evals.chunking.base import Span
from evals.chunking.c1_line_window import C1LineWindow, _merge
from evals.chunking.source import build_source
from evals.parsers.production_adapter import _SERVICE  # noqa: F401

from app.pipeline.chunker import chunk_document  # noqa: E402
from app.pipeline.extractor import ExtractionResult, FormField, PageResult, Table, TableCell  # noqa: E402

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

PROSE = ("The Borrower shall repay the Loans in full on the Maturity Date together with accrued interest "
         "and all other amounts then due under this Agreement. ") * 9


def _extraction(with_fields=False):
    p1 = "ARTICLE I DEFINITIONS\n" + PROSE + "\nApplicable Rate 2.50% 1.50%\nSection 1.2 Terms."
    p2 = "ARTICLE II LOANS\n" + " ".join(["word"] * 420) + "\nshort closing line"
    p3 = "Pricing Grid\nTier Rate\nI 1.0%\nII 1.5%"
    pages = [PageResult(page_number=n, text=t, word_count=len(t.split()), avg_confidence=100.0,
                        low_confidence_words=0, has_handwriting=False) for n, t in ((1, p1), (2, p2), (3, p3))]
    cells = [TableCell(r, c, x, 100.0) for (r, c, x) in
             [(1, 1, "Tier"), (1, 2, "Rate"), (2, 1, "I"), (2, 2, "1.0%"), (3, 1, "II"), (3, 2, "1.5%")]]
    fields = [FormField("Loan Amount", "110 000", 99, 99, 1)] if with_fields else []
    return ExtractionResult(session_id="docA", full_text="", pages=pages, total_pages=3,
                            tables=[Table(page=3, rows=3, cols=2, cells=cells)], form_fields=fields)


@pytest.mark.parametrize("with_fields", [False, True])
@pytest.mark.parametrize("size,overlap", [(400, 60), (120, 20), (40, 10)])
def test_n2_c1_matches_production(with_fields, size, overlap):
    er = _extraction(with_fields)
    src = build_source(er)
    ours = C1LineWindow(size, overlap).chunk(src)
    prod = chunk_document(src.cleaned, chunk_size=size, overlap=overlap)
    assert [(c.chunk_id, c.text, c.section, c.meta["production_page"]) for c in ours] == \
           [(c.chunk_id, c.text, c.section, c.page) for c in prod]


def test_f1_page_text_spans_point_at_chunk_text():
    src = build_source(_extraction())
    for ch in C1LineWindow(120, 20).chunk(src):
        for s in ch.spans:
            piece = src.text_at(s)
            assert piece.strip(), (ch.index, s)
            # every word of a span's source text appears in the chunk (tables: as markdown cells)
            assert set(piece.split()) <= set(ch.text.replace("|", " ").split()), (ch.index, piece)


def test_table_rows_map_to_their_page_text():
    src = build_source(_extraction())
    t = src.tables[0]
    assert [src.text_at(s) for s in t.row_spans] == ["Tier Rate", "I 1.0%", "II 1.5%"]
    table_chunk = next(c for c in C1LineWindow().chunk(src) if "[TABLE]" in c.text)
    assert all(s in table_chunk.spans or any(o.page == 3 and o.start <= s.start and s.end <= o.end
                                             for o in table_chunk.spans) for s in t.row_spans)


def test_long_line_pieces_get_their_own_word_spans():
    from evals.chunking.c1_line_window import _Line, _split_long
    text = "xx  " + "  ".join(f"w{i}" for i in range(7))          # double spaces: production rejoins with one
    pieces = _split_long(_Line(text, [Span(5, 100, 100 + len(text))]), 3)
    assert [p.text for p in pieces] == ["xx w0 w1", "w2 w3 w4", "w5 w6"]
    for p in pieces:
        s = p.spans[0]
        assert text[s.start - 100:s.end - 100].split() == p.text.split()
    table_row = _split_long(_Line("| a | b | c |", [Span(1, 0, 5)]), 2)   # table row: keep the row span
    assert all(p.spans == [Span(1, 0, 5)] for p in table_row)


def test_page_range_and_merge():
    assert _merge([Span(1, 10, 20), Span(1, 0, 9), Span(2, 0, 5), Span(1, 15, 30)]) == \
           [Span(1, 0, 30), Span(2, 0, 5)]
    src = build_source(_extraction())
    chunks = C1LineWindow().chunk(src)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
