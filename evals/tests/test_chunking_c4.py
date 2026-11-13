"""C4 layout-aware chunker: heading paths, tables, size bounds, provenance (F1)."""
import pytest

from evals.chunking.base import Element, SourceDoc, Span, TableSource
from evals.chunking.c4_layout import C4LayoutAware, heading_level
from evals.chunking.source import locate_elements
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_count
from evals.parsers.docling_doc import PageView

PROSE = ("The Borrower may prepay the Loans in whole or in part at any time without premium or "
         "penalty, subject to Section 3.05, upon notice to the Administrative Agent. ")


def _doc(pages: dict[int, list], form_fields=()) -> SourceDoc:
    """pages: {n: [(cls, text) | ("table", rows)]} -> SourceDoc with elements, spans, tables."""
    texts, elements, tables = {}, [], []
    for n, items in pages.items():
        lines, pos = [], 0
        for cls, val in items:
            if cls == "table":
                spans = []
                for row in val:
                    line = " ".join(c for c in row if c)
                    spans.append(Span(n, pos, pos + len(line)))
                    lines.append(line)
                    pos += len(line) + 1
                md = "\n".join("| " + " | ".join(r) + " |" for r in val)
                tables.append(TableSource(n, len(tables), md, [list(r) for r in val], spans))
                elements.append(Element(n, "table", "", None, len(tables) - 1))
            else:
                elements.append(Element(n, cls, val, Span(n, pos, pos + len(val))))
                lines.append(val)
                pos += len(val) + 1
        texts[n] = "\n".join(lines)
    return SourceDoc("d", texts, tables, list(form_fields), elements=elements)


def _check(doc, chunks):
    for c in chunks:
        assert c.meta["minilm_tokens"] == minilm_count(c.text) <= MAX_CONTENT_TOKENS, c.index
        flat = " ".join(c.text.replace("|", " ").split())
        for s in c.spans:
            assert all(w in flat for w in doc.text_at(s).split()), (c.index, s)


@pytest.mark.parametrize("text,level", [
    ("ARTICLE II", 1), ("Article IV Conditions", 1), ("EXHIBIT A", 1), ("Schedule 2", 1),
    ("3. Representations and Warranties.", 2), ("2.11 Prepayments.", 3), ("Section 2.11 Prepayments", 3),
    ("§ 1.01 Defined Terms", 3), ("2.11.1 Optional", 4), ("A. Origination Charges", 5), ("(a) Notices", 6),
    ("Closing Cost Details", 1), ("2019 Revolving Facility", 1),
])
def test_heading_level(text, level):
    assert heading_level(text)[0] == level


def test_heading_path_prefix_and_joined_titles():
    doc = _doc({1: [("section_header", "ARTICLE II"), ("section_header", "THE LOANS"),
                    ("section_header", "2.01 Commitments."), ("text", PROSE * 2),
                    ("section_header", "2.02 Prepayments."), ("text", PROSE * 2),
                    ("section_header", "ARTICLE III"), ("section_header", "3.01 Taxes."), ("text", PROSE * 2)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    texts = [c.text for c in chunks]
    assert texts[0].startswith("ARTICLE II THE LOANS > 2.01 Commitments.: The Borrower")
    assert texts[1].startswith("ARTICLE II THE LOANS > 2.02 Prepayments.: ")
    assert texts[2].startswith("ARTICLE III > 3.01 Taxes.: ")
    # every heading's span is attached to a chunk
    all_spans = [s for c in chunks for s in c.spans]
    for h in (e.span for e in doc.elements if e.cls == "section_header"):
        assert any(s.start <= h.start and h.end <= s.end for s in all_spans), h


def test_long_section_falls_back_to_sentence_windows():
    doc = _doc({1: [("section_header", "2.05 Prepayments."), ("text", PROSE * 20)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert len(chunks) > 3 and all(c.text.startswith("2.05 Prepayments.: ") for c in chunks)
    assert all(c.meta["minilm_tokens"] <= MAX_CONTENT_TOKENS for c in chunks)


def test_running_heading_is_demoted_and_furniture_dropped():
    pages = {n: [("page_header", "ACME CREDIT AGREEMENT"), ("section_header", "Execution Version"),
                 ("section_header", f"{n}.01 Item."), ("text", PROSE * 2), ("page_number", str(n))]
             for n in (1, 2, 3)}
    doc = _doc(pages)
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert not any("Execution Version" in c.section for c in chunks)
    assert any("Execution Version" in c.text for c in chunks)          # kept as body text
    assert not any("ACME CREDIT AGREEMENT" in c.text for c in chunks)
    assert [c.section for c in chunks] == ["1.01 Item.", "2.01 Item.", "3.01 Item."]


def test_heading_without_body_and_small_chunks_are_merged():
    doc = _doc({1: [("section_header", "2.02 [Reserved]."), ("section_header", "2.03 Fees."),
                    ("text", "Fees are payable quarterly."), ("section_header", "2.04 Interest."),
                    ("text", PROSE * 2)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert len(chunks) == 1
    assert "2.02 [Reserved]." in chunks[0].text and "Fees are payable quarterly." in chunks[0].text
    assert chunks[0].meta["merged"] == 3


def test_small_table_kept_whole_with_prefix():
    rows = [("Tier", "Ratio", "Margin"), ("I", "< 1.50:1", "1.25%"), ("II", ">= 1.50:1", "1.50%")]
    doc = _doc({1: [("section_header", "Pricing Grid"), ("table", rows)]})
    (c,) = C4LayoutAware().chunk(doc)
    _check(doc, [c])
    assert c.text == ("Pricing Grid: | Tier | Ratio | Margin |\n| --- | --- | --- |\n"
                      "| I | < 1.50:1 | 1.25% |\n| II | >= 1.50:1 | 1.50% |")
    assert c.meta["kind"] == "table" and c.meta["pieces"] == 1


def test_big_table_split_by_rows_with_header_repeated():
    rows = [("Lender", "Commitment", "Percentage")] + [
        (f"Bank number {i} National Association", f"${i},000,000.00", f"{i}.123456789%") for i in range(60)]
    doc = _doc({1: [("section_header", "SCHEDULE 2.01"), ("table", rows)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert len(chunks) > 1
    seen = []
    for c in chunks:
        lines = c.text.split("\n")
        assert lines[0] == "SCHEDULE 2.01: | Lender | Commitment | Percentage |"
        assert lines[1] == "| --- | --- | --- |"
        assert c.meta["header_repeated"] and c.meta["pieces"] == len(chunks)
        seen += lines[2:]
    assert seen == ["| " + " | ".join(r) + " |" for r in rows[1:]]       # every row once, in order


def test_overlong_row_is_cut_with_sub_spans():
    rows = [("Item", "Text"), ("1", " ".join(f"word{i}" for i in range(400)))]
    doc = _doc({1: [("table", rows)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert len(chunks) >= 2
    row_spans = [s for c in chunks for s in c.spans if s.start > 0]
    assert len(row_spans) >= 2 and row_spans[0].end <= row_spans[1].start


def test_prefix_is_capped():
    long_head = "ARTICLE IX " + " ".join(["MISCELLANEOUS PROVISIONS AND OTHER MATTERS"] * 20)
    doc = _doc({1: [("section_header", long_head), ("section_header", "9.01 Notices."), ("text", PROSE * 3)]})
    chunks = C4LayoutAware().chunk(doc)
    _check(doc, chunks)
    assert chunks[0].text.startswith("9.01 Notices.: ")
    assert chunks[0].meta["prefix_cut"] and chunks[0].meta["prefix_tokens"] <= 64


def test_form_fields_grouped():
    doc = _doc({1: [("text", PROSE)]}, form_fields=[("Loan Amount", "$162,000"), ("Rate", "3.875%"),
                                                    ("Empty", "")])
    chunks = C4LayoutAware().chunk(doc)
    form = [c for c in chunks if c.meta["kind"] == "form"]
    assert len(form) == 1 and form[0].text == "Form Fields: Loan Amount: $162,000\nRate: 3.875%"


def test_requires_layout_elements():
    with pytest.raises(ValueError):
        C4LayoutAware().chunk(SourceDoc("d", {1: "text"}))


def test_locate_elements_in_cleaned_text():
    view = PageView(page=1, size=(612, 792), elements=[
        {"id": "e1", "class": "section_header", "text": "﻿ARTICLE I", "bbox": [], "ref": ""},
        {"id": "e2", "class": "text", "text": "Shinhan Bank", "bbox": [], "ref": ""},   # cleaner removes
        {"id": "e3", "class": "text", "text": "Body text.", "bbox": [], "ref": ""},
        {"id": "e4", "class": "table", "text": "", "bbox": [], "ref": ""}],
        tables=[{"element_id": "e4"}])
    page = "﻿ARTICLE I\nBody text.\nA B"
    ts = TableSource(1, 0, "| A | B |", [["A", "B"]], [Span(1, 22, 25)])
    els, missing = locate_elements({1: view}, {1: page}, [ts])
    assert missing == 1
    assert [(e.cls, e.text, e.span, e.table) for e in els] == [
        ("section_header", "ARTICLE I", Span(1, 1, 10), None), ("text", "Shinhan Bank", None, None),
        ("text", "Body text.", Span(1, 11, 21), None), ("table", "", None, 0)]
