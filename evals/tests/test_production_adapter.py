"""Docling cache -> production ExtractionResult adapter, and the cleaning/validation metrics."""
import logging

import structlog

from evals.ingest_metrics.cleaning import _value_repairs, aggregate, run_doc
from evals.parsers.production_adapter import _page_confidence, to_extraction_result

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _bbox(l, t, r, b):
    return {"l": l, "t": t, "r": r, "b": b, "coord_origin": "TOPLEFT"}


def _cached(ocr=None):
    """A 3-page Docling document: page 1 text + a 2x2 table, page 2 blank, page 3 text."""
    texts = [
        {"self_ref": "#/texts/0", "label": "text", "text": "Loan amount 110 000 dollars",
         "prov": [{"page_no": 1, "bbox": _bbox(10, 10, 200, 20)}]},
        {"self_ref": "#/texts/1", "label": "text", "text": "Shinhan Bank",
         "prov": [{"page_no": 3, "bbox": _bbox(10, 10, 200, 20)}]},
        {"self_ref": "#/texts/2", "label": "text", "text": "Rate 5.0%",
         "prov": [{"page_no": 3, "bbox": _bbox(10, 30, 200, 40)}]},
    ]
    table = {"self_ref": "#/tables/0", "label": "table", "children": [],
             "prov": [{"page_no": 1, "bbox": _bbox(10, 50, 200, 100)}],
             "data": {"num_rows": 2, "num_cols": 2, "table_cells": [
                 {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0,
                  "end_col_offset_idx": 1, "text": "Pages", "column_header": True},
                 {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 1,
                  "end_col_offset_idx": 2, "text": "29 29"},
                 {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1,
                  "end_col_offset_idx": 2, "text": "1."}]}}
    body = [{"$ref": "#/texts/0"}, {"$ref": "#/tables/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}]
    doc = {"body": {"children": body}, "furniture": {"children": []}, "texts": texts, "tables": [table],
           "pictures": [], "groups": [],
           "pages": {str(i): {"size": {"width": 612, "height": 792}} for i in (1, 2, 3)}}
    return {"document": doc, "ocr_cells": ocr or {}}


def test_page_confidence_weights_text_layer_words_as_100():
    assert _page_confidence(10, []) == (100.0, 0)
    assert _page_confidence(4, [0.5, 0.9]) == (round((2 * 100 + 50 + 90) / 4, 2), 1)
    assert _page_confidence(0, []) == (0.0, 0)


def test_adapter_mirrors_textract_shape():
    er = to_extraction_result(_cached(), "doc1")
    assert [p.page_number for p in er.pages] == [1, 3]          # blank page 2 dropped
    assert er.total_pages == 2 and er.form_fields == []
    t = er.tables[0]
    assert (t.page, t.rows, t.cols) == (1, 2, 2)
    assert {(c.row, c.col) for c in t.cells} == {(1, 1), (1, 2), (2, 1), (2, 2)}   # 1-based, complete
    assert "[TABLE]" in er.full_text and "--- Page Break ---" in er.full_text
    assert er.overall_avg_confidence == 100.0


def test_adapter_uses_ocr_confidence():
    er = to_extraction_result(_cached({"3": [{"text": "Rate", "confidence": 0.4}]}), "doc1")
    p3 = next(p for p in er.pages if p.page_number == 3)
    assert p3.avg_confidence < 100 and p3.low_confidence_words == 1


def test_value_repairs_flag_corruption_only_when_numbers_change():
    assert _value_repairs("29 29") == {"number_space": 1, "number_corrupted": 1, "digit_letter": 0}
    assert _value_repairs("1.") == {"number_space": 1, "number_corrupted": 0, "digit_letter": 0}
    assert _value_repairs("36MONTHS")["digit_letter"] == 1
    assert _value_repairs("Section 4.1") == {"number_space": 0, "number_corrupted": 0, "digit_letter": 0}


def test_run_doc_counts_repairs_overcleaning_and_validation():
    er = to_extraction_result(_cached(), "doc1")
    refs = {3: "Rate 5.0%"}                      # header line removal makes page 3 closer -> not over-cleaned
    m = run_doc(er, lambda p: refs.get(p))
    assert m["repairs"]["header:shinhan_bank"] == 1
    assert m["repairs"]["table:number_space"] == 2 and m["repairs"]["table:number_corrupted"] == 1
    assert m["overclean_pages"] == [] and m["overclean_checked_pages"] == 2
    assert m["chars_removed"] > 0
    assert m["validation"]["passed"] is True     # has a table, confidence 100
    assert m["validation"]["codes"]["warning:MISSING_REQUIRED_FIELD"] == 10


def test_run_doc_flags_overcleaning_when_cleaner_removes_real_text():
    er = to_extraction_result(_cached(), "doc1")
    m = run_doc(er, lambda p: "Shinhan Bank\nRate 5.0%" if p == 3 else None)
    assert m["overclean_pages"] == [3]


def test_aggregate_rates():
    er = to_extraction_result(_cached(), "doc1")
    a = aggregate([run_doc(er), run_doc(er)])
    assert a["documents"] == 2 and a["repairs"]["header:shinhan_bank"] == 2
    assert a["table_cell_corruption_rate"] == 2 / 8
    assert a["docs_with_code"]["warning:MISSING_REQUIRED_FIELD"] == 2
