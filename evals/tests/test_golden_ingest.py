"""Tests for the ingestion golden-set schema validator and Docling pre-fill helpers."""
import copy

import pytest

from evals.golden.ingest.labeling import (
    _table_from_docling, prefill_from_docling, reading_order_refs, to_topleft,
)
from evals.golden.ingest.schema import map_docling_label, validate_labels


def _labels(**over):
    base = {
        "page_id": "d_p001", "doc_id": "d", "page": 1, "stratum": "table_heavy",
        "page_size": [612.0, 792.0], "status": "prefilled", "from_scratch": False,
        "reference_text_source": "text_layer", "reference_text": None,
        "layout_elements": [{"id": "e1", "class": "text", "bbox": [10, 10, 100, 50]},
                            {"id": "e2", "class": "table", "bbox": [10, 60, 300, 200]}],
        "reading_order": ["e1", "e2"],
        "tables": [{"id": "t1", "element_id": "e2", "bbox": [10, 60, 300, 200], "num_rows": 2,
                    "num_cols": 2, "header_rows": [0],
                    "cells": [{"row": 0, "col": 0, "row_span": 1, "col_span": 2, "text": "Fees"},
                              {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "A"},
                              {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "$5"}]}],
        "key_values": [], "header_footer_lines": [],
    }
    base.update(over)
    return base


class TestValidator:
    def test_valid_labels_pass(self):
        assert validate_labels(_labels()) == []

    def test_overlapping_span_detected(self):
        lab = _labels()
        lab["tables"][0]["cells"].append({"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "x"})
        assert any("overlapping" in e for e in validate_labels(lab))

    def test_missing_cell_detected(self):
        lab = _labels()
        lab["tables"][0]["cells"].pop()
        assert any("no cell" in e for e in validate_labels(lab))

    def test_cell_outside_grid(self):
        lab = _labels()
        lab["tables"][0]["cells"][1]["row_span"] = 2
        assert any("outside" in e for e in validate_labels(lab))

    def test_reading_order_bad_refs(self):
        errs = validate_labels(_labels(reading_order=["e1", "e9", "e1"]))
        assert any("missing element e9" in e for e in errs)
        assert any("twice" in e for e in errs)

    def test_bbox_outside_page(self):
        lab = _labels()
        lab["layout_elements"][0]["bbox"] = [10, 10, 700, 50]
        assert any("outside page" in e for e in validate_labels(lab))

    def test_unknown_class_and_verified_needs_labeler(self):
        lab = _labels(status="verified")
        lab["layout_elements"][0]["class"] = "banner"
        errs = validate_labels(lab)
        assert any("class" in e for e in errs)
        assert any("labeled_by" in e for e in errs)

    def test_missing_keys(self):
        lab = _labels()
        del lab["tables"]
        assert validate_labels(lab)[0].startswith("missing keys")


def test_label_mapping():
    assert map_docling_label("list_item") == "list"
    assert map_docling_label("footnote") == "text"
    assert map_docling_label("checkbox_selected") == "other"


def test_to_topleft_converts_bottomleft():
    assert to_topleft({"l": 10, "t": 700, "r": 50, "b": 680, "coord_origin": "BOTTOMLEFT"}, 792) == [10, 92, 50, 112]
    assert to_topleft({"l": 10, "t": 92, "r": 50, "b": 112, "coord_origin": "TOPLEFT"}, 792) == [10, 92, 50, 112]


def _doc():
    """Minimal DoclingDocument dict: header, heading, table (with a child text), footer number."""
    prov = lambda t, b: [{"page_no": 1, "bbox": {"l": 10, "t": t, "r": 200, "b": b, "coord_origin": "BOTTOMLEFT"}}]
    return {
        "pages": {"1": {"size": {"width": 612, "height": 792}}},
        "body": {"children": [{"$ref": "#/texts/1"}, {"$ref": "#/groups/0"}, {"$ref": "#/texts/3"}]},
        "furniture": {"children": [{"$ref": "#/texts/0"}]},
        "groups": [{"self_ref": "#/groups/0", "children": [{"$ref": "#/tables/0"}]}],
        "texts": [
            {"self_ref": "#/texts/0", "label": "page_header", "text": "ACME CREDIT AGREEMENT", "prov": prov(780, 770)},
            {"self_ref": "#/texts/1", "label": "section_header", "text": "Fees", "prov": prov(700, 690)},
            {"self_ref": "#/texts/2", "label": "text", "text": "in table", "prov": prov(600, 590)},
            {"self_ref": "#/texts/3", "label": "page_footer", "text": "12", "prov": prov(30, 20)},
        ],
        "tables": [{"self_ref": "#/tables/0", "label": "table", "children": [{"$ref": "#/texts/2"}],
                    "prov": prov(650, 500),
                    "data": {"num_rows": 2, "num_cols": 2, "table_cells": [
                        {"start_row_offset_idx": 0, "end_row_offset_idx": 1, "start_col_offset_idx": 0,
                         "end_col_offset_idx": 2, "text": "Fees", "column_header": True},
                        {"start_row_offset_idx": 1, "end_row_offset_idx": 2, "start_col_offset_idx": 1,
                         "end_col_offset_idx": 2, "text": "$5"}]}}],
        "pictures": [],
    }


def test_reading_order_follows_body_then_furniture_and_drops_table_children():
    assert reading_order_refs(_doc()) == ["#/texts/1", "#/tables/0", "#/texts/3", "#/texts/0"]


def test_prefill_builds_valid_labels():
    out = prefill_from_docling(_doc(), page=1)
    classes = [e["class"] for e in out["layout_elements"]]
    assert classes == ["section_header", "table", "page_number", "page_header"]   # "12" -> page_number
    assert out["header_footer_lines"] == ["12", "ACME CREDIT AGREEMENT"]
    t = out["tables"][0]
    assert t["header_rows"] == [0]
    assert len(t["cells"]) == 3          # header span + "$5" + filled empty (1,0)
    lab = _labels(**out)
    assert validate_labels(lab) == []
