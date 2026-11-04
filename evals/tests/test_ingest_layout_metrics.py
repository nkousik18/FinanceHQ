"""F8 tests for layout, key-value, header/footer and confidence metrics (known answers)."""
import pytest

from evals.ingest_metrics.confidence import document_ingestion_score, page_confidence, summarize
from evals.ingest_metrics.layout import (
    class_f1, header_footer, kendall_tau, key_values, reading_order_tau, section_header_recall,
)


def E(i, cls, y):
    return {"id": i, "class": cls, "bbox": [0, y, 100, y + 10]}


LAB = [E("l1", "section_header", 0), E("l2", "text", 20), E("l3", "text", 40), E("l4", "page_number", 90)]


class TestClassF1:
    def test_perfect(self):
        out = class_f1(LAB, [dict(e) for e in LAB])
        assert out["macro_f1"] == 1.0

    def test_one_misclassified(self):
        pred = [E("p1", "text", 0), E("p2", "text", 20), E("p3", "text", 40), E("p4", "page_number", 90)]
        out = class_f1(LAB, pred)
        assert out["per_class"]["section_header"]["recall"] == 0.0
        assert out["per_class"]["text"]["precision"] == pytest.approx(2 / 3)
        assert out["per_class"]["text"]["recall"] == 1.0

    def test_box_too_far_is_a_miss(self):
        out = class_f1([E("l", "text", 0)], [E("p", "text", 8)])     # IoU 2/18 < 0.5
        assert out["per_class"]["text"]["tp"] == 0


def test_section_header_recall_counts_title_as_heading():
    lab = [E("a", "section_header", 0), E("b", "title", 20), E("c", "section_header", 40)]
    pred = [E("x", "title", 0), E("y", "section_header", 20), E("z", "text", 40)]
    assert section_header_recall(lab, pred) == {"hit": 2, "total": 3, "recall": pytest.approx(2 / 3)}


class TestReadingOrder:
    def test_kendall_known_values(self):
        assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
        assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
        assert kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
        assert kendall_tau([1], [1]) is None

    def test_two_column_page_read_across_rows(self):
        # labels: left column (a, b) then right column (c, d); parser reads a, c, b, d
        lab = [{"id": "a", "class": "text", "bbox": [0, 0, 40, 10]}, {"id": "b", "class": "text", "bbox": [0, 20, 40, 30]},
               {"id": "c", "class": "text", "bbox": [60, 0, 100, 10]}, {"id": "d", "class": "text", "bbox": [60, 20, 100, 30]}]
        pred = [dict(e, id="p" + e["id"]) for e in lab]
        out = reading_order_tau(lab, ["a", "b", "c", "d"], pred, ["pa", "pc", "pb", "pd"])
        assert out == {"tau": pytest.approx(2 / 3), "n": 4}

    def test_furniture_ignored(self):
        out = reading_order_tau(LAB, ["l1", "l2", "l3", "l4"], [dict(e) for e in LAB], ["l4", "l1", "l2", "l3"])
        assert out == {"tau": 1.0, "n": 3}


def test_header_footer_multiset_text_match():
    out = header_footer(["ACME CREDIT AGREEMENT", "12"], ["Acme Credit Agreement", "Page 12"])
    assert (out["tp"], out["precision"], out["recall"]) == (1, 0.5, 0.5)


class TestKeyValues:
    LAB = [{"key": "Loan Amount", "value": "$162,000"}, {"key": "Loan Term", "value": "30 years"},
           {"key": "Purpose", "value": "Purchase"}]

    def test_no_predictions(self):
        out = key_values(self.LAB, [])
        assert out["recall"] == 0.0 and out["precision"] is None and out["value_exact_rate"] is None

    def test_partial(self):
        pred = [{"key": "Loan Amount:", "value": "$162,000"}, {"key": "Loan Term", "value": "15 years"},
                {"key": "Rate Lock", "value": "No"}]
        out = key_values(self.LAB, pred)
        assert (out["tp"], out["value_exact"]) == (2, 1)
        assert out["value_exact_rate"] == 0.5


class TestConfidence:
    CACHED = {
        "ocr_cells": {"1": [{"text": "Loan", "confidence": 0.95}, {"text": "Amount", "confidence": 0.85},
                            {"text": "$162,000", "confidence": 0.70}, {"text": " ", "confidence": 0.1}]},
        "confidence": {"pages": {"1": {"parse_score": None, "ocr_score": 0.83, "low_grade": "fair"},
                                 "2": {"parse_score": 1.0, "low_grade": "excellent"}}},
    }

    def test_summary(self):
        s = summarize([0.95, 0.85, 0.70])
        assert s["n"] == 3 and s["min"] == 0.70 and s["mean"] == pytest.approx(2.5 / 3)

    def test_page_rates_ignore_blank_cells(self):
        out = page_confidence(self.CACHED, 1)
        assert out["ocr"]["n"] == 3
        assert out["low_conf_rate_90"] == pytest.approx(2 / 3)
        assert out["low_conf_rate_80"] == pytest.approx(1 / 3)
        assert out["low_confidence_page"] is True          # p10 below 0.80

    def test_digital_page_has_no_ocr_numbers(self):
        out = page_confidence(self.CACHED, 2)
        assert out["ocr"]["n"] == 0 and out["low_conf_rate_90"] is None and out["low_confidence_page"] is False

    def test_document_score_kinds(self):
        assert document_ingestion_score(self.CACHED) == {"kind": "ocr", "score": pytest.approx(1 / 3), "n_words": 3}
        digital = {"confidence": {"pages": {"1": {"parse_score": 1.0}, "2": {"parse_score": 0.8}}}}
        assert document_ingestion_score(digital)["score"] == pytest.approx(0.9)
