"""F8 tests for box matching and table metrics, on hand-built tables with known answers."""
import copy

import pytest

from evals.ingest_metrics.geometry import greedy_match, iou, prf
from evals.ingest_metrics.tables import (
    cell_accuracy, cell_accuracy_counts, detection, header_rows_correct, teds,
)


def _t(bbox, grid, header_rows=(0,), spans=None):
    """grid: list of rows of cell texts. spans: {(r, c): (rs, cs)} for merged cells (covered cells omitted)."""
    spans = spans or {}
    covered = {(r + dr, c + dc) for (r, c), (rs, cs) in spans.items() for dr in range(rs) for dc in range(cs)} \
        - set(spans)
    cells = []
    for r, row in enumerate(grid):
        for c, txt in enumerate(row):
            if (r, c) in covered:
                continue
            rs, cs = spans.get((r, c), (1, 1))
            cells.append({"row": r, "col": c, "row_span": rs, "col_span": cs, "text": txt})
    return {"bbox": bbox, "num_rows": len(grid), "num_cols": len(grid[0]),
            "header_rows": list(header_rows), "cells": cells}


GOLD = _t([0, 0, 100, 100], [["Fee", "Amount"], ["Appraisal", "$405"]])   # the 2x2 table


class TestGeometry:
    def test_iou_known_values(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
        assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
        assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_greedy_is_one_to_one_best_first(self):
        labels = [{"bbox": [0, 0, 10, 10]}, {"bbox": [0, 0, 10, 9]}]
        preds = [{"bbox": [0, 0, 10, 10]}]
        assert greedy_match(labels, preds) == [(0, 0, 1.0)]

    def test_prf_handles_empty(self):
        assert prf(0, 0, 3) == {"precision": None, "recall": 0.0, "f1": 0.0, "tp": 0, "n_pred": 0, "n_label": 3}
        assert prf(2, 4, 2)["f1"] == pytest.approx(2 * 0.5 * 1 / 1.5)


class TestCellAccuracy:
    def test_perfect(self):
        assert cell_accuracy_counts(GOLD, copy.deepcopy(GOLD)) == (4, 4)

    def test_one_wrong_number(self):
        pred = copy.deepcopy(GOLD)
        pred["cells"][3]["text"] = "$406"
        assert cell_accuracy_counts(GOLD, pred) == (3, 4)

    def test_containment_and_case_punct_ignored(self):
        pred = copy.deepcopy(GOLD)
        pred["cells"][2]["text"] = "APPRAISAL:"
        assert cell_accuracy_counts(GOLD, pred) == (4, 4)

    def test_shifted_column_counts_wrong(self):
        pred = _t([0, 0, 100, 100], [["", "Fee", "Amount"], ["", "Appraisal", "$405"]])
        assert cell_accuracy_counts(GOLD, pred) == (0, 4)

    def test_missed_table_scores_zero_and_empty_cells_skipped(self):
        gold = _t([0, 0, 100, 100], [["Fee", ""], ["Appraisal", "$405"]])
        out = cell_accuracy([gold], [])
        assert out["total"] == 3 and out["correct"] == 0 and out["per_table"][0]["matched"] is False

    def test_merged_cell_lookup(self):
        gold = _t([0, 0, 100, 100], [["Loan Costs", ""], ["A", "$1"]], spans={(0, 0): (1, 2)})
        pred = copy.deepcopy(gold)
        assert cell_accuracy_counts(gold, pred) == (3, 3)


class TestTeds:
    def test_identical_is_one(self):
        assert teds(GOLD, copy.deepcopy(GOLD)) == pytest.approx(1.0)

    def test_missed_is_zero(self):
        assert teds(GOLD, None) == 0.0

    def test_structure_error_lowers_score(self):
        merged = _t([0, 0, 100, 100], [["Fee", "Amount"], ["Appraisal", "$405"]], spans={(0, 0): (1, 2)})
        assert 0.0 < teds(GOLD, merged) < 1.0

    def test_text_error_costs_less_than_structure_error(self):
        typo = copy.deepcopy(GOLD); typo["cells"][3]["text"] = "$406"
        extra_row = _t([0, 0, 100, 100], [["Fee", "Amount"], ["Appraisal", "$405"], ["Total", "$405"]])
        assert teds(GOLD, typo) > teds(GOLD, extra_row)


def test_detection_counts():
    preds = [copy.deepcopy(GOLD), _t([200, 200, 300, 300], [["x"]])]
    d = detection([GOLD], preds)
    assert (d["tp"], d["precision"], d["recall"]) == (1, 0.5, 1.0)


def test_header_rows():
    assert header_rows_correct(GOLD, copy.deepcopy(GOLD)) is True
    pred = copy.deepcopy(GOLD); pred["header_rows"] = []
    assert header_rows_correct(GOLD, pred) is False
    assert header_rows_correct(_t([0, 0, 1, 1], [["a"]], header_rows=()), _t([0, 0, 1, 1], [["a"]], header_rows=())) is None
