"""Tests for the agent settle-grid helper and the S0 report renderer."""
from evals.golden.ingest.schema import _table_errors
from evals.golden.ingest.settle_tables_agent import grid, to_rows
from evals.ingest_metrics.s0_report import render


def _table(g):
    return {"id": "t1", "bbox": [0, 0, 10, 10], **g}


def test_grid_spans_and_padding():
    g = grid([[("Title", 3)], [("Left", 1, 2), "a", "b"], ["c"]], [0])
    assert (g["num_rows"], g["num_cols"]) == (3, 3)
    assert _table_errors(_table(g), 100, 100) == []
    row2 = [c for c in g["cells"] if c["row"] == 2]
    assert [(c["col"], c["text"]) for c in row2] == [(1, "c"), (2, "")]   # col 0 covered by row span


def test_to_rows_round_trip():
    g = grid([["a", ("b", 2)], [("c", 1, 1), "d", "e"]])
    assert grid(to_rows(g))["cells"] == g["cells"]


def _summary():
    test = {"diff": 0.1, "lo": 0.02, "hi": 0.2, "p": 0.02, "n": 5, "winner": "P2", "rule": "CI excludes 0"}
    cfg = {"all": {"pages": 2, "cell_accuracy": {"point": 0.5, "lo": 0.4, "hi": 0.6, "n": 2},
                   "table_detection": {"f1": 1.0}, "class_f1": {}}}
    return {"dry_run": False,
            "golden": {"labels": 2, "verified": 2, "configs": {"P1": cfg, "P2": cfg},
                       "decision": test, "h10": {**test, "verdict": "supported"}},
            "corpus": {"configs": {"P1": {"edgar": {"pages": 3, "pages_per_minute": 10.0}, "missing_parse": []}},
                       "h6": {"P1": {"pairs": 3, "scan_cer_median": 0.05, "source_cer_median": 0.07,
                                     "ratio": 0.7, "supported": False}}}}


def test_render_contains_decision_and_skips_non_part_keys():
    md = render(_summary())
    assert "**P2**" in md and "Verdict: **supported**" in md
    assert "| P1 | edgar | 3 |" in md
    assert "missing_parse" not in md
