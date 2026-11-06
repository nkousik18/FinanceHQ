"""Tests for the P1 vs P2 table comparison used to settle the golden set (app/compare.py)."""
import copy
import json
from types import SimpleNamespace

from evals.golden.ingest.app.compare import compare_page, differing_cells, review_state
from evals.golden.ingest.app.server import LabelStore
from evals.tests.test_golden_ingest import _labels


def _table(texts_row1=("A", "$5"), header_span=2):
    lab = _labels()
    t = copy.deepcopy(lab["tables"][0])
    t["cells"][0]["col_span"] = header_span
    if header_span == 1:
        t["cells"].insert(1, {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": ""})
    t["cells"][-2]["text"], t["cells"][-1]["text"] = texts_row1
    return t


def _view(*tables):
    return SimpleNamespace(tables=list(tables))


def test_identical_tables_have_no_differences():
    assert differing_cells(_table(), _table()) == []


def test_text_and_span_differences_are_flagged():
    assert differing_cells(_table(("A", "$6")), _table()) == [[1, 1]]
    assert differing_cells(_table(header_span=1), _table()) == [[0, 0], [0, 1]]


def test_missing_other_table_flags_every_cell():
    assert len(differing_cells(_table(), None)) == 3


def test_compare_page_marks_disagreement_and_review_state():
    lab = _labels()
    cmp = compare_page(lab, {"P1": _view(_table()), "P2": _view(_table(("A", "$6")))})
    t = cmp["tables"][0]
    assert t["disagree"] and t["P2_diff"] == [[1, 1]] and t["label_acc"]["P1"] == 1.0
    assert review_state(cmp) == "todo"
    lab["tables"][0]["adjudication"] = "P2"
    assert review_state(compare_page(lab, {"P1": _view(_table()), "P2": _view(_table(("A", "$6")))})) == "done"


def test_agreeing_parsers_and_from_scratch_need_no_review():
    same = {"P1": _view(_table()), "P2": _view(_table())}
    assert review_state(compare_page(_labels(), same)) is None
    differ = {"P1": _view(_table()), "P2": _view()}
    assert review_state(compare_page(_labels(from_scratch=True), differ)) is None


def test_store_reports_review_and_keeps_adjudication_on_save(tmp_path):
    pages, images = tmp_path / "pages", tmp_path / "images"
    pages.mkdir(); images.mkdir()
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"pages": [{"page_id": "d_p001", "stratum": "table_heavy", "from_scratch": False}]}))
    (pages / "d_p001.json").write_text(json.dumps(_labels()))
    views = lambda doc_id, page: {"P1": _view(_table()), "P2": _view(_table(("A", "$6")))}
    store = LabelStore(pages, images, sel, labeller="tester", views=views)
    assert store.list_pages()[0]["review"] == "todo"
    lab = store.load("d_p001")
    lab["tables"][0]["adjudication"] = "P1"
    assert store.save("d_p001", lab)[0]
    assert store.list_pages()[0]["review"] == "done"
    assert store.compare("d_p001")["tables"][0]["adjudication"] == "P1"
