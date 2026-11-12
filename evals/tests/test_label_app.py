"""Tests for the labelling app's store and HTTP API (runs a real server on a free port)."""
import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from evals.golden.ingest.app.server import LabelStore, make_handler
from evals.tests.test_golden_ingest import _labels


@pytest.fixture
def store(tmp_path):
    pages, images = tmp_path / "pages", tmp_path / "images"
    pages.mkdir(); images.mkdir()
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"pages": [
        {"page_id": "d_p001", "stratum": "table_heavy", "from_scratch": False},
        {"page_id": "x_p002", "stratum": "scanned", "from_scratch": True}]}))
    (pages / "d_p001.json").write_text(json.dumps(_labels()))
    (pages / "x_p002.json").write_text(json.dumps(_labels(page_id="x_p002", layout_elements=[], reading_order=[], tables=[])))
    (images / "d_p001.png").write_bytes(b"\x89PNG fake")
    return LabelStore(pages, images, sel, labeller="tester")


def test_list_puts_from_scratch_first(store):
    ids = [p["page_id"] for p in store.list_pages()]
    assert ids == ["x_p002", "d_p001"]


def test_save_draft_with_problems_is_allowed(store):
    lab = _labels(reading_order=["e1", "nope"])
    ok, errors = store.save("d_p001", lab)
    assert ok and errors


def test_verified_with_problems_is_refused_and_file_unchanged(store):
    before = store.load("d_p001")
    ok, errors = store.save("d_p001", _labels(status="verified", reading_order=["nope"]))
    assert not ok and errors
    assert store.load("d_p001") == before


def test_verified_fills_labeller_and_date(store):
    ok, errors = store.save("d_p001", _labels(status="verified"))
    assert ok and errors == []
    saved = store.load("d_p001")
    assert saved["labeled_by"] == "tester" and saved["verified_on"]


def test_unknown_or_unsafe_page_ids_rejected(store):
    for pid in ("../../etc/passwd", "not_in_selection"):
        with pytest.raises(KeyError):
            store.load(pid)
    ok, errs = store.save("d_p001", _labels(page_id="x_p002"))
    assert not ok and "does not match" in errs[0]


def test_http_api_roundtrip(store):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, method="GET", body=None):
        req = urllib.request.Request(base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    try:
        status, html = call("/")
        assert status == 200 and b"Page Labeller" in html
        assert json.loads(call("/api/meta")[1])["labeller"] == "tester"
        assert len(json.loads(call("/api/pages")[1])) == 2
        assert json.loads(call("/api/labels/d_p001")[1])["page_id"] == "d_p001"
        assert call("/img/d_p001.png")[1].startswith(b"\x89PNG")
        assert json.loads(call("/api/validate", "POST", _labels(reading_order=["zz"]))[1])["errors"]
        status, body = call("/api/labels/d_p001", "PUT", _labels(status="verified", reading_order=["zz"]))
        assert status == 422 and not json.loads(body)["saved"]
        assert call("/api/labels/../selection", "GET")[0] == 404
    finally:
        httpd.shutdown()
