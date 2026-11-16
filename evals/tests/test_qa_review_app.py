"""QA draft review app: store logic + HTTP handler (evals/golden/qa/review_app/). No network, no corpus."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from evals.golden.qa.review_app.server import make_handler
from evals.golden.qa.review_app.store import ReviewStore, clean_final, draft_hash, highlights
from evals.golden.qa.schema import DictText, DocInfo

DOCS = {
    "d1_ca": DocInfo("d1_ca", "edgar", "d1", "original"),
    "d1_am1": DocInfo("d1_am1", "edgar", "d1", "amendment_1"),
}
TEXT = DictText({
    "d1_ca": {1: "SECTION 2.10 Fees.\nThe Borrower shall pay a commitment fee of 0.25% per annum\non the unused Commitments.",
              2: "The Maturity Date is June 30, 2024."},
    "d1_am1": {1: "Section 2.10 is amended: a commitment fee of 0.35% per annum."},
})


def cand(cid="lookup-001-a", **kw):
    c = {"cid": cid, "job_id": cid[:-2], "type": "lookup", "question": "What is the commitment fee rate?",
         "session_id": "d1", "session_docs": ["d1_am1", "d1_ca"],
         "evidence": [{"doc_id": "d1_ca", "page": 1, "span_text": "a commitment fee of 0.25% per annum"}],
         "reference_answer": "The commitment fee is 0.25% per annum.", "key_points": ["0.25% per annum"],
         "entities": ["0.25%"], "superseded_values": [], "amendment": False, "answer_numeric": 0.25,
         "numeric_tolerance": None, "unanswerable": False, "provenance": "llm_drafted_human_verified",
         "drafted_by": "m", "drafted_on": "2026-11-16", "shown_pages": [["d1_ca", 1]], "pair_note": None,
         "draft_issues": [], "review": None}
    c.update(kw)
    return c


@pytest.fixture
def store(tmp_path):
    cands = tmp_path / "candidates.jsonl"
    cands.write_text("".join(json.dumps(c) + "\n" for c in [
        cand(), cand("lookup-001-b", question="When is the Maturity Date?", evidence=[
            {"doc_id": "d1_ca", "page": 2, "span_text": "The Maturity Date is June 30, 2024."}],
            reference_answer="June 30, 2024.", entities=["June 30, 2024"], answer_numeric=None,
            shown_pages=[["d1_ca", 2]])]))
    return ReviewStore(cands, tmp_path / "reviews.json", DOCS, TEXT, "kousik")


def test_list_and_get(store):
    out = store.list()
    assert [i["status"] for i in out["items"]] == ["todo", "todo"]
    assert out["kept_by_type"]["lookup"] == 0 and out["total_quota"] == 120
    g = store.get("lookup-001-a")
    assert g["status"] == "todo" and g["pages"][0]["doc_id"] == "d1_ca"
    (s, e, i), = g["pages"][0]["highlights"]
    assert TEXT.page("d1_ca", 1)[s:e].startswith("a commitment fee") and i == 0
    assert g["pages"][0]["has_next"] and not g["pages"][0]["has_prev"]


def test_highlights_whitespace_insensitive():
    text = "pay a commitment\nfee of 0.25%  per annum"
    (s, e, i), = highlights(text, [(0, "a commitment fee of 0.25% per annum")])
    assert text[s:e].startswith("a commitment") and text[s:e].endswith("annum") and i == 0
    assert highlights(text, [(0, "not there")]) == []


def test_keep_as_is_and_edited(store):
    ok, errs = store.save("lookup-001-a", {"decision": "keep"})
    assert ok and errs == []
    rev = store.reviews()["lookup-001-a"]
    assert rev["decision"] == "keep" and not rev["edited"] and rev["reviewed_by"] == "kousik"
    final = {**store.get("lookup-001-b")["draft"], "question": "What is the Maturity Date of the facility?"}
    ok, _ = store.save("lookup-001-b", {"decision": "keep", "final": final, "note": "more specific"})
    rev = store.reviews()["lookup-001-b"]
    assert ok and rev["edited"] and rev["final"]["question"].endswith("facility?") and rev["note"] == "more specific"
    assert store.list()["kept_by_type"]["lookup"] == 2
    assert store.list()["items"][1]["question"].endswith("facility?")


def test_keep_refused_with_f3_errors_but_later_saves(store):
    bad = {**store.get("lookup-001-a")["draft"],
           "evidence": [{"doc_id": "d1_ca", "page": 1, "span_text": "a fee of 0.25%"}]}   # not verbatim
    ok, errs = store.save("lookup-001-a", {"decision": "keep", "final": bad})
    assert not ok and any("span_text not found" in e for e in errs)
    assert "lookup-001-a" not in store.reviews()
    ok, errs = store.save("lookup-001-a", {"decision": "later", "final": bad})
    assert ok and errs and store.reviews()["lookup-001-a"]["decision"] == "later"


def test_drop_needs_reason_and_bad_decision(store):
    assert store.save("lookup-001-a", {"decision": "drop"}) == (False, ["give a drop reason"])
    ok, _ = store.save("lookup-001-a", {"decision": "drop", "drop_reason": "trivial / not useful"})
    assert ok and store.list()["counts"] == {"drop": 1, "todo": 1}
    ok, errs = store.save("lookup-001-a", {"decision": "maybe"})
    assert not ok and "decision" in errs[0]


def test_stale_when_draft_changes(store, tmp_path):
    store.save("lookup-001-a", {"decision": "keep"})
    lines = store.candidates_path.read_text().splitlines()
    c = json.loads(lines[0])
    c["question"] = "Re-drafted question about the commitment fee rate?"
    store.candidates_path.write_text("\n".join([json.dumps(c)] + lines[1:]) + "\n")
    assert store.list()["items"][0]["status"] == "stale"
    assert store.reviews()["lookup-001-a"]["draft_hash"] != draft_hash(c)


def test_clean_final_coerces_form_values():
    f = clean_final({"question": " Q? ", "evidence": [{"doc_id": "d1_ca", "page": "2", "span_text": "x"}],
                     "key_points": ["a", " ", "b"], "entities": None, "answer_numeric": "$1,250.50",
                     "numeric_tolerance": ""})
    assert f["question"] == "Q?" and f["evidence"][0]["page"] == 2 and f["key_points"] == ["a", "b"]
    assert f["entities"] == [] and f["answer_numeric"] == 1250.5 and f["numeric_tolerance"] is None
    assert clean_final({"answer_numeric": "abc"})["answer_numeric"] == "abc"     # validator reports it


def test_unknown_ids_rejected(store):
    for bad in ("../../etc/passwd", "lookup-999-a", "LOOKUP-001-A"):
        with pytest.raises(KeyError):
            store.get(bad)
    with pytest.raises(KeyError):
        store.page_view("nope", 1)


def test_http_roundtrip(store):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None):
        req = urllib.request.Request(base + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        assert call("GET", "/")[0] == 200
        st, body = call("GET", "/api/list")
        assert st == 200 and len(json.loads(body)["items"]) == 2
        st, body = call("GET", "/api/page/d1_ca/2")
        assert st == 200 and "June 30" in json.loads(body)["text"]
        assert call("GET", "/api/cand/nope")[0] == 404
        st, body = call("POST", "/api/check/lookup-001-a", {"question": ""})
        assert st == 200 and json.loads(body)["errors"]
        st, body = call("PUT", "/api/review/lookup-001-a", {"decision": "keep", "final": {"question": ""}})
        assert st == 422 and not json.loads(body)["saved"]
        st, body = call("PUT", "/api/review/lookup-001-a", {"decision": "keep"})
        assert st == 200 and json.loads(body)["saved"]
    finally:
        httpd.shutdown()
