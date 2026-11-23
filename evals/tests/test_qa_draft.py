"""R3 drafting: LLM call loop, span filter, ledger guards (evals/golden/qa/draft.py, evals/llm/ledger.py).
No network: a fake client stands in for Groq."""
import json
from types import SimpleNamespace as NS

import pytest

from evals.golden.qa import draft
from evals.golden.qa.pages import Job
from evals.golden.qa.schema import DictText, DocInfo
from evals.llm.ledger import BudgetExceeded, Ledger, cost_usd

DOCS = {
    "d1_ca": DocInfo("d1_ca", "edgar", "d1", "original"),
    "d1_am1": DocInfo("d1_am1", "edgar", "d1", "amendment_1"),
    "scan_01": DocInfo("scan_01", "simulated_scan", "d1", "scan_of:d1_ca:p2"),
}
TEXT = DictText({
    "d1_ca": {1: "SECTION 2.10 Fees.\nThe Borrower shall pay a commitment fee of 0.25% per annum on the unused Commitments.",
              2: "The Maturity Date is June 30, 2024."},
    "d1_am1": {1: "Section 2.10 of the Agreement is hereby amended to read: The Borrower shall pay a commitment fee of 0.35% per annum."},
    "scan_01": {1: "The Maturity Date is June 3O, 2024."},
})
LOOKUP = Job("lookup-001", "lookup", "edgar", "d1", (("d1_ca", 1),))
AMEND = Job("cross_doc-001", "cross_doc", "edgar", "d1", (("d1_am1", 1), ("d1_ca", 1)),
            note="Section 2.10 Fees", amendment=True)
UNANS = Job("unanswerable-001", "unanswerable", "edgar", "d1", (("d1_ca", 2),))


def cand(**kw):
    c = {"question": "What is the commitment fee rate?", "reference_answer": "The commitment fee is 0.25% per annum.",
         "evidence": [{"doc_id": "d1_ca", "page": 1, "span_text": "a commitment fee of 0.25% per annum"}],
         "key_points": ["0.25% per annum"], "entities": ["0.25%"], "superseded_values": [],
         "amendment": False, "answer_numeric": 0.25, "numeric_tolerance": None}
    c.update(kw)
    return c


# ---- prompt and estimate ----------------------------------------------------------------------
def test_messages_show_pages_and_type_rule():
    msgs = draft.messages(AMEND, DOCS, TEXT)
    user = msgs[1]["content"]
    assert "=== d1_am1 page 1 ===" in user and "=== d1_ca page 1 ===" in user
    assert user.index("d1_am1 page 1") < user.index("d1_ca page 1")
    assert "Section 2.10 Fees" in user and "superseded_values" in user
    scan = draft.messages(Job("lookup-002", "lookup", "scan", "d1", (("scan_01", 1),)), DOCS, TEXT)
    assert "simulated scan" in scan[1]["content"]


def test_estimate_counts_prompt_and_guess():
    est = draft.estimate([LOOKUP, AMEND], DOCS, TEXT)
    assert est["jobs"] == 2 and est["candidates"] == 4
    assert est["prompt_tokens"] > 500                          # system prompt alone is several hundred
    assert est["completion_tokens_guess"] == 2 * (2 * draft.EST_OUT_PER_CANDIDATE + draft.EST_REASONING_PER_CALL)


# ---- parse and check --------------------------------------------------------------------------
def test_parse_content_tolerates_fences():
    raw = '```json\n{"candidates": [{"question": "q"}, "junk"]}\n```'
    assert draft.parse_content(raw) == [{"question": "q"}]
    with pytest.raises(ValueError):
        draft.parse_content("no json here")


def test_good_candidate_passes_with_no_issues():
    c = draft.to_candidate(LOOKUP, 0, cand(), DOCS, "openai/gpt-oss-120b", "2026-11-14")
    assert c["cid"] == "lookup-001-a" and c["session_docs"] == ["d1_am1", "d1_ca", "scan_01"]
    assert draft.check(c, LOOKUP, DOCS, TEXT) == ([], [])


def test_whitespace_insensitive_span_passes():
    c = draft.to_candidate(LOOKUP, 0, cand(evidence=[{"doc_id": "d1_ca", "page": "1",
                                                      "span_text": "Fees.  The Borrower shall pay"}]), DOCS, "m", "d")
    assert draft.check(c, LOOKUP, DOCS, TEXT)[0] == []


@pytest.mark.parametrize("ev, reason", [
    ([{"doc_id": "d1_ca", "page": 1, "span_text": "a commitment fee of 0.30% per annum"}], "not found verbatim"),
    ([{"doc_id": "d1_ca", "page": 2, "span_text": "The Maturity Date is June 30, 2024."}], "not a shown page"),
    ([], "no evidence"),
])
def test_rejects(ev, reason):
    c = draft.to_candidate(LOOKUP, 0, cand(evidence=ev), DOCS, "m", "d")
    rej, _ = draft.check(c, LOOKUP, DOCS, TEXT)
    assert rej and reason in rej[0]


def test_unanswerable_rules():
    ok = draft.to_candidate(UNANS, 0, cand(evidence=[], entities=[], answer_numeric=None,
                                           reference_answer="The documents do not state it."), DOCS, "m", "d")
    assert ok["unanswerable"] and draft.check(ok, UNANS, DOCS, TEXT) == ([], [])
    bad = draft.to_candidate(UNANS, 0, cand(evidence=[{"doc_id": "d1_ca", "page": 2,
                                                       "span_text": "The Maturity Date is June 30, 2024."}]), DOCS, "m", "d")
    assert "unanswerable candidate has evidence" in draft.check(bad, UNANS, DOCS, TEXT)[0]


def test_amendment_candidate_and_review_issues():
    good = cand(question="What is the commitment fee after the amendment?",
                reference_answer="0.35% per annum.", entities=["0.35%"], superseded_values=["0.25%"],
                amendment=True, answer_numeric=0.35,
                evidence=[{"doc_id": "d1_am1", "page": 1, "span_text": "a commitment fee of 0.35% per annum"},
                          {"doc_id": "d1_ca", "page": 1, "span_text": "a commitment fee of 0.25% per annum"}])
    c = draft.to_candidate(AMEND, 1, good, DOCS, "m", "d")
    assert c["cid"] == "cross_doc-001-b" and c["amendment"] and c["superseded_values"] == ["0.25%"]
    assert draft.check(c, AMEND, DOCS, TEXT) == ([], [])
    one_doc = {**good, "evidence": good["evidence"][:1], "entities": ["9.99%"]}
    rej, issues = draft.check(draft.to_candidate(AMEND, 0, one_doc, DOCS, "m", "d"), AMEND, DOCS, TEXT)
    assert not rej                                            # kept for review, not rejected
    assert any("2+ documents" in i for i in issues) and any("9.99%" in i for i in issues)


def test_superseded_values_dropped_outside_amendment_jobs():
    c = draft.to_candidate(LOOKUP, 0, cand(superseded_values=["x"], amendment=True), DOCS, "m", "d")
    assert c["superseded_values"] == [] and c["amendment"] is False


def test_numbers_coerced():
    c = draft.to_candidate(LOOKUP, 0, cand(answer_numeric="$1,250.5", numeric_tolerance="0"), DOCS, "m", "d")
    assert c["answer_numeric"] == 1250.5 and c["numeric_tolerance"] == 0.0
    assert draft.to_candidate(LOOKUP, 0, cand(answer_numeric="n/a"), DOCS, "m", "d")["answer_numeric"] is None


# ---- run loop with a fake client ---------------------------------------------------------------
class FakeClient:
    def __init__(self, contents, fail_at=None):
        self.contents, self.fail_at, self.calls = list(contents), fail_at, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls += 1
        if self.fail_at == self.calls:
            raise RuntimeError("rate limit: tokens per day")
        assert kw["model"] == draft.MODEL and kw["reasoning_effort"] == "low"
        usage = NS(prompt_tokens=1000, completion_tokens=900,
                   completion_tokens_details=NS(reasoning_tokens=400))
        return NS(choices=[NS(message=NS(content=self.contents.pop(0)), finish_reason="stop")],
                  usage=usage, model="openai/gpt-oss-120b")


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(draft, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(draft, "DRAFT_DIR", tmp_path / "drafts")
    monkeypatch.setattr(draft, "CANDIDATES", tmp_path / "drafts" / "candidates.jsonl")
    monkeypatch.setattr(draft, "REJECTED", tmp_path / "drafts" / "rejected.jsonl")
    return tmp_path


def test_run_saves_records_resumes_and_rebuilds(tmp_dirs):
    good = json.dumps({"candidates": [cand(), cand(evidence=[{"doc_id": "d1_ca", "page": 1, "span_text": "made up"}])]})
    ledger = Ledger(path=tmp_dirs / "ledger.jsonl")
    sleeps = []
    out = draft.run([LOOKUP, UNANS], DOCS, TEXT, FakeClient([good, "not json"]), ledger, sleep=sleeps.append)
    assert out["called"] == 2 and out["remaining"] == 0 and out["stopped"] is None
    assert len(sleeps) == 1 and sleeps[0] > 0                 # paced under the TPM limit, no wait after last
    rows = [json.loads(l) for l in (tmp_dirs / "ledger.jsonl").read_text().splitlines()]
    assert [r["purpose"] for r in rows] == ["r3_draft:lookup-001", "r3_draft:unanswerable-001"]
    assert rows[0]["reasoning_tokens"] == 400 and rows[0]["total_tokens"] == 1900
    # resume: nothing left to call
    again = draft.run([LOOKUP, UNANS], DOCS, TEXT, FakeClient([]), ledger, sleep=sleeps.append)
    assert again["called"] == 0
    rep = draft.rebuild([LOOKUP, UNANS], DOCS, TEXT)
    assert rep == {"drafted_jobs": 2, "candidates": 1, "rejected": 1, "unparseable_jobs": 1, "with_issues": 0}
    kept = json.loads(draft.CANDIDATES.read_text())
    assert kept["cid"] == "lookup-001-a" and kept["review"] is None
    m = draft.measured([LOOKUP, UNANS])
    assert m["calls"] == 2 and m["completion_mean"] == 900 and m["reasoning_mean"] == 400


def test_run_stops_on_api_error_and_budget(tmp_dirs):
    ledger = Ledger(path=tmp_dirs / "ledger.jsonl")
    out = draft.run([LOOKUP, UNANS], DOCS, TEXT, FakeClient(['{"candidates": []}'], fail_at=2), ledger,
                    sleep=lambda s: None)
    assert out["called"] == 1 and "tokens per day" in out["stopped"] and out["remaining"] == 1
    tight = Ledger(path=tmp_dirs / "ledger.jsonl", daily_budget=2500)    # 1,900 already used today
    out = draft.run([UNANS], DOCS, TEXT, FakeClient([]), tight, sleep=lambda s: None)
    assert out["called"] == 0 and "daily token budget" in out["stopped"]


def test_page_override_only_touches_named_job():
    out = draft.apply_overrides([LOOKUP, UNANS], {"unanswerable-001": (("d1_ca", 1),)})
    assert out[0] is LOOKUP and out[1].pages == (("d1_ca", 1),) and out[1].job_id == "unanswerable-001"
    assert draft.PAGE_OVERRIDES["table-007"] == (("deal_07_ca", 101),)


class JsonValidateFailed(Exception):
    """Shape of groq.BadRequestError for code json_validate_failed (body = inner error dict)."""
    def __init__(self, gen):
        super().__init__("Failed to generate JSON")
        self.body = {"message": "Failed to generate JSON", "type": "invalid_request_error",
                     "code": "json_validate_failed", "failed_generation": gen}


def test_run_salvages_json_validate_failed(tmp_dirs):
    # page text with a raw \x07 bullet; the model copies it into the span -> Groq's strict JSON check fails
    text = DictText({"d1_ca": {1: "Estimated Closing Costs $5,099 \x07Includes Loan Costs."}})
    gen = '{"candidates": [' + json.dumps(cand(
        question="What are the estimated closing costs?", reference_answer="$5,099.",
        evidence=[{"doc_id": "d1_ca", "page": 1, "span_text": "XX"}], entities=["$5,099"],
        answer_numeric=5099)).replace('"XX"', '"Estimated Closing Costs $5,099 \x07Includes Loan Costs."') + ']}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(gen)                                             # strict parse fails, like Groq's
    client = FakeClient([])
    client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(JsonValidateFailed(gen))
    ledger = Ledger(path=tmp_dirs / "ledger.jsonl")
    out = draft.run([LOOKUP], DOCS, text, client, ledger, sleep=lambda s: None)
    assert out["called"] == 1 and out["stopped"] is None
    raw = json.loads((draft.RAW_DIR / "lookup-001.json").read_text())
    assert raw["usage_estimated"] and raw["completion_tokens"] > draft.EST_REASONING_PER_CALL
    assert json.loads((tmp_dirs / "ledger.jsonl").read_text())["total_tokens"] > 0   # budget still counts it
    rep = draft.rebuild([LOOKUP], DOCS, text)
    assert rep["candidates"] == 1 and rep["rejected"] == 0               # span still verbatim incl. \x07


def test_other_bad_request_still_stops(tmp_dirs):
    class Other(Exception):
        body = {"code": "context_length_exceeded"}
    client = FakeClient([])
    client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(Other("too long"))
    out = draft.run([LOOKUP], DOCS, TEXT, client, Ledger(path=tmp_dirs / "l.jsonl"), sleep=lambda s: None)
    assert out["called"] == 0 and "too long" in out["stopped"]


# ---- ledger --------------------------------------------------------------------------------------
def test_ledger_caps(tmp_path):
    assert cost_usd("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)
    lg = Ledger(path=tmp_path / "l.jsonl", daily_budget=10_000)
    lg.record("x", "openai/gpt-oss-120b", "openai/gpt-oss-120b", 6000, 3000, ts="2026-11-14T10:00:00+00:00")
    lg.check("openai/gpt-oss-120b", 500, 400, day="2026-11-14")
    with pytest.raises(BudgetExceeded, match="daily"):
        lg.check("openai/gpt-oss-120b", 600, 500, day="2026-11-14")
    lg.check("openai/gpt-oss-120b", 600, 500, day="2026-11-15")          # new UTC day
    assert Ledger(path=tmp_path / "l.jsonl").tokens_today("2026-11-14") == 9000   # reloaded from file
    with pytest.raises(BudgetExceeded, match="run cap"):
        Ledger(path=tmp_path / "l.jsonl", run_cap=0.001).check("openai/gpt-oss-120b", 10_000, 0, day="x")
    with pytest.raises(BudgetExceeded, match="total cap"):
        Ledger(path=tmp_path / "l.jsonl", total_cap=0.002).check("openai/gpt-oss-120b", 1000, 0, day="x")
    Ledger(path=tmp_path / "l.jsonl", total_cap=0.002, allow_over_total=True).check("openai/gpt-oss-120b", 1000, 0, day="x")
