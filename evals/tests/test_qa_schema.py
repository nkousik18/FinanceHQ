"""QA golden-set schema and F3 validator (evals/golden/qa/schema.py)."""
import copy

import pytest

from evals.golden.qa.schema import (DictText, DocInfo, mix_errors, quota_errors, validate_question,
                                    validate_set, value_in)

DOCS = {
    "deal_07_ca": DocInfo("deal_07_ca", "edgar", "deal_07", "original"),
    "deal_07_am2": DocInfo("deal_07_am2", "edgar", "deal_07", "amendment_2"),
    "scan_05": DocInfo("scan_05", "simulated_scan", "deal_07", "scan_of:deal_07_ca:p40"),
    "cfpb_cd_h25b": DocInfo("cfpb_cd_h25b", "consumer", "cfpb_purchase_fixed", "form"),
}
TEXT = DictText({
    "deal_07_ca": {40: "Applicable Margin\nLevel I SOFR Loans 1.25%\nThe Maturity Date is March 15, 2027."},
    "deal_07_am2": {3: 'The definition of "Applicable Margin" is amended:  Level I SOFR Loans 1.50%.'},
    "scan_05": {1: "Level I SOFR Loans 1.25%"},
    "cfpb_cd_h25b": {1: "Loan Amount $162,000\nInterest Rate 3.875%"},
})


def _q(**kw) -> dict:
    q = {
        "qid": "q0001", "split": "test", "type": "lookup",
        "question": "What is the loan amount?", "session_id": "cfpb_purchase_fixed",
        "session_docs": ["cfpb_cd_h25b"],
        "evidence": [{"doc_id": "cfpb_cd_h25b", "page": 1, "span_text": "Loan Amount $162,000"}],
        "reference_answer": "The loan amount is $162,000.", "key_points": ["$162,000"],
        "entities": ["$162,000"], "superseded_values": [], "amendment": False,
        "answer_numeric": None, "numeric_tolerance": None, "unanswerable": False,
        "provenance": "llm_drafted_human_verified", "verified_by": "kousik", "verified_on": "2026-11-20",
    }
    q.update(kw)
    return q


def _amend(**kw) -> dict:
    base = dict(
        type="cross_doc", session_id="deal_07", session_docs=["deal_07_ca", "deal_07_am2"],
        question="What is the Level I SOFR margin after the Second Amendment?",
        evidence=[{"doc_id": "deal_07_ca", "page": 40, "span_text": "Level I SOFR Loans 1.25%"},
                  {"doc_id": "deal_07_am2", "page": 3, "span_text": "Level I SOFR Loans 1.50%."}],
        reference_answer="1.50% (amended from 1.25%).", key_points=["1.50%"], entities=["1.50%"],
        superseded_values=["1.25%"], amendment=True)
    base.update(kw)
    return _q(**base)


def _errs(q):
    return validate_question(q, DOCS, TEXT)


def test_valid_questions_pass():
    assert _errs(_q()) == []
    assert _errs(_amend()) == []
    un = _q(type="unanswerable", unanswerable=True, evidence=[], reference_answer="Not stated.",
            key_points=[], entities=[])
    assert _errs(un) == []


@pytest.mark.parametrize("value,target,ok", [
    ("$162,000", "Loan Amount $162,000", True),
    ("162000", "Loan Amount $162,000", True),
    ("3.875%", "Interest Rate 3.875 percent", True),
    ("1.50%", "margin of 1.25%", False),
    ("Bank of America, N.A.", "BANK OF AMERICA, N.A., as Agent", True),
    ("JPMorgan", "Wells Fargo", False),
])
def test_value_in_normalizes(value, target, ok):
    assert value_in(value, target) is ok


def test_span_must_be_in_cleaned_page_text():
    q = _q(evidence=[{"doc_id": "cfpb_cd_h25b", "page": 1, "span_text": "Loan Amount $165,000"}])
    assert any("span_text not found" in e for e in _errs(q))
    ws = _q(evidence=[{"doc_id": "cfpb_cd_h25b", "page": 1, "span_text": "Loan  Amount\n$162,000"}])
    assert _errs(ws) == []
    wrong_page = _q(evidence=[{"doc_id": "cfpb_cd_h25b", "page": 2, "span_text": "Loan Amount $162,000"}])
    assert any("not found" in e for e in _errs(wrong_page))


def test_entity_must_be_in_evidence_or_answer():
    assert any("entity '3.875%'" in e for e in _errs(_q(entities=["$162,000", "3.875%"])))


def test_superseded_values_must_exist_in_session_docs():
    assert any("superseded value '2.00%'" in e for e in _errs(_amend(superseded_values=["2.00%"])))


def test_amendment_rules():
    assert any("without superseded_values" in e for e in _errs(_amend(superseded_values=[])))
    no_am_doc = _amend(evidence=[{"doc_id": "deal_07_ca", "page": 40, "span_text": "Level I SOFR Loans 1.25%"},
                                 {"doc_id": "scan_05", "page": 1, "span_text": "Level I SOFR Loans 1.25%"}],
                       session_docs=["deal_07_ca", "scan_05"], reference_answer="1.50%")
    assert any("amendment document" in e for e in _errs(no_am_doc))
    assert any("non-amendment" in e for e in _errs(_q(superseded_values=["$162,000"])))


def test_structure_rules():
    assert any("missing field verified_by" in e for e in _errs({k: v for k, v in _q().items() if k != "verified_by"}))
    assert any("not verified" in e for e in _errs(_q(verified_by="")))
    assert any("bad type" in e for e in _errs(_q(type="trivia")))
    assert any("not in session_docs" in e for e in _errs(_q(evidence=[{"doc_id": "deal_07_ca", "page": 40, "span_text": "x"}])))
    assert any("not in session" in e for e in _errs(_q(session_docs=["cfpb_cd_h25b", "deal_07_ca"])))
    assert any("2+ documents" in e for e in _errs(_amend(evidence=_amend()["evidence"][1:])))
    assert any("has evidence" in e for e in _errs(_q(type="unanswerable", unanswerable=True)))
    assert any("must match type" in e for e in _errs(_q(unanswerable=True)))
    assert any("answer_numeric" in e for e in _errs(_q(type="calculation")))
    assert any("numeric_tolerance" in e for e in _errs(_q(numeric_tolerance=0.01)))
    assert _errs(_q(type="calculation", answer_numeric=162000, numeric_tolerance=0)) == []


def test_duplicate_qids_fail():
    errs = validate_set([_q(), _q()], DOCS, TEXT)
    assert "q0001: duplicate qid" in errs


def _mix(n_dev: dict, n_test: dict) -> list[dict]:
    qs, i = [], 0
    for split, counts in (("dev", n_dev), ("test", n_test)):
        for t, n in counts.items():
            for _ in range(n):
                i += 1
                qs.append({"qid": f"q{i:04d}", "split": split, "type": t})
    return qs


def test_split_mix_within_one_of_proportional():
    assert mix_errors(_mix({"lookup": 8, "table": 8}, {"lookup": 17, "table": 17})) == []
    errs = mix_errors(_mix({"lookup": 4, "table": 12}, {"lookup": 21, "table": 13}))
    assert any("split dev" in e and "lookup" in e for e in errs)


def test_retired_questions_ignored_in_mix():
    qs = _mix({"lookup": 1}, {"lookup": 2})
    qs.append({"qid": "q9", "split": "dev", "type": "table", "retired": True})
    assert mix_errors(qs) == []


def test_final_quotas_report_shortfalls():
    errs = quota_errors([_q()], DOCS)
    assert any(e.startswith("type table") for e in errs)
    assert any("amendment cross_doc" in e for e in errs)
    assert any("scan evidence" in e for e in errs)


def test_validate_set_does_not_mutate():
    qs = [_q(), _amend(qid="q0002")]
    before = copy.deepcopy(qs)
    validate_set(qs, DOCS, TEXT)
    assert qs == before
