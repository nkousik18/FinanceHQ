"""§8.3 retrieval metrics (evals/retrieval/metrics.py) on hand-built documents and coverage lists."""
import math

import pytest

from evals.chunking.base import Chunk, SourceDoc, Span
from evals.retrieval import metrics as M

ORIG = "Section 2.07 Fees. The Borrower shall pay a closing fee of $62,500 on the Closing Date."
AMEND = "Section 2.07 is amended: the Borrower shall pay an annual facility fee of $75,000."
OTHER = "Section 9.01 Notices. All notices shall be in writing and delivered by hand."

SOURCES = {"ca": SourceDoc("ca", {1: ORIG, 2: OTHER}), "am": SourceDoc("am", {1: AMEND})}
E = frozenset()


def _whole(doc: str, page: int, text: str, i: int, words: int | None = None) -> Chunk:
    body = text if words is None else " ".join(["w"] * words)
    return Chunk(f"{doc}-{i}", doc, i, body, [Span(page, 0, len(text))])


C_ORIG = _whole("ca", 1, ORIG, 0)
C_OTHER = _whole("ca", 2, OTHER, 1)
C_AMEND = _whole("am", 1, AMEND, 0)


def _q(**kw) -> dict:
    q = {"qid": "q1", "type": "lookup", "session_id": "deal_07", "unanswerable": False, "amendment": False,
         "evidence": [{"doc_id": "ca", "page": 1, "span_text": "closing fee of $62,500"}],
         "entities": ["$62,500"]}
    q.update(kw)
    return q


# ---- budget ------------------------------------------------------------------------------------
def test_budget_cut_mirrors_production_router():
    from app.prompts.router import MAX_CONTEXT_WORDS
    assert M.MAX_CONTEXT_WORDS == MAX_CONTEXT_WORDS
    cs = [_whole("ca", 1, ORIG, i, words=w) for i, w in enumerate([500, 600, 200, 50])]
    assert M.budget_cut(cs) == 2                       # 500+600=1100; +200 > 1200 stops (no skipping to the 50)
    assert M.budget_cut([_whole("ca", 1, ORIG, 0, words=2000)]) == 1   # first chunk always kept
    assert M.budget_cut([]) == 0


# ---- pure rank metrics -------------------------------------------------------------------------
COV = [E, frozenset({0}), E, frozenset({0, 1}), E]      # 2 spans; relevant at ranks 2 and 4


def test_recall_hit_precision():
    assert M.recall_at(COV, 2, 1) == 0 and M.recall_at(COV, 2, 3) == 0.5 and M.recall_at(COV, 2, 5) == 1
    assert M.hit_at(COV, 1) == 0 and M.hit_at(COV, 3) == 1
    assert M.precision_at(COV, 5) == pytest.approx(0.4)
    assert M.precision_at(COV, 10) == pytest.approx(0.2)        # short list still divides by k


def test_context_precision_mrr():
    assert M.context_precision(COV) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert M.mrr_at(COV) == 0.5
    assert M.mrr_at([E] * 10 + [frozenset({0})]) == 0          # beyond rank 10
    assert M.context_precision([E, E]) == 0


def test_ndcg_and_map_use_index_relevant_count():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    assert M.ndcg_at(COV, 2) == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert M.ndcg_at(COV, 3) == pytest.approx(dcg / (1 + 1 / math.log2(3) + 1 / math.log2(4)))
    assert M.map_at(COV, 2) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert M.map_at(COV, 4) == pytest.approx((1 / 2 + 2 / 4) / 4)   # 2 relevant chunks never retrieved
    assert M.ndcg_at([frozenset({0})], 1) == 1 and M.ndcg_at(COV, 0) == 0


def test_entity_recall_normalizes_numbers():
    assert M.entity_recall(["$62,500", "Closing Date"], "a fee of 62,500 due on the CLOSING DATE") == 1
    assert M.entity_recall(["$75,000"], "fee of $62,500") == 0
    assert M.entity_recall([], "x") is None


def test_redundancy():
    a = _whole("ca", 1, "alpha beta gamma", 0)
    b = _whole("ca", 1, "beta gamma delta epsilon", 1)
    assert M.redundancy([a, b]) == pytest.approx(2 / 3)
    assert M.redundancy([a]) is None


# ---- coverage ----------------------------------------------------------------------------------
def test_coverage_uses_section_8_3_rule():
    loc = M.locate_all([M.Evidence("ca", 1, "closing fee of $62,500")], SOURCES)
    assert M.coverage(C_ORIG, loc, SOURCES) == {0}
    assert M.coverage(C_OTHER, loc, SOURCES) == E                       # other page
    half = Chunk("ca-h", "ca", 9, "x", [Span(1, 0, ORIG.index("$62,500"))])
    assert M.coverage(half, loc, SOURCES) == {0}                        # >= 50% of the span
    little = Chunk("ca-l", "ca", 9, "x", [Span(1, 0, ORIG.index("fee of"))])
    assert M.coverage(little, loc, SOURCES) == E
    assert M.count_relevant([C_ORIG, C_OTHER, half, little], loc, SOURCES) == 2


# ---- one question ------------------------------------------------------------------------------
def test_score_question_lookup():
    r = M.score_question(_q(), "edgar", [C_OTHER, C_ORIG], [0.9, 0.8], SOURCES)
    m = r.metrics
    assert r.n_budget == 2 and r.unlocated == 0
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["recall@budget"] == 1
    assert m["mrr@10"] == 0.5 and m["entity_recall"] == 1
    words_other, words_orig = len(OTHER.split()), len(ORIG.split())
    assert m["noise_ratio"] == pytest.approx(words_other / (words_other + words_orig))
    assert m["context_words"] == words_other + words_orig
    assert m["cross_doc_coverage"] is None and m["amend_coverage"] is None
    assert [row["covers"] for row in r.ranked] == [[], [0]] and r.ranked[1]["score"] == 0.8


def test_recall_at_budget_drops_chunks_past_the_budget():
    big = _whole("ca", 2, OTHER, 1, words=1190)
    r = M.score_question(_q(), "edgar", [big, C_ORIG], None, SOURCES)
    assert r.n_budget == 1
    assert r.metrics["recall@budget"] == 0 and r.metrics["recall@3"] == 1
    assert r.metrics["entity_recall"] == 0 and r.metrics["noise_ratio"] == 1


def test_unanswerable_excluded_from_coverage_metrics():
    q = _q(unanswerable=True, evidence=[], entities=[])
    m = M.score_question(q, "edgar", [C_ORIG, C_OTHER], None, SOURCES).metrics
    assert m["recall@budget"] is None and m["mrr@10"] is None and m["entity_recall"] is None
    assert m["context_words"] > 0 and m["redundancy"] is not None


def test_cross_doc_and_amendment_coverage():
    q = _q(type="cross_doc", amendment=True, evidence=[
        {"doc_id": "ca", "page": 1, "span_text": "closing fee of $62,500"},
        {"doc_id": "am", "page": 1, "span_text": "annual facility fee of $75,000"}])
    both = M.score_question(q, "edgar", [C_AMEND, C_ORIG], None, SOURCES, amendment_docs={"am"}).metrics
    assert both["cross_doc_coverage"] == 1 and both["amend_coverage"] == 1
    assert both["amend_only_superseded"] == 0
    stale = M.score_question(q, "edgar", [C_ORIG, C_OTHER], None, SOURCES, amendment_docs={"am"}).metrics
    assert stale["cross_doc_coverage"] == 0 and stale["amend_coverage"] == 0
    assert stale["amend_only_superseded"] == 1 and stale["recall@budget"] == 0.5


def test_unlocated_span_counts_as_not_covered():
    q = _q(evidence=[{"doc_id": "ca", "page": 1, "span_text": "not on the page"}])
    r = M.score_question(q, "edgar", [C_ORIG], None, SOURCES)
    assert r.unlocated == 1 and r.metrics["recall@budget"] == 0


# ---- aggregation -------------------------------------------------------------------------------
def test_breakdown_by_type_and_part():
    r1 = M.score_question(_q(qid="a"), "edgar", [C_ORIG], None, SOURCES)
    r2 = M.score_question(_q(qid="b", type="table"), "consumer", [C_OTHER], None, SOURCES)
    r3 = M.score_question(_q(qid="c", unanswerable=True, evidence=[]), "edgar", [C_OTHER], None, SOURCES)
    b = M.breakdown([r1, r2, r3])
    assert b["overall"]["recall@budget"] == {"mean": 0.5, "n": 2}
    assert b["overall"]["context_words"]["n"] == 3
    assert b["type=table"]["recall@budget"]["mean"] == 0
    assert b["part=edgar"]["recall@budget"] == {"mean": 1.0, "n": 1}


def test_question_part():
    parts = {"scan_27": "simulated_scan", "ca": "edgar"}
    assert M.question_part(_q(), parts) == "edgar"
    assert M.question_part(_q(evidence=[{"doc_id": "scan_27", "page": 1, "span_text": "x"}]), parts) == \
        "simulated_scan"
    assert M.question_part(_q(session_id="cfpb_le_cd_1"), parts) == "consumer"


def test_index_coverage_counts_relevant_chunks_and_fragmented_spans():
    ev = [M.Evidence("ca", 1, "closing fee of $62,500"), M.Evidence("am", 1, "annual facility fee of $75,000"),
          M.Evidence("ca", 1, "nowhere in the text")]
    loc = M.locate_all(ev, SOURCES)
    assert M.index_coverage([C_ORIG, C_OTHER], loc, SOURCES) == (1, 1)    # am span fragmented; unlocated not counted
    assert M.index_coverage([C_ORIG, C_AMEND], loc, SOURCES) == (2, 0)
