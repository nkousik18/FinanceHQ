"""R5 step 1: R-bm25 (tokenizer, ranking) and R-hybrid (RRF), plus their runner wiring."""
import json

import numpy as np
import pytest

from evals import run as R
from evals.chunking.base import Chunk
from evals.retrieval.index_cache import IndexCache
from evals.retrievers import RETRIEVERS, BM25Retriever, HybridRetriever, SessionIndex
from evals.retrievers.bm25 import normalize_number, tokenize
from evals.retrievers.hybrid import rrf
from evals.tests.test_retrieval_runner import FakeCorpus, _q, fake_embed


def _index(texts):
    chunks = [Chunk(f"c{i}", "d", i, t, []) for i, t in enumerate(texts)]
    return SessionIndex.build([(chunks, fake_embed(texts))])


# ---- tokenizer ---------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, norm", [("62,500", "62500"), ("62,500.00", "62500"), ("3.50", "3.5"),
                                       ("100.00", "100"), ("2.11", "2.11"), ("1,000,000", "1000000"),
                                       ("100", "100"), ("0.25", "0.25")])
def test_normalize_number(raw, norm):
    assert normalize_number(raw) == norm


def test_tokenize_keeps_identifiers_whole_and_drops_stopwords():
    toks = tokenize("The closing fee of $62,500.00 under Section 2.11(b) is 3.50%.")
    assert toks == ["closing", "fee", "62500", "under", "section", "2.11", "b", "3.5"]


def test_tokenize_does_not_stem():
    assert tokenize("Borrowers borrowing") == ["borrowers", "borrowing"]


def test_same_amount_written_two_ways_matches():
    assert tokenize("$62,500")[0] == tokenize("62500.00")[0]


def test_non_thousands_comma_splits():
    assert tokenize("1,2") == ["1", "2"]


# ---- R-bm25 ------------------------------------------------------------------------------------
def test_bm25_finds_exact_identifier():
    idx = _index(["Section 6.12 Leverage. Ratio shall not exceed 3.50 to 1.00.",
                  "Section 2.07 Fees. Closing fee of $62,500.",
                  "Section 2.11 Prepayments."])
    hits = BM25Retriever().search(idx, None, 10, query="What does Section 2.11 say?")
    assert hits.ids[0] == 2 and hits.scores == sorted(hits.scores, reverse=True)
    assert set(hits.timings) == {"bm25"}


def test_bm25_returns_only_matching_chunks():
    idx = _index(["closing fee", "leverage ratio", "unrelated words"])
    hits = BM25Retriever().search(idx, None, 10, query="closing fee amount")
    assert hits.ids == [0]


def test_bm25_no_query_tokens_returns_nothing():
    idx = _index(["closing fee", "leverage ratio"])
    assert BM25Retriever().search(idx, None, 10, query="what is the").ids == []


def test_bm25_ties_keep_chunk_order_and_index_is_reused():
    idx = _index(["fee", "fee", "other"])
    assert BM25Retriever().search(idx, None, 10, query="fee").ids == [0, 1]
    model = idx.extras["bm25"]
    BM25Retriever().search(idx, None, 10, query="fee")
    assert idx.extras["bm25"] is model


def test_bm25_respects_k():
    idx = _index([f"fee {i}" for i in range(30)])
    assert len(BM25Retriever().search(idx, None, 5, query="fee").ids) == 5


# ---- RRF / R-hybrid ----------------------------------------------------------------------------
def test_rrf_hand_worked():
    # a: 1/61 + 1/63 ; b: 1/62 + 1/61 ; c: 1/63 ; d: 1/62
    fused = rrf([["a", "b", "c"], ["b", "d", "a"]])
    assert [i for i, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_ties_prefer_better_rank_then_dense():
    # x only in dense at rank 1, y only in bm25 at rank 1: equal scores -> dense first
    assert [i for i, _ in rrf([["x"], ["y"]])] == ["x", "y"]
    # disjoint lists interleave: rank 1s first (dense, then bm25), then rank 2s
    assert [i for i, _ in rrf([["p", "q"], ["r", "s"]])] == ["p", "r", "q", "s"]


def test_hybrid_merges_dense_and_bm25():
    idx = _index(["closing fee paid on closing date", "leverage ratio covenant", "Section 2.11 prepayments"])
    q = "Section 2.11 closing fee"
    hits = HybridRetriever().search(idx, fake_embed([q])[0], 10, query=q)
    assert set(hits.ids) == {0, 1, 2}                       # dense ranks all; bm25 adds nothing new
    assert set(hits.timings) == {"vector_search", "bm25", "fusion"}
    assert hits.scores == sorted(hits.scores, reverse=True)
    assert len(HybridRetriever().search(idx, fake_embed([q])[0], 2, query=q).ids) == 2


def test_hybrid_uses_twenty_deep_lists():
    texts = [f"filler text {i}" for i in range(40)] + ["Section 9.99 exact"]
    idx = _index(texts)
    q = "Section 9.99"
    hits = HybridRetriever().search(idx, fake_embed([q])[0], 20, query=q)
    assert 40 in hits.ids                                    # BM25 hit fused in
    assert len(hits.ids) == 20


# ---- runner wiring -----------------------------------------------------------------------------
def test_registry_has_r5_step1_retrievers():
    assert {"R-dense", "R-bm25", "R-hybrid"} <= set(RETRIEVERS)
    assert RETRIEVERS["R-bm25"].needs_vector is False


@pytest.mark.parametrize("retriever, comps", [("R-bm25", {"bm25", "total"}),
                                              ("R-hybrid", {"query_embed", "vector_search", "bm25",
                                                            "fusion", "total"})])
def test_run_stage_with_lexical_retrievers(tmp_path, retriever, comps):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n")
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S2", "P2", [R.RunConfig("x", "C3", retriever, 5)])
    s = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                    embed=fake_embed, cache=cache)["x"]
    assert s["breakdown"]["overall"]["recall@budget"]["mean"] == 1.0
    assert set(s["latency_ms"]) == comps
    assert s["retriever_params"] == RETRIEVERS[retriever].params
    row = json.loads((tmp_path / "out" / "s2" / "dev" / "x" / "questions.jsonl").read_text())
    assert set(row["timings_s"]) == comps



def test_prepare_builds_bm25_at_index_time():
    from evals.retrievers import Reranked
    for r in (BM25Retriever(), HybridRetriever(),
              Reranked(BM25Retriever(), lambda q, t: np.zeros(len(t)), lambda q, t: 1)):
        idx = _index(["closing fee", "leverage"])
        r.prepare(idx)
        assert "bm25" in idx.extras
