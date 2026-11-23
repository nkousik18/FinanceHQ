"""R5 step 3: +rerank wrapper (fake scorer for the logic; one test with the real cross-encoder)."""
import json

import numpy as np
import pytest

from evals import run as R
from evals.chunking.base import Chunk
from evals.retrieval.index_cache import IndexCache
from evals.retrievers import (BM25Retriever, DenseRetriever, ParentChildRetriever, Reranked, SessionIndex,
                              make_retriever)
from evals.retrievers.rerank import MAX_LENGTH, RERANK_DEPTH
from evals.tests.test_retrieval_runner import FakeCorpus, _q, fake_embed
from evals.tests.test_retriever_pc import _c, _c4, _doc


def _index(texts):
    chunks = [Chunk(f"c{i}", "d", i, t, []) for i, t in enumerate(texts)]
    return SessionIndex.build([(chunks, fake_embed(texts))])


def by_length(query, texts):          # longer text = better: an order unrelated to dense
    return np.array([len(t) for t in texts], float)


def short_pairs(query, text):
    return 10


class Spy:
    """Base retriever that records the k it was asked for."""
    retriever_id, params, needs_vector = "spy", {}, True

    def __init__(self):
        self.dense, self.asked = DenseRetriever(), []

    def search(self, index, qv, k, query=""):
        self.asked.append(k)
        return self.dense.search(index, qv, k)


def test_rerank_reorders_by_scorer_and_keeps_k():
    texts = ["fee", "closing fee paid at closing", "a closing fee", "leverage ratio covenant text"]
    idx = _index(texts)
    hits = Reranked(DenseRetriever(), by_length, short_pairs).search(idx, fake_embed(["fee"])[0], 2, query="fee")
    assert hits.ids == [3, 1] and hits.scores == [28.0, 27.0]
    assert "rerank" in hits.timings and "vector_search" in hits.timings
    assert hits.info == {"rerank_candidates": 4, "rerank_truncated": 0}


def test_rerank_takes_twenty_candidates_even_for_small_k():
    spy = Spy()
    Reranked(spy, by_length, short_pairs).search(_index(["a", "b"]), fake_embed(["a"])[0], 5)
    assert spy.asked == [RERANK_DEPTH]
    Reranked(spy, by_length, short_pairs).search(_index(["a", "b"]), fake_embed(["a"])[0], 30)
    assert spy.asked[-1] == 30


def test_rerank_ties_keep_base_order():
    idx = _index(["closing fee", "fee", "closing"])
    base = DenseRetriever().search(idx, fake_embed(["closing fee"])[0], 3).ids
    hits = Reranked(DenseRetriever(), lambda q, t: np.zeros(len(t)), short_pairs).search(
        idx, fake_embed(["closing fee"])[0], 3)
    assert hits.ids == base


def test_rerank_counts_truncated_pairs():
    idx = _index(["short", "x " * 600])
    hits = Reranked(DenseRetriever(), by_length, lambda q, t: len(t.split()) + 3).search(
        idx, fake_embed(["q"])[0], 2, query="q")
    assert hits.info["rerank_truncated"] == 1 and MAX_LENGTH == 512


def test_rerank_over_bm25_and_pc_keeps_their_needs():
    r = Reranked(BM25Retriever(), by_length, short_pairs)
    assert r.needs_vector is False and r.retriever_id == "R-bm25+rerank" and "rerank" in r.params
    src = _doc()
    kids = [_c("k0", 1, 0, 40, "closing fee x"), _c("k2", 2, 0, 30, "leverage")]
    idx = SessionIndex.build([(kids, fake_embed([k.text for k in kids]))])
    from evals.retrievers.parent_child import ParentMap
    idx.extras["parents"] = ParentMap({"d": src}, {"d": _c4(src)})
    pc = Reranked(ParentChildRetriever(), by_length, short_pairs)
    assert pc.needs_parents is True
    hits = pc.search(idx, fake_embed(["closing fee"])[0], 10, query="closing fee")
    assert [c.section for c in hits.chunks] == ["Sec 1", "Sec 2"] and hits.ids == []   # P1 longer than P2


def test_make_retriever():
    assert isinstance(make_retriever("R-dense"), DenseRetriever)
    assert isinstance(make_retriever("R-hybrid", rerank=True), Reranked)


# ---- runner ------------------------------------------------------------------------------------
def test_config_rerank_flag(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stage: S2\nruns:\n  - {name: a, chunker: C3, retriever: R-bm25, top_k: 5, rerank: true}\n")
    assert R.load_config(p).runs[0].rerank is True
    with pytest.raises(ValueError):
        R.RunConfig("a", "C3", "R-dense", 5, rerank="yes")


def test_run_stage_with_rerank(tmp_path, monkeypatch):
    monkeypatch.setattr("evals.retrievers.rerank.cross_encoder_scores", by_length)
    monkeypatch.setattr("evals.retrievers.rerank.pair_tokens", short_pairs)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n")
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S2", "P2", [R.RunConfig("x", "C3", "R-dense", 5, rerank=True)])
    s = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                    embed=fake_embed, cache=cache)["x"]
    assert "rerank" in s["latency_ms"] and s["retriever_params"]["rerank"]["depth"] == 20
    assert s["retriever_info"]["rerank_truncated"] == 0 and s["retriever_info"]["rerank_candidates"] > 0
    row = json.loads((tmp_path / "out" / "s2" / "dev" / "x" / "questions.jsonl").read_text())
    assert row["retriever_info"]["rerank_candidates"] > 0
    from evals.tracking import retrieval_metrics
    assert "retriever.rerank_truncated" in retrieval_metrics(s)


# ---- real model --------------------------------------------------------------------------------
def test_real_cross_encoder_prefers_the_answer():
    try:
        from evals.retrievers.rerank import cross_encoder_scores, pair_tokens
        s = cross_encoder_scores("What is the closing fee?",
                                 ["The Total Leverage Ratio shall not exceed 3.50 to 1.00.",
                                  "The Borrower shall pay a closing fee of $62,500 on the Closing Date."])
    except OSError as e:                                   # model not downloadable here
        pytest.skip(f"cross-encoder unavailable: {e}")
    assert s[1] > s[0]
    assert pair_tokens("q", "word " * 600) > MAX_LENGTH
