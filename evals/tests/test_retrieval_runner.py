"""R4 step 2: scan views, chunk/embedding cache, R-dense, and the runner (no parse cache, no MiniLM)."""
import json
import zlib

import numpy as np
import pytest

from evals import run as R
from evals.chunking.base import SourceDoc
from evals.chunking.c3_fixed_tokens import C3FixedTokens
from evals.retrieval.index_cache import IndexCache, cache_key
from evals.retrieval.views import DocView, scan_source, session_view
from evals.retrievers.dense import DenseRetriever, SessionIndex

PART = {"ca": "edgar", "am": "edgar", "scan_1": "simulated_scan", "scan_2": "simulated_scan"}
ROLE = {"ca": "original", "am": "amendment_1", "scan_1": "scan_of:ca:p2", "scan_2": "scan_of:am:p1"}
PAGES = {
    "ca": {1: "Section 2.07 Fees. The Borrower shall pay a closing fee of $62,500 on the Closing Date.",
           2: "Section 6.12 Leverage. The Total Leverage Ratio shall not exceed 3.50 to 1.00."},
    "am": {1: "Section 2.07 is amended: the Borrower shall pay an annual facility fee of $75,000."},
    "scan_1": {1: "Section 6.12 Leverage. The Total Leverage Ratio shall not exceed 3.50 to 1.00."},
    "scan_2": {1: "Section 2.07 is amended: the Borrower shall pay an annual facility fee of $75,000."},
}


def _q(qid="q1", evidence=None, **kw):
    q = {"qid": qid, "split": "dev", "type": "lookup", "question": "What is the closing fee?",
         "session_id": "deal_x", "session_docs": ["am", "ca", "scan_1", "scan_2"],
         "evidence": evidence if evidence is not None else
         [{"doc_id": "ca", "page": 1, "span_text": "closing fee of $62,500"}],
         "entities": ["$62,500"], "amendment": False, "unanswerable": False}
    q.update(kw)
    return q


# ---- views -------------------------------------------------------------------------------------
def test_scan_source_parses_manifest_role():
    assert scan_source("scan_of:cfpb_cd_h25b:p2") == ("cfpb_cd_h25b", 2)
    assert scan_source("original") is None


def test_digital_view_leaves_scans_out():
    assert session_view(_q(), PART, ROLE) == [DocView("am"), DocView("ca")]


def test_scan_view_swaps_the_cited_scan_for_its_source_page():
    q = _q(evidence=[{"doc_id": "scan_1", "page": 1, "span_text": "3.50 to 1.00"}])
    assert session_view(q, PART, ROLE) == [DocView("am"), DocView("ca", frozenset({2})), DocView("scan_1")]


def test_source_page_cited_too_is_kept():
    q = _q(evidence=[{"doc_id": "scan_1", "page": 1, "span_text": "3.50"},
                     {"doc_id": "ca", "page": 2, "span_text": "3.50"}])
    assert DocView("ca") in session_view(q, PART, ROLE)


def test_scan_without_role_fails_loudly():
    q = _q(evidence=[{"doc_id": "scan_1", "page": 1, "span_text": "x"}])
    with pytest.raises(ValueError):
        session_view(q, PART, {**ROLE, "scan_1": "form"})


# ---- fakes -------------------------------------------------------------------------------------
VOCAB = ["closing", "fee", "leverage", "ratio", "amended", "annual", "facility", "borrower", "section"]


def fake_embed(texts):
    """Bag-of-vocabulary vectors, L2-normalized (deterministic, no model)."""
    out = np.zeros((len(texts), 384), np.float32)
    for i, t in enumerate(texts):
        low = t.lower()
        for j, w in enumerate(VOCAB):
            out[i, j] = low.count(w)
        out[i, 383] = 0.01 + (zlib.crc32(t.encode()) % 7) / 1000   # break ties, never zero
        out[i] /= np.linalg.norm(out[i])
    return out


class FakeCorpus:
    part_of, role_of = PART, ROLE

    def __init__(self):
        self.loads = []

    def parser_key(self, doc_id):
        return f"parse-{doc_id}"

    def load_source(self, doc_id, drop_pages):
        self.loads.append((doc_id, tuple(sorted(drop_pages))))
        pages = {p: t for p, t in PAGES[doc_id].items() if p not in drop_pages}
        return SourceDoc(doc_id, pages), f"parse-{doc_id}"

    def amendment_docs(self):
        return {"am"}


# ---- cache -------------------------------------------------------------------------------------
def test_cache_key_changes_with_each_input():
    c, c2 = C3FixedTokens(), C3FixedTokens(size=128)
    base = cache_key(c, "p")
    assert base == cache_key(C3FixedTokens(), "p")
    assert len({base, cache_key(c2, "p"), cache_key(c, "q"), cache_key(c, "p", frozenset({2})),
                cache_key(c, "p", cleaning_version="v1"), cache_key(c, "p", embedder="other")}) == 6


def test_cache_builds_once_then_loads_from_disk(tmp_path):
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path)
    a = cache.get(C3FixedTokens(), "ca")
    assert not a.from_cache and cache.built == 1 and len(a.chunks) == len(a.vectors) > 0
    fresh = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path)
    b = fresh.get(C3FixedTokens(), "ca")
    assert b.from_cache and fresh.built == 0 and corpus.loads == [("ca", ())]
    assert [c.to_dict() for c in b.chunks] == [c.to_dict() for c in a.chunks]
    assert np.array_equal(a.vectors, b.vectors) and b.source.pages == PAGES["ca"]
    dropped = fresh.get(C3FixedTokens(), "ca", frozenset({2}))
    assert 2 not in dropped.source.pages and all(s.page == 1 for c in dropped.chunks for s in c.spans)


def test_cache_rebuilds_a_mismatched_entry(tmp_path):
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path)
    key = cache.get(C3FixedTokens(), "ca").key
    np.save(tmp_path / f"{key}.npy", np.zeros((0, 384), np.float32))
    again = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path)
    assert not again.get(C3FixedTokens(), "ca").from_cache


# ---- R-dense -----------------------------------------------------------------------------------
def test_dense_retriever_ranks_by_cosine():
    from evals.chunking.base import Chunk
    chunks = [Chunk(f"c{i}", "d", i, t, []) for i, t in enumerate(["leverage ratio", "closing fee", "section"])]
    idx = SessionIndex.build([(chunks, fake_embed([c.text for c in chunks]))])
    hits = DenseRetriever().search(idx, fake_embed(["what is the closing fee"])[0], 10)
    assert hits.ids[0] == 1 and len(hits.ids) == 3 and hits.scores == sorted(hits.scores, reverse=True)
    assert "vector_search" in hits.timings


# ---- runner ------------------------------------------------------------------------------------
def test_load_config_validates(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("stage: S1\nruns:\n  - {name: a, chunker: C3, retriever: R-dense, top_k: 5}\n")
    cfg = R.load_config(p)
    assert cfg.table_config == "P2" and cfg.runs[0].top_k == 5
    p.write_text("stage: S1\nruns:\n  - {name: a, chunker: C9, retriever: R-dense, top_k: 5}\n")
    with pytest.raises(ValueError):
        R.load_config(p)
    p.write_text("stage: S1\nruns:\n  - {name: a, chunker: C3, retriever: R-dense, top_k: 5}\n"
                 "  - {name: a, chunker: C2, retriever: R-dense, top_k: 5}\n")
    with pytest.raises(ValueError):
        R.load_config(p)


def test_s1_config_lists_all_five_chunkers():
    cfg = R.load_config(R.REPO / "evals" / "configs" / "s1_chunkers.yaml")
    assert [r.chunker for r in cfg.runs] == ["C1", "C2", "C3", "C4", "C5"]
    assert {r.retriever for r in cfg.runs} == {"R-dense"}


def test_run_stage_end_to_end(tmp_path):
    qs = [_q("q1"),
          _q("q2", type="lookup", question="What is the leverage ratio?",
             evidence=[{"doc_id": "scan_1", "page": 1, "span_text": "Total Leverage Ratio shall not exceed 3.50"}],
             entities=["3.50"]),
          _q("q3", type="unanswerable", question="Who is the audit firm?", evidence=[], entities=[],
             unanswerable=True),
          _q("q4", split="test")]
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text("".join(json.dumps(q) + "\n" for q in qs))
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S1", "P2", [R.RunConfig("C3-dense", "C3", "R-dense", 5)])
    out = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                      embed=fake_embed, cache=cache)
    s = out["C3-dense"]
    assert s["questions"] == 3 and s["unlocated_spans"] == 0 and s["sessions_built"] == 2
    assert s["breakdown"]["overall"]["recall@budget"] == {"mean": 1.0, "n": 2}
    assert s["breakdown"]["part=simulated_scan"]["recall@budget"]["n"] == 1
    rows = [json.loads(l) for l in (tmp_path / "out" / "s1" / "dev" / "C3-dense" / "questions.jsonl").open()]
    q2 = next(r for r in rows if r["qid"] == "q2")
    assert q2["view"] == [["am", []], ["ca", [2]], ["scan_1", []]]
    assert q2["ranked"][0]["doc_id"] == "scan_1" and q2["n_relevant"] >= 1
    assert set(q2["timings_s"]) == {"query_embed", "vector_search", "total"}
    summary = json.loads((tmp_path / "out" / "s1" / "dev" / "C3-dense" / "summary.json").read_text())
    assert summary["run"]["chunker"] == "C3" and summary["cache"]["built"] == 4
    assert summary["questions_sha256"] == R.file_sha256(qpath)
    # second stage run: every document view comes from the cache
    out2 = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                       embed=fake_embed, cache=IndexCache(corpus.load_source, fake_embed, corpus.parser_key,
                                                          root=tmp_path / "cache"))
    assert out2["C3-dense"]["cache"] == {"built": 0, "loaded": 4}
    assert out2["C3-dense"]["breakdown"] == s["breakdown"]


def test_run_stage_rejects_unknown_run_names(tmp_path):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n")
    cfg = R.StageConfig("S1", "P2", [R.RunConfig("a", "C3", "R-dense", 5)])
    with pytest.raises(ValueError):
        R.run_stage(cfg, "dev", only=["nope"], questions_path=qpath, out_root=tmp_path, corpus=FakeCorpus(),
                    embed=fake_embed, cache=IndexCache(FakeCorpus().load_source, fake_embed,
                                                       FakeCorpus().parser_key, root=tmp_path / "c"))


# ---- MLflow (step 3) ---------------------------------------------------------------------------
def test_metric_names_are_mlflow_safe():
    from evals.tracking import metric_name
    assert metric_name("", "recall@budget") == "recall_at_budget"
    assert metric_name("type=cross_doc", "hit@10") == "type.cross_doc.hit_at_10"
    assert metric_name("part=simulated_scan", "mrr@10") == "part.simulated_scan.mrr_at_10"


def test_flat_params_nests_config():
    from evals.tracking import flat_params
    assert flat_params({"chunker": "C3", "chunker_params": {"size": 128}, "x": {}}, "run.") == \
        {"run.chunker": "C3", "run.chunker_params.size": "128", "run.x": "{}"}


def test_run_stage_logs_one_mlflow_run_per_config(tmp_path):
    import mlflow
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n")
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S1", "P2", [R.RunConfig("C3-dense", "C3", "R-dense", 5),
                                     R.RunConfig("C2-dense", "C2", "R-dense", 5)])
    store = tmp_path / "mlflow"
    try:
        out = R.run_stage(cfg, "dev", limit=1, questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                          embed=fake_embed, cache=cache, mlflow_store=store)
        runs = mlflow.search_runs(experiment_names=["loandoc_retrieval_evals"], output_format="list")
        assert len(runs) == 2
        r = next(x for x in runs if x.info.run_id == out["C3-dense"]["mlflow_run_id"])
        assert r.data.params["run.chunker"] == "C3" and r.data.params["run.top_k"] == "5"
        assert r.data.params["questions_sha256"] == R.file_sha256(qpath)
        assert r.data.tags["smoke"] == "True" and r.info.run_name == "S1 C3-dense dev SMOKE"
        assert r.data.metrics["recall_at_budget"] == 1.0 and r.data.metrics["n_answerable"] == 1
        assert "latency.total.p50_ms" in r.data.metrics and "type.lookup.mrr_at_10" in r.data.metrics
        arts = {a.path for a in mlflow.MlflowClient().list_artifacts(r.info.run_id)}
        assert arts == {"summary.json", "questions.jsonl"}
    finally:
        mlflow.set_tracking_uri("")
