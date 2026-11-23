"""R5 step 5: FAISS index types (ann.py), recall vs exact, S3 runner wiring, pooled study, S3 report."""
import json

import numpy as np
import pytest

from evals import run as R
from evals.chunking.base import Chunk
from evals.retrieval.index_cache import IndexCache
from evals.retrievers import DenseRetriever, HybridRetriever, SessionIndex, ann
from evals.tests.test_retrieval_runner import FakeCorpus, _q, fake_embed


def _vecs(n, d=384, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---- specs -------------------------------------------------------------------------------------
def test_parse_spec_defaults_and_validation():
    assert ann.parse_spec(None) == {"type": "flat"}
    assert ann.parse_spec({"type": "hnsw", "ef_search": 16}) == {"type": "hnsw", "M": 32, "ef_construction": 200,
                                                                 "ef_search": 16}
    assert ann.parse_spec({"type": "ivf"}) == {"type": "ivf", "nlist": "sqrt", "nprobe": 4}
    for bad in ({"type": "pq"}, {"type": "hnsw", "nprobe": 2}, {"type": "ivf", "nprobe": 0},
                {"type": "hnsw", "M": "x"}):
        with pytest.raises(ValueError):
            ann.parse_spec(bad)


def test_nlist_for():
    assert ann.nlist_for(10122, "sqrt") == 101 and ann.nlist_for(3, "sqrt") == 2
    assert ann.nlist_for(5, 50) == 5 and ann.nlist_for(1, "sqrt") == 1


def test_overlap():
    assert ann.overlap([1, 2, 3], [3, 2, 9, 8]) == 0.5 and ann.overlap([1], []) is None


# ---- builds ------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", [{"type": "flat"}, {"type": "hnsw", "ef_search": 128},
                                  {"type": "ivf", "nprobe": 1000}])
def test_index_types_match_exact_when_exhaustive(spec):
    x, q = _vecs(300), _vecs(5, seed=1)
    exact = ann.build(x, {"type": "flat"}).search(q, 10)[1]
    got = ann.build(x, spec).search(q, 10)[1]
    for a, b in zip(got, exact):                 # IVF probing every list / HNSW with ef >> n on 300 points
        assert ann.overlap(list(a), list(b)) >= (1.0 if spec["type"] != "hnsw" else 0.9)


def test_ivf_one_probe_is_approximate_and_builds_reproducibly():
    x, q = _vecs(2000), _vecs(20, seed=1)
    exact = ann.build(x, {"type": "flat"}).search(q, 20)[1]
    a = ann.build(x, {"type": "ivf", "nprobe": 1}).search(q, 20)[1]
    b = ann.build(x, {"type": "ivf", "nprobe": 1}).search(q, 20)[1]
    assert (a == b).all()
    assert np.mean([ann.overlap(list(r), list(e)) for r, e in zip(a, exact)]) < 1.0


def test_tiny_index_types_build():
    for spec in ({"type": "ivf"}, {"type": "hnsw"}):
        idx = ann.build(_vecs(3), spec)
        assert idx.ntotal == 3 and len([i for i in idx.search(_vecs(1, seed=2), 20)[1][0] if i >= 0]) == 3


# ---- session index + retrievers ----------------------------------------------------------------
def _session(spec):
    texts = ["closing fee", "leverage ratio", "section prepayments", "annual facility fee"]
    chunks = [Chunk(f"c{i}", "d", i, t, []) for i, t in enumerate(texts)]
    return SessionIndex.build([(chunks, fake_embed(texts))], index=spec)


def test_flat_session_has_no_exact_copy_and_no_overlap_info():
    idx = _session(None)
    assert "exact" not in idx.extras
    assert DenseRetriever().search(idx, fake_embed(["fee"])[0], 3).info == {}


def test_approximate_session_reports_recall_vs_exact():
    idx = _session({"type": "hnsw"})
    assert "exact" in idx.extras
    hits = DenseRetriever().search(idx, fake_embed(["fee"])[0], 3)
    assert hits.info == {"exact_overlap": 1.0, "exact_overlap_n": 1}
    hy = HybridRetriever().search(idx, fake_embed(["fee"])[0], 3, query="fee")
    assert hy.info["exact_overlap_n"] == 1


def test_run_config_validates_index():
    with pytest.raises(ValueError):
        R.RunConfig("x", "C3", "R-dense", 5, index={"type": "lsh"})
    assert R.RunConfig("x", "C3", "R-dense", 5, index={"type": "ivf"}).index == {"type": "ivf"}


def test_run_stage_with_ivf(tmp_path):
    qpath = tmp_path / "q.jsonl"
    qpath.write_text(json.dumps(_q()) + "\n" + json.dumps(_q("q2")) + "\n")
    corpus = FakeCorpus()
    cache = IndexCache(corpus.load_source, fake_embed, corpus.parser_key, root=tmp_path / "cache")
    cfg = R.StageConfig("S3", "P2", [R.RunConfig("ivf", "C3", "R-hybrid", 5, index={"type": "ivf", "nprobe": 1})])
    s = R.run_stage(cfg, "dev", questions_path=qpath, out_root=tmp_path / "out", corpus=corpus,
                    embed=fake_embed, cache=cache)["ivf"]
    assert s["retriever_info"]["exact_overlap_n"] == 2 and 0 <= s["retriever_info"]["exact_overlap"] <= 2
    assert s["run"]["index"] == {"type": "ivf", "nprobe": 1}


# ---- pooled study ------------------------------------------------------------------------------
def test_measure_pooled():
    from evals.index_study import KS, measure, spec_name
    x, q = _vecs(500), _vecs(8, seed=3)
    flat = ann.build(x, {"type": "flat"})
    exact = {k: [list(map(int, r)) for r in flat.search(q, k)[1]] for k in KS}
    r = measure(x, q, {"type": "flat"}, exact, repeats=1)
    assert r["name"] == "flat" and r["recall_vs_exact@20"] == {"mean": 1.0, "min": 1.0} and r["bytes"] > 0
    r = measure(x, q, {"type": "ivf", "nprobe": 1}, exact, repeats=1)
    assert r["nlist_used"] == 22 and r["recall_vs_exact@10"]["mean"] <= 1.0
    assert spec_name({"type": "hnsw", "ef_search": 16}) == "hnsw-M32-ef16"


# ---- report ------------------------------------------------------------------------------------
def test_render_s3(tmp_path):
    from evals import report as RP
    from evals.tests.test_report import _write_run
    _write_run(tmp_path, "flat", [1.0] * 4, stage="s3")
    _write_run(tmp_path, "ivf-np1", [1.0, 1.0, 1.0, 0.0], stage="s3",
               info={"exact_overlap": 3.0, "exact_overlap_n": 4})
    runs = RP.load_runs("S3", "dev", ["flat", "ivf-np1"], root=tmp_path)
    for n, idx in (("flat", None), ("ivf-np1", {"type": "ivf", "nprobe": 1})):
        runs[n]["summary"]["run"]["index"] = idx
        runs[n]["summary"]["latency_ms"]["vector_search"] = {"p50": 0.0001, "p95": 0.0002}
        for row in runs[n]["rows"].values():
            row["index_chunks"] = 100
    pooled = {"n_vectors": 1000, "chunker": "C2", "n_docs": 3, "dim": 384, "n_queries": 4, "repeats": 5,
              "faiss_version": "1.8.0", "search_threads": 8, "questions_sha256": "a" * 64,
              "manifest_sha256": "m" * 64, "embedder": "minilm", "git_sha": "f" * 40,
              "results": [{"spec": {"type": "flat"}, "build_s": 0.01, "bytes": 2**20,
                           "recall_vs_exact@10": {"mean": 1.0, "min": 1.0},
                           "recall_vs_exact@20": {"mean": 1.0, "min": 1.0}, "search_ms": {"p50": 0.1, "p95": 0.2}}]}
    md = RP.render_s3("dev", runs, pooled)
    assert "**No pick**" in md and "IVF nlist=sqrt nprobe=1" in md and "| 0.750 |" in md
    assert "| Flat (exact) | 0.01 | 1.0 | 1.000 (min 1.00) |" in md and "1,000 `C2` vectors" in md
