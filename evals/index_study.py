"""
S3 pooled index study (scope v2 §7 "Index types", S-2026-11-22-33): one FAISS index over the whole
dev corpus, to show the recall-vs-latency curve that per-session indexes (3–1,489 chunks) are too
small to show.

    venv/bin/python -m evals.index_study --config evals/configs/s3_index_types.yaml

Pool = the run chunker's chunks of every document in a dev question's session view, each document
once with no dropped pages (scans included as their own documents). Queries = the dev questions
(MiniLM, as production). For each index spec in the config's `pooled.specs`:
  - build time (single-threaded, see ann.py) and serialized size
  - per query: top 10 and top 20 vs exact IndexFlatIP top 10 / 20 (§8.3 recall vs exact search)
  - search latency: each query searched `repeats` times, its median kept; p50 / p95 over queries
    (single-threaded FAISS, see dense._faiss)
No pick (§9.3): the output is the curve. Writes data/results/r4/s3/<split>/pooled/summary.json and one
MLflow run per spec (experiment loandoc_retrieval_evals, tag `study=pooled`).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from evals import run as R
from evals.retrievers import ann

log = logging.getLogger("evals.index_study")
KS = (10, 20)


def spec_name(spec: dict) -> str:
    s = ann.parse_spec(spec)
    if s["type"] == "hnsw":
        return f"hnsw-M{s['M']}-ef{s['ef_search']}"
    if s["type"] == "ivf":
        return f"ivf-{s['nlist']}-np{s['nprobe']}"
    return "flat"


def pool_vectors(questions: list[dict], corpus, cache, chunker) -> tuple[np.ndarray, list[str]]:
    from evals.retrieval.views import session_view
    docs = sorted({v.doc_id for q in questions for v in session_view(q, corpus.part_of, corpus.role_of)})
    parts = [cache.get(chunker, d) for d in docs]
    x = np.vstack([p.vectors for p in parts if len(p.vectors)]).astype(np.float32)
    return x, docs


def measure(x: np.ndarray, queries: np.ndarray, spec: dict, exact: dict[int, list[list[int]]],
            repeats: int = 3) -> dict:
    t0 = time.perf_counter()
    index = ann.build(x, spec)
    build_s = time.perf_counter() - t0
    from evals.retrievers.dense import _faiss
    size = int(_faiss().serialize_index(index).size)
    times, found = [], []
    for q in queries:
        qv = q.reshape(1, -1)
        index.search(qv, max(KS))                         # warm-up
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            _, ids = index.search(qv, max(KS))
            ts.append(time.perf_counter() - t)
        times.append(float(np.median(ts)))
        found.append([int(i) for i in ids[0] if i >= 0])
    out = {"name": spec_name(spec), "spec": ann.parse_spec(spec), "build_s": round(build_s, 3),
           "bytes": size, "search_ms": {"p50": 1000 * float(np.percentile(times, 50)),
                                        "p95": 1000 * float(np.percentile(times, 95))}}
    if out["spec"]["type"] == "ivf":
        out["nlist_used"] = ann.nlist_for(len(x), out["spec"]["nlist"])
    for k in KS:
        ov = [ann.overlap(f[:k], e) for f, e in zip(found, exact[k])]
        out[f"recall_vs_exact@{k}"] = {"mean": float(np.mean(ov)), "min": float(np.min(ov))}
    return out


def run_study(config: Path, split: str = "dev", mlflow: bool = True, out_root: Path = R.OUT_ROOT) -> dict:
    raw = yaml.safe_load(config.read_text())
    pooled = raw["pooled"]
    corpus = R.Corpus(raw.get("table_config", "P2"))
    embed = R.MiniLM()
    from evals.retrieval.index_cache import IndexCache
    cache = IndexCache(corpus.load_source, embed, corpus.parser_key)
    from evals.golden.qa.validate import QUESTIONS, read_questions
    questions = R.select_questions(read_questions(QUESTIONS), split, None)
    x, docs = pool_vectors(questions, corpus, cache, R.CHUNKERS[pooled["chunker"]]())
    queries = embed([q["question"] for q in questions]).astype(np.float32)
    flat = ann.build(x, {"type": "flat"})
    exact = {k: [[int(i) for i in row if i >= 0] for row in flat.search(queries, k)[1]] for k in KS}
    from evals.retrievers.dense import _faiss
    faiss = _faiss()
    results = []
    for spec in pooled["specs"]:
        r = measure(x, queries, spec, exact, pooled.get("repeats", 3))
        log.info("%s: recall vs exact@20 %.3f, search p50 %.3f ms, build %.2f s", r["name"],
                 r["recall_vs_exact@20"]["mean"], r["search_ms"]["p50"], r["build_s"])
        results.append(r)
    summary = {"stage": raw["stage"], "study": "pooled", "split": split, "chunker": pooled["chunker"],
               "n_vectors": int(len(x)), "dim": int(x.shape[1]), "n_docs": len(docs), "n_queries": len(queries),
               "repeats": pooled.get("repeats", 3), "faiss_version": faiss.__version__,
               "search_threads": faiss.omp_get_max_threads(), "git_sha": R.git_sha(),
               "manifest_sha256": R.file_sha256(R.MANIFEST), "questions_sha256": R.file_sha256(QUESTIONS),
               "embedder": R.EMBEDDER_ID, "table_config": corpus.table_config, "results": results}
    out = out_root / raw["stage"].lower() / split / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if mlflow:
        log_mlflow(summary)
    return summary


def log_mlflow(summary: dict) -> None:
    from evals.tracking import RETRIEVAL_EXPERIMENT, flat_params, metric_name, use_experiment
    mlflow = use_experiment(RETRIEVAL_EXPERIMENT)
    for r in summary["results"]:
        with mlflow.start_run(run_name=f"{summary['stage']} pooled {r['name']} {summary['split']}"):
            mlflow.set_tags({"stage": summary["stage"], "split": summary["split"], "study": "pooled",
                             "smoke": "False", "index": r["spec"]["type"]})
            mlflow.log_params({**flat_params(r["spec"], "index."),
                               **flat_params({k: summary[k] for k in ("chunker", "n_vectors", "n_queries", "repeats",
                                                                      "faiss_version", "search_threads", "git_sha",
                                                                      "manifest_sha256", "questions_sha256",
                                                                      "embedder")})})
            m = {"build_s": r["build_s"], "bytes": r["bytes"],
                 "search_p50_ms": r["search_ms"]["p50"], "search_p95_ms": r["search_ms"]["p95"]}
            for k in KS:
                m[metric_name(f"recall_vs_exact@{k}", "mean")] = r[f"recall_vs_exact@{k}"]["mean"]
                m[metric_name(f"recall_vs_exact@{k}", "min")] = r[f"recall_vs_exact@{k}"]["min"]
            mlflow.log_metrics(m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev"], default="dev")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_study(a.config, a.split, mlflow=not a.no_mlflow)
    return 0


if __name__ == "__main__":
    sys.exit(main())
