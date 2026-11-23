"""
Eval runner (scope v2 §9.2), retrieval stages (S1 now; S2/S3 reuse it).

    venv/bin/python -m evals.run --config evals/configs/s1_chunkers.yaml --split dev
    venv/bin/python -m evals.run --config evals/configs/s1_chunkers.yaml --split dev --runs C2-dense --limit 5

Config (YAML): `stage`, `table_config`, and `runs`: an explicit list (never a generated product) of
{name, chunker, chunker_params?, retriever, top_k}. Per run and question:
  1. session view (views.py: digital docs, or the cited scan swapped in for its source page)
  2. each document view's chunks + vectors from the chunk/embedding cache (index_cache.py)
  3. one FAISS index per view (like production's per-session index), query embedded with MiniLM
  4. retrieve max(top_k, 10) chunks; only the first top_k can enter the 1,200-word budget
  5. §8.3 metrics (retrieval/metrics.py), relevant chunks counted over the whole session index

Output (gitignored) data/results/r4/<stage>/<split>/<run>/: questions.jsonl (per question: metrics,
ranked chunk ids, scores, covered spans, timings) and summary.json (config, git SHA, manifest and
golden-set hashes, metric breakdowns, latency p50/p95, cache stats). MLflow logging: R4 step 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import structlog
import yaml

from evals.chunking.check_corpus import CHUNKERS
from evals.chunking.source import CLEANING_VERSION
from evals.corpus.manifest import MANIFEST, read_manifest
from evals.golden.qa.validate import QUESTIONS, read_questions
from evals.retrieval import metrics as M
from evals.retrieval.index_cache import EMBEDDER_ID, IndexCache
from evals.retrieval.views import DocView, session_view
from evals.retrievers.dense import RETRIEVERS, SessionIndex

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "data" / "results" / "r4"
RANK_DEPTH = 10                       # @10 metrics need at least 10 ranked chunks
log = logging.getLogger("evals.run")


# ---- config ------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str
    chunker: str
    retriever: str
    top_k: int
    chunker_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.chunker not in CHUNKERS:
            raise ValueError(f"run {self.name}: unknown chunker {self.chunker!r}")
        if self.retriever not in RETRIEVERS:
            raise ValueError(f"run {self.name}: unknown retriever {self.retriever!r}")
        if not (isinstance(self.top_k, int) and self.top_k >= 1):
            raise ValueError(f"run {self.name}: top_k must be a positive int")


@dataclass
class StageConfig:
    stage: str
    table_config: str
    runs: list[RunConfig]


def load_config(path: Path) -> StageConfig:
    raw = yaml.safe_load(path.read_text())
    runs = [RunConfig(**r) for r in raw.get("runs") or []]
    if not runs:
        raise ValueError(f"{path}: no runs")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate run names")
    return StageConfig(raw["stage"], raw.get("table_config", "P2"), runs)


# ---- provenance --------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---- corpus access (real parse cache) ----------------------------------------------------------
_KEY = re.compile(r'"cache_key":\s*"([0-9a-f]{64})"')


class Corpus:
    """Manifest facts + parse-cache access for the index cache."""

    def __init__(self, table_config: str = "P2"):
        self.table_config = table_config
        rows = read_manifest()
        self.part_of = {r.doc_id: r.part for r in rows}
        self.role_of = {r.doc_id: r.role for r in rows}
        self._keys: dict[str, str] = {}

    def _config(self, doc_id: str):
        from evals.parsers.config import CONFIGS, downstream_config
        return CONFIGS[downstream_config(self.table_config, self.part_of[doc_id])]

    def parser_key(self, doc_id: str) -> str:
        """The parse's cache key, read from the file head (meta is written first)."""
        if doc_id not in self._keys:
            from evals.parsers.config import cache_path
            path = cache_path(self._config(doc_id), doc_id)
            if not path.exists():
                raise FileNotFoundError(f"{doc_id}: no parse cache at {path}")
            with path.open() as fh:
                m = _KEY.search(fh.read(4096))
            if not m:
                raise ValueError(f"{doc_id}: no cache_key in {path}")
            self._keys[doc_id] = m[1]
        return self._keys[doc_id]

    def load_source(self, doc_id: str, drop_pages: frozenset[int]):
        from evals.chunking.source import load_source
        from evals.parsers.config import load_cached
        cached = load_cached(self._config(doc_id), doc_id)
        if cached is None:
            raise FileNotFoundError(f"{doc_id}: no parse cache")
        return load_source(cached, doc_id, drop_pages=drop_pages), cached["meta"]["cache_key"]

    def amendment_docs(self) -> set[str]:
        return {d for d, r in self.role_of.items() if r.startswith("amendment")}


class MiniLM:
    """Production embedder settings (app/pipeline/embedder.py): normalized, float32, CPU."""

    def __init__(self):
        from evals.chunking.run_metrics import Embedder
        self._e = Embedder()

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self._e(texts)


# ---- one run -----------------------------------------------------------------------------------
def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def run_one(rc: RunConfig, questions: list[dict], corpus, cache: IndexCache, embed) -> tuple[list[dict], dict]:
    chunker = CHUNKERS[rc.chunker](**rc.chunker_params)
    retriever = RETRIEVERS[rc.retriever]()
    amend = corpus.amendment_docs()
    sessions: dict[tuple[DocView, ...], tuple[SessionIndex, dict]] = {}
    rows, results, lat = [], [], {"query_embed": [], "vector_search": [], "total": []}
    t_run = time.perf_counter()
    for q in questions:
        view = tuple(session_view(q, corpus.part_of, corpus.role_of))
        if view not in sessions:
            parts = [cache.get(chunker, v.doc_id, v.drop_pages) for v in view]
            idx = SessionIndex.build([(p.chunks, p.vectors) for p in parts])
            sessions[view] = (idx, {p.source.doc_id: p.source for p in parts})
        idx, sources = sessions[view]
        t0 = time.perf_counter()
        qvec = embed([q["question"]])[0]
        t_embed = time.perf_counter() - t0
        hits = retriever.search(idx, qvec, max(rc.top_k, RANK_DEPTH))
        t_total = time.perf_counter() - t0
        ranked = [idx.chunks[i] for i in hits.ids]
        n_rel = None
        if q["evidence"] and not q.get("unanswerable"):
            ev = [M.Evidence(e["doc_id"], e["page"], e["span_text"]) for e in q["evidence"]]
            n_rel = M.count_relevant(idx.chunks, M.locate_all(ev, sources), sources)
        part = M.question_part(q, corpus.part_of)
        r = M.score_question(q, part, ranked, hits.scores, sources, n_relevant=n_rel,
                             amendment_docs=amend, budget_pool=rc.top_k)
        timings = {"query_embed": t_embed, **hits.timings, "total": t_total}
        for k in lat:
            lat[k].append(timings.get(k, 0.0))
        results.append(r)
        rows.append({"qid": q["qid"], "type": q["type"], "part": part, "split": q["split"],
                     "view": [[v.doc_id, sorted(v.drop_pages)] for v in view],
                     "index_chunks": len(idx.chunks), "n_relevant": n_rel, "unlocated": r.unlocated,
                     "n_budget": r.n_budget, "metrics": r.metrics, "ranked": r.ranked,
                     "timings_s": timings})
    summary = {
        "breakdown": M.breakdown(results),
        "latency_ms": {k: {"p50": None if not v else 1000 * _pct(v, 50),
                           "p95": None if not v else 1000 * _pct(v, 95)} for k, v in lat.items()},
        "questions": len(questions), "unlocated_spans": sum(r.unlocated for r in results),
        "sessions_built": len(sessions), "seconds": round(time.perf_counter() - t_run, 1),
    }
    return rows, summary


# ---- stage -------------------------------------------------------------------------------------
def select_questions(questions: list[dict], split: str, limit: int | None) -> list[dict]:
    qs = [q for q in questions if not q.get("retired") and (split == "all" or q["split"] == split)]
    return qs[:limit] if limit else qs


def write_run(out: Path, rows: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "questions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def run_stage(cfg: StageConfig, split: str, only: list[str] | None = None, limit: int | None = None,
              questions_path: Path = QUESTIONS, out_root: Path = OUT_ROOT, corpus=None, embed=None,
              cache: IndexCache | None = None) -> dict[str, dict]:
    questions = select_questions(read_questions(questions_path), split, limit)
    if not questions:
        raise ValueError(f"no questions in split {split!r}")
    corpus = corpus or Corpus(cfg.table_config)
    embed = embed or MiniLM()
    cache = cache or IndexCache(corpus.load_source, embed, corpus.parser_key)
    prov = {"git_sha": git_sha(), "manifest_sha256": file_sha256(MANIFEST),
            "questions_sha256": file_sha256(questions_path), "embedder": EMBEDDER_ID,
            "table_config": cfg.table_config, "cleaning_version": CLEANING_VERSION,
            "split": split, "limit": limit, "stage": cfg.stage}
    runs = [r for r in cfg.runs if not only or r.name in only]
    if only and len(runs) != len(only):
        raise ValueError(f"unknown runs: {sorted(set(only) - {r.name for r in runs})}")
    out: dict[str, dict] = {}
    for rc in runs:
        built0, loaded0 = cache.built, cache.loaded
        log.info("run %s (%s + %s, top_k %d) on %d questions", rc.name, rc.chunker, rc.retriever,
                 rc.top_k, len(questions))
        rows, summary = run_one(rc, questions, corpus, cache, embed)
        summary = {"run": asdict(rc), **prov, **summary,
                   "cache": {"built": cache.built - built0, "loaded": cache.loaded - loaded0}}
        write_run(out_root / cfg.stage.lower() / split / rc.name, rows, summary)
        o = summary["breakdown"]["overall"]
        log.info("run %s: recall@budget %.3f, mrr@10 %.3f, entity recall %.3f (%s s)", rc.name,
                 o["recall@budget"]["mean"], o["mrr@10"]["mean"], o["entity_recall"]["mean"],
                 summary["seconds"])
        out[rc.name] = summary
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--runs", help="comma-separated run names (default: all in the config)")
    ap.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    if a.split == "test":
        log.warning("test split: S1–S3 pick on dev; test is for S4 (scope §9.3)")
    cfg = load_config(a.config)
    run_stage(cfg, a.split, a.runs.split(",") if a.runs else None, a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
