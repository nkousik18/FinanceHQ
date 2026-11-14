"""
§8.2 chunk metrics over the corpus, per chunker, overall and by corpus part (scope v2 §8.2).

    venv/bin/python -m evals.chunking.run_metrics                    # C1–C5, with embedding/index
    venv/bin/python -m evals.chunking.run_metrics --chunkers C2 --no-index

Documents: downstream_config("P2", part), cleaner v2 (same as check_corpus). Definitions:
`evals/chunking/metrics.py`. Headline (★) truncation rates get document-bootstrap 95% CIs
(`evals/stats.ratio_ci`, documents as units). Evidence fragmentation needs the QA set (R3):
reported as pending here; `metrics.fragmentation` is ready for it.

Index size and build time: every chunk embedded with the production embedder (MiniLM,
normalized, batch 64, CPU), one FAISS IndexFlatIP per chunker over the whole corpus (production
builds one per session; Flat size is linear, so the total is the same). Memory = vectors × 384 ×
4 bytes; disk = the written index file.

Outputs (gitignored): data/results/r2/chunk_metrics.json, chunk_metrics_docs.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import structlog

from evals.chunking import metrics as M
from evals.chunking.check_corpus import CHUNKERS
from evals.chunking.source import CLEANING_VERSION, load_source
from evals.chunking.tokens import MINILM_NAME
from evals.corpus.manifest import read_manifest
from evals.parsers.config import CONFIGS, downstream_config, load_cached
from evals.stats import ratio_ci

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "r2"
log = logging.getLogger("run_metrics")

COUNTS = ("chunks", "truncated", "tokens", "tokens_dropped", "page_words", "page_words_never_embedded",
          "unaligned_words", "tables", "tables_split", "tables_broken", "boundaries", "mid_sentence",
          "indexed_tokens", "duplicate_tokens", "chunk_seconds", "embed_seconds")


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MINILM_NAME, device="cpu")

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def doc_metrics(src, chunks, chunk_seconds: float, embed=None) -> tuple[dict, M.Sizes, np.ndarray | None]:
    sz = M.sizes(chunks)
    tr = M.truncation(src, chunks, sz.minilm)
    tb = M.table_breaks(src, chunks)
    sb = M.sentence_breaks(src, chunks)
    rd = M.redundancy(src, chunks)
    vecs, embed_s = None, 0.0
    if embed is not None and chunks:
        t0 = time.perf_counter()
        vecs = embed([c.text for c in chunks])
        embed_s = time.perf_counter() - t0
    row = {**asdict(tr), "tables": tb.tables, "tables_split": tb.split, "tables_broken": tb.broken,
           "boundaries": sb.boundaries, "mid_sentence": sb.mid_sentence,
           "indexed_tokens": rd.indexed_tokens, "duplicate_tokens": rd.duplicate_tokens,
           "chunk_seconds": round(chunk_seconds, 3), "embed_seconds": round(embed_s, 3)}
    return row, sz, vecs


def _pct(a: float, b: float) -> float | None:
    return round(100 * a / b, 2) if b else None


def _dist(xs: list[int]) -> list[int]:
    a = np.array(xs or [0])
    return [int(np.percentile(a, q)) for q in (5, 50, 95)] + [int(a.max())]


def _ci_pct(nums, dens) -> dict:
    ci = ratio_ci(nums, dens)
    return {k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in ci.items()}


def summarize(rows: list[dict], sizes: list[M.Sizes]) -> dict:
    """Pooled metrics over documents (rows and sizes aligned)."""
    tot = {k: sum(r[k] for r in rows) for k in COUNTS}
    mini = [n for s in sizes for n in s.minilm]
    return {
        "docs": len(rows), "chunks": tot["chunks"],
        "size_p5_p50_p95_max": {"minilm": _dist(mini), "tiktoken": _dist([n for s in sizes for n in s.tiktoken]),
                                "words": _dist([n for s in sizes for n in s.words])},
        "truncation_rate_pct": _ci_pct([r["truncated"] for r in rows], [r["chunks"] for r in rows]),
        "tokens_dropped_per_copy_pct": _pct(tot["tokens_dropped"], tot["tokens"]),
        "page_words_never_embedded_pct": _ci_pct([r["page_words_never_embedded"] for r in rows],
                                                 [r["page_words"] for r in rows]),
        "unaligned_words": tot["unaligned_words"],
        "tables": tot["tables"], "tables_split_pct": _pct(tot["tables_split"], tot["tables"]),
        "table_break_rate_pct": _pct(tot["tables_broken"], tot["tables"]),
        "boundaries": tot["boundaries"], "sentence_break_rate_pct": _pct(tot["mid_sentence"], tot["boundaries"]),
        "overlap_redundancy_pct": _pct(tot["duplicate_tokens"], tot["indexed_tokens"]),
        "evidence_fragmentation_pct": "pending QA set (R3)",
        "chunk_seconds": round(tot["chunk_seconds"], 1), "embed_seconds": round(tot["embed_seconds"], 1),
    }


def index_stats(vecs: list[np.ndarray]) -> dict:
    import faiss
    if not vecs:
        return {}
    x = np.vstack(vecs)
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    add_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "index.faiss"
        faiss.write_index(index, str(path))
        disk = path.stat().st_size
    return {"vectors": int(index.ntotal), "dim": int(x.shape[1]), "memory_mb": round(x.nbytes / 2**20, 2),
            "disk_mb": round(disk / 2**20, 2), "faiss_add_seconds": round(add_s, 3)}


def run(chunker_ids: list[str], with_index: bool = True, table_config: str = "P2") -> tuple[dict, list[dict]]:
    chunkers = {k: CHUNKERS[k]() for k in chunker_ids}
    embed = Embedder() if with_index else None
    per: dict[str, list[tuple[str, dict, M.Sizes]]] = {k: [] for k in chunker_ids}
    vecs: dict[str, list[np.ndarray]] = {k: [] for k in chunker_ids}
    doc_rows, missing = [], []
    for row in read_manifest():
        cfg = downstream_config(table_config, row.part)
        cached = load_cached(CONFIGS[cfg], row.doc_id)
        if cached is None:
            missing.append(row.doc_id)
            continue
        src = load_source(cached, row.doc_id)
        for k, ch in chunkers.items():
            t0 = time.perf_counter()
            chunks = ch.chunk(src)
            m, sz, v = doc_metrics(src, chunks, time.perf_counter() - t0, embed)
            per[k].append((row.part, m, sz))
            if v is not None:
                vecs[k].append(v)
            doc_rows.append({"chunker": k, "doc_id": row.doc_id, "part": row.part, "config": cfg, **m})
        log.warning("%s (%s): done", row.doc_id, cfg)

    out = {"table_config": table_config, "cleaning_version": CLEANING_VERSION, "missing_parse": missing,
           "chunkers": {}}
    for k, items in per.items():
        parts = sorted({p for p, _, _ in items})
        res = summarize([m for _, m, _ in items], [s for _, _, s in items])
        res["by_part"] = {p: summarize([m for q, m, _ in items if q == p], [s for q, _, s in items if q == p])
                          for p in parts}
        res["params"] = chunkers[k].params
        if with_index:
            res["index"] = index_stats(vecs[k])
            n_docs = max(len(items), 1)
            res["index"]["embed_seconds_per_doc"] = round(res["embed_seconds"] / n_docs, 2)
        out["chunkers"][k] = res
    return out, doc_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunkers", nargs="+", default=list(CHUNKERS), choices=sorted(CHUNKERS))
    ap.add_argument("--no-index", action="store_true", help="skip embedding and the FAISS index")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    res, docs = run(args.chunkers, with_index=not args.no_index)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunk_metrics.json").write_text(json.dumps(res, indent=1))
    with (OUT / "chunk_metrics_docs.jsonl").open("w") as fh:
        for r in docs:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: {x: v[x] for x in v if x != "by_part"} for k, v in res["chunkers"].items()}, indent=1))
    return 1 if res["missing_parse"] else 0


if __name__ == "__main__":
    sys.exit(main())
