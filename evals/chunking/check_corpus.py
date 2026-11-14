"""
Run chunkers over the whole corpus and check provenance (scope v2 §6.1, F1).

    venv/bin/python -m evals.chunking.check_corpus                 # C1 C2 C3 C4 C5
    venv/bin/python -m evals.chunking.check_corpus --chunkers C4

Documents: downstream_config("P2", part) (P2 digital, P2V scans), cleaner v2.
Checks per chunk (all chunkers): has text; every span non-empty, inside its page, not
blank-padded; every word of a span's text occurs in the chunk text (table pipes ignored).
C2/C3/C5 (exact): every span's text is a substring of the chunk text, and every non-blank page
character is in some chunk. C2–C5: the MiniLM count of the chunk text equals the chunker's
own count and fits 254. Coverage (% of non-blank page characters in no chunk span) is reported
for all; C4 leaves out page headers/footers/numbers on purpose, so it is not an error there.
Exit 1 on any failure. The §8.2 metrics: evals/chunking/run_metrics.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import structlog

from evals.chunking.base import has_text, is_blank
from evals.chunking.c1_line_window import C1LineWindow
from evals.chunking.c2_sentence_window import C2SentenceWindow
from evals.chunking.c3_fixed_tokens import C3FixedTokens
from evals.chunking.c4_layout import C4LayoutAware
from evals.chunking.c5_semantic import C5Semantic
from evals.chunking.source import load_source
from evals.chunking.tokens import MAX_CONTENT_TOKENS, minilm_counts
from evals.corpus.manifest import read_manifest
from evals.parsers.config import CONFIGS, downstream_config, load_cached

CHUNKERS = {"C1": C1LineWindow, "C2": C2SentenceWindow, "C3": C3FixedTokens, "C4": C4LayoutAware, "C5": C5Semantic}
EXACT = {"C2", "C3", "C5"}            # chunk text is a slice of the page stream
COUNTED = {"C2", "C3", "C4", "C5"}    # chunker reports its own MiniLM count and must fit the embedder
log = logging.getLogger("check_corpus")


def check_doc(src, chunks, exact: bool) -> tuple[list[str], int, int]:
    """(errors, uncovered non-blank chars, total non-blank chars)."""
    errs: list[str] = []
    covered = {p: np.zeros(len(t), bool) for p, t in src.pages.items()}
    for c in chunks:
        if not has_text(c.text):
            errs.append(f"{c.chunk_id}: empty text")
        flat = " ".join(c.text.replace("|", " ").split())
        for s in c.spans:
            page = src.pages.get(s.page)
            if page is None or not (0 <= s.start < s.end <= len(page)):
                errs.append(f"{c.chunk_id}: bad span {s}")
                continue
            piece = page[s.start:s.end]
            if not has_text(piece) or is_blank(piece[0]) or is_blank(piece[-1]):
                errs.append(f"{c.chunk_id}: empty/padded span {s}")
            if exact and piece not in c.text:
                errs.append(f"{c.chunk_id}: span text not in chunk {s}")
            elif not all(w in flat for w in piece.replace("|", " ").split()):
                errs.append(f"{c.chunk_id}: span words not in chunk {s}")
            covered[s.page][s.start:s.end] = True
    uncovered = total = 0
    for p, t in src.pages.items():
        blank = np.fromiter((is_blank(ch) for ch in t), bool, len(t))
        total += int((~blank).sum())
        gaps = np.flatnonzero(~covered[p] & ~blank)
        uncovered += len(gaps)
        if exact and len(gaps):
            errs.append(f"{src.doc_id} p{p}: {len(gaps)} chars in no chunk (first at {gaps[0]})")
    return errs, uncovered, total


def run(chunker_ids: list[str], table_config: str = "P2") -> dict:
    rows = read_manifest()
    chunkers = {k: CHUNKERS[k]() for k in chunker_ids}
    stats = {k: {"docs": 0, "chunks": 0, "spans": 0, "chunks_without_span": 0, "errors": [],
                 "tokens": [], "seconds": 0.0, "uncovered": 0, "chars": 0} for k in chunker_ids}
    missing = []
    for row in rows:
        cfg = downstream_config(table_config, row.part)
        cached = load_cached(CONFIGS[cfg], row.doc_id)
        if cached is None:
            missing.append(row.doc_id)
            continue
        src = load_source(cached, row.doc_id)
        for k, ch in chunkers.items():
            t0 = time.perf_counter()
            chunks = ch.chunk(src)
            st = stats[k]
            st["seconds"] += time.perf_counter() - t0
            st["docs"] += 1
            st["chunks"] += len(chunks)
            st["spans"] += sum(len(c.spans) for c in chunks)
            st["chunks_without_span"] += sum(not c.spans for c in chunks)
            errs, unc, tot = check_doc(src, chunks, k in EXACT)
            st["errors"] += errs
            st["uncovered"] += unc
            st["chars"] += tot
            counts = minilm_counts([c.text for c in chunks])
            st["tokens"] += counts
            if k in COUNTED:
                for c, n in zip(chunks, counts):
                    if n != c.meta["minilm_tokens"] or n > MAX_CONTENT_TOKENS:
                        st["errors"].append(f"{c.chunk_id}: {n} tokens (chunker said "
                                            f"{c.meta['minilm_tokens']})")
        log.info("%s (%s): done", row.doc_id, cfg)
    out = {"table_config": table_config, "missing_parse": missing, "chunkers": {}}
    for k, st in stats.items():
        tok = np.array(st.pop("tokens") or [0])
        out["chunkers"][k] = {
            **{x: st[x] for x in ("docs", "chunks", "spans", "chunks_without_span")},
            "seconds": round(st["seconds"], 1),
            "minilm_tokens_p5_p50_p95_max": [int(np.percentile(tok, q)) for q in (5, 50, 95)] + [int(tok.max())],
            "over_limit_pct": round(100 * float((tok > MAX_CONTENT_TOKENS).mean()), 1),
            "uncovered_chars_pct": round(100 * st["uncovered"] / max(st["chars"], 1), 2),
            "errors": len(st["errors"]), "first_errors": st["errors"][:10],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunkers", nargs="+", default=list(CHUNKERS), choices=sorted(CHUNKERS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    res = run(args.chunkers)
    print(json.dumps(res, indent=1))
    ok = not res["missing_parse"] and all(v["errors"] == 0 for v in res["chunkers"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
