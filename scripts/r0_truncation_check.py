"""
R0 — MiniLM truncation check for the production chunker (C1).

Question: what fraction of C1 chunks are longer than all-MiniLM-L6-v2's input
limit, and how much text is silently dropped (never embedded)?

Input: a saved Textract AnalyzeDocument response (no AWS call is made). The
response is run through the production path: _assemble_result -> clean_extraction
-> chunk_document (default C1 settings) -> MiniLM tokenizer.

Usage (from repo root):
    venv/bin/python scripts/r0_truncation_check.py \
        --textract scripts/output/loan1_raw.json \
        --out docs/r0_truncation.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fastapi_service"))

from app.pipeline.extractor import _assemble_result  # noqa: E402
from app.pipeline.cleaner import clean_extraction  # noqa: E402
from app.pipeline.chunker import (  # noqa: E402
    chunk_document,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from app.pipeline.embedder import MODEL_NAME  # noqa: E402


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _overlap_len(prev: list[str], cur: list[str]) -> int:
    """Longest suffix of prev that is a prefix of cur (the chunker's overlap)."""
    for k in range(min(len(prev), len(cur)), 0, -1):
        if prev[-k:] == cur[:k]:
            return k
    return 0


def _unique_coverage(chunks, per_chunk) -> dict:
    """
    Lay chunks end to end on one word axis (removing each overlap), mark the
    words each chunk actually embeds (its first `words - words_dropped`), and
    count document words that no chunk embeds.
    """
    covered: list[bool] = []
    prev_words: list[str] = []
    for c, p in zip(chunks, per_chunk):
        words = c.text.split()
        start = len(covered) - _overlap_len(prev_words, words)
        covered.extend([False] * (start + len(words) - len(covered)))
        for i in range(len(words) - p["words_dropped"]):
            covered[start + i] = True
        prev_words = words
    return {"total": len(covered), "never": covered.count(False)}


def measure(textract_json: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    raw = json.loads(textract_json.read_text())
    blocks = raw["Blocks"]
    extraction = _assemble_result(session_id="r0-truncation", blocks=blocks)
    cleaned = clean_extraction(extraction)
    chunks = chunk_document(cleaned)  # production defaults = C1

    model = SentenceTransformer(MODEL_NAME)
    tok = model.tokenizer
    max_len = model.max_seq_length  # includes [CLS] and [SEP]
    content_budget = max_len - 2

    per_chunk = []
    for c in chunks:
        ids = tok(c.text, add_special_tokens=False)["input_ids"]
        n = len(ids)
        kept = min(n, content_budget)
        # Approximate the words that survive: decode the kept tokens.
        kept_words = len(tok.decode(ids[:kept]).split()) if n > content_budget else c.word_count
        per_chunk.append({
            "chunk_id": c.chunk_id,
            "words": c.word_count,
            "est_tokens_chunker": int(c.word_count / 0.75),
            "minilm_tokens": n,
            "truncated": n > content_budget,
            "tokens_dropped": max(0, n - content_budget),
            "words_dropped": max(0, c.word_count - kept_words),
        })

    unique = _unique_coverage(chunks, per_chunk)

    n_chunks = len(per_chunk)
    tokens = [p["minilm_tokens"] for p in per_chunk]
    total_tokens = sum(tokens)
    total_words = sum(p["words"] for p in per_chunk)
    truncated = [p for p in per_chunk if p["truncated"]]

    def q(vals, pct):
        s = sorted(vals)
        return s[min(len(s) - 1, int(round(pct / 100 * (len(s) - 1))))] if s else 0

    return {
        "document": textract_json.name,
        "pages": cleaned.total_pages,
        "chunker": {"id": "C1", "chunk_size": DEFAULT_CHUNK_SIZE, "overlap": DEFAULT_CHUNK_OVERLAP,
                    "size_unit": "words / 0.75"},
        "embedder": {"model": MODEL_NAME, "max_seq_length": max_len,
                     "content_tokens_after_special": content_budget},
        "chunks": n_chunks,
        "minilm_tokens_per_chunk": {"p5": q(tokens, 5), "p50": q(tokens, 50), "p95": q(tokens, 95),
                                    "max": max(tokens) if tokens else 0,
                                    "mean": round(statistics.mean(tokens), 1) if tokens else 0},
        "tokens_per_word_ratio": round(total_tokens / total_words, 3) if total_words else 0,
        "truncation_rate_pct": _pct(len(truncated), n_chunks),
        "tokens_never_embedded_pct": _pct(sum(p["tokens_dropped"] for p in per_chunk), total_tokens),
        "words_never_embedded_pct": _pct(sum(p["words_dropped"] for p in per_chunk), total_words),
        "unique_words_in_document": unique["total"],
        "unique_words_never_embedded_pct": _pct(unique["never"], unique["total"]),
        "note": ("*_never_embedded_pct count each chunk copy separately (overlap double-counted). "
                 "unique_words_never_embedded_pct de-duplicates overlap: it is the share of the "
                 "document's words that no chunk embeds at all."),
        "per_chunk": per_chunk,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--textract", type=Path, default=REPO / "scripts/output/loan1_raw.json")
    ap.add_argument("--out", type=Path, default=REPO / "docs/r0_truncation.json")
    args = ap.parse_args()

    result = measure(args.textract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    summary = {k: v for k, v in result.items() if k != "per_chunk"}
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
