"""
Parser confidence metrics (scope v2 §8.1, Auto): read straight from the parse cache.

Two sources (Decision Log 2026-11-04):
- OCR cells (Tesseract): per-word confidence 0-1, on OCR'd pages only
- Docling's confidence report: per-page parse / layout / table / ocr scores and a grade
"""
from __future__ import annotations

import math

LOW_90, LOW_80 = 0.90, 0.80
LOWEST_GRADES = {"poor"}


def _quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(vals: list[float]) -> dict:
    return {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None,
            "p10": _quantile(vals, 0.10), "min": min(vals) if vals else None}


def ocr_word_confidences(cached: dict, page: int | None = None) -> list[float]:
    cells = cached.get("ocr_cells", {})
    pages = [str(page)] if page is not None else list(cells)
    return [c["confidence"] for p in pages for c in cells.get(p, []) if c.get("text", "").strip()]


def page_confidence(cached: dict, page: int) -> dict:
    words = ocr_word_confidences(cached, page)
    rep = cached.get("confidence", {}).get("pages", {}).get(str(page), {})
    s = summarize(words)
    low_page = (s["p10"] is not None and s["p10"] < LOW_80) or rep.get("low_grade") in LOWEST_GRADES
    return {
        "ocr": s,
        "low_conf_rate_90": sum(w < LOW_90 for w in words) / len(words) if words else None,
        "low_conf_rate_80": sum(w < LOW_80 for w in words) / len(words) if words else None,
        "docling": {k: rep.get(k) for k in ("parse_score", "layout_score", "table_score", "ocr_score",
                                            "mean_grade", "low_grade")},
        "low_confidence_page": bool(low_page),
    }


def document_ingestion_score(cached: dict) -> dict:
    """
    OCR'd documents: % of words with confidence >= 0.90.
    Digital documents: mean Docling page parse_score.
    The two kinds are reported separately and never pooled (§8.7).
    """
    words = ocr_word_confidences(cached)
    if words:
        return {"kind": "ocr", "score": sum(w >= LOW_90 for w in words) / len(words), "n_words": len(words)}
    scores = [p.get("parse_score") for p in cached.get("confidence", {}).get("pages", {}).values()
              if p.get("parse_score") is not None]
    return {"kind": "digital", "score": sum(scores) / len(scores) if scores else None, "n_pages": len(scores)}
