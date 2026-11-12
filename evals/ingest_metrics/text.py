"""
Text accuracy metrics (scope v2 §8.1): CER, WER, numeric fidelity, text-layer usability.

Reference text: the PDF's embedded text layer for digital pages; for simulated
scans, the text layer of the page they were rendered from.
"""
from __future__ import annotations

from collections import Counter

from rapidfuzz.distance import Levenshtein

from evals.ingest_metrics.normalize import norm_match, norm_text, numeric_tokens

MIN_REF_WORDS = 50
MAX_NONPRINTABLE = 0.05


def cer(hyp: str, ref: str) -> float | None:
    """Character edit distance / reference length, after norm_text. None if the reference is empty."""
    h, r = norm_text(hyp), norm_text(ref)
    if not r:
        return None
    return Levenshtein.distance(h, r) / len(r)


def wer(hyp: str, ref: str) -> float | None:
    """Word edit distance / reference word count, after norm_text."""
    h, r = norm_text(hyp).split(), norm_text(ref).split()
    if not r:
        return None
    return Levenshtein.distance(h, r) / len(r)


def bag_word_recall(hyp: str, ref: str) -> float | None:
    """Diagnostic: share of reference words found in the hypothesis as a multiset (case and
    punctuation ignored). Unlike CER it ignores reading order, so the two together separate
    'wrong characters' from 'different order'."""
    r = Counter(norm_match(ref).split())
    if not r:
        return None
    h = Counter(norm_match(hyp).split())
    return sum((r & h).values()) / sum(r.values())


def numeric_fidelity(hyp: str, ref: str) -> float | None:
    """Share of the reference's numeric tokens found in the hypothesis, compared as multisets
    (so reading order can't affect it). None if the reference has no numbers."""
    r = Counter(numeric_tokens(ref))
    if not r:
        return None
    h = Counter(numeric_tokens(hyp))
    return sum((r & h).values()) / sum(r.values())


def text_layer_usable(text: str) -> tuple[bool, str]:
    """A text layer is a usable reference if it has >= 50 words and <= 5% non-printable characters."""
    words = len(text.split())
    if words < MIN_REF_WORDS:
        return False, f"only {words} words"
    nonprint = sum(1 for ch in text if not ch.isprintable() and ch not in "\n\t\r")
    if text and nonprint / len(text) > MAX_NONPRINTABLE:
        return False, f"{nonprint / len(text):.1%} non-printable"
    return True, ""
