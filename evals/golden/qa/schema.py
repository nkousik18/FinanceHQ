"""
QA golden-set schema and validator (scope v2 §5.1–5.3, F3).

One question per line in evals/golden/qa/questions.jsonl:

{
  "qid": "q0042", "split": "test" | "dev", "type": "cross_doc",
  "question": "...",
  "session_id": "deal_07",                       # manifest deal_id
  "session_docs": ["deal_07_ca", "deal_07_am1", ...],
  "evidence": [{"doc_id": "deal_07_am2", "page": 3, "span_text": "..."}],
  "reference_answer": "...", "key_points": ["..."], "entities": ["..."],
  "superseded_values": ["..."],                  # amendment questions only
  "amendment": false,                            # added to §5.2 (Decision Log 2026-11-13)
  "answer_numeric": null, "numeric_tolerance": null,
  "unanswerable": false,
  "provenance": "llm_drafted_human_verified",
  "verified_by": "kousik", "verified_on": "2026-11-20",
  "retired": false                               # optional; a fix = new qid + old one retired
}

F3 checks (`validate_set`): span_text found in the cleaned page text (parse
downstream_config("P2", part), cleaner v2; exact, else whitespace-insensitive); every entity
found in the evidence spans or the reference answer; every superseded value found in the
session's documents; qids unique; each split's type mix within ±1 of proportional. Plus
schema checks per question. `final=True` also enforces the §5.1 quotas (sizes, type counts,
corpus shares, scans, amendment questions).

Matching of entities and superseded values (scope §8.3 "after normalization"): a value with
numbers matches if all its numeric tokens (evals/ingest_metrics/normalize.numeric_tokens) occur
in the target; otherwise its normalized text (norm_match) must be a substring of the target's.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from evals.ingest_metrics.normalize import norm_match, numeric_tokens

TYPES = ("lookup", "table", "calculation", "cross_doc", "explain", "unanswerable")
SPLITS = ("dev", "test")
PROVENANCE = {"llm_drafted_human_verified", "human_written"}
REQUIRED = ("qid", "split", "type", "question", "session_id", "session_docs", "evidence",
            "reference_answer", "key_points", "entities", "superseded_values", "amendment",
            "answer_numeric", "numeric_tolerance", "unanswerable", "provenance",
            "verified_by", "verified_on")

# §5.1 quotas (final set)
TYPE_COUNTS = {"lookup": 25, "table": 25, "calculation": 20, "cross_doc": 15, "explain": 15,
               "unanswerable": 20}
SPLIT_SIZES = {"dev": 40, "test": 80}
MIN_EDGAR_SHARE = 0.70
MIN_CONSUMER_SHARE = 0.20
MIN_SCAN_QUESTIONS = 8
MIN_AMENDMENT_QUESTIONS = 10
MIX_TOLERANCE = 1

class CorpusText(Protocol):
    """Cleaned page text of the corpus documents."""
    def page(self, doc_id: str, page: int) -> str | None: ...
    def pages(self, doc_id: str) -> list[str]: ...


class DictText:
    """CorpusText over {doc_id: {page: text}} (tests, small fixtures)."""
    def __init__(self, texts: dict[str, dict[int, str]]):
        self.texts = texts

    def page(self, doc_id: str, page: int) -> str | None:
        return self.texts.get(doc_id, {}).get(page)

    def pages(self, doc_id: str) -> list[str]:
        return list(self.texts.get(doc_id, {}).values())


@dataclass(frozen=True)
class DocInfo:
    doc_id: str
    part: str            # edgar | consumer | simulated_scan
    session: str         # manifest deal_id
    role: str            # original | amendment_<n> | form | scan_of:...


def value_in(value: str, target: str) -> bool:
    """Normalized match of an entity / superseded value inside a text."""
    nums = numeric_tokens(value)
    if nums:
        have = Counter(numeric_tokens(target))
        return all(have[n] >= c for n, c in Counter(nums).items())
    v = norm_match(value)
    return bool(v) and v in norm_match(target)


def locate_span(page_text: str | None, span_text: str) -> bool:
    if not page_text or not span_text.strip():
        return False
    if span_text.strip() in page_text:
        return True
    return " ".join(span_text.split()) in " ".join(page_text.split())


# ---- one question --------------------------------------------------------------------------
def validate_question(q: dict, docs: dict[str, DocInfo], text: CorpusText) -> list[str]:
    qid = q.get("qid", "?")
    errs = [f"{qid}: missing field {f}" for f in REQUIRED if f not in q]
    if errs:
        return errs
    e = lambda msg: errs.append(f"{qid}: {msg}")  # noqa: E731
    if q["split"] not in SPLITS:
        e(f"bad split {q['split']!r}")
    if q["type"] not in TYPES:
        e(f"bad type {q['type']!r}")
    if q["provenance"] not in PROVENANCE:
        e(f"bad provenance {q['provenance']!r}")
    if not q["verified_by"] or not q["verified_on"]:
        e("not verified (nothing enters the set without verification)")
    if not str(q["question"]).strip():
        e("empty question")

    session = [d for d in docs.values() if d.session == q["session_id"]]
    if not session:
        e(f"unknown session {q['session_id']!r}")
    for d in q["session_docs"]:
        if d not in docs or docs[d].session != q["session_id"]:
            e(f"session_docs: {d} is not in session {q['session_id']}")

    unans = q["type"] == "unanswerable"
    if q["unanswerable"] != unans:
        e("unanswerable flag must match type 'unanswerable'")
    if unans:
        if q["evidence"]:
            e("unanswerable question has evidence")
        if q["amendment"] or q["superseded_values"]:
            e("unanswerable question can't be an amendment question")
        return errs
    if not q["evidence"]:
        e("no evidence")
    if not str(q["reference_answer"]).strip():
        e("empty reference_answer")
    if not q["key_points"]:
        e("no key_points")

    spans: list[str] = []
    for i, ev in enumerate(q["evidence"]):
        d, p, t = ev.get("doc_id"), ev.get("page"), ev.get("span_text", "")
        if d not in q["session_docs"]:
            e(f"evidence[{i}]: {d} not in session_docs")
            continue
        if not locate_span(text.page(d, p), t):
            e(f"evidence[{i}]: span_text not found in cleaned text of {d} p{p}")
        spans.append(t)
    if q["type"] == "cross_doc" and len({ev.get("doc_id") for ev in q["evidence"]}) < 2:
        e("cross_doc needs evidence from 2+ documents")

    target = "\n".join(spans + [q["reference_answer"]])
    for ent in q["entities"]:
        if not value_in(ent, target):
            e(f"entity {ent!r} not in evidence spans or reference answer")

    if q["amendment"]:
        if not q["superseded_values"]:
            e("amendment question without superseded_values")
        if not any(docs[ev["doc_id"]].role.startswith("amendment") for ev in q["evidence"] if ev.get("doc_id") in docs):
            e("amendment question without evidence from an amendment document")
    elif q["superseded_values"]:
        e("superseded_values on a non-amendment question")
    if q["superseded_values"]:
        session_text = "\n".join(t for d in q["session_docs"] for t in text.pages(d))
        for v in q["superseded_values"]:
            if not value_in(v, session_text):
                e(f"superseded value {v!r} not found in the session's documents")

    num, tol = q["answer_numeric"], q["numeric_tolerance"]
    if num is not None and not isinstance(num, (int, float)):
        e("answer_numeric must be a number or null")
    if tol is not None and (num is None or not isinstance(tol, (int, float)) or tol < 0):
        e("numeric_tolerance needs answer_numeric and must be >= 0")
    if q["type"] == "calculation" and num is None:
        e("calculation question without answer_numeric")
    return errs


# ---- the set -------------------------------------------------------------------------------
def mix_errors(questions: list[dict]) -> list[str]:
    """Each split's type mix within ±1 of proportional to the whole set."""
    live = [q for q in questions if not q.get("retired")]
    n = len(live)
    errs = []
    if not n:
        return errs
    by_type = Counter(q["type"] for q in live)
    for split in SPLITS:
        in_split = [q for q in live if q["split"] == split]
        have = Counter(q["type"] for q in in_split)
        for t, total in by_type.items():
            want = total * len(in_split) / n
            if abs(have[t] - want) > MIX_TOLERANCE:
                errs.append(f"split {split}: {have[t]} {t} questions, proportional is {want:.1f}")
    return errs


def mix_report(questions: list[dict], docs: dict[str, DocInfo]) -> dict:
    live = [q for q in questions if not q.get("retired")]
    session_part = Counter("consumer" if q["session_id"].startswith("cfpb") else "edgar" for q in live)
    return {
        "questions": len(live), "retired": len(questions) - len(live),
        "by_split": dict(Counter(q["split"] for q in live)),
        "by_type": dict(Counter(q["type"] for q in live)),
        "by_session_part": dict(session_part),
        "scan_evidence": sum(any(docs.get(ev["doc_id"]) and docs[ev["doc_id"]].part == "simulated_scan"
                                 for ev in q["evidence"]) for q in live),
        "amendment": sum(bool(q["amendment"]) for q in live),
    }


def quota_errors(questions: list[dict], docs: dict[str, DocInfo]) -> list[str]:
    live = [q for q in questions if not q.get("retired")]
    rep = mix_report(questions, docs)
    errs = []
    for t, want in TYPE_COUNTS.items():
        if rep["by_type"].get(t, 0) != want:
            errs.append(f"type {t}: {rep['by_type'].get(t, 0)} questions, quota {want}")
    for s, want in SPLIT_SIZES.items():
        if rep["by_split"].get(s, 0) != want:
            errs.append(f"split {s}: {rep['by_split'].get(s, 0)} questions, quota {want}")
    n = max(len(live), 1)
    if rep["by_session_part"].get("edgar", 0) / n < MIN_EDGAR_SHARE:
        errs.append(f"EDGAR share {rep['by_session_part'].get('edgar', 0)}/{n} < {MIN_EDGAR_SHARE:.0%}")
    if rep["by_session_part"].get("consumer", 0) / n < MIN_CONSUMER_SHARE:
        errs.append(f"consumer share {rep['by_session_part'].get('consumer', 0)}/{n} < {MIN_CONSUMER_SHARE:.0%}")
    if rep["scan_evidence"] < MIN_SCAN_QUESTIONS:
        errs.append(f"{rep['scan_evidence']} questions with scan evidence < {MIN_SCAN_QUESTIONS}")
    amend_cross = sum(bool(q["amendment"]) and q["type"] == "cross_doc" for q in live)
    if amend_cross < MIN_AMENDMENT_QUESTIONS:
        errs.append(f"{amend_cross} amendment cross_doc questions < {MIN_AMENDMENT_QUESTIONS}")
    return errs


def validate_set(questions: list[dict], docs: dict[str, DocInfo], text: CorpusText,
                 final: bool = False) -> list[str]:
    errs: list[str] = []
    seen = Counter(q.get("qid") for q in questions)
    errs += [f"{qid}: duplicate qid" for qid, c in seen.items() if c > 1]
    for q in questions:
        errs += validate_question(q, docs, text)
    errs += mix_errors(questions)
    if final:
        errs += quota_errors(questions, docs)
    return errs
