"""
Review store for the QA drafts (scope v2 §5.3 step 3: every question human-verified).

Reads `drafts/candidates.jsonl` (rebuilt by draft.py on every run, never edited here) and keeps
Kousik's decisions in `drafts/reviews.json`, keyed by candidate id:

{"cross_doc-001-a": {
    "decision": "keep" | "drop" | "later",
    "final": {question, reference_answer, evidence, key_points, entities, superseded_values,
              answer_numeric, numeric_tolerance},          # keep: the question as approved (edited or not)
    "edited": true,                                       # final differs from the draft
    "drop_reason": "...", "note": "...",
    "reviewed_by": "kousik", "reviewed_on": "2026-11-17",
    "draft_hash": "..."}}                                 # draft fields when reviewed -> stale check

A "keep" is refused while the F3 checks (`schema.validate_question`) find a problem; "later" and
"drop" save as they are. If draft.py re-drafts a job and a candidate's draft changes, its review
is shown as stale. The finalise step (assign qids, dev/test split) reads `final` from here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from evals.golden.qa.schema import TYPE_COUNTS, DocInfo, validate_question

EDITABLE = ("question", "reference_answer", "evidence", "key_points", "entities", "superseded_values",
            "answer_numeric", "numeric_tolerance")
DECISIONS = ("keep", "drop", "later")
DROP_REASONS = ("wrong answer", "ambiguous question", "trivial / not useful", "duplicate",
                "bad evidence span", "answer is elsewhere in the file", "other")
CID_RE = re.compile(r"^[a-z_]+-\d{3}-[a-j]$")


def draft_hash(cand: dict) -> str:
    return hashlib.sha256(json.dumps({k: cand.get(k) for k in EDITABLE}, sort_keys=True).encode()).hexdigest()[:16]


def span_regex(span: str) -> re.Pattern | None:
    """Whitespace-insensitive pattern for a span (same rule as schema.locate_span)."""
    toks = span.split()
    return re.compile(r"\s+".join(re.escape(t) for t in toks)) if toks else None


def highlights(page_text: str | None, spans: list[tuple[int, str]]) -> list[list[int]]:
    """[start, end, evidence_index] of every span found in the page text."""
    out = []
    for i, span in spans:
        rx = span_regex(span)
        m = rx.search(page_text or "") if rx else None
        if m:
            out.append([m.start(), m.end(), i])
    return sorted(out)


class ReviewStore:
    """File access for the review app; kept apart from HTTP so it can be unit-tested."""

    def __init__(self, candidates: Path, reviews: Path, docs: dict[str, DocInfo], text, reviewer: str,
                 pdf_dir: Path | None = None, image_dir: Path | None = None):
        self.candidates_path, self.reviews_path = candidates, reviews
        self.docs, self.text, self.reviewer = docs, text, reviewer
        self.pdf_dir, self.image_dir = pdf_dir, image_dir

    # ---- reading -------------------------------------------------------------------------------
    def candidates(self) -> list[dict]:
        if not self.candidates_path.exists():
            return []
        return [json.loads(l) for l in self.candidates_path.read_text().splitlines() if l.strip()]

    def reviews(self) -> dict[str, dict]:
        return json.loads(self.reviews_path.read_text()) if self.reviews_path.exists() else {}

    def _cand(self, cid: str) -> dict:
        if not CID_RE.match(cid):
            raise KeyError(cid)
        for c in self.candidates():
            if c["cid"] == cid:
                return c
        raise KeyError(cid)

    def status(self, cand: dict, rev: dict | None) -> str:
        if not rev:
            return "todo"
        if rev.get("draft_hash") != draft_hash(cand):
            return "stale"
        return rev["decision"]

    def list(self) -> dict:
        revs = self.reviews()
        items = []
        for c in self.candidates():
            r = revs.get(c["cid"])
            items.append({"cid": c["cid"], "type": c["type"], "session_id": c["session_id"],
                          "question": (r or {}).get("final", {}).get("question") or c["question"],
                          "issues": len(c.get("draft_issues") or []), "status": self.status(c, r),
                          "edited": bool(r and r.get("edited")), "scan": any(
                              self.docs.get(d) and self.docs[d].part == "simulated_scan" for d, _ in c["shown_pages"])})
        kept = Counter(i["type"] for i in items if i["status"] == "keep")
        return {"items": items, "counts": dict(Counter(i["status"] for i in items)),
                "kept_by_type": {t: kept.get(t, 0) for t in TYPE_COUNTS}, "quota": TYPE_COUNTS,
                "total_quota": sum(TYPE_COUNTS.values())}

    def page_view(self, doc_id: str, page: int, spans: list[tuple[int, str]] = ()) -> dict:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        t = self.text.page(doc_id, page)
        info = self.docs[doc_id]
        return {"doc_id": doc_id, "page": page, "part": info.part, "role": info.role,
                "text": t, "highlights": highlights(t, list(spans)),
                "has_prev": self.text.page(doc_id, page - 1) is not None,
                "has_next": self.text.page(doc_id, page + 1) is not None}

    def get(self, cid: str) -> dict:
        c = self._cand(cid)
        r = self.reviews().get(cid)
        shown = c["shown_pages"]
        spans = (r or {}).get("final", {}).get("evidence") or c["evidence"]
        pages = [self.page_view(d, p, [(i, e["span_text"]) for i, e in enumerate(spans)
                                       if e.get("doc_id") == d and e.get("page") == p]) for d, p in shown]
        return {"candidate": c, "review": r, "status": self.status(c, r), "pages": pages,
                "draft": {k: c.get(k) for k in EDITABLE}}

    # ---- checking and saving -------------------------------------------------------------------
    def as_question(self, cand: dict, final: dict) -> dict:
        return {**cand, **{k: final.get(k, cand.get(k)) for k in EDITABLE}, "qid": cand["cid"], "split": "dev",
                "verified_by": self.reviewer, "verified_on": date.today().isoformat()}

    def check(self, cid: str, final: dict) -> list[str]:
        q = self.as_question(self._cand(cid), clean_final(final))
        return [m.split(": ", 1)[1] for m in validate_question(q, self.docs, self.text)]

    def save(self, cid: str, body: dict) -> tuple[bool, list[str]]:
        cand = self._cand(cid)
        decision = body.get("decision")
        if decision not in DECISIONS:
            return False, [f"decision must be one of {DECISIONS}"]
        final = clean_final(body.get("final") or {k: cand.get(k) for k in EDITABLE})
        errors = self.check(cid, final)
        if decision == "keep" and errors:
            return False, errors
        if decision == "drop" and not (body.get("drop_reason") or "").strip():
            return False, ["give a drop reason"]
        rec = {"decision": decision, "final": final,
               "edited": any(final.get(k) != cand.get(k) for k in EDITABLE),
               "drop_reason": body.get("drop_reason") or None, "note": (body.get("note") or "").strip() or None,
               "reviewed_by": self.reviewer, "reviewed_on": date.today().isoformat(),
               "draft_hash": draft_hash(cand)}
        revs = self.reviews()
        revs[cid] = rec
        self.reviews_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.reviews_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(revs, indent=1, ensure_ascii=False, sort_keys=True))
        tmp.replace(self.reviews_path)
        return True, errors

    # ---- page images -----------------------------------------------------------------------------
    def image(self, doc_id: str, page: int) -> bytes:
        """PNG of a PDF page (rendered once, cached under image_dir)."""
        if doc_id not in self.docs or not self.pdf_dir or not self.image_dir:
            raise KeyError(doc_id)
        out = self.image_dir / f"{doc_id}_p{page}.png"
        if not out.exists():
            import fitz  # PyMuPDF
            with fitz.open(self.pdf_dir / f"{doc_id}.pdf") as d:
                if not 1 <= page <= len(d):
                    raise KeyError(page)
                out.parent.mkdir(parents=True, exist_ok=True)
                d[page - 1].get_pixmap(dpi=110).save(out)
        return out.read_bytes()


def clean_final(final: dict) -> dict:
    """Coerce the edited fields to the schema's types (the form sends strings and lists)."""
    def lst(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    def num(v):
        if v in (None, ""):
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        try:
            return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
        except ValueError:
            return str(v)            # left as text so the validator reports it
    ev = []
    for e in final.get("evidence") or []:
        try:
            page = int(e.get("page"))
        except (TypeError, ValueError):
            page = e.get("page")
        ev.append({"doc_id": str(e.get("doc_id") or ""), "page": page, "span_text": str(e.get("span_text") or "")})
    return {"question": str(final.get("question") or "").strip(),
            "reference_answer": str(final.get("reference_answer") or "").strip(),
            "evidence": ev, "key_points": lst(final.get("key_points")), "entities": lst(final.get("entities")),
            "superseded_values": lst(final.get("superseded_values")),
            "answer_numeric": num(final.get("answer_numeric")), "numeric_tolerance": num(final.get("numeric_tolerance"))}
