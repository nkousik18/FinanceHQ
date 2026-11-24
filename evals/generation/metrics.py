"""
Deterministic generation metrics (scope v2 §8.4) — no judge. Per question; None = not applicable.

- numeric_em (calculation questions with `answer_numeric`): the value on the answer's
  "Final answer:" line (else the answer's last number) is within `numeric_tolerance` of
  `answer_numeric`. "million"/"billion" after a number scale it; "%" is dropped (90% -> 90).
- refused: the answer is a refusal — it contains REFUSAL, or its first sentence matches a decline
  pattern ("does not contain/state/specify …", "not found/stated/specified …", "cannot be
  determined …"). Later sentences don't count: "X is 5%; the agreement does not state Y" answers.
- refusal_correct (unanswerable questions): refused.
- false_refusal (answerable questions whose evidence was fully in the context, recall@budget = 1):
  refused. Partly retrieved evidence is excluded: refusing then may be right.
- stale (amendment questions with `superseded_values`, not refused): the answer states a superseded
  value and none of the question's current `entities`. Matching = `schema.value_in` (numbers
  normalized, text norm-matched).
- citation_precision: valid cited chunks that cover an evidence span / all citations (an id not in
  the context counts as a wrong citation). None without citations or for unanswerable questions.
- citation_recall: covering context chunks that are cited / covering context chunks. None when the
  context has no covering chunk or the answer refused.
- answer_entity_recall (answerable questions with `entities`): entities found in the answer
  (`schema.value_in`) / entities; a refusal scores 0. A judge-free correctness proxy: it rewards
  stating the right values, not reasoning, and ignores extra wrong claims.
- format_ok: refusals use the exact REFUSAL string; other answers cite >= 1 chunk, cite no unknown
  id, and calculation answers have a "Final answer:" line.
"""
from __future__ import annotations

import re

from evals.generation.prompt import FINAL_PREFIX, REFUSAL
from evals.golden.qa.schema import value_in
from evals.ingest_metrics.normalize import numeric_tokens

_CITE = re.compile(r"\[([^\[\]]{1,80})\]")
_ID = re.compile(r"\bS\d{1,3}\b")
_DECLINE = re.compile(
    r"not found in the (document|context)|"
    r"(does|do|did) not (contain|state|specify|provide|include|mention|disclose|indicate|say)|"
    r"(is|are) not (stated|specified|provided|found|available|given|disclosed|mentioned|included)|"
    r"no (information|mention|data|figure|value)|"
    r"cannot (be )?(determine|determined|calculate|calculated|answer|answered|find|found)|"
    r"unable to (determine|find|calculate|answer)|"
    r"insufficient information", re.I)
_SCALE = re.compile(r"^\s*(million|billion|thousand)\b", re.I)
_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_FINAL = re.compile(rf"^\W*{re.escape(FINAL_PREFIX)}\s*(.+)$", re.I | re.M)
_NUM_SPAN = re.compile(r"-?\$?\(?\d[\d,]*(\.\d+)?%?")


def citations(answer: str) -> list[str]:
    """Local ids cited, in order, with repeats removed. `[S1, S3]` and `[S1][S3]` both work; a
    bracket with no S-id (e.g. `[Reserved]`) is ignored; `[S9]` not in context is kept (invalid)."""
    out: list[str] = []
    for m in _CITE.finditer(answer):
        for sid in _ID.findall(m.group(1)):
            if sid not in out:
                out.append(sid)
    return out


def first_sentence(text: str) -> str:
    t = _CITE.sub("", text).strip()
    m = re.search(r"[.?!](\s|$)", t)
    return t[:m.end()] if m else t


def is_refusal(answer: str) -> bool:
    a = answer.strip()
    if REFUSAL.lower() in a.lower():
        return True
    return bool(_DECLINE.search(first_sentence(a)))


def parse_number(text: str) -> float | None:
    """First number in `text` (after normalization), scaled by a following million/billion."""
    toks = numeric_tokens(text)
    if not toks:
        return None
    tok = toks[0].rstrip("%")
    try:
        val = float(tok)
    except ValueError:                       # a date token
        return None
    m = _NUM_SPAN.search(text)
    if m and (s := _SCALE.match(text[m.end():])):
        val *= _SCALES[s.group(1).lower()]
    return val


def final_value(answer: str) -> tuple[float | None, bool]:
    """(value, from_final_line). Falls back to the last number in the answer."""
    lines = _FINAL.findall(answer)
    if lines:
        v = parse_number(lines[-1])
        if v is not None:
            return v, True
    text = _CITE.sub(" ", answer)
    spans = list(_NUM_SPAN.finditer(text))
    for m in reversed(spans):
        v = parse_number(text[m.start():])
        if v is not None:
            return v, False
    return None, False


def numeric_em(answer: str, target: float, tol: float | None) -> bool:
    v, _ = final_value(answer)
    return v is not None and abs(v - float(target)) <= (tol or 0) + 1e-9


def score_answer(q: dict, answer: str, context: list, recall_at_budget: float | None) -> dict:
    """Metrics for one answer. `context`: ContextChunk list in prompt order."""
    refused = is_refusal(answer)
    unanswerable = bool(q.get("unanswerable"))
    ids = {c.local_id: c for c in context}
    cited = citations(answer)
    valid = [s for s in cited if s in ids]
    covering = {c.local_id for c in context if c.covers}
    m: dict = {"refused": refused, "n_citations": len(cited), "invalid_citations": len(cited) - len(valid),
               "numeric_em": None, "final_line": None, "refusal_correct": None, "false_refusal": None,
               "stale": None, "citation_precision": None, "citation_recall": None,
               "answer_entity_recall": None}
    if q["type"] == "calculation" and q.get("answer_numeric") is not None:
        m["numeric_em"] = numeric_em(answer, q["answer_numeric"], q.get("numeric_tolerance"))
        m["final_line"] = final_value(answer)[1]
    ents = q.get("entities") or []
    if not unanswerable and ents:
        m["answer_entity_recall"] = 0.0 if refused else sum(value_in(e, answer) for e in ents) / len(ents)
    if unanswerable:
        m["refusal_correct"] = refused
    elif recall_at_budget == 1.0:
        m["false_refusal"] = refused
    if q.get("superseded_values") and not unanswerable and not refused:
        old = any(value_in(v, answer) for v in q["superseded_values"])
        new = any(value_in(e, answer) for e in q.get("entities") or [])
        m["stale"] = old and not new
    if not unanswerable and cited:
        m["citation_precision"] = sum(s in covering for s in valid) / len(cited)
    if not unanswerable and covering and not refused:
        m["citation_recall"] = len(covering & set(valid)) / len(covering)
    if refused:
        fmt = REFUSAL in answer
    else:
        fmt = bool(valid) and not (len(cited) - len(valid))
        if q["type"] == "calculation":
            fmt = fmt and bool(_FINAL.search(answer))
    m["format_ok"] = fmt
    return m


BOOL_METRICS = ("numeric_em", "refusal_correct", "false_refusal", "stale", "format_ok", "refused")
MEAN_METRICS = ("citation_precision", "citation_recall", "answer_entity_recall")
