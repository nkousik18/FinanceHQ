"""
R3 drafting, step 3 of 3: ask the LLM for candidate questions and filter them (scope v2 §5.3).

    venv/bin/python -m evals.golden.qa.draft                 # dry run: job plan + token estimate
    venv/bin/python -m evals.golden.qa.draft --run --limit 3 # pilot: first 3 jobs not yet drafted
    venv/bin/python -m evals.golden.qa.draft --run           # all remaining jobs (stops at the budget)
    venv/bin/python -m evals.golden.qa.draft --rebuild       # re-filter saved responses, no calls

Jobs = pairs.py (12 cross_doc) + pages.py (80 single-page), 2 candidates each. Model:
openai/gpt-oss-120b on Groq, reasoning effort low (Kousik 2026-11-13). Every call goes through
the D7 ledger (`evals.llm.ledger`): run/total $ caps and the daily token budget.

Resumable: each job's raw response is saved in data/results/r3/raw/<job_id>.json and never asked
again. Candidates are rebuilt from all saved responses on every run:
- `evals/golden/qa/drafts/candidates.jsonl`: passed the span filter, for Kousik's review
- `evals/golden/qa/drafts/rejected.jsonl`: auto-rejected, with the reason
**Auto-reject (§5.3 step 2):** an evidence span not found verbatim (exact or whitespace-
insensitive, `schema.locate_span`) in the cleaned text of the page it cites, or citing a page the
model wasn't shown; an unanswerable candidate with evidence; an answerable one without.
Other validator findings (entity not in spans, cross_doc from one doc, …) are kept as
`draft_issues` for review, not rejected. Nothing here writes questions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

from evals.golden.qa.pages import OUT_DIR, Job, load_corpus_infos, select_jobs
from evals.golden.qa.pairs import amendment_pairs, form_pairs, select_pair_jobs
from evals.golden.qa.schema import DocInfo, locate_span, validate_question

log = logging.getLogger("qa_draft")

MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "low"
TEMPERATURE = 0.4
MAX_COMPLETION_TOKENS = 6000
TPM_LIMIT = 8000              # Groq free-tier tokens/minute for this model: pace calls under it

# Token estimate for a call (dry run). Prompt = counted with the gpt-oss tokenizer; completion =
# a guess until the pilot measures it: JSON per candidate + hidden reasoning at effort "low".
EST_OUT_PER_CANDIDATE = 300
EST_REASONING_PER_CALL = 800

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = OUT_DIR / "raw"
DRAFT_DIR = Path(__file__).resolve().parent / "drafts"
CANDIDATES = DRAFT_DIR / "candidates.jsonl"
REJECTED = DRAFT_DIR / "rejected.jsonl"

SYSTEM = """You write evaluation questions for a retrieval system over loan documents (US corporate credit agreements and their amendments, and CFPB mortgage disclosure forms).
You are shown the cleaned text of one or two document pages, each under a header "=== doc_id page N ===".

Rules for every question:
- Answerable from the shown text alone (except type "unanswerable"), and specific: a reader of the whole loan file must be able to tell which value is meant. Name the facility, party or section when needed. Do not mention "the page" or "the text" in the question.
- Write it the way a loan analyst or borrower would ask it.
- evidence: one or more {"doc_id", "page", "span_text"}. span_text is copied EXACTLY, character for character, from the page text: one contiguous passage (a sentence, clause or table row, 5-60 words) that contains the answer. Keep typos, OCR errors, odd spacing and capitalisation as they are. Never paraphrase or join pieces from different places into one span; use several evidence items instead. Never shorten a span with "..." or "…": if the passage is too long, choose a shorter contiguous sentence or clause from it that still holds the value.
- reference_answer: a complete, correct answer in 1-3 sentences.
- key_points: the 1-4 facts a complete answer must contain.
- entities: every amount, rate, date and party name in the reference answer, written as it appears in the evidence. Each entity is short (a value or a name, at most about 8 words), never a whole clause or definition.
- answer_numeric: the answer as a plain number (no $ or %, e.g. 2.5 for 2.50%) when the answer is one number, else null. numeric_tolerance: null unless answer_numeric is set and rounding is involved.
- superseded_values: [] and amendment: false unless the instructions say otherwise.
Return JSON only: {"candidates": [ {question, reference_answer, evidence, key_points, entities, superseded_values, amendment, answer_numeric, numeric_tolerance} ]}"""

TYPE_RULES = {
    "lookup": "Type lookup: each question asks for a single fact stated in running text (a rate, date, amount, party name, or what a defined term covers).",
    "table": "Type table: each question asks for a value that appears only inside a table on the page (the span is the table row holding it). The question must say which row and which column/meaning it wants.",
    "calculation": "Type calculation: each question needs arithmetic over two or more values on the page (a sum, difference, ratio, percentage or change). Give one evidence item per input value, answer_numeric = the result, numeric_tolerance = 0 for exact sums/differences, else about 0.5% of the result. key_points: the inputs and the result.",
    "explain": "Type explain: each question asks about a clause or condition in prose (what triggers it, what is required, what happens, exceptions). Answer in 2-3 sentences; key_points are its 2-4 essential conditions or consequences.",
    "unanswerable": "Type unanswerable: each question is plausible for this kind of document and this deal (it uses the real party/loan names) but is NOT answered here and very likely not anywhere in the loan file: e.g. a figure such documents don't contain, an event after the documents, a party's internal matters. Avoid anything a credit agreement or this form normally states somewhere (rates, dates, covenants, definitions, fees). evidence: [], reference_answer: a sentence saying the documents do not state it, key_points: [that sentence], entities: [], answer_numeric: null.",
    "cross_doc_amendment": "Type cross_doc, amendment: the FIRST page is from an amendment, the SECOND from the original credit agreement; they concern {note}. Each question asks for a term as it stands AFTER the amendment, where the amendment changed a concrete value (a date, rate, amount, ratio, period or party). Evidence: the amendment span with the new value AND the original span with the old value. superseded_values: the old value(s) the amendment replaced, copied from the original page. amendment: true. If the amendment changes no concrete value, ask what the amendment changed versus the original and give the replaced wording's key value as superseded_values.",
    "cross_doc_form": "Type cross_doc: the FIRST page is a Loan Estimate, the SECOND the Closing Disclosure for the same loan ({note}). Each question needs both pages (e.g. how an estimated amount compares with the final amount, or the change between them). Evidence from both pages. amendment: false, superseded_values: [].",
}


# ---- jobs and page text ------------------------------------------------------------------------
def build_jobs(docs: dict[str, DocInfo], text) -> tuple[list[Job], dict[str, int]]:
    """All drafting jobs: cross_doc pairs first (their pages are then avoided), then single pages."""
    edgar = [d for d, i in docs.items() if i.part == "edgar"]
    texts = {d: page_map(text, d) for d in edgar}
    roles = {d: (docs[d].session, docs[d].role) for d in edgar}
    infos = load_corpus_infos()
    used: dict = {}
    pair_jobs, short1 = select_pair_jobs(amendment_pairs(texts, roles), form_pairs(infos), used=used)
    page_jobs, short2 = select_jobs(infos, used=used)
    return pair_jobs + page_jobs, {**short1, **short2}


def page_map(text, doc_id: str) -> dict[int, str]:
    """page -> text for one document (CacheText or schema.DictText)."""
    if hasattr(text, "_pages"):
        return text._pages(doc_id)
    return dict(text.texts.get(doc_id, {}))


def type_rule(job: Job) -> str:
    if job.qtype == "cross_doc":
        return TYPE_RULES["cross_doc_amendment" if job.amendment else "cross_doc_form"].format(note=job.note)
    return TYPE_RULES[job.qtype]


def page_block(doc_id: str, page: int, docs: dict[str, DocInfo], text) -> str:
    info = docs[doc_id]
    tag = " (simulated scan: OCR text, may contain recognition errors)" if info.part == "simulated_scan" else ""
    return f"=== {doc_id} page {page}{tag} ===\n{text.page(doc_id, page) or ''}"


def messages(job: Job, docs: dict[str, DocInfo], text) -> list[dict]:
    pages = "\n\n".join(page_block(d, p, docs, text) for d, p in job.pages)
    user = (f"{type_rule(job)}\nWrite {job.n} different questions of this type.\n\n{pages}\n\n"
            f"Return JSON only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ---- estimate ----------------------------------------------------------------------------------
def count_tokens(msgs: list[dict]) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_harmony")
    return sum(len(enc.encode(m["content"])) + 4 for m in msgs) + 3


def estimate(jobs: list[Job], docs: dict[str, DocInfo], text) -> dict:
    prompt = [count_tokens(messages(j, docs, text)) for j in jobs]
    completion = [j.n * EST_OUT_PER_CANDIDATE + EST_REASONING_PER_CALL for j in jobs]
    from evals.llm.ledger import cost_usd
    total = sum(prompt) + sum(completion)
    return {"jobs": len(jobs), "candidates": sum(j.n for j in jobs),
            "prompt_tokens": sum(prompt), "prompt_per_call_max": max(prompt, default=0),
            "completion_tokens_guess": sum(completion), "total_tokens": total,
            "days_at_190k": round(total / 190_000, 2),
            "cost_usd_list_price": round(cost_usd(MODEL, sum(prompt), sum(completion)), 4)}


# ---- parse and filter --------------------------------------------------------------------------
def parse_content(content: str) -> list[dict]:
    """Candidates from the model's JSON (tolerates a code fence or text around the object)."""
    s = content.strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError("no JSON object in response")
    data = json.loads(m.group(0), strict=False)   # raw control chars (e.g. \x07 bullets in CFPB text)
    cands = data.get("candidates", data if isinstance(data, list) else [])
    return [c for c in cands if isinstance(c, dict)]


def _list(v) -> list:
    return v if isinstance(v, list) else ([] if v in (None, "") else [v])


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(str(v).replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def to_candidate(job: Job, idx: int, c: dict, docs: dict[str, DocInfo], drafted_by: str, drafted_on: str) -> dict:
    unans = job.qtype == "unanswerable"
    ev = [{"doc_id": e.get("doc_id"), "page": int(e["page"]) if str(e.get("page", "")).isdigit() else e.get("page"),
           "span_text": str(e.get("span_text", ""))} for e in _list(c.get("evidence")) if isinstance(e, dict)]
    return {
        "cid": f"{job.job_id}-{'abcdefghij'[idx]}", "job_id": job.job_id, "type": job.qtype,
        "question": str(c.get("question", "")).strip(),
        "session_id": job.session_id,
        "session_docs": sorted(d for d, i in docs.items() if i.session == job.session_id),
        "evidence": ev,
        "reference_answer": str(c.get("reference_answer", "")).strip(),
        "key_points": [str(x) for x in _list(c.get("key_points"))],
        "entities": [str(x) for x in _list(c.get("entities"))],
        "superseded_values": [str(x) for x in _list(c.get("superseded_values"))] if job.amendment else [],
        "amendment": bool(job.amendment and c.get("amendment", True)),
        "answer_numeric": _num(c.get("answer_numeric")),
        "numeric_tolerance": _num(c.get("numeric_tolerance")),
        "unanswerable": unans,
        "provenance": "llm_drafted_human_verified",
        "drafted_by": drafted_by, "drafted_on": drafted_on,
        "shown_pages": [list(p) for p in job.pages], "pair_note": job.note,
    }


def check(cand: dict, job: Job, docs: dict[str, DocInfo], text) -> tuple[list[str], list[str]]:
    """(reject reasons, review issues)."""
    reject: list[str] = []
    shown = {(d, p) for d, p in job.pages}
    if not cand["question"]:
        reject.append("empty question")
    if cand["unanswerable"] and cand["evidence"]:
        reject.append("unanswerable candidate has evidence")
    if not cand["unanswerable"] and not cand["evidence"]:
        reject.append("no evidence")
    for i, e in enumerate(cand["evidence"]):
        if (e["doc_id"], e["page"]) not in shown:
            reject.append(f"evidence[{i}] cites {e['doc_id']} p{e['page']}, not a shown page")
        elif not locate_span(text.page(e["doc_id"], e["page"]), e["span_text"]):
            reject.append(f"evidence[{i}] span not found verbatim in {e['doc_id']} p{e['page']}")
    if reject:
        return reject, []
    q = {**cand, "qid": cand["cid"], "split": "dev", "verified_by": "pending", "verified_on": "pending"}
    issues = [m.split(": ", 1)[1] for m in validate_question(q, docs, text)]
    return [], [m for m in issues if not m.startswith("not verified")]


def rebuild(jobs: list[Job], docs: dict[str, DocInfo], text) -> dict:
    """Re-filter every saved raw response into candidates.jsonl / rejected.jsonl."""
    keep, rej, errors = [], [], []
    for job in jobs:
        f = RAW_DIR / f"{job.job_id}.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())
        try:
            cands = parse_content(raw["content"])
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append({"job_id": job.job_id, "error": str(exc)})
            continue
        for i, c in enumerate(cands[:10]):
            cand = to_candidate(job, i, c, docs, raw.get("returned_model", MODEL), raw["ts"][:10])
            reasons, issues = check(cand, job, docs, text)
            if reasons:
                rej.append({**cand, "reject_reasons": reasons})
            else:
                keep.append({**cand, "draft_issues": issues, "review": None})
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES.write_text("".join(json.dumps(c) + "\n" for c in keep))
    REJECTED.write_text("".join(json.dumps(c) + "\n" for c in rej + errors))
    return {"drafted_jobs": sum((RAW_DIR / f"{j.job_id}.json").exists() for j in jobs),
            "candidates": len(keep), "rejected": len(rej), "unparseable_jobs": len(errors),
            "with_issues": sum(bool(c["draft_issues"]) for c in keep)}


# ---- calling the model -------------------------------------------------------------------------
def groq_client():
    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv(REPO / ".env")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (.env)")
    return Groq(api_key=key, max_retries=3)


def call_model(client, msgs: list[dict]) -> dict:
    r = client.chat.completions.create(
        model=MODEL, messages=msgs, temperature=TEMPERATURE, reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"}, max_completion_tokens=MAX_COMPLETION_TOKENS)
    u = r.usage
    details = getattr(u, "completion_tokens_details", None)
    return {"content": r.choices[0].message.content or "", "returned_model": r.model,
            "finish_reason": r.choices[0].finish_reason,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None}


def salvage(exc: Exception, msgs: list[dict], est_prompt: int) -> dict | None:
    """Groq 400 `json_validate_failed` still returns the generation. It fails Groq's strict JSON
    check when a span copies a raw control character from the page (e.g. \x07 bullets in CFPB
    forms); our parser accepts those. Usage is not returned, so tokens are estimated."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error", body) if isinstance(body.get("error", body), dict) else body
    gen = err.get("failed_generation")
    if err.get("code") != "json_validate_failed" or not gen:
        return None
    import tiktoken
    out_tokens = len(tiktoken.get_encoding("o200k_harmony").encode(gen)) + EST_REASONING_PER_CALL
    return {"content": gen, "returned_model": MODEL, "finish_reason": "json_validate_failed_salvaged",
            "prompt_tokens": est_prompt, "completion_tokens": out_tokens, "reasoning_tokens": None,
            "usage_estimated": True}


def run(jobs: list[Job], docs: dict[str, DocInfo], text, client, ledger, limit: int | None = None,
        tpm: int = TPM_LIMIT, sleep=time.sleep) -> dict:
    """Draft jobs without a saved response, in order, until done, `limit`, or a budget stops it."""
    from evals.llm.ledger import BudgetExceeded
    todo = [j for j in jobs if not (RAW_DIR / f"{j.job_id}.json").exists()][:limit]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done, stop = 0, None
    for job in todo:
        msgs = messages(job, docs, text)
        est_p = count_tokens(msgs)
        try:
            ledger.check(MODEL, est_p, job.n * EST_OUT_PER_CANDIDATE + EST_REASONING_PER_CALL)
        except BudgetExceeded as exc:
            stop = str(exc)
            break
        t0 = time.monotonic()
        try:
            out = call_model(client, msgs)
        except Exception as exc:                      # rate limit / API error: stop, resume later
            out = salvage(exc, msgs, est_p)
            if out is None:
                stop = f"{type(exc).__name__}: {exc}"
                break
            log.warning("%s: json_validate_failed, generation salvaged (usage estimated)", job.job_id)
        row = ledger.record(f"r3_draft:{job.job_id}", MODEL, out["returned_model"], out["prompt_tokens"],
                            out["completion_tokens"], out["reasoning_tokens"])
        (RAW_DIR / f"{job.job_id}.json").write_text(json.dumps({**out, "job": job.to_dict(), "ts": row["ts"],
                                                                "est_prompt_tokens": est_p}, indent=1))
        done += 1
        log.info("%s: %d+%d tokens (reasoning %s), %s", job.job_id, out["prompt_tokens"],
                 out["completion_tokens"], out["reasoning_tokens"], out["finish_reason"])
        wait = row["total_tokens"] / tpm * 60 - (time.monotonic() - t0)
        if wait > 0 and job is not todo[-1]:
            sleep(wait)
    return {"called": done, "remaining": len([j for j in jobs if not (RAW_DIR / f"{j.job_id}.json").exists()]),
            "stopped": stop, "run_tokens": ledger.run_tokens, "run_cost_usd": round(ledger.run_cost, 4),
            "tokens_today": ledger.tokens_today()}


def measured(jobs: list[Job]) -> dict | None:
    """Actual tokens per call from saved responses (to re-estimate after the pilot)."""
    rows = [json.loads((RAW_DIR / f"{j.job_id}.json").read_text()) for j in jobs
            if (RAW_DIR / f"{j.job_id}.json").exists()]
    if not rows:
        return None
    n = len(rows)
    return {"calls": n, "prompt_mean": round(sum(r["prompt_tokens"] for r in rows) / n),
            "completion_mean": round(sum(r["completion_tokens"] for r in rows) / n),
            "reasoning_mean": round(sum(r["reasoning_tokens"] or 0 for r in rows) / n),
            "est_prompt_ratio": round(sum(r["prompt_tokens"] for r in rows) / max(1, sum(r["est_prompt_tokens"] for r in rows)), 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="call the model (default: dry run)")
    ap.add_argument("--limit", type=int, help="at most N new jobs")
    ap.add_argument("--rebuild", action="store_true", help="only re-filter saved responses")
    ap.add_argument("--daily-budget", type=int, help="override the daily token budget")
    ap.add_argument("--show-prompt", help="print the prompt of one job id")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    from evals.golden.qa.validate import CacheText, corpus_docs
    docs = corpus_docs()
    text = CacheText(docs)
    jobs, short = build_jobs(docs, text)
    (OUT_DIR / "jobs.json").write_text(json.dumps([j.to_dict() for j in jobs], indent=1))
    if args.show_prompt:
        job = next(j for j in jobs if j.job_id == args.show_prompt)
        for m in messages(job, docs, text):
            print(f"--- {m['role']} ---\n{m['content']}")
        return 0
    out: dict = {"shortfall": short}
    if args.run:
        from evals.llm.ledger import Ledger
        ledger = Ledger(**({"daily_budget": args.daily_budget} if args.daily_budget else {}))
        out["run"] = run(jobs, docs, text, groq_client(), ledger, args.limit)
    if args.run or args.rebuild:
        out["filter"] = rebuild(jobs, docs, text)
    remaining = [j for j in jobs if not (RAW_DIR / f"{j.job_id}.json").exists()]
    out["estimate_remaining"] = estimate(remaining, docs, text)
    out["measured"] = measured(jobs)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
