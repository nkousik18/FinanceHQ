"""
LLM judge for S4 answers (scope v2 §8.4, §8.6). Judge: `openai/gpt-oss-120b`, reasoning effort low,
temperature 0, JSON mode. Rubric version JUDGE_VERSION; one call per answer grades everything:

- claims:        the answer split into atomic factual claims, each `supported` by the context or not
                 -> faithfulness = supported / claims; hallucination count = unsupported claims
- correctness:   correct / partial / wrong against the reference answer (scored 1 / 0.5 / 0)
- key_points:    present or not, per reference key point -> completeness = present / key points
- relevance:     1-5, does the answer address the question asked

Skipped calls (deterministic, no tokens), `judged_by: "rule"`:
- unanswerable question + refusal -> correct, no claims, relevance 5
- answerable question + refusal   -> wrong, completeness 0, no claims; relevance not graded (None)
An answerable-question refusal is wrong by the reference answer whatever the judge would say;
faithfulness is undefined without claims.

The judge sees the same context as the generator (rebuilt from the retrieval run, labelled S1…Sn),
so faithfulness is "supported by what the generator saw". Output: `judgments.jsonl` next to
`answers.jsonl`; resumable; every call goes through the D7 ledger. No judge number is reported
before calibration passes (§8.6 gate).

    venv/bin/python -m evals.generation.judge --config evals/configs/s4_generation.yaml --split dev --limit 2
    ... --confirm-cost
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path

from evals.generation import run as GR
from evals.generation.prompt import build_context

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_VERSION = "judge-v1"
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 4000
TPM = 8000                         # Groq free-tier tokens/minute for gpt-oss-120b (R3 drafting)
EST_COMPLETION = 900               # incl. hidden reasoning, until judgments exist (S-2026-11-23-25)
CORRECTNESS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
log = logging.getLogger("evals.judge")

SYSTEM = """You grade answers produced by a question-answering system over loan documents. Be strict and literal. Reply with one JSON object only."""

USER = """Grade the ANSWER to the QUESTION.

DOCUMENT CONTEXT (what the system saw; chunks labelled S1, S2, …):
{context}

QUESTION: {question}

REFERENCE ANSWER (ground truth): {reference}

REFERENCE KEY POINTS:
{key_points}

ANSWER (citations like [S2] refer to the context chunks):
{answer}

Instructions:
1. claims: split the ANSWER into its atomic factual claims (values, dates, parties, conditions, calculation results). Skip restatements of the question and the final "Final answer:" line if it only repeats a claim. For each claim set "supported": true only if the DOCUMENT CONTEXT states or directly implies it (arithmetic on values in the context counts as supported if the arithmetic is right). Judge support against the context, not the reference.
2. correctness against the REFERENCE ANSWER: "correct" = all essential facts match (wording may differ; equivalent numbers like $1,000,000 and $1 million match); "partial" = some essential facts right but at least one missing or wrong; "wrong" = the main fact is wrong or missing, or the answer declines although the reference gives an answer. If the reference says the documents do not contain the answer, a refusal is "correct" and any invented value is "wrong".
3. key_points: for each reference key point, in order, true if the ANSWER contains it (same meaning).
4. relevance (1-5): 5 = directly answers the question asked; 3 = partly on topic or padded; 1 = does not address the question.
5. rationale: one or two sentences.

JSON schema:
{{"claims": [{{"claim": "...", "supported": true}}], "correctness": "correct|partial|wrong", "key_points": [true, false], "relevance": 5, "rationale": "..."}}"""


def messages(q: dict, answer: str, ctx) -> list[dict]:
    kp = "\n".join(f"- {k}" for k in q.get("key_points") or []) or "- (none)"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(context=build_context(ctx) or "(empty)", question=q["question"],
                                                    reference=q["reference_answer"], key_points=kp, answer=answer)}]


def count_tokens(msgs: list[dict]) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_harmony")
    return sum(len(enc.encode(m["content"])) + 4 for m in msgs) + 3


class JudgeParseError(ValueError):
    pass


def parse(content: str, n_key_points: int) -> dict:
    """Validate the judge JSON. Raises JudgeParseError on anything off-schema."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        raise JudgeParseError("no JSON object")
    try:
        d = json.loads(m.group(0), strict=False)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"bad JSON: {exc}") from exc
    claims = d.get("claims")
    if not isinstance(claims, list) or any(not isinstance(c, dict) or not isinstance(c.get("supported"), bool)
                                           for c in claims):
        raise JudgeParseError("claims must be a list of {claim, supported: bool}")
    if d.get("correctness") not in CORRECTNESS:
        raise JudgeParseError(f"correctness {d.get('correctness')!r}")
    kp = d.get("key_points")
    if not isinstance(kp, list) or any(not isinstance(x, bool) for x in kp):
        raise JudgeParseError("key_points must be a list of bools")
    if len(kp) != n_key_points:
        raise JudgeParseError(f"key_points: {len(kp)} values for {n_key_points} key points")
    rel = d.get("relevance")
    if not isinstance(rel, int) or isinstance(rel, bool) or not 1 <= rel <= 5:
        raise JudgeParseError(f"relevance {rel!r}")
    return {"claims": [{"claim": str(c.get("claim", "")), "supported": c["supported"]} for c in claims],
            "correctness": d["correctness"], "key_points": kp, "relevance": rel,
            "rationale": str(d.get("rationale", ""))}


def scores(g: dict) -> dict:
    """Metrics from a parsed (or rule) grade."""
    claims = g["claims"]
    sup = sum(c["supported"] for c in claims)
    kp = g["key_points"]
    return {"faithfulness": sup / len(claims) if claims else None,
            "hallucinations": len(claims) - sup,
            "correctness": CORRECTNESS[g["correctness"]],
            "completeness": (sum(kp) / len(kp)) if kp else None,
            "relevance": g["relevance"]}


def rule_grade(q: dict, a: dict) -> dict | None:
    """Grade without a call when the answer is a refusal (see module docstring)."""
    if not a["metrics"]["refused"]:
        return None
    n = len(q.get("key_points") or [])
    if q.get("unanswerable"):
        return {"claims": [], "correctness": "correct", "key_points": [True] * n, "relevance": 5,
                "rationale": "rule: unanswerable question, answer refused"}
    return {"claims": [], "correctness": "wrong", "key_points": [False] * n, "relevance": None,
            "rationale": "rule: answerable question, answer refused"}


def _scores_rule(g: dict) -> dict:
    s = scores({**g, "relevance": g["relevance"] or 1})
    s["relevance"] = g["relevance"]
    return s


def call_judge(client, msgs: list[dict]) -> dict:
    r = client.chat.completions.create(
        model=JUDGE_MODEL, messages=msgs, temperature=0, reasoning_effort=REASONING_EFFORT,
        response_format={"type": "json_object"}, max_completion_tokens=MAX_COMPLETION_TOKENS)
    u = r.usage
    det = getattr(u, "completion_tokens_details", None)
    return {"content": r.choices[0].message.content or "", "returned_model": r.model,
            "finish_reason": r.choices[0].finish_reason, "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None}


def _failed_generation(exc: Exception) -> str | None:
    """Groq 400 json_validate_failed still carries the generation (raw control chars, R3 gotcha)."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    err = body.get("error", body) if isinstance(body.get("error", body), dict) else body
    return err.get("failed_generation") if err.get("code") == "json_validate_failed" else None


def judge_run(cfg, split: str, questions: dict, client, ledger, store, only=None, limit=None, confirm=False,
              retrieval_root: Path = GR.RETRIEVAL_ROOT, out_root: Path = GR.OUT_ROOT, tpm: int = TPM,
              sleep=time.sleep, keys: set[tuple[str, str]] | None = None) -> dict:
    """Judge saved answers. `keys`: only these (run, qid) pairs (the calibration sample)."""
    from evals.llm.ledger import BudgetExceeded, cost_usd
    runs = [r for r in cfg.runs if not only or r.name in only]
    todo, prev = [], []
    for r in runs:
        d = GR.out_dir(cfg, r, split, out_root)
        answers = [a for a in GR.read_jsonl(d / "answers.jsonl") if "error" not in a and a["qid"] in questions
                   and (keys is None or (r.name, a["qid"]) in keys)]
        answers = answers[:limit] if limit else answers
        done_rows = GR.read_jsonl(d / "judgments.jsonl")
        prev += [j for j in done_rows if j.get("judged_by") == "llm"]
        done = {j["qid"] for j in done_rows if j.get("judge_version") == JUDGE_VERSION and j.get("grade")}   # parse errors retry
        rsum = json.loads((GR.retrieval_dir(r, split, retrieval_root) / "summary.json").read_text())
        rrows = {x["qid"]: x for x in GR.read_jsonl(GR.retrieval_dir(r, split, retrieval_root) / "questions.jsonl")}
        todo += [(r, d, a, rsum, rrows[a["qid"]]) for a in answers if a["qid"] not in done]
    # rule grades first (free), then estimate the calls
    calls = []
    for r, d, a, rsum, rrow in todo:
        q = questions[a["qid"]]
        g = rule_grade(q, a)
        if g is not None:
            if confirm:
                _append(d, _record(a, g, "rule", None))
            continue
        ctx = GR.context_for(rrow, rsum, store)
        calls.append((r, d, a, messages(q, a["answer"], ctx)))
    est_p = sum(count_tokens(m) for *_, m in calls)
    comp = statistics.mean(j["completion_tokens"] for j in prev) if prev else EST_COMPLETION
    est_c = round(comp * len(calls))
    info = {"answers_todo": len(todo), "rule_graded": len(todo) - len(calls), "calls_todo": len(calls),
            "est_prompt_tokens": est_p, "est_completion_tokens": est_c, "est_total_tokens": est_p + est_c,
            "est_usd": round(cost_usd(JUDGE_MODEL, est_p, est_c), 4), "tokens_today": ledger.tokens_today(),
            "daily_budget": ledger.daily_budget}
    log.info("judge estimate: %s", info)
    if not confirm or not calls:
        return {**info, "called": 0, "stopped": None if calls else "nothing to call",
                **({"note": "dry run: pass --confirm-cost to call the judge"} if calls and not confirm else {})}
    called, parse_failures, stop = 0, 0, None
    for i, (r, d, a, msgs) in enumerate(calls):
        p_est = count_tokens(msgs)
        try:
            ledger.check(JUDGE_MODEL, p_est, round(comp))
        except BudgetExceeded as exc:
            stop = str(exc)
            break
        t0 = time.monotonic()
        try:
            out = call_judge(client, msgs)
        except Exception as exc:
            gen = _failed_generation(exc)
            if gen is None:
                stop = f"{type(exc).__name__}: {exc}"
                break
            out = {"content": gen, "returned_model": JUDGE_MODEL, "finish_reason": "json_validate_failed_salvaged",
                   "prompt_tokens": p_est, "completion_tokens": round(comp), "reasoning_tokens": None,
                   "usage_estimated": True}
        lrow = ledger.record(f"r6_judge:{r.name}:{a['qid']}", JUDGE_MODEL, out["returned_model"],
                             out["prompt_tokens"], out["completion_tokens"], out["reasoning_tokens"])
        q = questions[a["qid"]]
        try:
            g = parse(out["content"], len(q.get("key_points") or []))
            rec = _record(a, g, "llm", out)
        except JudgeParseError as exc:
            parse_failures += 1
            rec = {**_record(a, None, "llm", out), "parse_error": str(exc)}
        rec["cost_usd"] = lrow["cost_usd"]
        _append(d, rec)
        called += 1
        log.info("%s %s: %s, faith %s, %d+%d tok (reasoning %s)", r.name, a["qid"],
                 rec.get("grade", {}).get("correctness") if rec.get("grade") else "PARSE-ERROR",
                 rec.get("scores", {}).get("faithfulness") if rec.get("scores") else None,
                 out["prompt_tokens"], out["completion_tokens"], out["reasoning_tokens"])
        wait = (out["prompt_tokens"] + out["completion_tokens"]) / tpm * 60 - (time.monotonic() - t0)
        if wait > 0 and i < len(calls) - 1:
            sleep(wait)
    return {**info, "called": called, "parse_failures": parse_failures, "stopped": stop,
            "run_tokens": ledger.run_tokens, "run_cost_usd": round(ledger.run_cost, 4),
            "tokens_today_after": ledger.tokens_today()}


def _record(a: dict, g: dict | None, by: str, out: dict | None) -> dict:
    rec = {"qid": a["qid"], "run": a["run"], "type": a["type"], "judge_version": JUDGE_VERSION,
           "judged_by": by, "judge_model": JUDGE_MODEL if by == "llm" else None, "grade": g,
           "scores": (scores(g) if by == "llm" else _scores_rule(g)) if g else None}
    if out is not None:
        rec.update({k: out.get(k) for k in ("returned_model", "finish_reason", "prompt_tokens",
                                            "completion_tokens", "reasoning_tokens")})
        rec["raw"] = out["content"]
        if out.get("usage_estimated"):
            rec["usage_estimated"] = True
    return rec


def _append(d: Path, rec: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "judgments.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--runs")
    ap.add_argument("--limit", type=int, help="first N answers of each run")
    ap.add_argument("--confirm-cost", action="store_true")
    ap.add_argument("--daily-budget", type=int)
    ap.add_argument("--sample", type=int, help="dev only: judge the first N answers of the frozen calibration sample")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from evals.golden.qa.validate import QUESTIONS, read_questions
    from evals.llm.ledger import Ledger
    cfg = GR.load_config(a.config)
    questions = {q["qid"]: q for q in read_questions(QUESTIONS) if q["split"] == a.split}
    ledger = Ledger(**({"daily_budget": a.daily_budget} if a.daily_budget else {}))
    client = GR.groq_client() if a.confirm_cost else None
    keys = None
    if a.sample:
        if a.split != "dev":
            raise SystemExit("--sample is the dev calibration sample")
        from evals.generation.calibration import sample_keys
        keys = set(sample_keys(a.sample))
    res = judge_run(cfg, a.split, questions, client, ledger, GR.ChunkStore("P2"),
                    a.runs.split(",") if a.runs else None, a.limit, a.confirm_cost, keys=keys)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
