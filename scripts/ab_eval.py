"""
A/B Evaluation — compare LLM models on loan document Q&A.

Tests 3 models × 7 questions across 6 automated metrics:
  groundedness, hallucination, not-found compliance,
  structure, relevance, verbosity, latency

Each run is logged to MLflow (experiment: financehq_ab_eval):
  - One run per model × question (21 runs total)
  - One summary run per model (3 runs)

Run:
    cd /Users/Masters/Projects/FinanceHQ

    # Start MLflow UI first (separate terminal):
    mlflow ui --port 5000

    python scripts/ab_eval.py

Outputs:
    scripts/ab_results.json   — raw scores + responses
    scripts/ab_report.md      — human-readable report
    MLflow UI                 — http://localhost:5000
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from bytez import Bytez

# Allow importing app modules from the fastapi_service directory
sys.path.insert(0, str(Path(__file__).parent.parent / "fastapi_service"))
os.chdir(Path(__file__).parent.parent)   # so .env is found by pydantic-settings

from app.tracking.mlflow_tracker import FinanceHQTracker

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_KEY = "19408716817b70780ddaaea1a7e32eb6"
RESULTS_PATH = Path(__file__).parent / "ab_results.json"
REPORT_PATH  = Path(__file__).parent / "ab_report.md"

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct":  "Llama-3.1-8B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen/Qwen2.5-7B-Instruct":           "Qwen2.5-7B",
}

# ------------------------------------------------------------------
# Loan document context (realistic synthetic excerpt)
# ------------------------------------------------------------------

CONTEXT = """
## Applicant Details

- **Applicant Name**: Rajesh Kumar
- **Co-Applicant Name**: Priya Kumar
- **Applicant Monthly Income**: ₹85,000
- **Co-Applicant Monthly Income**: ₹42,000
- **Date of Application**: 14 March 2025
- **Branch**: Koramangala, Bengaluru

## Loan Details

- **Loan Amount Sanctioned**: ₹12,00,000
- **Annual Interest Rate**: 8.75%
- **Loan Tenure**: 60 months
- **EMI**: ₹24,842
- **Repayment Mode**: ECS (Electronic Clearing Service)
- **Processing Fee**: ₹12,000 (1% of loan amount)

## Terms and Conditions — Missed Payment

If the borrower fails to pay any EMI on the due date, a penal interest of 2% per month shall be levied on the overdue amount for the period of default. Continued default for more than 90 days will trigger NPA classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

## Financial Summary

- **Total Monthly Obligations (EMI)**: ₹24,842
- **Fixed Obligation Income Ratio (FOIR)**: 29.2%
- **Net Annual Income (Combined)**: ₹15,24,000
- **Property Value (Collateral)**: ₹28,00,000
- **LTV Ratio**: 42.8%
""".strip()

# ------------------------------------------------------------------
# 7 evaluation questions — one per intent + edge cases
# ------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "Q1",
        "intent": "lookup",
        "question": "What is the annual interest rate on this loan?",
        "expected_keywords": ["8.75", "8.75%"],
        "requires_not_found": False,
    },
    {
        "id": "Q2",
        "intent": "lookup",
        "question": "Who is the co-applicant and what is their monthly income?",
        "expected_keywords": ["priya", "42,000", "42000"],
        "requires_not_found": False,
    },
    {
        "id": "Q3",
        "intent": "calculate",
        "question": "What is the total amount paid over the full loan tenure?",
        "expected_keywords": ["24,842", "60", "14,90,520", "1490520", "14,90"],
        "requires_not_found": False,
    },
    {
        "id": "Q4",
        "intent": "compare",
        "question": "Compare the applicant and co-applicant monthly income.",
        "expected_keywords": ["85,000", "42,000", "rajesh", "priya"],
        "requires_not_found": False,
    },
    {
        "id": "Q5",
        "intent": "explain",
        "question": "What happens if I miss an EMI payment?",
        "expected_keywords": ["2%", "penal", "90 days", "npa", "sarfaesi"],
        "requires_not_found": False,
    },
    {
        "id": "Q6",
        "intent": "summarise",
        "question": "Give me a summary of this loan document.",
        "expected_keywords": ["rajesh", "12,00,000", "8.75", "60 months", "24,842"],
        "requires_not_found": False,
    },
    {
        "id": "Q7",
        "intent": "not_found",
        "question": "What is the credit score of the applicant?",
        "expected_keywords": [],
        "requires_not_found": True,  # correct answer = acknowledge it's not in the document
    },
]

# Prompt template (matches our LOOKUP template style — ground the model)
PROMPT_TEMPLATE = """\
You are a precise loan document assistant. Answer only from the provided document context.
- If the exact value is present, quote it directly.
- If performing a calculation, show your steps and state the final answer with units.
- If comparing, present a clear structured comparison.
- If the information is not in the document, say "Not found in the document."
- Do not use outside knowledge.

Document Context:
{context}

Question: {question}

Answer:"""


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

NOT_FOUND_PHRASES = [
    "not found", "not mentioned", "not provided", "not in the document",
    "not available", "not stated", "no information", "cannot find",
    "does not mention", "not specified",
]

STRUCTURE_MARKERS = [
    r"\|",           # markdown table
    r"^\s*[-*]\s",   # bullet points
    r"^\d+\.",       # numbered list
    r":\s*₹",        # key: value with currency
    r"vs\.?|versus", # comparison word
]


def _keywords_in_text(keywords: list[str], text: str) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 2)


def _hallucination_score(answer: str, context: str) -> float:
    """
    Extract all numbers from the answer.
    Check what fraction appear in the context.
    High fraction = grounded. Low fraction = potential hallucination.
    Returns a groundedness score (1.0 = no invented numbers).
    """
    # Find all numbers/amounts in the answer
    answer_numbers = set(re.findall(r"[\d,]+(?:\.\d+)?", answer))
    if not answer_numbers:
        return 1.0  # No numbers to verify — can't hallucinate numbers
    context_numbers = set(re.findall(r"[\d,]+(?:\.\d+)?", context))
    grounded = sum(1 for n in answer_numbers if n in context_numbers)
    return round(grounded / len(answer_numbers), 2)


def _not_found_compliance(answer: str) -> bool:
    """Returns True if the answer correctly declines to answer."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES)


def _structure_score(answer: str) -> float:
    """
    Checks for structured formatting in the answer.
    Returns 0.0–1.0 based on how many structure markers appear.
    """
    hits = sum(
        1 for pattern in STRUCTURE_MARKERS
        if re.search(pattern, answer, re.MULTILINE | re.IGNORECASE)
    )
    return round(min(hits / 2, 1.0), 2)  # 2+ markers = full score


def _relevance_score(question: str, answer: str) -> float:
    """
    Jaccard similarity between content words in the question and answer.
    Filters out stop words.
    """
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "did", "will", "would", "could",
        "should", "of", "in", "on", "at", "to", "for", "and", "or",
        "but", "if", "this", "that", "what", "how", "me", "my", "i",
        "give", "tell", "show", "please", "can", "you",
    }
    q_words = {w.lower() for w in re.findall(r"\w+", question) if w.lower() not in stop}
    a_words = {w.lower() for w in re.findall(r"\w+", answer) if w.lower() not in stop}
    if not q_words or not a_words:
        return 0.0
    intersection = q_words & a_words
    union = q_words | a_words
    return round(len(intersection) / len(union), 2)


def score_response(
    question_meta: dict,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """Compute all automated metrics for one response."""
    word_count = len(answer.split())
    groundedness = _keywords_in_text(question_meta["expected_keywords"], answer)
    number_groundedness = _hallucination_score(answer, context)
    not_found_ok = (
        _not_found_compliance(answer)
        if question_meta["requires_not_found"]
        else None
    )
    structure = _structure_score(answer)
    relevance = _relevance_score(question_meta["question"], answer)

    return {
        "groundedness":          groundedness,           # keyword hits from expected answer
        "number_groundedness":   number_groundedness,    # invented numbers check
        "not_found_compliance":  not_found_ok,           # None if n/a
        "structure_score":       structure,              # formatting quality
        "relevance_score":       relevance,              # Jaccard question↔answer
        "word_count":            word_count,
    }


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

@dataclass
class EvalResult:
    model: str
    model_short: str
    question_id: str
    intent: str
    question: str
    answer: str
    latency_ms: float
    error: str | None
    groundedness: float
    number_groundedness: float
    not_found_compliance: bool | None
    structure_score: float
    relevance_score: float
    word_count: int


def run_eval(tracker: FinanceHQTracker) -> list[EvalResult]:
    sdk = Bytez(API_KEY)
    results: list[EvalResult] = []
    total = len(MODELS) * len(QUESTIONS)
    count = 0

    for model_id in MODELS:
        short = MODEL_SHORT[model_id]
        model = sdk.model(model_id)
        print(f"\n{'='*60}")
        print(f"Model: {short}")
        print(f"{'='*60}")

        for q in QUESTIONS:
            count += 1
            prompt = PROMPT_TEMPLATE.format(
                context=CONTEXT,
                question=q["question"],
            )
            print(f"  [{count}/{total}] {q['id']} ({q['intent']}) — {q['question'][:55]}...")

            answer = ""
            error = None
            latency_ms = 0.0

            # Retry up to 3 times for rate limits / capacity errors
            for attempt in range(3):
                t0 = time.monotonic()
                r = model.run(
                    [{"role": "user", "content": prompt}],
                    params={"max_new_tokens": 400, "temperature": 0.1},
                )
                latency_ms = round((time.monotonic() - t0) * 1000, 1)

                if r.error:
                    print(f"    attempt {attempt+1} error: {r.error[:80]}")
                    if attempt < 2:
                        time.sleep(3)
                    else:
                        error = r.error
                else:
                    if isinstance(r.output, dict):
                        answer = r.output.get("content", "")
                    else:
                        answer = str(r.output or "")
                    break

            if error:
                result = EvalResult(
                    model=model_id, model_short=short,
                    question_id=q["id"], intent=q["intent"],
                    question=q["question"], answer="",
                    latency_ms=latency_ms, error=error,
                    groundedness=0, number_groundedness=0,
                    not_found_compliance=None,
                    structure_score=0, relevance_score=0, word_count=0,
                )
                results.append(result)
                tracker.log_eval_run(
                    model_id=model_id, model_short=short,
                    question_id=q["id"], intent=q["intent"],
                    question=q["question"], answer="", prompt=prompt,
                    latency_ms=latency_ms, error=error,
                    groundedness=0, number_groundedness=0,
                    structure_score=0, relevance_score=0,
                    word_count=0, not_found_compliance=None,
                )
                continue

            metrics = score_response(q, answer, CONTEXT)
            print(f"    latency={latency_ms}ms  words={metrics['word_count']}  "
                  f"ground={metrics['groundedness']}  relevance={metrics['relevance_score']}")

            result = EvalResult(
                model=model_id, model_short=short,
                question_id=q["id"], intent=q["intent"],
                question=q["question"], answer=answer,
                latency_ms=latency_ms, error=None,
                **metrics,
            )
            results.append(result)

            # Log this run to MLflow
            tracker.log_eval_run(
                model_id=model_id, model_short=short,
                question_id=q["id"], intent=q["intent"],
                question=q["question"], answer=answer, prompt=prompt,
                latency_ms=latency_ms, error=None,
                groundedness=metrics["groundedness"],
                number_groundedness=metrics["number_groundedness"],
                structure_score=metrics["structure_score"],
                relevance_score=metrics["relevance_score"],
                word_count=metrics["word_count"],
                not_found_compliance=metrics["not_found_compliance"],
            )

            # Small pause between requests — free tier is 1 at a time
            time.sleep(1)

    return results


# ------------------------------------------------------------------
# Report generator
# ------------------------------------------------------------------

def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def generate_report(results: list[EvalResult]) -> str:
    lines: list[str] = []

    lines += [
        "# A/B Evaluation Report — FinanceHQ LLM Models",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d')}  ",
        f"**Models tested**: {', '.join(MODEL_SHORT.values())}  ",
        f"**Questions**: {len(QUESTIONS)} × {len(MODELS)} models = {len(QUESTIONS) * len(MODELS)} total responses  ",
        "",
        "---",
        "",
    ]

    # ---- Per-model aggregate table ----
    lines += ["## Overall Model Scores", ""]
    lines += [
        "| Model | Groundedness | Number Accuracy | Structure | Relevance | Avg Latency (ms) | Avg Words |",
        "|---|---|---|---|---|---|---|",
    ]

    model_summaries: dict[str, dict] = {}
    for m_id, short in MODEL_SHORT.items():
        m_results = [r for r in results if r.model == m_id and not r.error]
        if not m_results:
            continue
        summary = {
            "groundedness":        _avg([r.groundedness for r in m_results]),
            "number_groundedness": _avg([r.number_groundedness for r in m_results]),
            "structure_score":     _avg([r.structure_score for r in m_results]),
            "relevance_score":     _avg([r.relevance_score for r in m_results]),
            "latency_ms":          _avg([r.latency_ms for r in m_results]),
            "word_count":          _avg([r.word_count for r in m_results]),
        }
        model_summaries[short] = summary
        lines.append(
            f"| {short} "
            f"| {summary['groundedness']:.2f} "
            f"| {summary['number_groundedness']:.2f} "
            f"| {summary['structure_score']:.2f} "
            f"| {summary['relevance_score']:.2f} "
            f"| {summary['latency_ms']:.0f} "
            f"| {summary['word_count']:.0f} |"
        )
    lines += [""]

    # ---- Best model ----
    if model_summaries:
        def _composite(s: dict) -> float:
            return (
                s["groundedness"]        * 0.30 +
                s["number_groundedness"] * 0.25 +
                s["relevance_score"]     * 0.25 +
                s["structure_score"]     * 0.10 +
                max(0, 1 - s["latency_ms"] / 10000) * 0.10  # latency bonus, normalised
            )
        best = max(model_summaries, key=lambda m: _composite(model_summaries[m]))
        lines += [
            f"**Best overall model: {best}** "
            f"(composite score weights: groundedness 30%, number accuracy 25%, relevance 25%, structure 10%, speed 10%)",
            "",
            "---",
            "",
        ]

    # ---- Per-question breakdown ----
    lines += ["## Per-Question Breakdown", ""]

    for q in QUESTIONS:
        qid = q["id"]
        lines += [
            f"### {qid} — {q['intent'].upper()}",
            f"**Question**: {q['question']}",
            "",
        ]

        if q["requires_not_found"]:
            lines += [
                "**Expected behaviour**: Model should acknowledge the information is not in the document.",
                "",
            ]

        for m_id, short in MODEL_SHORT.items():
            match = next((r for r in results if r.model == m_id and r.question_id == qid), None)
            if not match:
                continue

            lines += [f"**{short}**"]
            if match.error:
                lines += [f"> ERROR: {match.error}", ""]
                continue

            lines += [f"> {match.answer.strip()}", ""]

            metric_parts = [
                f"groundedness={match.groundedness}",
                f"number_accuracy={match.number_groundedness}",
                f"structure={match.structure_score}",
                f"relevance={match.relevance_score}",
                f"words={match.word_count}",
                f"latency={match.latency_ms}ms",
            ]
            if match.not_found_compliance is not None:
                complied = "PASS" if match.not_found_compliance else "FAIL"
                metric_parts.insert(0, f"not_found={complied}")

            lines += [f"*Metrics: {' | '.join(metric_parts)}*", ""]

        lines += ["---", ""]

    # ---- Intent-level analysis ----
    lines += ["## Intent-Level Analysis", ""]
    intents = ["lookup", "calculate", "compare", "explain", "summarise", "not_found"]
    lines += [
        "| Intent | Best Model | Avg Groundedness | Avg Relevance |",
        "|---|---|---|---|",
    ]

    for intent in intents:
        intent_results = [r for r in results if r.intent == intent and not r.error]
        if not intent_results:
            continue

        by_model: dict[str, list[EvalResult]] = {}
        for r in intent_results:
            by_model.setdefault(r.model_short, []).append(r)

        best_m = max(
            by_model,
            key=lambda m: _avg([r.groundedness + r.relevance_score for r in by_model[m]])
        )
        avg_ground = _avg([r.groundedness for r in intent_results])
        avg_rel    = _avg([r.relevance_score for r in intent_results])
        lines.append(f"| {intent} | {best_m} | {avg_ground:.2f} | {avg_rel:.2f} |")

    lines += [""]

    # ---- Response type observations ----
    lines += [
        "## Response Type Observations",
        "",
        "| Model | Verbosity | Tends To | Hallucination Risk | Not-Found Compliance |",
        "|---|---|---|---|---|",
    ]

    for m_id, short in MODEL_SHORT.items():
        m_results = [r for r in results if r.model == m_id and not r.error]
        if not m_results:
            continue

        avg_words = _avg([r.word_count for r in m_results])
        avg_num_g = _avg([r.number_groundedness for r in m_results])
        nf_results = [r for r in m_results if r.not_found_compliance is not None]
        nf_pass = sum(1 for r in nf_results if r.not_found_compliance)
        nf_str = f"{nf_pass}/{len(nf_results)}" if nf_results else "N/A"

        verbosity = "Verbose" if avg_words > 100 else ("Concise" if avg_words < 50 else "Moderate")
        hall_risk = "Low" if avg_num_g >= 0.80 else ("Medium" if avg_num_g >= 0.60 else "High")

        # Tendency: check if it adds unsolicited context
        avg_ground = _avg([r.groundedness for r in m_results])
        tendency = "Stays on topic" if avg_ground >= 0.7 else "Adds context" if avg_words > 100 else "Minimal answers"

        lines.append(f"| {short} | {verbosity} ({avg_words:.0f} words) | {tendency} | {hall_risk} | {nf_str} |")

    lines += [
        "",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | How It's Computed |",
        "|---|---|",
        "| **Groundedness** | Fraction of expected answer keywords found in the response |",
        "| **Number Accuracy** | Fraction of numbers in the response that appear in the source context |",
        "| **Structure Score** | Presence of markdown formatting: bullets, tables, key-value pairs (0–1) |",
        "| **Relevance Score** | Jaccard similarity between content words in question and answer |",
        "| **Not-Found Compliance** | For Q7 (missing info): did the model correctly decline to answer? |",
        "| **Word Count** | Total words in the raw response |",
        "| **Latency** | Wall-clock time from API call to response (ms) |",
        "",
        "> All metrics are automated. No LLM-as-judge calls — scores are deterministic and reproducible.",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("FinanceHQ LLM A/B Evaluation")
    print(f"Models: {', '.join(MODEL_SHORT.values())}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Total calls: {len(MODELS) * len(QUESTIONS)}")
    print()

    # Initialise MLflow tracker
    tracker = FinanceHQTracker()
    print(f"MLflow experiment: financehq_ab_eval")
    print(f"MLflow UI: http://localhost:5000  (run: mlflow ui --port 5000)\n")

    results = run_eval(tracker)

    # Save raw results
    raw = [asdict(r) for r in results]
    RESULTS_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    print(f"\nRaw results saved → {RESULTS_PATH}")

    # Generate markdown report
    report = generate_report(results)
    REPORT_PATH.write_text(report)
    print(f"Report saved       → {REPORT_PATH}")

    # Log one summary run per model to MLflow
    print("\nLogging model summaries to MLflow...")
    for m_id, short in MODEL_SHORT.items():
        m_results = [r for r in results if r.model == m_id and not r.error]
        if not m_results:
            continue

        avg_g  = _avg([r.groundedness        for r in m_results])
        avg_ng = _avg([r.number_groundedness  for r in m_results])
        avg_s  = _avg([r.structure_score      for r in m_results])
        avg_r  = _avg([r.relevance_score      for r in m_results])
        avg_l  = _avg([r.latency_ms           for r in m_results])
        avg_w  = _avg([r.word_count           for r in m_results])
        errors = sum(1 for r in results if r.model == m_id and r.error)

        composite = (
            avg_g  * 0.30 +
            avg_ng * 0.25 +
            avg_r  * 0.25 +
            avg_s  * 0.10 +
            max(0, 1 - avg_l / 10000) * 0.10
        )

        tracker.log_eval_summary(
            model_short=short,
            model_id=m_id,
            avg_groundedness=avg_g,
            avg_number_groundedness=avg_ng,
            avg_structure_score=avg_s,
            avg_relevance_score=avg_r,
            avg_latency_ms=avg_l,
            avg_word_count=avg_w,
            composite_score=round(composite, 4),
            total_questions=len(m_results),
            error_count=errors,
        )

    # Print summary to stdout
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for m_id, short in MODEL_SHORT.items():
        m_results = [r for r in results if r.model == m_id and not r.error]
        if m_results:
            avg_g = _avg([r.groundedness for r in m_results])
            avg_r = _avg([r.relevance_score for r in m_results])
            avg_l = _avg([r.latency_ms for r in m_results])
            errors = sum(1 for r in results if r.model == m_id and r.error)
            print(f"{short:<18} ground={avg_g:.2f}  relevance={avg_r:.2f}  "
                  f"latency={avg_l:.0f}ms  errors={errors}")

    print(f"\nView results: mlflow ui --port 5000  →  http://localhost:5000")
