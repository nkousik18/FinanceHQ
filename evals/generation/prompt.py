"""
Citation prompt (scope v2 D5) and context assembly.

Context = the chunks the retrieval run marked `in_budget`, in rank order. That flag comes from
`evals.retrieval.metrics.budget_cut`, a copy of production `router._build_context` (greedy to 1,200
words, first chunk always in, stop at the first chunk that doesn't fit), so the generator sees exactly
what production would send for that ranking.

Chunks are labelled with short local ids `S1, S2, …` (rank order) instead of the long chunk ids
(`deal_10_am1-C2-75`): fewer prompt tokens and fewer mangled ids. The runner maps them back to
chunk ids, so citation metrics are per chunk as D5 specifies.

One template for every question type (production routes by intent; the eval needs one fixed prompt
so configurations differ only in retrieval). Fixed strings the metrics rely on:
- refusal: exactly REFUSAL ("Not found in the document.", production's lookup-template phrase)
- calculation answers end with a line "Final answer: <value>"
"""
from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "cite-v1"
REFUSAL = "Not found in the document."
FINAL_PREFIX = "Final answer:"

TEMPLATE = """\
You are a precise assistant answering questions about loan documents (credit agreements, their amendments, and mortgage disclosures).

Rules:
- Answer only from the document context below. Do not use outside knowledge.
- After each claim, cite the source chunk(s) it comes from in square brackets, e.g. [S2] or [S1][S3]. Cite only chunk ids shown in the context.
- If the context does not contain the answer, reply exactly: {refusal}
- When an amendment changed a value, give the value in force at the time the question asks about.
- If the question needs a calculation, show the steps briefly, then end with one line: "{final} <number with unit>".
- Be concise.

Document context:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class ContextChunk:
    local_id: str          # "S1"
    chunk_id: str
    doc_id: str
    page: int | None
    text: str
    covers: tuple[int, ...]   # evidence-span indexes this chunk covers (from the retrieval run)


def build_context(chunks: list[ContextChunk]) -> str:
    """Same layout as production (`header\\ntext`, joined by `---`), with the local id in the header."""
    parts = [f"[{c.local_id} | {c.doc_id} | Page {c.page if c.page is not None else '?'}]\n{c.text}"
             for c in chunks]
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, chunks: list[ContextChunk]) -> str:
    return TEMPLATE.format(refusal=REFUSAL, final=FINAL_PREFIX, context=build_context(chunks),
                           question=question)
