"""
Prompt router — selects and assembles the right prompt for a detected intent.

Given:
  - classified intents (from IntentClassifier)
  - retrieved chunks (from Retriever)
  - user question

Returns:
  - assembled prompt string ready to send to the LLM
  - the intent that was used (primary)
  - the prompt variant label (for MLflow A/B tracking)
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.intent_classifier import Intent, ClassifiedIntent
from app.retrieval.retriever import RetrievedChunk
from app.prompts.templates.lookup import LOOKUP_PROMPT
from app.prompts.templates.calculate import CALCULATE_PROMPT
from app.prompts.templates.compare import COMPARE_PROMPT
from app.prompts.templates.explain import EXPLAIN_PROMPT
from app.prompts.templates.summarise import SUMMARISE_PROMPT
from app.core.logging import get_logger

logger = get_logger(__name__)

# Map intent → (template, variant_label)
# variant_label is used for MLflow A/B tracking in Phase 3
_TEMPLATE_MAP: dict[Intent, tuple[str, str]] = {
    Intent.LOOKUP:    (LOOKUP_PROMPT,    "lookup_v1"),
    Intent.CALCULATE: (CALCULATE_PROMPT, "calculate_v1"),
    Intent.COMPARE:   (COMPARE_PROMPT,   "compare_v1"),
    Intent.EXPLAIN:   (EXPLAIN_PROMPT,   "explain_v1"),
    Intent.SUMMARISE: (SUMMARISE_PROMPT, "summarise_v1"),
}

MAX_CONTEXT_WORDS = 1200   # keep prompt within reasonable token budget


@dataclass
class RoutedPrompt:
    prompt: str
    intent: Intent
    variant: str               # e.g. "lookup_v1" — used for A/B tracking
    chunks_used: int
    context_words: int


def _build_context(chunks: list[RetrievedChunk], max_words: int) -> tuple[str, int]:
    """
    Assemble retrieved chunks into a context string within the word budget.
    Chunks are already sorted by score (rank 1 = best).
    """
    sections: list[str] = []
    total_words = 0

    for rc in chunks:
        chunk_words = rc.chunk.word_count
        if total_words + chunk_words > max_words and sections:
            break
        header = f"[Chunk {rc.rank} | Page {rc.chunk.page} | Score {rc.score:.2f}]"
        sections.append(f"{header}\n{rc.chunk.text}")
        total_words += chunk_words

    return "\n\n---\n\n".join(sections), total_words


def route(
    question: str,
    intents: list[ClassifiedIntent],
    chunks: list[RetrievedChunk],
) -> RoutedPrompt:
    """
    Select prompt template based on primary intent, assemble with context.

    For multi-intent queries, the primary intent (highest priority) drives
    template selection. Secondary intents are logged for MLflow tracking.
    """
    if not intents:
        # Safety fallback
        primary_intent = Intent.SUMMARISE
        variant = "summarise_v1"
    else:
        primary_intent = intents[0].intent
        variant = _TEMPLATE_MAP[primary_intent][1]

    template, variant = _TEMPLATE_MAP[primary_intent]
    context, context_words = _build_context(chunks, MAX_CONTEXT_WORDS)
    prompt = template.format(context=context, question=question)

    secondary = [i.intent.value for i in intents[1:]] if len(intents) > 1 else []

    logger.info(
        "prompt_routed",
        primary_intent=primary_intent.value,
        secondary_intents=secondary,
        variant=variant,
        chunks_used=len(chunks),
        context_words=context_words,
    )

    return RoutedPrompt(
        prompt=prompt,
        intent=primary_intent,
        variant=variant,
        chunks_used=len(chunks),
        context_words=context_words,
    )
