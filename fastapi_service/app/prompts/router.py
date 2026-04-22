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

Variant selection
-----------------
Each intent has two prompt variants (v1: detailed+instructional,
v2: concise+structured). The VariantSelector reads MLflow run data to pick
the better-performing variant per intent (see app/tracking/ab_selector.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.intent_classifier import Intent, ClassifiedIntent
from app.retrieval.retriever import RetrievedChunk
from app.prompts.templates.lookup import LOOKUP_PROMPT
from app.prompts.templates.lookup_v2 import LOOKUP_PROMPT_V2
from app.prompts.templates.calculate import CALCULATE_PROMPT
from app.prompts.templates.calculate_v2 import CALCULATE_PROMPT_V2
from app.prompts.templates.compare import COMPARE_PROMPT
from app.prompts.templates.compare_v2 import COMPARE_PROMPT_V2
from app.prompts.templates.explain import EXPLAIN_PROMPT
from app.prompts.templates.explain_v2 import EXPLAIN_PROMPT_V2
from app.prompts.templates.summarise import SUMMARISE_PROMPT
from app.prompts.templates.summarise_v2 import SUMMARISE_PROMPT_V2
from app.tracking.ab_selector import get_selector
from app.core.logging import get_logger

logger = get_logger(__name__)

# variant label → prompt template string
_VARIANT_TEMPLATES: dict[str, str] = {
    "lookup_v1":    LOOKUP_PROMPT,
    "lookup_v2":    LOOKUP_PROMPT_V2,
    "calculate_v1": CALCULATE_PROMPT,
    "calculate_v2": CALCULATE_PROMPT_V2,
    "compare_v1":   COMPARE_PROMPT,
    "compare_v2":   COMPARE_PROMPT_V2,
    "explain_v1":   EXPLAIN_PROMPT,
    "explain_v2":   EXPLAIN_PROMPT_V2,
    "summarise_v1": SUMMARISE_PROMPT,
    "summarise_v2": SUMMARISE_PROMPT_V2,
}

# intent → available variants (first entry is the fallback)
_INTENT_VARIANTS: dict[Intent, list[str]] = {
    Intent.LOOKUP:    ["lookup_v1",    "lookup_v2"],
    Intent.CALCULATE: ["calculate_v1", "calculate_v2"],
    Intent.COMPARE:   ["compare_v1",   "compare_v2"],
    Intent.EXPLAIN:   ["explain_v1",   "explain_v2"],
    Intent.SUMMARISE: ["summarise_v1", "summarise_v2"],
}

MAX_CONTEXT_WORDS = 1200   # keep prompt within reasonable token budget


@dataclass
class RoutedPrompt:
    prompt: str
    intent: Intent
    variant: str               # e.g. "lookup_v2" — logged to MLflow for A/B tracking
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
    Select prompt variant based on primary intent + A/B selector, then
    assemble with retrieved context.

    For multi-intent queries, the primary intent (highest priority) drives
    template selection. Secondary intents are logged for observability.
    """
    if not intents:
        primary_intent = Intent.SUMMARISE
    else:
        primary_intent = intents[0].intent

    available_variants = _INTENT_VARIANTS[primary_intent]

    try:
        selector = get_selector()
        variant = selector.choose(primary_intent.value, available_variants)
    except Exception as exc:
        # Selector failure must never break the query path
        logger.warning("ab_selector_unavailable", error=str(exc))
        variant = available_variants[0]

    template = _VARIANT_TEMPLATES[variant]
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
