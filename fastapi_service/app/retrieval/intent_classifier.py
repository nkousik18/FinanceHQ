"""
Intent classifier — zero-shot using MiniLM cosine similarity.

No LLM call. No training data. Reuses the same MiniLM model already
loaded for embeddings.

How it works:
  1. Each intent has a set of natural language descriptions
  2. Embed the query and all intent descriptions
  3. Cosine similarity → score per intent
  4. Return intents above threshold, sorted by priority

Intents (priority order — higher priority handled first):
  calculate > lookup > compare > explain > summarise

Multi-intent: if multiple intents score above MULTI_INTENT_THRESHOLD,
all are returned in priority order. Caller decides how to handle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.pipeline.embedder import get_embedder
from app.core.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Similarity thresholds
# ------------------------------------------------------------------

PRIMARY_THRESHOLD = 0.30      # minimum score to be considered at all
MULTI_INTENT_THRESHOLD = 0.28 # secondary intents must score above this


# ------------------------------------------------------------------
# Intent definitions
# ------------------------------------------------------------------

class Intent(str, Enum):
    CALCULATE  = "calculate"
    LOOKUP     = "lookup"
    COMPARE    = "compare"
    EXPLAIN    = "explain"
    SUMMARISE  = "summarise"


# Priority order — index 0 = highest priority
INTENT_PRIORITY: list[Intent] = [
    Intent.CALCULATE,
    Intent.LOOKUP,
    Intent.COMPARE,
    Intent.EXPLAIN,
    Intent.SUMMARISE,
]

# Natural language descriptions used for zero-shot matching.
# More descriptions per intent = better coverage.
INTENT_DESCRIPTIONS: dict[Intent, list[str]] = {
    Intent.CALCULATE: [
        "calculate the total repayment amount",
        "compute the monthly EMI payment",
        "how much will I owe after all payments",
        "work out the total cost of borrowing",
        "sum up all charges and fees",
        "what will the outstanding balance be after N months",
        "compute the debt service coverage ratio",
        "how much will the loan cost me overall",
    ],
    Intent.LOOKUP: [
        "what is the interest rate on this loan",
        "what is the annual interest rate",
        "find the loan amount in the document",
        "what is the loan tenure",
        "who is the primary applicant",
        "what is the branch name",
        "what date was the loan sanctioned",
        "retrieve a specific field value from the document",
        "what specific value appears in the document",
        "look up a particular piece of information",
    ],
    Intent.COMPARE: [
        "compare the applicant and co-applicant",
        "how does the income compare to the EMI",
        "difference between applicant and guarantor",
        "which is higher",
        "contrast the two values",
        "how do these figures compare",
        "compare applicant details",
    ],
    Intent.EXPLAIN: [
        "explain what happens if I miss a payment",
        "what does this clause mean",
        "why is this required",
        "explain the terms and conditions",
        "what does this section mean",
        "describe the repayment conditions",
        "explain the penalty",
        "what are the consequences",
    ],
    Intent.SUMMARISE: [
        "summarise the document",
        "give me an overview",
        "what is this document about",
        "summarise the key points",
        "briefly describe the loan",
        "what are the main terms",
        "give a summary of this document",
        "what are the highlights",
    ],
}


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class ClassifiedIntent:
    intent: Intent
    score: float
    priority: int    # lower = higher priority (0 = most important)
    is_primary: bool


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class IntentClassifier:
    def __init__(self) -> None:
        self._embedder = get_embedder()
        self._intent_embeddings = self._precompute_intent_embeddings()
        logger.info("intent_classifier_ready", intents=len(INTENT_DESCRIPTIONS))

    def _precompute_intent_embeddings(self) -> dict[Intent, np.ndarray]:
        """
        Pre-embed all intent descriptions once at startup.
        Each intent → mean of its description embeddings.
        """
        result: dict[Intent, np.ndarray] = {}
        for intent, descriptions in INTENT_DESCRIPTIONS.items():
            embeddings = self._embedder.embed(descriptions)   # (N, 384)
            result[intent] = embeddings.mean(axis=0)          # (384,) centroid
        logger.debug("intent_embeddings_precomputed", count=len(result))
        return result

    def classify(self, query: str) -> list[ClassifiedIntent]:
        """
        Classify the query into one or more intents.

        Returns intents sorted by priority (not by score).
        If no intent scores above threshold, returns [SUMMARISE] as fallback.
        """
        query_vec = self._embedder.embed_one(query)   # (384,) normalised

        scores: dict[Intent, float] = {}
        for intent, centroid in self._intent_embeddings.items():
            # Cosine similarity = dot product (both L2-normalised)
            norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            score = float(np.dot(query_vec, norm_centroid))
            scores[intent] = score

        logger.debug(
            "intent_scores",
            query=query[:60],
            scores={k.value: round(v, 3) for k, v in scores.items()},
        )

        # Find primary (highest scoring above threshold)
        best_intent = max(scores, key=lambda i: scores[i])
        best_score = scores[best_intent]

        if best_score < PRIMARY_THRESHOLD:
            # Nothing matched — fall back to summarise
            logger.info(
                "intent_fallback",
                query=query[:60],
                best_score=round(best_score, 3),
            )
            return [ClassifiedIntent(
                intent=Intent.SUMMARISE,
                score=best_score,
                priority=INTENT_PRIORITY.index(Intent.SUMMARISE),
                is_primary=True,
            )]

        # Primary intent always goes first (result[0] == primary)
        primary = ClassifiedIntent(
            intent=best_intent,
            score=round(best_score, 3),
            priority=INTENT_PRIORITY.index(best_intent),
            is_primary=True,
        )

        # Collect secondary intents above multi-intent threshold, sorted by priority
        secondaries: list[ClassifiedIntent] = []
        for intent in INTENT_PRIORITY:
            if intent == best_intent:
                continue
            score = scores[intent]
            if score >= MULTI_INTENT_THRESHOLD:
                secondaries.append(ClassifiedIntent(
                    intent=intent,
                    score=round(score, 3),
                    priority=INTENT_PRIORITY.index(intent),
                    is_primary=False,
                ))
        secondaries.sort(key=lambda x: x.priority)

        active = [primary] + secondaries

        logger.info(
            "intent_classified",
            query=query[:60],
            intents=[f"{c.intent.value}({c.score})" for c in active],
            primary=active[0].intent.value if active else None,
        )

        return active


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

from functools import lru_cache

@lru_cache()
def get_intent_classifier() -> IntentClassifier:
    return IntentClassifier()
