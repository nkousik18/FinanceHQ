"""
A/B variant selector — reads MLflow run data to route each query to the
best-performing prompt variant for its detected intent.

Optimization target: avg total_latency_ms across recent runs.
Once the LLM-as-judge evaluator is added, replace latency with a quality
score (e.g. groundedness × relevance) for a more meaningful signal.

Selection strategy
------------------
Exploration phase  — any variant with fewer than MIN_RUNS runs gets traffic
                     until it reaches the threshold (round-robin exploration).
Exploitation phase — once all variants have MIN_RUNS+ runs, pick the one
                     with the lowest average latency.
Epsilon-greedy     — with probability EXPLORE_RATE a random variant is chosen
                     regardless of performance, so all variants keep receiving
                     traffic and the selector can react to prompt changes.

Caching
-------
MLflow is queried at most once per CACHE_TTL_SECONDS (default 5 min).
A stale cache is acceptable — latency differences between variants are small
and we prefer low overhead over perfect freshness.

Fallback
--------
Any MLflow error (server down, no runs yet) falls back silently to the first
variant in the list (always v1 by convention).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import lru_cache

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EXPLORE_RATE      = 0.10   # 10 % random exploration
MIN_RUNS          = 10     # runs needed before exploitation starts
CACHE_TTL_SECONDS = 300    # 5 minutes between MLflow queries
QUERY_EXPERIMENT  = "financehq_rag_queries"
MAX_RUNS_PER_VARIANT = 500  # cap MLflow search to keep it fast


@dataclass
class _VariantStats:
    variant: str
    run_count: int = 0
    avg_latency_ms: float = float("inf")


@dataclass
class _Cache:
    # intent value (str) → chosen variant label
    selections: dict[str, str] = field(default_factory=dict)
    expires_at: float = 0.0

    def is_fresh(self) -> bool:
        return time.monotonic() < self.expires_at


class VariantSelector:
    """
    Reads MLflow to pick the best prompt variant per intent.
    Thread-safe for read access (FastAPI runs sync endpoints in a thread pool).
    """

    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._cache = _Cache()

    def choose(self, intent: str, variants: list[str]) -> str:
        """
        Return the variant to use for `intent` chosen from `variants`.

        Always returns a value — falls back to variants[0] on any error.
        """
        if len(variants) == 1:
            return variants[0]

        # Epsilon-greedy: explore randomly with probability EXPLORE_RATE
        if random.random() < EXPLORE_RATE:
            chosen = random.choice(variants)
            logger.debug("ab_selector_explore", intent=intent, chosen=chosen)
            return chosen

        if not self._cache.is_fresh():
            self._refresh(variants)

        chosen = self._cache.selections.get(intent, variants[0])
        logger.debug("ab_selector_exploit", intent=intent, chosen=chosen)
        return chosen

    # ------------------------------------------------------------------
    # Cache refresh
    # ------------------------------------------------------------------

    def _refresh(self, all_variants: list[str]) -> None:
        try:
            self._cache.selections = self._compute_selections(all_variants)
            self._cache.expires_at = time.monotonic() + CACHE_TTL_SECONDS
            logger.info("ab_selector_cache_refreshed", selections=self._cache.selections)
        except Exception as exc:
            # MLflow unavailable — keep stale cache, log and carry on
            logger.warning("ab_selector_mlflow_error", error=str(exc))
            self._cache.expires_at = time.monotonic() + 60  # retry in 1 min

    def _compute_selections(self, all_variants: list[str]) -> dict[str, str]:
        """
        Query MLflow and return the best variant per intent.
        Returns {} if the experiment doesn't exist yet (no runs logged).
        """
        exp = self._client.get_experiment_by_name(QUERY_EXPERIMENT)
        if exp is None:
            return {}

        # Fetch stats for every known variant in one pass
        variant_stats: dict[str, dict[str, _VariantStats]] = {}
        # shape: {intent_value: {variant_label: _VariantStats}}

        for variant in all_variants:
            runs = self._client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.variant = '{variant}'",
                max_results=MAX_RUNS_PER_VARIANT,
            )
            for run in runs:
                intent = run.data.tags.get("intent", "unknown")
                if intent not in variant_stats:
                    variant_stats[intent] = {}
                if variant not in variant_stats[intent]:
                    variant_stats[intent][variant] = _VariantStats(variant=variant)

                stats = variant_stats[intent][variant]
                latency = run.data.metrics.get("total_latency_ms")
                if latency is not None:
                    # Incremental mean update
                    stats.run_count += 1
                    stats.avg_latency_ms = (
                        stats.avg_latency_ms * (stats.run_count - 1) + latency
                    ) / stats.run_count

        selections: dict[str, str] = {}
        for intent, stats_by_variant in variant_stats.items():
            selections[intent] = self._pick(stats_by_variant, all_variants)

        return selections

    def _pick(
        self,
        stats_by_variant: dict[str, _VariantStats],
        all_variants: list[str],
    ) -> str:
        """
        Given per-variant stats for one intent, return the variant to exploit.

        If any variant is still in the exploration phase (< MIN_RUNS),
        return the one with the fewest runs to keep building its baseline.
        Otherwise return the variant with the lowest average latency.
        """
        # Check exploration phase: any variant below MIN_RUNS?
        under_explored = [
            s for s in stats_by_variant.values() if s.run_count < MIN_RUNS
        ]
        if under_explored:
            least_seen = min(under_explored, key=lambda s: s.run_count)
            return least_seen.variant

        # All variants have enough data — exploit best latency
        best = min(stats_by_variant.values(), key=lambda s: s.avg_latency_ms)
        return best.variant


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

@lru_cache()
def get_selector() -> VariantSelector:
    return VariantSelector()
