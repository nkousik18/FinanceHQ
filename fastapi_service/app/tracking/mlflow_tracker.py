"""
MLflow tracking for FinanceHQ.

Two experiments:
  financehq_ab_eval       — one run per model × question in the A/B eval script
  financehq_rag_queries   — one run per live /query API call

Usage — eval script (synchronous):
    tracker = FinanceHQTracker()
    tracker.log_eval_run(...)

Usage — FastAPI endpoint (non-blocking):
    background_tasks.add_task(tracker.log_query_run, ...)
"""
from __future__ import annotations

import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

AB_EVAL_EXPERIMENT    = "financehq_ab_eval"
QUERY_EXPERIMENT      = "financehq_rag_queries"


class FinanceHQTracker:
    """
    Thin wrapper around the MLflow client.

    All MLflow calls are synchronous — the caller decides whether to run
    them in-band (eval script) or in a background thread (FastAPI endpoint).
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        settings = get_settings()
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        self._client = MlflowClient(tracking_uri=uri)
        self._ensure_experiments()
        logger.info("mlflow_tracker_ready", tracking_uri=uri)

    # ------------------------------------------------------------------
    # Experiment setup
    # ------------------------------------------------------------------

    def _ensure_experiments(self) -> None:
        """Create experiments if they don't exist."""
        for name in (AB_EVAL_EXPERIMENT, QUERY_EXPERIMENT):
            if not self._client.get_experiment_by_name(name):
                self._client.create_experiment(name)
                logger.info("mlflow_experiment_created", name=name)

    # ------------------------------------------------------------------
    # A/B Eval logging
    # ------------------------------------------------------------------

    def log_eval_run(
        self,
        *,
        model_id: str,
        model_short: str,
        question_id: str,
        intent: str,
        question: str,
        answer: str,
        prompt: str,
        latency_ms: float,
        groundedness: float,
        number_groundedness: float,
        structure_score: float,
        relevance_score: float,
        word_count: int,
        not_found_compliance: bool | None,
        error: str | None,
        temperature: float = 0.1,
        max_new_tokens: int = 400,
    ) -> str:
        """
        Log one model × question eval run.
        Returns the MLflow run_id.
        """
        run_name = f"{model_short}_{question_id}"

        with mlflow.start_run(
            experiment_id=self._get_experiment_id(AB_EVAL_EXPERIMENT),
            run_name=run_name,
        ) as run:
            # --- Tags (used for filtering/grouping in the MLflow UI) ---
            mlflow.set_tags({
                "model_short":  model_short,
                "intent":       intent,
                "question_id":  question_id,
                "run_type":     "eval",
                "status":       "error" if error else "ok",
            })

            # --- Params (fixed inputs — appear in the Params tab) ---
            mlflow.log_params({
                "model_id":       model_id,
                "model_short":    model_short,
                "question_id":    question_id,
                "intent":         intent,
                "temperature":    temperature,
                "max_new_tokens": max_new_tokens,
            })

            # --- Metrics (numeric scores — appear in the Metrics tab) ---
            if not error:
                metrics: dict[str, float] = {
                    "groundedness":        groundedness,
                    "number_groundedness": number_groundedness,
                    "structure_score":     structure_score,
                    "relevance_score":     relevance_score,
                    "word_count":          float(word_count),
                    "latency_ms":          latency_ms,
                }
                if not_found_compliance is not None:
                    metrics["not_found_compliance"] = float(not_found_compliance)
                mlflow.log_metrics(metrics)

            # --- Artifacts (text files — downloadable from the Artifacts tab) ---
            mlflow.log_text(question, "question.txt")
            if answer:
                mlflow.log_text(answer, "answer.txt")
            if prompt:
                mlflow.log_text(prompt, "prompt.txt")
            if error:
                mlflow.log_text(error, "error.txt")

        logger.debug(
            "mlflow_eval_run_logged",
            run_name=run_name,
            run_id=run.info.run_id,
            intent=intent,
        )
        return run.info.run_id

    def log_eval_summary(
        self,
        *,
        model_short: str,
        model_id: str,
        avg_groundedness: float,
        avg_number_groundedness: float,
        avg_structure_score: float,
        avg_relevance_score: float,
        avg_latency_ms: float,
        avg_word_count: float,
        composite_score: float,
        total_questions: int,
        error_count: int,
    ) -> str:
        """
        Log one aggregate summary run per model.
        Useful for the top-level model comparison view in MLflow.
        """
        run_name = f"{model_short}_SUMMARY"

        with mlflow.start_run(
            experiment_id=self._get_experiment_id(AB_EVAL_EXPERIMENT),
            run_name=run_name,
        ) as run:
            mlflow.set_tags({
                "model_short": model_short,
                "run_type":    "eval_summary",
            })
            mlflow.log_params({
                "model_id":        model_id,
                "model_short":     model_short,
                "total_questions": total_questions,
            })
            mlflow.log_metrics({
                "avg_groundedness":        avg_groundedness,
                "avg_number_groundedness": avg_number_groundedness,
                "avg_structure_score":     avg_structure_score,
                "avg_relevance_score":     avg_relevance_score,
                "avg_latency_ms":          avg_latency_ms,
                "avg_word_count":          avg_word_count,
                "composite_score":         composite_score,
                "error_count":             float(error_count),
            })

        logger.info(
            "mlflow_eval_summary_logged",
            model=model_short,
            composite_score=composite_score,
            run_id=run.info.run_id,
        )
        return run.info.run_id

    # ------------------------------------------------------------------
    # Production query logging
    # ------------------------------------------------------------------

    def log_query_run(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        prompt: str,
        intent: str,
        variant: str,
        model_id: str,
        top_k: int,
        chunks_used: int,
        context_words: int,
        llm_latency_ms: float,
        total_latency_ms: float,
        retrieved_chunks: list[dict] | None = None,
    ) -> str:
        """
        Log one production /query call.
        Designed to be called from a FastAPI BackgroundTask — runs in a
        thread pool executor so it never blocks the HTTP response.
        """
        with mlflow.start_run(
            experiment_id=self._get_experiment_id(QUERY_EXPERIMENT),
        ) as run:
            mlflow.set_tags({
                "intent":    intent,
                "variant":   variant,
                "run_type":  "query",
            })
            mlflow.log_params({
                "session_id": session_id,
                "model_id":   model_id,
                "intent":     intent,
                "variant":    variant,
                "top_k":      top_k,
            })
            mlflow.log_metrics({
                "chunks_used":      float(chunks_used),
                "context_words":    float(context_words),
                "llm_latency_ms":   llm_latency_ms,
                "total_latency_ms": total_latency_ms,
            })
            mlflow.log_text(question, "question.txt")
            mlflow.log_text(answer,   "answer.txt")
            mlflow.log_text(prompt,   "prompt.txt")
            if retrieved_chunks:
                import json
                mlflow.log_text(
                    json.dumps(retrieved_chunks, indent=2, ensure_ascii=False),
                    "retrieved_chunks.json",
                )

        logger.debug(
            "mlflow_query_run_logged",
            intent=intent,
            variant=variant,
            run_id=run.info.run_id,
        )
        return run.info.run_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_experiment_id(self, name: str) -> str:
        exp = self._client.get_experiment_by_name(name)
        if exp is None:
            return self._client.create_experiment(name)
        return exp.experiment_id


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

from functools import lru_cache

@lru_cache()
def get_tracker() -> FinanceHQTracker:
    return FinanceHQTracker()
