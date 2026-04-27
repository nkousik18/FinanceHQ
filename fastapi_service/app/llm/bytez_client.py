"""
Groq LLM client — wraps the Groq SDK for inference and streaming.

Usage:
    from app.llm.bytez_client import get_llm_client, stream_tokens, LLMInferenceError

    # Sync (POST /query)
    client = get_llm_client()
    response = client.complete(prompt)   # LLMResponse

    # Async streaming (POST /query/stream)
    async for token in stream_tokens(prompt):
        ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from groq import Groq, AsyncGroq

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMInferenceError(Exception):
    pass

# Backward-compat alias so existing tests don't break
BytezInferenceError = LLMInferenceError


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class GroqClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")

        self._client   = Groq(api_key=settings.groq_api_key)
        self._model_id = settings.groq_model
        self._max_tokens   = settings.groq_max_tokens
        self._temperature  = settings.groq_temperature

        logger.info(
            "groq_client_ready",
            model=self._model_id,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def complete(self, prompt: str) -> LLMResponse:
        t0 = time.monotonic()
        try:
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_completion_tokens=self._max_tokens,
                stream=False,
            )
            text    = completion.choices[0].message.content or ""
            usage   = completion.usage
            latency = round((time.monotonic() - t0) * 1000, 1)

            logger.info(
                "groq_inference_ok",
                model=self._model_id,
                latency_ms=round(latency),
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
            )

            return LLMResponse(
                text=text,
                model=self._model_id,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                latency_ms=latency,
            )
        except Exception as exc:
            logger.error("groq_inference_failed", error=str(exc))
            raise LLMInferenceError(f"Groq inference failed: {exc}") from exc


async def stream_tokens(prompt: str):
    """
    Async generator — yields raw token strings from Groq's streaming API.
    Used by POST /query/stream.
    Raises LLMInferenceError on failure.
    """
    settings = get_settings()
    client   = AsyncGroq(api_key=settings.groq_api_key)
    try:
        stream = await client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.groq_temperature,
            max_completion_tokens=settings.groq_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as exc:
        logger.error("groq_stream_failed", error=str(exc))
        raise LLMInferenceError(f"Groq stream failed: {exc}") from exc


@lru_cache()
def get_llm_client() -> GroqClient:
    return GroqClient()
