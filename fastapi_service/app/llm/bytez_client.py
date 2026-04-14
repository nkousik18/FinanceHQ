"""
Bytez LLM client — wraps the Bytez SDK for inference.

Usage:
    from app.llm.bytez_client import get_llm_client

    client = get_llm_client()
    response = client.complete(prompt)   # LLMResponse
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from bytez import Bytez

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RETRIES = 2
RETRY_DELAYS = [1.0, 3.0]   # seconds between attempts


class BytezInferenceError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None    # None when Bytez doesn't return usage
    output_tokens: int | None
    latency_ms: float


class BytezClient:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.bytez_api_key:
            raise ValueError("BYTEZ_API_KEY is not set")

        self._sdk = Bytez(settings.bytez_api_key)
        self._model_id = settings.bytez_model
        self._max_tokens = settings.bytez_max_tokens
        self._temperature = settings.bytez_temperature
        self._model = self._sdk.model(self._model_id)

        logger.info(
            "bytez_client_ready",
            model=self._model_id,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def complete(self, prompt: str) -> LLMResponse:
        """
        Send a prompt to the LLM and return the response.
        Retries up to RETRIES times on rate-limit or capacity errors.
        Raises BytezInferenceError if all attempts fail.
        """
        messages = [{"role": "user", "content": prompt}]
        last_error: str = ""

        for attempt in range(RETRIES + 1):
            t0 = time.monotonic()
            result = self._model.run(
                messages,
                params={
                    "max_new_tokens": self._max_tokens,
                    "temperature": self._temperature,
                },
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if result.error:
                last_error = result.error
                logger.warning(
                    "bytez_inference_error",
                    attempt=attempt + 1,
                    error=result.error,
                    model=self._model_id,
                )
                if attempt < RETRIES:
                    time.sleep(RETRY_DELAYS[attempt])
                continue

            text = ""
            if isinstance(result.output, dict):
                text = result.output.get("content", "")
            elif isinstance(result.output, str):
                text = result.output

            # Usage tokens — Bytez may or may not return these
            usage = getattr(result, "usage", None) or {}
            input_tokens = usage.get("prompt_tokens") if usage else None
            output_tokens = usage.get("completion_tokens") if usage else None

            logger.info(
                "bytez_inference_ok",
                model=self._model_id,
                latency_ms=round(latency_ms),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return LLMResponse(
                text=text,
                model=self._model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 1),
            )

        raise BytezInferenceError(
            f"Bytez inference failed after {RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        )


@lru_cache()
def get_llm_client() -> BytezClient:
    return BytezClient()
