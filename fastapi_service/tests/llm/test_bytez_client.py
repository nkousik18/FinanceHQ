"""
Tests for app/llm/bytez_client.py
Run: pytest tests/llm/test_bytez_client.py -v
"""
import pytest
from unittest.mock import MagicMock, patch, call

from app.llm.bytez_client import BytezClient, BytezInferenceError, LLMResponse, RETRY_DELAYS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_ok_result(content: str = "The interest rate is 8.5%.") -> MagicMock:
    result = MagicMock()
    result.error = None
    result.output = {"role": "assistant", "content": content}
    result.usage = None
    return result


def _make_error_result(message: str) -> MagicMock:
    result = MagicMock()
    result.error = message
    result.output = None
    result.usage = None
    return result


def _make_client(model_run_return) -> BytezClient:
    """
    Build a BytezClient with the Bytez SDK fully mocked.
    model_run_return: single value or list of values returned by model.run()
    """
    mock_model = MagicMock()
    if isinstance(model_run_return, list):
        mock_model.run.side_effect = model_run_return
    else:
        mock_model.run.return_value = model_run_return

    mock_sdk = MagicMock()
    mock_sdk.model.return_value = mock_model

    with patch("app.llm.bytez_client.Bytez", return_value=mock_sdk), \
         patch("app.llm.bytez_client.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            bytez_api_key="test-key",
            bytez_model="meta-llama/Llama-3.1-8B-Instruct",
            bytez_max_tokens=512,
            bytez_temperature=0.1,
        )
        client = BytezClient()
        client._model = mock_model   # ensure mock is wired in
    return client


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------

class TestBytezClientComplete:

    def test_returns_llm_response(self):
        client = _make_client(_make_ok_result())
        resp = client.complete("What is the rate?")
        assert isinstance(resp, LLMResponse)

    def test_text_extracted_from_dict_output(self):
        client = _make_client(_make_ok_result("The rate is 8.5%."))
        resp = client.complete("What is the rate?")
        assert resp.text == "The rate is 8.5%."

    def test_text_extracted_from_string_output(self):
        result = _make_ok_result()
        result.output = "plain string answer"
        client = _make_client(result)
        resp = client.complete("What is the rate?")
        assert resp.text == "plain string answer"

    def test_model_name_on_response(self):
        client = _make_client(_make_ok_result())
        resp = client.complete("Question?")
        assert resp.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_latency_is_positive_float(self):
        client = _make_client(_make_ok_result())
        resp = client.complete("Question?")
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0

    def test_tokens_none_when_no_usage(self):
        client = _make_client(_make_ok_result())
        resp = client.complete("Question?")
        assert resp.input_tokens is None
        assert resp.output_tokens is None

    def test_tokens_populated_when_usage_present(self):
        result = _make_ok_result()
        result.usage = {"prompt_tokens": 120, "completion_tokens": 40}
        client = _make_client(result)
        resp = client.complete("Question?")
        assert resp.input_tokens == 120
        assert resp.output_tokens == 40

    def test_params_passed_to_sdk(self):
        ok = _make_ok_result()
        client = _make_client(ok)
        client.complete("Question?")
        client._model.run.assert_called_once()
        _, kwargs = client._model.run.call_args
        assert "params" in kwargs
        assert kwargs["params"]["max_new_tokens"] == 512
        assert kwargs["params"]["temperature"] == 0.1


# ------------------------------------------------------------------
# Retry behaviour
# ------------------------------------------------------------------

class TestBytezClientRetry:

    def test_retries_on_error_then_succeeds(self):
        """First call errors, second succeeds — should return the answer."""
        results = [
            _make_error_result("rate limited"),
            _make_ok_result("Retry worked."),
        ]
        with patch("app.llm.bytez_client.time.sleep"):  # don't actually wait
            client = _make_client(results)
            resp = client.complete("Question?")
        assert resp.text == "Retry worked."

    def test_sleeps_between_retries(self):
        results = [
            _make_error_result("rate limited"),
            _make_ok_result("ok"),
        ]
        with patch("app.llm.bytez_client.time.sleep") as mock_sleep:
            client = _make_client(results)
            client.complete("Question?")
        mock_sleep.assert_called_once_with(RETRY_DELAYS[0])

    def test_raises_after_all_retries_exhausted(self):
        results = [_make_error_result("capacity error")] * 3
        with patch("app.llm.bytez_client.time.sleep"):
            client = _make_client(results)
            with pytest.raises(BytezInferenceError):
                client.complete("Question?")

    def test_error_message_contains_last_error(self):
        results = [_make_error_result("out of capacity")] * 3
        with patch("app.llm.bytez_client.time.sleep"):
            client = _make_client(results)
            with pytest.raises(BytezInferenceError, match="out of capacity"):
                client.complete("Question?")

    def test_no_retry_on_success(self):
        client = _make_client(_make_ok_result())
        client.complete("Question?")
        assert client._model.run.call_count == 1


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

class TestGetLlmClient:

    def test_singleton_returns_same_instance(self):
        from app.llm.bytez_client import get_llm_client

        mock_model = MagicMock()
        mock_sdk = MagicMock()
        mock_sdk.model.return_value = mock_model

        with patch("app.llm.bytez_client.Bytez", return_value=mock_sdk), \
             patch("app.llm.bytez_client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                bytez_api_key="test-key",
                bytez_model="meta-llama/Llama-3.1-8B-Instruct",
                bytez_max_tokens=512,
                bytez_temperature=0.1,
            )
            get_llm_client.cache_clear()
            a = get_llm_client()
            b = get_llm_client()

        assert a is b
