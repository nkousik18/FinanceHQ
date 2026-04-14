"""
Tests for app/api/query.py  (POST /query)
Run: pytest tests/api/test_query.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.retrieval.retriever import RetrievedChunk, RetrieverError
from app.retrieval.intent_classifier import Intent, ClassifiedIntent
from app.prompts.router import RoutedPrompt
from app.llm.bytez_client import LLMResponse, BytezInferenceError
from app.pipeline.chunker import Chunk


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def _make_chunk(text: str = "The interest rate is 8.5% per annum.", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id="sess-001-0",
        session_id="sess-001",
        index=0,
        text=text,
        page=page,
        section="Loan Details",
        word_count=len(text.split()),
        token_estimate=len(text.split()),
    )


def _make_retrieved(text: str = "The interest rate is 8.5% per annum.") -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=_make_chunk(text), score=0.91, rank=1)]


def _make_intents(intent: Intent = Intent.LOOKUP) -> list[ClassifiedIntent]:
    return [ClassifiedIntent(
        intent=intent,
        score=0.85,
        priority=1,
        is_primary=True,
    )]


def _make_routed(intent: Intent = Intent.LOOKUP) -> RoutedPrompt:
    return RoutedPrompt(
        prompt="assembled prompt text",
        intent=intent,
        variant=f"{intent.value}_v1",
        chunks_used=1,
        context_words=8,
    )


def _make_llm_response(text: str = "The interest rate is 8.5% per annum.") -> LLMResponse:
    return LLMResponse(
        text=text,
        model="meta-llama/Llama-3.1-8B-Instruct",
        input_tokens=None,
        output_tokens=None,
        latency_ms=1200.0,
    )


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------

class TestQueryEndpointHappyPath:

    def test_returns_200(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.status_code == 200

    def test_answer_in_response(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response("8.5% per annum.")

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.json()["answer"] == "8.5% per annum."

    def test_intent_field_is_string(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed(Intent.LOOKUP)), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents(Intent.LOOKUP)
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.json()["intent"] == "lookup"

    def test_variant_field_populated(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.json()["variant"] == "lookup_v1"

    def test_chunks_used_matches_retrieval(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.json()["chunks_used"] == 1

    def test_latency_ms_is_positive(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        assert resp.json()["latency_ms"] > 0

    def test_all_response_fields_present(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the interest rate?",
            })
        data = resp.json()
        for field in ("answer", "intent", "variant", "chunks_used", "context_words", "latency_ms"):
            assert field in data


# ------------------------------------------------------------------
# Intent routing — correct template selected per intent
# ------------------------------------------------------------------

class TestQueryIntentRouting:

    @pytest.mark.parametrize("intent,expected_variant", [
        (Intent.LOOKUP,    "lookup_v1"),
        (Intent.CALCULATE, "calculate_v1"),
        (Intent.COMPARE,   "compare_v1"),
        (Intent.EXPLAIN,   "explain_v1"),
        (Intent.SUMMARISE, "summarise_v1"),
    ])
    def test_variant_matches_intent(self, client, intent, expected_variant):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed(intent)), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents(intent)
            mock_llm.return_value.complete.return_value = _make_llm_response()

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "Some question",
            })
        assert resp.json()["variant"] == expected_variant
        assert resp.json()["intent"] == intent.value


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------

class TestQueryEndpointErrors:

    def test_404_when_session_not_found(self, client):
        with patch("app.api.query.retrieve",
                   side_effect=RetrieverError("No index for session")):
            resp = client.post("/query", json={
                "session_id": "missing-session",
                "question": "What is the rate?",
            })
        assert resp.status_code == 404
        assert "No index for session" in resp.json()["detail"]

    def test_502_when_llm_fails(self, client):
        with patch("app.api.query.retrieve", return_value=_make_retrieved()), \
             patch("app.api.query.get_intent_classifier") as mock_clf, \
             patch("app.api.query.route", return_value=_make_routed()), \
             patch("app.api.query.get_llm_client") as mock_llm:
            mock_clf.return_value.classify.return_value = _make_intents()
            mock_llm.return_value.complete.side_effect = BytezInferenceError("out of capacity")

            resp = client.post("/query", json={
                "session_id": "sess-001",
                "question": "What is the rate?",
            })
        assert resp.status_code == 502

    def test_422_on_empty_question(self, client):
        resp = client.post("/query", json={
            "session_id": "sess-001",
            "question": "",
        })
        assert resp.status_code == 422

    def test_422_on_missing_session_id(self, client):
        resp = client.post("/query", json={"question": "What is the rate?"})
        assert resp.status_code == 422

    def test_422_on_missing_question(self, client):
        resp = client.post("/query", json={"session_id": "sess-001"})
        assert resp.status_code == 422

    def test_422_on_top_k_too_high(self, client):
        resp = client.post("/query", json={
            "session_id": "sess-001",
            "question": "What is the rate?",
            "top_k": 99,
        })
        assert resp.status_code == 422

    def test_422_on_top_k_zero(self, client):
        resp = client.post("/query", json={
            "session_id": "sess-001",
            "question": "What is the rate?",
            "top_k": 0,
        })
        assert resp.status_code == 422


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}
