"""
Tests for app/retrieval/intent_classifier.py
Run: pytest fastapi_service/tests/retrieval/test_intent_classifier.py -v
"""
import pytest
from app.retrieval.intent_classifier import (
    IntentClassifier, Intent, ClassifiedIntent,
    PRIMARY_THRESHOLD, MULTI_INTENT_THRESHOLD,
)


@pytest.fixture(scope="module")
def classifier():
    return IntentClassifier()


class TestIntentClassifier:

    def test_lookup_intent(self, classifier):
        result = classifier.classify("What is the interest rate on this loan?")
        assert result[0].intent == Intent.LOOKUP

    def test_calculate_intent(self, classifier):
        result = classifier.classify("Calculate the total amount I will pay over 36 months")
        assert result[0].intent == Intent.CALCULATE

    def test_summarise_intent(self, classifier):
        result = classifier.classify("Give me a summary of this document")
        assert result[0].intent == Intent.SUMMARISE

    def test_explain_intent(self, classifier):
        result = classifier.classify("Explain what happens if I miss a payment")
        assert result[0].intent == Intent.EXPLAIN

    def test_compare_intent(self, classifier):
        result = classifier.classify("Compare the applicant and co-applicant income")
        assert result[0].intent == Intent.COMPARE

    def test_returns_list(self, classifier):
        result = classifier.classify("What is the loan amount?")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_each_result_is_classified_intent(self, classifier):
        result = classifier.classify("What is the loan amount?")
        for r in result:
            assert isinstance(r, ClassifiedIntent)

    def test_primary_intent_is_flagged(self, classifier):
        result = classifier.classify("What is the branch name?")
        assert result[0].is_primary is True

    def test_score_is_float(self, classifier):
        result = classifier.classify("What is the EMI?")
        assert isinstance(result[0].score, float)

    def test_fallback_on_nonsense_query(self, classifier):
        result = classifier.classify("xyzzy blorp wibble")
        # Falls back to SUMMARISE
        assert result[0].intent == Intent.SUMMARISE

    def test_priority_order_respected_for_multi_intent(self, classifier):
        # Calculate + lookup style question — calculate should win priority
        result = classifier.classify("Calculate the total interest and tell me the rate")
        priorities = [r.priority for r in result]
        assert priorities == sorted(priorities)

    def test_empty_query_returns_fallback(self, classifier):
        result = classifier.classify("   ")
        assert len(result) >= 1
        assert result[0].intent == Intent.SUMMARISE
