"""
Test retrieval + intent classification + prompt routing end to end.
Requires the FAISS index to already be in S3 (run test_chunk_index.py first).

Usage:
    python scripts/test_retrieval.py loan1 "What is the loan amount?"
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fastapi_service"))

from dotenv import load_dotenv
from app.core.logging import setup_logging, get_logger
from app.retrieval.retriever import retrieve
from app.retrieval.intent_classifier import get_intent_classifier
from app.prompts.router import route

load_dotenv()
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"), environment="development")
logger = get_logger("test_retrieval")

TEST_QUESTIONS = [
    "What is the loan amount requested?",
    "What is the applicant's gross monthly income?",
    "Calculate the total repayment over 36 months at the given EMI",
    "Compare the applicant and co-applicant details",
    "Explain the repayment mode selected",
    "Summarise the key loan details",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_retrieval.py <session_id> [question]")
        sys.exit(1)

    session_id = sys.argv[1]
    questions = [sys.argv[2]] if len(sys.argv) > 2 else TEST_QUESTIONS

    classifier = get_intent_classifier()

    for question in questions:
        print(f"\n{'='*65}")
        print(f"Q: {question}")
        print(f"{'='*65}")

        # Step 1: Classify intent
        intents = classifier.classify(question)
        print(f"Intent(s): {[f'{i.intent.value}({i.score:.2f})' for i in intents]}")

        # Step 2: Retrieve chunks
        chunks = retrieve(session_id, question, top_k=4)
        print(f"Top chunks retrieved: {len(chunks)}")
        for rc in chunks[:2]:
            preview = rc.chunk.text[:100].replace("\n", " ")
            print(f"  [{rc.rank}] score={rc.score:.3f} p{rc.chunk.page} | {preview}...")

        # Step 3: Route to prompt
        routed = route(question, intents, chunks)
        print(f"Template: {routed.variant}")
        print(f"Context words: {routed.context_words}")
        print(f"\nAssembled prompt (first 400 chars):")
        print(routed.prompt[:400])
        print("...")


if __name__ == "__main__":
    main()
