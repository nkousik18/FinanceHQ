# LoanDoc AI — Project Overview

## What This Is

An end-to-end Retrieval-Augmented Generation (RAG) system for loan documents. Users upload a PDF loan document, the system extracts and indexes its content, and users can ask natural-language questions about it. The system answers using only the content of the uploaded document — no hallucination from general training data.

## Why It Exists (Portfolio Goal)

This project demonstrates the full MLOps lifecycle of a production RAG system:

- Document ingestion and processing pipeline
- Embedding and vector-based retrieval
- LLM inference with prompt engineering
- Prompt A/B testing with quantitative evaluation
- Experiment tracking via MLflow
- Containerization and cloud deployment

The target audience for the demo: hiring managers at ML/MLOps/AI engineering roles.

## Use Case

**Input:** A loan agreement, mortgage document, or financial contract (PDF)  
**Output:** Accurate, document-grounded answers to natural-language questions

Example queries:
- "What is the interest rate on this loan?"
- "What are the prepayment penalty conditions?"
- "Summarize the repayment schedule."
- "What happens if I miss a payment?"

## Core Constraints

| Constraint | Decision |
|---|---|
| No paid GPU | Use Bytez API for LLM inference (pay-per-token, no idle cost) |
| Single storage layer | AWS S3 — PDFs, extracted text, FAISS index, MLflow artifacts |
| Deployable for free | Render (backend + frontend) |
| Demonstrable MLOps | MLflow tracking + Bytez evaluator + A/B prompt testing |

## What This Is NOT

- Not a production lending system
- Not multi-tenant (single user per session scope)
- Not fine-tuned on loan data — uses RAG, not training
