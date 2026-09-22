Here's the same treatment for FinanceHQ.

---

Main page

FinanceHQ
Python · FastAPI · AWS Textract · FAISS · MiniLM · Groq · MLflow · React · AWS EC2 · GitHub Actions

🔗 Live demo · GitHub

The problem

Loan documents are long, dense, and full of numbers that actually matter, rates, penalty clauses, prepayment terms, buried in pages of legal language. Reading one to answer a single question is slow, and a general-purpose chatbot asked the same question will confidently guess from training data instead of the document actually in front of it. In a financial context, a wrong number is worse than no answer.

What it does

FinanceHQ turns a loan PDF, or several, into a session you can ask real questions against. Upload triggers a six-stage background pipeline, Textract extraction, cleaning, validation, chunking, embedding, indexing, and every query after that retrieves only from what's actually in the document, classifies the question's intent, routes it to an A/B-selected prompt template, and streams a grounded answer back token by token. Every retrieval, prompt variant, and latency number is logged to MLflow as it happens. Multiple documents in one session share one merged index, so a question can span both ("compare the EMI on loan A and loan B").

Architecture

PDF Upload
   → S3 (uploads/{session}/{doc}/original.pdf)
   → AWS Textract (AnalyzeDocument, TABLES + FORMS; sync ≤1 page, async job + poll ≥2 pages)
   → Clean + validate (confidence thresholds, hard-block only on ERROR severity)
   → Chunk (400-token sliding window, 60-token overlap) → embed (MiniLM-L6-v2, 384-dim)
   → merge into session FAISS index (IndexFlatIP) → S3 (chunks.json, embeddings.npy, faiss.index)

Question
   → retrieve top-k (FAISS, in-memory cache per session)
   → classify intent (zero-shot cosine similarity, same MiniLM instance, 5 categories)
   → route to prompt template (ε-greedy A/B variant selector)
   → Groq (llama-3.3-70b-versatile) → SSE stream to browser
   → async → MLflow (params, metrics, artifacts logged per query, never blocks the response)

Deployed across two AWS EC2 t2.micro instances, React UI on one, FastAPI + MiniLM + MLflow on the other, split specifically to fit the free tier's 1GB RAM ceiling. GitHub Actions SSHes into both on every push to main.

Tech stack

┌────────────────────────┬──────────────────────────────────────────────────────────────────────┐
│         Layer          │                              Technology                              │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ API framework          │ FastAPI (async)                                                      │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ LLM inference           │ Groq (llama-3.3-70b-versatile, production); Bytez (offline model eval)│
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ PDF extraction          │ AWS Textract (AnalyzeDocument, TABLES + FORMS, sync + async polling) │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Embeddings              │ sentence-transformers/all-MiniLM-L6-v2, local, 384-dim               │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Vector index            │ FAISS IndexFlatIP, in-process, serialized to S3, session-merged      │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Experiment tracking     │ MLflow, self-hosted, 3 experiments                                   │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Storage                 │ AWS S3, single source of truth for PDFs, indexes, session status     │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Frontend                │ React (Vite, TypeScript, Tailwind, Framer Motion)                    │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Web / infra             │ nginx + gunicorn (UI), nginx + uvicorn (API), 2× AWS EC2 t2.micro    │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ CI/CD                   │ GitHub Actions, parallel SSH deploy to both instances on push        │
├────────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Testing                 │ pytest + unittest.mock, ~122 tests, all external deps mocked         │
└────────────────────────┴──────────────────────────────────────────────────────────────────────┘

Key features

- Six-stage ingestion pipeline (S3 upload → Textract → clean → validate → chunk → embed + index), each stage transition persisted to a per-session status.json in S3, so the API stays fully stateless and any replica can answer a status check.
- Multi-document sessions: the FAISS index is merged, not replaced, across every PDF uploaded to a session, so one query searches across all of them at once.
- Zero-shot intent classification across 5 categories (lookup, calculate, compare, explain, summarise) via MiniLM cosine similarity, no extra LLM call, no training data, reusing the embedding model that's already loaded in memory.
- 5 prompt templates × 2 variants each, routed by an ε-greedy (ε=0.1) A/B selector that reads live latency and quality from MLflow and shifts traffic toward whichever variant is actually winning.
- Real SSE token streaming from FastAPI straight to the browser, served through an explicitly un-buffered nginx config so tokens arrive as they're generated instead of batching at the proxy.
- Two separate evaluation surfaces: a 3-model offline comparison script (7 hand-authored questions, 6 automated metrics, zero human labelling, zero LLM-as-judge) used during model selection, and a live production A/B test across 3 Groq models that the ε-greedy selector reads continuously.

Challenges & how I solved them

- A single free-tier EC2 instance couldn't hold the whole stack. React/Django UI + FastAPI + MiniLM + MLflow together sat close to a t2.micro's 1GB RAM ceiling, causing OOM crashes I only found by watching a real deploy, not from any local signal. Fixed by splitting into two instances by weight, UI on one (~150MB), FastAPI + MiniLM + MLflow on the other (~700MB), with the browser calling the backend instance directly rather than proxying through the UI instance.
- Installing PyTorch the normal way exhausted the deploy box's disk before the app ever ran once. `pip install -r requirements.fastapi.txt` pulled the default CUDA build (~423MB) onto an 8GB disk with no GPU to use it. Fixed by installing the CPU-only wheel first, in the setup script itself, so the CUDA build is never fetched.
- Token streaming worked locally and silently stalled in production. `/query/stream` streamed cleanly over `uvicorn --reload` but arrived in one late burst through nginx, because nginx buffers proxied responses by default. Fixed with an explicit SSE-safe block, `proxy_buffering off`, chunked transfer encoding on, proxy cache disabled, same code, a config line away from looking broken.

Results

- Textract with TABLES + FORMS took extraction quality from roughly 5.5/10 to 8.5/10 on real loan-form PDFs (handwriting, multi-column layouts, table cells) versus a plain text-layer extractor.
- ~122 automated tests passing, every external dependency (S3, Textract, Groq/Bytez, MLflow) mocked; the 2 pre-existing failures are bugs in test expectations, not production code, and only the 2 files that need live AWS credentials are skipped in CI.
- Live production A/B, 21 MLflow-tracked runs, ε-greedy selector: Llama 3.3 70B is the current winner at 87% groundedness and 2.02s latency, now receiving 100% of live traffic over Llama 3.1 8B and Llama 3.1 70B.
- A separate offline 3-model evaluation caught a real hallucination: asked to compute total loan repayment, Llama-3.1-8B ignored the EMI value actually present in context and applied a memorized interest-rate formula instead, landing on ₹5,37,84,800 against a correct ₹14,90,520, caught by an automated number-groundedness check, not a human reviewer. All 3 models correctly refused the one question with no answer in the document (a credit score never mentioned), 100% not-found compliance.

What's next

- Fix the retriever's in-memory cache eviction gap: adding a second PDF to a session doesn't currently invalidate the cached FAISS index, so queries can search a stale index until the process restarts. The fix (`evict_session()` at the end of the pipeline) is identified, not yet shipped.
- Add a per-session lock around the FAISS merge step; two simultaneous uploads to the same session currently race on the same S3 read-modify-write, and one upload's chunks can be silently dropped from the merged index.
- Persist the Textract async job ID to session status so a mid-poll process restart resumes the job instead of leaving a document stuck in EXTRACTING forever.
- Assign Elastic IPs to both EC2 instances instead of hand-updating `FASTAPI_URL` and two GitHub secrets every time a stop/start changes a public IP.
- Wire up the deprioritized LLM-as-judge evaluator for post-query faithfulness/relevance scoring; the offline metric-based eval currently covers this instead.

[Read the full engineering write-up →]

---

Deep-dive (linked from the button above)

The one hard problem

The plan was to run the whole stack, React/Django UI, FastAPI RAG backend, MiniLM, MLflow, the same way it ran locally in Docker Compose: one process group, one box. I assumed a free-tier EC2 instance would hold that the same way my laptop did.

It didn't. A t2.micro caps at 1GB of RAM, and FastAPI plus MiniLM plus MLflow together sat close to 700MB before the UI process or the OS took anything, no headroom left. I only found this by actually deploying and watching the instance die, `systemctl status` reporting a crashed service with nothing informative in the applica­tion logs, because the kernel had OOM-killed the process, not the app. Nothing about it showed up locally, my dev machine has more RAM than the entire free-tier budget.

The fix was to stop treating "the stack" as one deployable unit and split it by actual memory weight: Instance 1 runs only the UI (~150MB) behind nginx and gunicorn. Instance 2 runs FastAPI, MiniLM, and MLflow together (~700MB) behind its own nginx. The browser talks to Instance 2 directly for `/query` and `/query/stream`, FASTAPI_URL is injected into the UI at render time, rather than routing every API call back through Instance 1 first. Two boxes, still free tier, still under $1/month, just split along the line that actually mattered.

How I solved it

Getting there took a second pass at the deploy script, too. `pip install -r requirements.fastapi.txt` on a clean EC2 box pulls PyTorch's default build, which includes the ~423MB CUDA wheel, on an instance with no GPU and an 8GB disk. On the first real deploy attempt, `pip install` itself ran out of disk before the app ever started once. The fix was ordering: install the CPU-only PyTorch wheel explicitly first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then the rest of `requirements.fastapi.txt`, so the CUDA build is never fetched at all. A one-line reorder in the setup script, invisible until you've actually watched a fresh instance run out of disk mid-install.

Streaming needed its own fix once both instances were up. `/query/stream` worked perfectly against `uvicorn --reload` in local dev, tokens landing in the browser as fast as Groq produced them. In production, behind nginx, the same endpoint would sit silent for the full response time and then dump everything at once, because nginx buffers proxied responses by default, and a buffered SSE stream isn't streaming, it's just a slow regular response with extra ceremony. The fix was an explicit SSE-safe nginx block on Instance 2: `proxy_buffering off`, `chunked_transfer_encoding on`, `proxy_cache off`, `proxy_read_timeout 120s`. Same FastAPI code, same Groq call, the only difference between "streaming" and "silently not streaming" was four lines of proxy config.

One bug that taught me something

EC2 public IPs aren't stable across a stop/start, and I learned this the operationally expensive way: I stopped both instances to avoid idle billing, started them again later, and FinanceHQ was down with no error in either app's logs, both services reported healthy. The actual break was one layer up: Instance 2's public IP had changed, so Instance 1's `FASTAPI_URL` (baked into its `.env`) and the two GitHub Actions SSH secrets were all silently pointing at an address that no longer existed. Nothing in the application had a bug; the infrastructure underneath it had just moved.

The immediate fix is a runbook, now written down instead of relearned each time: pull the new public IP from the console, update `FASTAPI_URL` in Instance 1's `.env`, update `INSTANCE1_IP`/`INSTANCE2_IP` in GitHub secrets if either changed, restart the UI service. The real fix, reserving an Elastic IP per instance so this class of break can't happen again, is scoped and sitting in What's Next; I haven't done it yet because the free-tier project hasn't been stopped and restarted since I wrote the runbook, and I'd rather ship the retriever cache fix first.

What I'd do differently

I'd reserve Elastic IPs for both instances from the first deploy instead of discovering the churn the hard way after a stop/start. I'd build the per-session lock into the FAISS merge path from day one instead of accepting "sequential uploads only" as a v1 shortcut that now needs a deliberate follow-up. And I'd call `evict_session()` as part of the original multi-doc pipeline design instead of retrofitting cache invalidation after multi-doc sessions were already shipped and the gap had already been found by reading the code, not by a failing test.
