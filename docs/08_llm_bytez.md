# LLM Inference — Bytez API

## What Bytez Is

Bytez is an API platform that provides access to open-source LLMs (Mistral, Llama, Falcon, etc.) via a REST API. You pay per token, with no idle GPU cost. The same API is used for both inference (generating answers) and evaluation (scoring answers).

**Key advantage over self-hosted vLLM:** Zero infrastructure. No GPU VM to manage, no idle billing. Pay only when a request is made.

## Inference Integration

### API Contract

Bytez uses an OpenAI-compatible `/v1/chat/completions` endpoint format. This means the integration looks identical to calling OpenAI — swapping inference providers in the future requires only a URL/key change.

```python
import httpx

BYTEZ_API_URL = "https://api.bytez.com/models/v1"   # verify current endpoint
BYTEZ_API_KEY = os.environ["BYTEZ_API_KEY"]
BYTEZ_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # or llama3, configurable

async def llm_complete(prompt: str) -> str:
    payload = {
        "model": BYTEZ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,   # low temperature = more deterministic for factual QA
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{BYTEZ_API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {BYTEZ_API_KEY}"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def llm_stream(prompt: str):
    payload = {
        "model": BYTEZ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{BYTEZ_API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {BYTEZ_API_KEY}"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
```

### Streaming to Django

FastAPI streams to Django via Server-Sent Events (SSE):

```python
# FastAPI endpoint
@app.post("/query/stream")
async def query_stream(request: Request):
    ...
    async def event_generator():
        async for token in llm_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Django receives SSE and renders tokens progressively using JavaScript `EventSource`.

**Note on Django streaming:** Django's standard `HttpResponse` buffers. Use `StreamingHttpResponse` to pass the SSE stream through. For WebSocket-based streaming, Django Channels is required — but SSE is simpler and sufficient for this use case.

## Model Selection

Default model: **Mistral-7B-Instruct-v0.3**

Why Mistral 7B:
- Strong instruction following
- Good at factual extraction from context
- Small enough for fast inference
- Familiar from the original project

Alternative models available via Bytez (configurable without code changes via env var):
- `meta-llama/Llama-3.1-8B-Instruct` — slightly better reasoning
- `google/gemma-2-9b-it` — strong at structured output
- `mistralai/Mixtral-8x7B-Instruct-v0.1` — higher quality, higher cost

Model is set via `BYTEZ_MODEL` env var, making A/B testing between models possible at the infrastructure level.

## Cost Optimization

### Token Counting Before Sending

Every prompt is token-counted before the API call. The count is logged to MLflow. If prompt + max_response tokens exceed the budget threshold, context is trimmed.

```python
estimated_input_tokens = count_tokens(prompt)
estimated_cost = (estimated_input_tokens / 1000) * BYTEZ_INPUT_COST_PER_1K
# Log to MLflow: estimated_cost, actual_input_tokens, actual_output_tokens
```

### Response Token Cap

`max_tokens=512` per query. For loan document Q&A, answers are typically 50–200 tokens. Setting 512 prevents runaway generations.

### Caching Identical Queries

Simple in-process cache for identical (question, session_id) pairs:

```python
import hashlib

_query_cache: dict[str, str] = {}

def cache_key(question: str, session_id: str) -> str:
    return hashlib.md5(f"{session_id}:{question.strip().lower()}".encode()).hexdigest()
```

TTL: session lifetime (no expiry needed — document content doesn't change).

This means if a user asks the same question twice, the second call costs $0 and returns instantly. Also avoids redundant MLflow runs for identical queries.

### Cost Estimate

| Volume | Input tokens | Output tokens | Estimated cost |
|---|---|---|---|
| 100 queries/month | ~300K | ~50K | ~$0.10–$0.50 |
| 500 queries/month | ~1.5M | ~250K | ~$0.50–$2.50 |

At portfolio/demo scale: **< $1/month**.

## Error Handling

```python
class BytezInferenceError(Exception):
    pass

# Retry logic: 2 retries with exponential backoff
# If all retries fail: return a graceful error message to the user
# Log the failure to MLflow as a failed run

RETRIES = 2
RETRY_DELAY = [1.0, 3.0]  # seconds

for attempt in range(RETRIES + 1):
    try:
        result = await llm_complete(prompt)
        break
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:  # rate limit
            await asyncio.sleep(RETRY_DELAY[attempt])
        else:
            raise BytezInferenceError(f"Bytez API error: {e}")
```

## Environment Variables

```bash
BYTEZ_API_KEY=...
BYTEZ_MODEL=mistralai/Mistral-7B-Instruct-v0.3
BYTEZ_MAX_TOKENS=512
BYTEZ_TEMPERATURE=0.1
```

All model parameters are env-var-driven. No magic numbers in code.
