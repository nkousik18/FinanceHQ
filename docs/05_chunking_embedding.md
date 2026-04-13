# Chunking and Embedding

## Chunking

### Why Chunking Matters

The LLM has a context window limit. You cannot feed the entire 15,000-word loan document as context. Chunking splits the document into segments small enough to fit in context, large enough to contain complete thoughts.

Bad chunks break retrieval:
- Too small → chunks lack context ("4.5%" with no surrounding sentence)
- Too large → too much noise, dilutes the relevant content, costs more tokens

### Strategy: Sliding Window with Sentence Boundary Respect

```
Document:
[sentence 1][sentence 2][sentence 3][sentence 4][sentence 5][sentence 6]

Chunk 1: [s1][s2][s3][s4]               ← chunk_size = 4 sentences
Chunk 2:         [s3][s4][s5][s6]       ← overlap = 2 sentences
Chunk 3:                 [s5][s6][s7][s8]
```

**Sentence tokenization:** Uses `nltk.sent_tokenize`. Never splits mid-sentence.

### Parameters

| Parameter | Default | Tunable via MLflow A/B? |
|---|---|---|
| `chunk_size_tokens` | 400 | Yes |
| `chunk_overlap_tokens` | 80 | Yes |
| `min_chunk_tokens` | 50 | No |

Chunks smaller than `min_chunk_tokens` (fragments, single lines, headers alone) are discarded.

### Token Counting

Use `tiktoken` (OpenAI's tokenizer) as the token counter — it's the industry standard proxy for "how many tokens will this cost."

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))
```

### Chunk Schema

```json
{
  "chunk_id": 12,
  "text": "The borrower shall repay the principal amount of $250,000 with interest at 4.5% per annum over a period of 30 years...",
  "token_count": 94,
  "char_start": 5420,
  "char_end": 5980,
  "page_ref": 3
}
```

`page_ref` is determined by mapping `char_start` back to page boundaries from the raw text stage.

### Why Not Semantic Chunking?

Semantic chunking uses embedding similarity to find natural topic boundaries. It produces better chunks conceptually but:
1. Requires an embedding call per boundary candidate — slow and costly
2. Loan documents are already well-structured (sections, clauses) — sliding window already produces coherent chunks
3. The chunk parameter tuning via A/B testing (see `09_evaluation_abtesting.md`) achieves the same optimization goal empirically

---

## Embedding

### Model: `sentence-transformers/all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Dimensions | 384 |
| Max input tokens | 256 (sentences longer than this are truncated) |
| Inference | CPU, ~5ms per chunk |
| Cost | $0 |
| Quality | Good for semantic similarity on domain text |

### Why Not a Larger Model?

| Model | Dims | Cost | Latency |
|---|---|---|---|
| MiniLM-L6-v2 | 384 | Free | ~5ms |
| `text-embedding-ada-002` | 1536 | $0.0001/1K tokens | API call |
| `text-embedding-3-large` | 3072 | $0.00013/1K tokens | API call |
| `e5-large` | 1024 | Free | ~50ms (larger model) |

For loan documents (structured, domain-specific), MiniLM-L6-v2 performs well. The quality difference over ada-002 is marginal for factual retrieval tasks. The cost difference is everything.

### Embedding Process

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # L2 normalize for cosine similarity via dot product
        show_progress_bar=False,
    )
    return embeddings  # shape: (num_chunks, 384)
```

`normalize_embeddings=True` is critical. FAISS `IndexFlatIP` (inner product) on normalized vectors gives exact cosine similarity.

### Query Embedding

At query time, the same model embeds the question:
```python
def embed_query(question: str) -> np.ndarray:
    vec = model.encode([question], normalize_embeddings=True)[0]
    return vec  # shape: (384,)
```

The model is loaded once at FastAPI startup and kept in memory (singleton). It is ~90MB on disk.

### Storing Embeddings

```python
np.save(local_path, embeddings)   # save as .npy
# then upload to S3
```

Loading:
```python
data = s3_client.download_file(session_id, "chunks", "chunk_embeddings.npy")
embeddings = np.load(io.BytesIO(data))   # shape: (num_chunks, 384)
```

### FAISS Index Construction

```python
import faiss

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def serialize_index(index: faiss.IndexFlatIP) -> bytes:
    buf = faiss.serialize_index(index)
    return buf.tobytes()

def deserialize_index(data: bytes) -> faiss.IndexFlatIP:
    buf = np.frombuffer(data, dtype=np.uint8)
    return faiss.deserialize_index(buf)
```

Index is serialized to bytes → uploaded to `s3://loandoc-bucket/chunks/{session_id}/faiss.index`.

### In-Memory Cache

The FastAPI process caches loaded FAISS indexes:

```python
_index_cache: dict[str, tuple[faiss.Index, list[dict]]] = {}

def get_index(session_id: str) -> tuple[faiss.Index, list[dict]]:
    if session_id not in _index_cache:
        # download from S3 and deserialize
        _index_cache[session_id] = (index, chunks)
    return _index_cache[session_id]
```

Cache is never explicitly invalidated (session indexes don't change once built). Cache is bounded by process memory — for a portfolio project with a few sessions, this is fine.
