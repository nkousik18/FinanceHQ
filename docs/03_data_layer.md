# Data Layer — AWS S3

## S3 as the Single Source of Truth

Every artifact produced by the pipeline lives in S3. No local disk is used for persistence. This means:
- The FastAPI container can be restarted or redeployed without losing data
- Multiple container instances could serve the same data (future horizontal scaling)
- MLflow artifacts are also on S3 — the tracking server is stateless except for its DB

## Bucket Structure

```
s3://loandoc-bucket/
│
├── uploads/
│   └── {user_id}/
│       └── {session_id}/
│           └── original.pdf
│
├── extracted/
│   └── {session_id}/
│       ├── raw_text.txt          # direct PyMuPDF output
│       ├── cleaned_text.txt      # after cleaning pipeline
│       └── validation_report.json
│
├── chunks/
│   └── {session_id}/
│       ├── chunks.json           # chunk metadata + text
│       ├── chunk_embeddings.npy  # raw numpy array (num_chunks, 384)
│       └── faiss.index           # serialized FAISS index
│
├── sessions/
│   └── {session_id}/
│       └── metadata.json         # session state, timestamps, status
│
└── mlflow/
    └── artifacts/
        └── {experiment_id}/
            └── {run_id}/
                └── ...           # MLflow artifact store
```

## Session ID Format

```
{user_id}_{YYYYMMDD}_{HHMMSS}_{random_4_chars}
```

Example: `user54_20250115_142300_a3f2`

- Globally unique
- Human-readable (useful for debugging)
- Encodes the user and time (no need for a separate lookup for basic attribution)

## Session Metadata Schema

```json
{
  "session_id": "user54_20250115_142300_a3f2",
  "user_id": "user54",
  "original_filename": "home_loan_agreement_2024.pdf",
  "status": "READY",
  "created_at": "2025-01-15T14:23:00Z",
  "pipeline_completed_at": "2025-01-15T14:23:45Z",
  "s3_keys": {
    "pdf": "uploads/user54/user54_20250115.../original.pdf",
    "raw_text": "extracted/user54_.../raw_text.txt",
    "cleaned_text": "extracted/user54_.../cleaned_text.txt",
    "chunks": "chunks/user54_.../chunks.json",
    "embeddings": "chunks/user54_.../chunk_embeddings.npy",
    "faiss_index": "chunks/user54_.../faiss.index"
  },
  "stats": {
    "page_count": 8,
    "char_count": 14200,
    "chunk_count": 47,
    "embedding_dim": 384
  }
}
```

## Status Values

| Status | Meaning |
|---|---|
| `UPLOADING` | PDF upload in progress |
| `PROCESSING` | Pipeline running (extraction → chunking → embedding) |
| `READY` | Pipeline complete, ready for queries |
| `FAILED` | Pipeline failed at some stage — check `validation_report.json` |

## S3 Access Pattern

All S3 access goes through a single `S3Client` wrapper in FastAPI:

```python
class S3Client:
    def upload_file(session_id, stage, filename, data) -> str
    def download_file(session_id, stage, filename) -> bytes
    def exists(session_id, stage, filename) -> bool
    def get_session_metadata(session_id) -> dict
    def update_session_status(session_id, status)
```

This wrapper is the only place `boto3` is imported. No direct S3 calls outside this class.

## IAM Permissions Required

Minimum S3 policy for the FastAPI service IAM role:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:PutObject",
    "s3:GetObject",
    "s3:DeleteObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::loandoc-bucket",
    "arn:aws:s3:::loandoc-bucket/*"
  ]
}
```

MLflow also needs `s3:PutObject` and `s3:GetObject` on the `mlflow/artifacts/` prefix.

## Cost Estimates (S3)

For a portfolio project with light usage:

| Operation | Estimated volume | Cost |
|---|---|---|
| PUT (upload) | ~50/month | ~$0.00 |
| GET (retrieval, per query) | ~500/month | ~$0.01 |
| Storage | <1 GB | ~$0.02/month |

Total S3 cost: **< $0.05/month**
