export interface Step {
  icon: string;
  color: string;
  bg: string;
  border: string;
  label: string;
  sub: string;
  title: string;
  files: string[];
  tech: string[];
  points: string[];
}

export const STEPS: Step[] = [
  {
    icon: "📄", color: "#60A5FA", bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.25)",
    label: "PDF Upload", sub: "→ S3",
    title: "PDF Upload → S3 Storage",
    files: ["s3_client.py", "keys.py"],
    tech: ["boto3", "lru_cache"],
    points: [
      "Validates file type and size before any processing begins",
      "All S3 key strings centralised in keys.py — no magic strings anywhere else in the codebase",
      "Thin boto3 wrapper raises S3Error on failure; lru_cache keeps the client singleton alive",
    ],
  },
  {
    icon: "☁️", color: "#FB923C", bg: "rgba(251,146,60,0.12)", border: "rgba(251,146,60,0.25)",
    label: "Textract OCR", sub: "Extract",
    title: "AWS Textract Extraction",
    files: ["extractor.py", "validator.py"],
    tech: ["AWS Textract", "PyMuPDF"],
    points: [
      "AnalyzeDocument with TABLES + FORMS — handles handwriting and multi-column layouts",
      "1-page → sync DetectDocumentText · multi-page → async job with S3 result polling",
      "Validator discards blocks with ERROR-level confidence before passing text downstream",
    ],
  },
  {
    icon: "🔧", color: "#A78BFA", bg: "rgba(167,139,250,0.12)", border: "rgba(167,139,250,0.25)",
    label: "Text Cleaning", sub: "Normalise",
    title: "Text Cleaning + Normalisation",
    files: ["cleaner.py"],
    tech: ["regex", "structlog"],
    points: [
      "Fixes Textract artefacts: number-space noise, repeated page headers, empty checkbox tokens",
      "Preserves section headings needed for heading-aware chunking in the next stage",
      "Strips raw Textract metadata blocks not needed beyond this point",
    ],
  },
  {
    icon: "🧩", color: "#C084FC", bg: "rgba(192,132,252,0.12)", border: "rgba(192,132,252,0.25)",
    label: "Chunk + Embed", sub: "MiniLM",
    title: "Chunking + MiniLM Embedding",
    files: ["chunker.py", "embedder.py"],
    tech: ["MiniLM-L6-v2", "sentence-transformers", "NumPy"],
    points: [
      "~400-token sliding window, 60-token overlap, nearest heading stored as section label",
      "MiniLM-L6-v2 produces 384-dim L2-normalised vectors — inner product == cosine similarity",
      "Multi-doc: vstack new embeddings onto the existing array before re-indexing",
    ],
  },
  {
    icon: "🗄️", color: "#22D3EE", bg: "rgba(34,211,238,0.12)", border: "rgba(34,211,238,0.25)",
    label: "FAISS Index", sub: "Vector DB",
    title: "FAISS Vector Index",
    files: ["indexer.py"],
    tech: ["FAISS IndexFlatIP", "NumPy", "S3 persist"],
    points: [
      "IndexFlatIP on L2-normalised vectors — exact cosine search, no approximation error",
      "Full rebuild on each upload (incremental add not supported with S3 serialisation)",
      "Serialised to S3 after build; loaded into memory on first query, cached per session",
    ],
  },
  {
    icon: "🔍", color: "#34D399", bg: "rgba(52,211,153,0.12)", border: "rgba(52,211,153,0.25)",
    label: "Intent + Retrieve", sub: "Zero-shot",
    title: "Intent Classification + Retrieval",
    files: ["retriever.py", "intent_classifier.py"],
    tech: ["MiniLM reuse", "FAISS top-k", "MLflow tags"],
    points: [
      "Zero-shot intent: cosine similarity of query vs 5 intent-description embeddings — no extra LLM call",
      "Primary intent drives template selection; secondaries logged to MLflow for offline analysis",
      "top-k=5 default, configurable 1–20 per request; chunk scores returned for tracing",
    ],
  },
  {
    icon: "⚡", color: "#F87171", bg: "rgba(248,113,113,0.12)", border: "rgba(248,113,113,0.25)",
    label: "Prompt → LLM", sub: "SSE stream",
    title: "Prompt Routing → Groq LLM → SSE",
    files: ["router.py", "ab_selector.py", "groq_client.py"],
    tech: ["Groq llama-3.3-70b", "AsyncGroq SSE", "ε-greedy A/B"],
    points: [
      "5 intents × 2 variants = 10 templates — v1 detailed prose, v2 structured concise output",
      "ε-greedy selector (10% explore) reads MLflow latency per variant to exploit the winner",
      "AsyncGroq streams real tokens via SSE; MLflow logged via run_in_executor after stream closes",
    ],
  },
];
