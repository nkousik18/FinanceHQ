"""
Token counting for chunkers and §8.2 metrics: MiniLM (the production embedder's own tokenizer)
and tiktoken.

MiniLM counts here are **content tokens** (no [CLS]/[SEP]). The embedder's max_seq_length is 256
including those 2 special tokens (R0 §2), so a chunk is embedded whole iff its count <= 254.
"""
from __future__ import annotations

import logging
from functools import lru_cache

MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # production embedder.MODEL_NAME, hub id
MAX_SEQ_LENGTH = 256                                     # SentenceTransformer max_seq_length
SPECIAL_TOKENS = 2                                       # [CLS] + [SEP]
MAX_CONTENT_TOKENS = MAX_SEQ_LENGTH - SPECIAL_TOKENS     # 254
TIKTOKEN_ENCODING = "cl100k_base"

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def minilm_tokenizer():
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()          # silences "sequence longer than 512" on whole documents
    tok = AutoTokenizer.from_pretrained(MINILM_NAME)
    if not tok.is_fast:
        raise RuntimeError("MiniLM fast tokenizer required (offset mapping)")
    return tok


@lru_cache(maxsize=1)
def _tiktoken():
    import tiktoken
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def minilm_counts(texts: list[str]) -> list[int]:
    if not texts:
        return []
    enc = minilm_tokenizer()(texts, add_special_tokens=False)
    return [len(ids) for ids in enc["input_ids"]]


def minilm_count(text: str) -> int:
    return minilm_counts([text])[0]


def tiktoken_count(text: str) -> int:
    return len(_tiktoken().encode(text, disallowed_special=()))


def minilm_offsets(text: str) -> tuple[list[tuple[int, int]], list[int | None]]:
    """Content tokens of `text`: char offsets and pre-tokenizer word index per token."""
    enc = minilm_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["offset_mapping"]), enc.word_ids()
