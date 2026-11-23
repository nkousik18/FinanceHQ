"""
§8.3 retrieval metrics (scope v2), per question, from a ranked chunk list. No retriever code here:
the runner (R4 step 2+) retrieves, this module scores.

Coverage (§8.3, same rule as R2's evidence fragmentation, `evals.chunking.metrics.covers`): a
chunk covers an evidence span when it is from the span's doc, its page range includes the span's
page, and >= 50% of the span's non-blank characters fall inside the chunk's spans.

Context budget: production's `_build_context` (fastapi_service/app/prompts/router.py) — chunks in
rank order, added while the running word count stays <= 1,200; the first chunk is always added;
the first chunk that doesn't fit stops the loop (no skipping ahead). Words = whitespace split of
the chunk text (production `_word_count`).

Definitions (k = cut-off; "relevant" = the chunk covers >= 1 evidence span):
- recall@k:          covered spans in top k / all spans
- hit@k:             1 if >= 1 span covered in top k
- precision@k:       relevant chunks in top k / k
- context precision: RAGAS: mean of precision@i over the ranks i <= k holding a relevant chunk
                     (0 if none)
- mrr@10:            1 / rank of the first relevant chunk (0 if none in top 10)
- ndcg@10:           binary gains, log2 discount; ideal = min(10, relevant chunks in the whole
                     session index)
- map@10:            sum(precision@i * rel_i, i <= 10) / min(10, relevant chunks in the index)
- recall@budget (★): covered spans within the budget / all spans  — the S1/S2 decision metric
- entity recall (★): entities found (schema.value_in: numbers normalized, names norm-matched) in
                     the budget context text / all entities
- noise ratio:       words of budget chunks that cover nothing / budget words
- redundancy:        mean over budget chunk pairs of |shared tokens| / |smaller chunk's tokens|,
                     tokens = distinct norm_match words (None with < 2 chunks)
- cross-doc coverage (cross_doc questions): every evidence doc has a covering chunk in the budget
- amendment coverage (amendment questions): an amending-doc span is covered in the budget; also
                     `amend_only_superseded`: an original-doc span is covered and no amending span is
- context_words:     budget words

Unanswerable questions: every coverage-based metric is None (excluded from means); context_words and
redundancy are still reported. An evidence span that can't be located in the cleaned page text
counts as not covered and is reported in `unlocated` (the F3 validator should make this 0).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from evals.chunking.base import Chunk, SourceDoc, Span
from evals.chunking.metrics import Evidence, covers, locate_evidence
from evals.golden.qa.schema import value_in
from evals.ingest_metrics.normalize import norm_match

MAX_CONTEXT_WORDS = 1200          # production prompts/router.py MAX_CONTEXT_WORDS
KS = (1, 3, 5, 10)
RANK_CUTOFF = 10                  # mrr / ndcg / map / context precision


def chunk_words(chunk: Chunk) -> int:
    return len(chunk.text.split())


def budget_cut(ranked: list[Chunk], max_words: int = MAX_CONTEXT_WORDS) -> int:
    """How many top-ranked chunks production puts in the prompt (mirrors router._build_context)."""
    total = n = 0
    for c in ranked:
        w = chunk_words(c)
        if total + w > max_words and n:
            break
        total += w
        n += 1
    return n


@dataclass
class Located:
    """A question's evidence spans located in the cleaned page text (None: not found)."""
    evidence: list[Evidence]
    spans: list[Span | None]


def locate_all(evidence: list[Evidence], sources: dict[str, SourceDoc]) -> Located:
    spans = [locate_evidence(sources[ev.doc_id], ev) if ev.doc_id in sources else None
             for ev in evidence]
    return Located(evidence, spans)


def coverage(chunk: Chunk, loc: Located, sources: dict[str, SourceDoc]) -> frozenset[int]:
    """Indices of the evidence spans this chunk covers."""
    out = set()
    for i, (ev, span) in enumerate(zip(loc.evidence, loc.spans)):
        if span is not None and chunk.doc_id == ev.doc_id and covers(sources[ev.doc_id], chunk, span):
            out.add(i)
    return frozenset(out)


def count_relevant(chunks: list[Chunk], loc: Located, sources: dict[str, SourceDoc]) -> int:
    """Relevant chunks in a whole index (the ideal-ranking size for ndcg / map)."""
    return sum(bool(coverage(c, loc, sources)) for c in chunks)


def index_coverage(chunks: list[Chunk], loc: Located, sources: dict[str, SourceDoc]) -> tuple[int, int]:
    """Over a whole index: (relevant chunks, located spans no single chunk covers). The second is
    §8.2 evidence fragmentation's numerator, per question."""
    covered: set[int] = set()
    n_rel = 0
    for c in chunks:
        cv = coverage(c, loc, sources)
        n_rel += bool(cv)
        covered |= cv
    located = {i for i, s in enumerate(loc.spans) if s is not None}
    return n_rel, len(located - covered)


# ---- rank metrics on a coverage list (pure; cov[i] = spans covered by the chunk at rank i+1) ----
def recall_at(cov: list[frozenset[int]], n_spans: int, k: int) -> float:
    return len(frozenset().union(*cov[:k])) / n_spans


def hit_at(cov: list[frozenset[int]], k: int) -> float:
    return float(any(cov[:k]))


def precision_at(cov: list[frozenset[int]], k: int) -> float:
    return sum(bool(c) for c in cov[:k]) / k


def context_precision(cov: list[frozenset[int]], k: int = RANK_CUTOFF) -> float:
    rel = [bool(c) for c in cov[:k]]
    if not any(rel):
        return 0.0
    hits, total = 0, 0.0
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1
            total += hits / i
    return total / sum(rel)


def mrr_at(cov: list[frozenset[int]], k: int = RANK_CUTOFF) -> float:
    for i, c in enumerate(cov[:k], 1):
        if c:
            return 1.0 / i
    return 0.0


def ndcg_at(cov: list[frozenset[int]], n_relevant: int, k: int = RANK_CUTOFF) -> float:
    ideal = min(k, n_relevant)
    if ideal == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(cov[:k], 1) if c)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg


def map_at(cov: list[frozenset[int]], n_relevant: int, k: int = RANK_CUTOFF) -> float:
    denom = min(k, n_relevant)
    if denom == 0:
        return 0.0
    hits, total = 0, 0.0
    for i, c in enumerate(cov[:k], 1):
        if c:
            hits += 1
            total += hits / i
    return total / denom


# ---- context metrics ---------------------------------------------------------------------------
def entity_recall(entities: list[str], context: str) -> float | None:
    if not entities:
        return None
    return sum(value_in(e, context) for e in entities) / len(entities)


def redundancy(chunks: list[Chunk]) -> float | None:
    toks = [set(norm_match(c.text).split()) for c in chunks]
    pairs = [(a, b) for a, b in combinations(toks, 2) if a and b]
    if not pairs:
        return None
    return sum(len(a & b) / min(len(a), len(b)) for a, b in pairs) / len(pairs)


# ---- one question ------------------------------------------------------------------------------
@dataclass
class QuestionResult:
    qid: str
    type: str
    part: str
    metrics: dict[str, float | None]
    n_budget: int
    ranked: list[dict] = field(default_factory=list)     # chunk_id, rank, score, covers, in_budget
    unlocated: int = 0


def score_question(q: dict, part: str, ranked: list[Chunk], scores: list[float] | None,
                   sources: dict[str, SourceDoc], n_relevant: int | None = None,
                   amendment_docs: set[str] = frozenset(),
                   max_words: int = MAX_CONTEXT_WORDS, budget_pool: int | None = None) -> QuestionResult:
    """Score one question's ranked retrieval. `sources` must hold every session doc that appears in
    the evidence or the ranking. `n_relevant`: relevant chunks in the session index (ndcg/map
    ideal); defaults to the relevant chunks in `ranked`. `amendment_docs`: doc_ids whose role is an
    amendment (amendment coverage). `budget_pool`: production's top_k — only the first
    `budget_pool` chunks can enter the budget (the ranking may be deeper, for the @10 metrics);
    default: all of `ranked`."""
    evidence = [Evidence(e["doc_id"], e["page"], e["span_text"]) for e in q["evidence"]]
    loc = locate_all(evidence, sources)
    cov = [coverage(c, loc, sources) for c in ranked]
    nb = budget_cut(ranked[:budget_pool] if budget_pool is not None else ranked, max_words)
    budget = ranked[:nb]
    words = [chunk_words(c) for c in budget]
    m: dict[str, float | None] = {"context_words": float(sum(words)),
                                  "redundancy": redundancy(budget)}
    answerable = not q.get("unanswerable") and bool(evidence)

    keys = [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS] + [f"precision@{k}" for k in KS] + \
        ["recall@budget", "context_precision", "mrr@10", "ndcg@10", "map@10", "entity_recall",
         "noise_ratio", "cross_doc_coverage", "amend_coverage", "amend_only_superseded"]
    m.update({k: None for k in keys})
    if answerable:
        n = len(evidence)
        rel = n_relevant if n_relevant is not None else sum(bool(c) for c in cov)
        for k in KS:
            m[f"recall@{k}"] = recall_at(cov, n, k)
            m[f"hit@{k}"] = hit_at(cov, k)
            m[f"precision@{k}"] = precision_at(cov, k)
        m["recall@budget"] = recall_at(cov, n, nb)
        m["context_precision"] = context_precision(cov)
        m["mrr@10"] = mrr_at(cov)
        m["ndcg@10"] = ndcg_at(cov, rel)
        m["map@10"] = map_at(cov, rel)
        m["entity_recall"] = entity_recall(q.get("entities") or [], "\n".join(c.text for c in budget))
        total = sum(words)
        m["noise_ratio"] = (sum(w for w, c in zip(words, cov) if not c) / total) if total else None
        covered = frozenset().union(*cov[:nb])
        if q["type"] == "cross_doc":
            need = {ev.doc_id for ev in evidence}
            got = {evidence[i].doc_id for i in covered}
            m["cross_doc_coverage"] = float(need <= got)
        if q.get("amendment"):
            amend = {i for i, ev in enumerate(evidence) if ev.doc_id in amendment_docs}
            orig = set(range(n)) - amend
            m["amend_coverage"] = float(bool(amend & covered))
            m["amend_only_superseded"] = float(bool(orig & covered) and not (amend & covered))

    ranked_rows = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "rank": i + 1,
                    "score": None if scores is None else float(scores[i]),
                    "covers": sorted(cov[i]), "in_budget": i < nb}
                   for i, c in enumerate(ranked)]
    return QuestionResult(q["qid"], q["type"], part, m, nb, ranked_rows,
                          unlocated=sum(s is None for s in loc.spans))


# ---- aggregation -------------------------------------------------------------------------------
def mean_metrics(results: list[QuestionResult]) -> dict[str, dict]:
    """Per metric: mean over the questions where it's defined, and that n."""
    vals: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for k, v in r.metrics.items():
            if v is not None:
                vals[k].append(v)
    return {k: {"mean": sum(v) / len(v), "n": len(v)} for k, v in sorted(vals.items())}


def breakdown(results: list[QuestionResult]) -> dict[str, dict]:
    """Means overall, by question type and by corpus part (§8.3: break every metric down)."""
    out = {"overall": mean_metrics(results)}
    for attr in ("type", "part"):
        groups: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in results:
            groups[getattr(r, attr)].append(r)
        for g, rs in sorted(groups.items()):
            out[f"{attr}={g}"] = mean_metrics(rs)
    return out


def question_part(q: dict, part_of_doc: dict[str, str]) -> str:
    """Corpus part of a question: simulated_scan if any evidence is on a scan, consumer for CFPB
    sessions, else edgar (finalise.py strata without the amendment split)."""
    if any(part_of_doc.get(e["doc_id"]) == "simulated_scan" for e in q["evidence"]):
        return "simulated_scan"
    return "consumer" if q["session_id"].startswith("cfpb") else "edgar"
