"""
S4 generation runner (scope v2 §8.4, §8.5, §9.5, D5, D7). Resumable.

For each configuration it reads the retrieval run's per-question ranking (`data/results/r4/<stage>/
<split>/<run>/questions.jsonl`), rebuilds the context from the chunk cache (the `in_budget` chunks,
i.e. production's budget rule), sends the citation prompt to the generator with native streaming,
and appends one line per answer to `data/results/r6/<stage>/<split>/<run>/answers.jsonl`.

- **Resume:** questions already in answers.jsonl are skipped; run again the next UTC day.
- **Order:** question by question, every configuration per question, so a run stopped by the
  daily budget leaves paired results (same questions answered by every configuration).
- **Cost guard (D7):** prints the estimate and refuses to call without `--confirm-cost`; every call
  goes through the ledger (daily token budget, $ caps). Estimates use the measured mean tokens of
  earlier answers once there are some.
- **Rate limits:** calls are paced under `tpm` tokens/minute; a 429 stops the run (resume later).
- `--summarise` writes summary.json per configuration from the saved answers (no calls).

    venv/bin/python -m evals.generation.run --config evals/configs/s4_generation.yaml --split dev --limit 5
    venv/bin/python -m evals.generation.run --config ... --split test --confirm-cost
    venv/bin/python -m evals.generation.run --config ... --split test --summarise
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.generation import metrics as GM
from evals.generation.prompt import PROMPT_VERSION, ContextChunk, build_prompt
from evals.generation.stream import stream_call

REPO = Path(__file__).resolve().parents[2]
RETRIEVAL_ROOT = REPO / "data" / "results" / "r4"
OUT_ROOT = REPO / "data" / "results" / "r6"
TOKENS_PER_WORD = 1.35            # prompt estimate until answers exist (S-2026-11-23-25)
EST_COMPLETION = 300              # completion estimate until answers exist
log = logging.getLogger("evals.generation")


# ---- config ------------------------------------------------------------------------------------
@dataclass
class GenRun:
    name: str
    retrieval: dict               # {stage, run}: the retrieval run whose ranking feeds this config


@dataclass
class GenConfig:
    stage: str
    model: str
    temperature: float
    max_tokens: int
    tpm: int
    runs: list[GenRun]


def load_config(path: Path) -> GenConfig:
    raw = yaml.safe_load(path.read_text())
    runs = [GenRun(**r) for r in raw.get("runs") or []]
    if not runs or len({r.name for r in runs}) != len(runs):
        raise ValueError(f"{path}: need runs with unique names")
    for r in runs:
        if set(r.retrieval) != {"stage", "run"}:
            raise ValueError(f"{path}: run {r.name}: retrieval must be {{stage, run}}")
    return GenConfig(raw["stage"], raw["model"], float(raw["temperature"]), int(raw["max_tokens"]),
                     int(raw.get("tpm", 6000)), runs)


def retrieval_dir(run: GenRun, split: str, root: Path = RETRIEVAL_ROOT) -> Path:
    return root / run.retrieval["stage"].lower() / split / run.retrieval["run"]


def out_dir(cfg: GenConfig, run: GenRun, split: str, root: Path = OUT_ROOT) -> Path:
    return root / cfg.stage.lower() / split / run.name


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


# ---- context -----------------------------------------------------------------------------------
class ChunkStore:
    """Chunk text by (chunker, doc view) from the R4 index cache; built lazily (MiniLM only loads
    on a cache miss)."""

    def __init__(self, table_config: str):
        self.table_config = table_config
        self._cache = None

    def _get_cache(self):
        if self._cache is None:
            from evals.retrieval.index_cache import IndexCache
            from evals.run import Corpus, MiniLM
            corpus = Corpus(self.table_config)
            embed_holder: dict = {}

            def embed(texts):
                if "m" not in embed_holder:
                    embed_holder["m"] = MiniLM()
                return embed_holder["m"](texts)
            self._cache = IndexCache(corpus.load_source, embed, corpus.parser_key)
        return self._cache

    def chunks(self, chunker_id: str, chunker_params: dict, doc_id: str, drop: list[int]) -> dict:
        from evals.chunking.check_corpus import CHUNKERS
        di = self._get_cache().get(CHUNKERS[chunker_id](**chunker_params), doc_id, frozenset(drop))
        return {c.chunk_id: c for c in di.chunks}


def context_for(row: dict, summary: dict, store) -> list[ContextChunk]:
    """The in-budget chunks of one retrieval row, in rank order, as prompt chunks S1…Sn."""
    run = summary["run"]
    if run.get("parent_chunker"):
        raise ValueError("parent-child runs are not supported by the generation harness")
    drops = {d: p for d, p in row["view"]}
    out: list[ContextChunk] = []
    for r in row["ranked"]:
        if not r["in_budget"]:
            continue
        by_id = store.chunks(run["chunker"], run.get("chunker_params") or {}, r["doc_id"], drops.get(r["doc_id"], []))
        c = by_id[r["chunk_id"]]
        out.append(ContextChunk(f"S{len(out) + 1}", c.chunk_id, c.doc_id, c.page_start, c.text, tuple(r["covers"])))
    return out


# ---- estimate ----------------------------------------------------------------------------------
def estimate(prompts_words: list[int], done: list[dict]) -> tuple[int, int]:
    """(prompt tokens, completion tokens) for the calls to make. Measured ratio once answers exist."""
    ok = [a for a in done if a.get("prompt_tokens")]
    if ok:
        ratio = sum(a["prompt_tokens"] for a in ok) / sum(a["prompt_words"] for a in ok)
        comp = statistics.mean(a["completion_tokens"] for a in ok)
    else:
        ratio, comp = TOKENS_PER_WORD, EST_COMPLETION
    return round(sum(prompts_words) * ratio), round(comp * len(prompts_words))


# ---- run ---------------------------------------------------------------------------------------
def plan(cfg: GenConfig, split: str, only: list[str] | None, limit: int | None, questions: dict,
         retrieval_root: Path, out_root: Path):
    """Per run: retrieval summary + rows (question order of the retrieval run), done qids."""
    runs = [r for r in cfg.runs if not only or r.name in only]
    if only and len(runs) != len(only):
        raise ValueError(f"unknown runs: {sorted(set(only) - {r.name for r in runs})}")
    jobs = {}
    for r in runs:
        rdir = retrieval_dir(r, split, retrieval_root)
        if not (rdir / "summary.json").exists():
            raise SystemExit(f"no retrieval run at {rdir} — run evals.run on split {split} first")
        summary = json.loads((rdir / "summary.json").read_text())
        if summary.get("limit"):
            raise SystemExit(f"{rdir} is a smoke retrieval run (limit {summary['limit']})")
        rows = [row for row in read_jsonl(rdir / "questions.jsonl") if row["qid"] in questions]
        rows = rows[:limit] if limit else rows
        done = {a["qid"] for a in read_jsonl(out_dir(cfg, r, split, out_root) / "answers.jsonl")
                if "error" not in a}
        jobs[r.name] = (r, summary, rows, done)
    return jobs


def generate(cfg: GenConfig, split: str, questions: dict, client, ledger, store, only=None, limit=None,
             confirm: bool = False, retrieval_root: Path = RETRIEVAL_ROOT, out_root: Path = OUT_ROOT,
             sleep=time.sleep) -> dict:
    from evals.llm.ledger import BudgetExceeded
    jobs = plan(cfg, split, only, limit, questions, retrieval_root, out_root)
    # interleave: question by question, every configuration
    order: list[str] = []
    for _, (_, _, rows, _) in jobs.items():
        for row in rows:
            if row["qid"] not in order:
                order.append(row["qid"])
    todo = [(qid, name) for qid in order for name, (_, _, rows, done) in jobs.items()
            if qid not in done and any(r["qid"] == qid for r in rows)]
    prev = [a for name, (r, *_rest) in jobs.items()
            for a in read_jsonl(out_dir(cfg, r, split, out_root) / "answers.jsonl") if "error" not in a]
    # words of the prompts to send (contexts are cheap to build: cached chunks)
    built = {}
    for qid, name in todo:
        r, summary, rows, _ = jobs[name]
        row = next(x for x in rows if x["qid"] == qid)
        ctx = context_for(row, summary, store)
        built[(qid, name)] = (row, ctx, build_prompt(questions[qid]["question"], ctx))
    est_p, est_c = estimate([len(p.split()) for _, _, p in built.values()], prev)
    from evals.llm.ledger import cost_usd
    est_usd = cost_usd(cfg.model, est_p, est_c)
    info = {"calls_todo": len(todo), "est_prompt_tokens": est_p, "est_completion_tokens": est_c,
            "est_total_tokens": est_p + est_c, "est_usd": round(est_usd, 4),
            "tokens_today": ledger.tokens_today(), "daily_budget": ledger.daily_budget}
    log.info("estimate: %s", info)
    if not confirm or not todo:
        return {**info, "called": 0, "stopped": None if todo else "nothing to do",
                **({"note": "dry run: pass --confirm-cost to call the model"} if todo and not confirm else {})}
    per_call_c = est_c / len(todo)
    called, stop = 0, None
    for qid, name in todo:
        r, summary, rows, _ = jobs[name]
        row, ctx, prompt = built[(qid, name)]
        p_est = round(len(prompt.split()) * (est_p / max(1, sum(len(p.split()) for *_, p in built.values()))))
        try:
            ledger.check(cfg.model, p_est, round(per_call_c))
        except BudgetExceeded as exc:
            stop = str(exc)
            break
        t0 = time.monotonic()
        try:
            res = stream_call(client, cfg.model, prompt, cfg.temperature, cfg.max_tokens)
        except Exception as exc:                         # rate limit / API error: stop, resume later
            stop = f"{type(exc).__name__}: {exc}"
            if "RateLimit" not in type(exc).__name__:
                _append(out_dir(cfg, r, split, out_root), {"qid": qid, "error": stop, "ts": ledger.now()})
            break
        pt = res.prompt_tokens if res.prompt_tokens is not None else p_est
        ct = res.completion_tokens if res.completion_tokens is not None else round(per_call_c)
        lrow = ledger.record(f"r6_gen:{cfg.stage}:{name}:{qid}", cfg.model, res.returned_model or cfg.model,
                             pt, ct, res.reasoning_tokens)
        q = questions[qid]
        m = GM.score_answer(q, res.answer, ctx, row["metrics"].get("recall@budget"))
        retr_s = row.get("timings_s", {}).get("total")
        rec = {"qid": qid, "type": q["type"], "part": row.get("part"), "split": split, "run": name,
               "prompt_version": PROMPT_VERSION, "model": cfg.model, "ts": lrow["ts"],
               "context": [{"sid": c.local_id, "chunk_id": c.chunk_id, "covers": list(c.covers)} for c in ctx],
               "prompt_words": len(prompt.split()), "answer": res.answer, "raw": res.raw,
               "cited_chunk_ids": [next((c.chunk_id for c in ctx if c.local_id == s), s)
                                   for s in GM.citations(res.answer)],
               "recall@budget": row["metrics"].get("recall@budget"), "metrics": m,
               "timings_s": {**res.timings, "retrieval": retr_s,
                             "e2e_total": (res.timings["e2e"] + retr_s) if retr_s is not None else None},
               "returned_model": res.returned_model, "finish_reason": res.finish_reason,
               "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": res.reasoning_tokens,
               "reasoning_chars": res.reasoning_chars, "usage_estimated": res.prompt_tokens is None,
               "cost_usd": lrow["cost_usd"]}
        _append(out_dir(cfg, r, split, out_root), rec)
        called += 1
        log.info("%s %s: %d+%d tok, ttft %.2fs, e2e %.2fs, refused=%s cites=%d", name, qid, pt, ct,
                 res.timings["ttft"] or -1, res.timings["e2e"], m["refused"], m["n_citations"])
        wait = (pt + ct) / cfg.tpm * 60 - (time.monotonic() - t0)
        if wait > 0 and (qid, name) != todo[-1]:
            sleep(wait)
    return {**info, "called": called, "stopped": stop, "run_tokens": ledger.run_tokens,
            "run_cost_usd": round(ledger.run_cost, 4), "tokens_today_after": ledger.tokens_today()}


def _append(d: Path, rec: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with (d / "answers.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---- summary -----------------------------------------------------------------------------------
def _stat(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "mean": statistics.mean(xs) if xs else None,
            "median": statistics.median(xs) if xs else None}


def summarise_answers(answers: list[dict]) -> dict:
    ok = [a for a in answers if "error" not in a]
    errors = len(answers) - len(ok)

    def block(rows):
        out = {}
        for k in GM.BOOL_METRICS + GM.MEAN_METRICS:
            vals = [float(a["metrics"][k]) for a in rows if a["metrics"].get(k) is not None]
            out[k] = {"n": len(vals), "mean": statistics.mean(vals) if vals else None}
        return out
    types = sorted({a["type"] for a in ok})
    cost = sum(a["cost_usd"] for a in ok)
    from evals.llm.ledger import PRICES
    s = {"answered": len(ok), "errors": errors,
         "error_rate": errors / len(answers) if answers else None,
         "overall": block(ok), "by_type": {t: block([a for a in ok if a["type"] == t]) for t in types},
         "latency_s": {k: _stat([a["timings_s"].get(k) for a in ok])
                       for k in ("ttft", "ttfs", "inter_token", "tokens_per_s", "e2e", "retrieval", "e2e_total")},
         "tokens": {k: _stat([a.get(k) for a in ok]) for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
         "cost": {"total_usd": cost, "per_query_usd": cost / len(ok) if ok else None,
                  "per_1k_queries_usd": 1000 * cost / len(ok) if ok else None}}
    if ok:
        pin, pout = PRICES[ok[0]["model"]]
        inp = sum(a["prompt_tokens"] for a in ok) * pin / 1e6
        s["cost"]["input_usd"], s["cost"]["output_usd"] = inp, cost - inp
        s["model"], s["prompt_version"] = ok[0]["model"], ok[0]["prompt_version"]
    return s


def summarise(cfg: GenConfig, split: str, out_root: Path = OUT_ROOT) -> dict:
    out = {}
    for r in cfg.runs:
        d = out_dir(cfg, r, split, out_root)
        answers = read_jsonl(d / "answers.jsonl")
        if not answers:
            continue
        s = {"run": r.name, "retrieval": r.retrieval, "split": split, "temperature": cfg.temperature,
             "max_tokens": cfg.max_tokens, **summarise_answers(answers)}
        (d / "summary.json").write_text(json.dumps(s, indent=2))
        out[r.name] = s
    return out


# ---- CLI ---------------------------------------------------------------------------------------
def groq_client():
    import os

    from dotenv import load_dotenv
    from groq import Groq
    load_dotenv(REPO / ".env")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not set (.env)")
    return Groq(api_key=key, max_retries=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--runs", help="comma-separated run names (default: all)")
    ap.add_argument("--limit", type=int, help="first N questions of each retrieval run")
    ap.add_argument("--confirm-cost", action="store_true", help="actually call the model (D7)")
    ap.add_argument("--summarise", action="store_true", help="write summary.json from saved answers; no calls")
    ap.add_argument("--daily-budget", type=int, help="override the ledger's daily token guard")
    ap.add_argument("--qids-file", type=Path, help="JSON with a 'qids' list: only these questions (calibration)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = load_config(a.config)
    if a.summarise:
        print(json.dumps({k: {"answered": v["answered"], "overall": v["overall"]} for k, v in
                          summarise(cfg, a.split).items()}, indent=1))
        return 0
    from evals.golden.qa.validate import QUESTIONS, read_questions
    from evals.llm.ledger import Ledger
    questions = {q["qid"]: q for q in read_questions(QUESTIONS) if not q.get("retired") and q["split"] == a.split}
    if a.qids_file:
        keep = set(json.loads(a.qids_file.read_text())["qids"])
        questions = {k: v for k, v in questions.items() if k in keep}
    ledger = Ledger(**({"daily_budget": a.daily_budget} if a.daily_budget else {}))
    table_config = "P2"
    store = ChunkStore(table_config)
    client = groq_client() if a.confirm_cost else None
    res = generate(cfg, a.split, questions, client, ledger, store, a.runs.split(",") if a.runs else None,
                   a.limit, a.confirm_cost)
    print(json.dumps(res, indent=1))
    if res.get("called"):
        summarise(cfg, a.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
