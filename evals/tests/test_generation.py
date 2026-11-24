"""R6 generation harness: prompt, stream timings, deterministic metrics, resumable runner.
No network: a fake streaming client stands in for Groq (F8 known-answer tests)."""
import json
from types import SimpleNamespace as NS

import pytest

from evals.generation import metrics as GM
from evals.generation import run as GR
from evals.generation.prompt import FINAL_PREFIX, REFUSAL, ContextChunk, build_prompt
from evals.generation.stream import compute_timings, first_sentence_end, stream_call, strip_think
from evals.llm.ledger import Ledger


def ctx(*covers):
    return [ContextChunk(f"S{i + 1}", f"d-C2-{i}", "d", 1, f"text {i}", tuple(c)) for i, c in enumerate(covers)]


# ---- prompt ------------------------------------------------------------------------------------
def test_prompt_has_rules_ids_and_question():
    p = build_prompt("What is the fee?", ctx([0], []))
    assert "[S1 | d | Page 1]\ntext 0" in p and "[S2 | d | Page 1]" in p
    assert REFUSAL in p and FINAL_PREFIX in p and p.rstrip().endswith("Question: What is the fee?\n\nAnswer:")


# ---- stream ------------------------------------------------------------------------------------
def test_strip_think_and_sentence_end():
    assert strip_think("<think>hmm</think>The fee is 2%.") == "The fee is 2%."
    assert strip_think("<think>never closed") == ""
    assert first_sentence_end("Rate is 3.75") is None          # decimal point, not an end
    assert first_sentence_end("Rate is 3.") is None             # mid-stream: digits may follow
    assert first_sentence_end("Rate is 3.", final=True) == 10
    assert first_sentence_end("It is 5%. Next") == 9


def test_timings_known_answer():
    ev = [(0.5, "content", "<think>plan"), (0.7, "content", "</think>"), (1.0, "content", "Fee is 3"),
          (1.1, "content", ".75% [S1]"), (1.2, "content", ". More"), (1.5, "content", " text.")]
    t = compute_timings(ev, 1.6, completion_tokens=10)
    assert t["ttft_any"] == 0.5 and t["ttft"] == 1.0
    assert t["ttfs"] == 1.2                                     # "3.75" didn't end the sentence
    assert t["inter_token"] == pytest.approx((1.5 - 1.0) / 3)
    assert t["tokens_per_s"] == pytest.approx(10 / 0.5) and t["e2e"] == 1.6 and t["content_chunks"] == 4


def test_timings_reasoning_delta_and_end_of_stream_sentence():
    ev = [(0.2, "reasoning", "think"), (0.9, "content", "Not found in the document.")]
    t = compute_timings(ev, 1.0, None)
    assert t["ttft_any"] == 0.2 and t["ttft"] == 0.9 and t["ttfs"] == 0.9 and t["tokens_per_s"] is None


def fake_chunk(content=None, reasoning=None, finish=None, usage=None):
    return NS(model="qwen/qwen3.8-27b", x_groq=NS(usage=usage) if usage else None, usage=None,
              choices=[NS(finish_reason=finish, delta=NS(content=content, reasoning=reasoning))])


class FakeClient:
    def __init__(self, texts, prompt_tokens=100, completion_tokens=20):
        self.texts, self.calls = list(texts), []
        self.pt, self.ct = prompt_tokens, completion_tokens
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        pieces = [text[i:i + 4] for i in range(0, len(text), 4)]
        u = NS(prompt_tokens=self.pt, completion_tokens=self.ct, completion_tokens_details=None)
        return iter([fake_chunk(p) for p in pieces] + [fake_chunk(finish="stop", usage=u)])


def test_stream_call_collects_answer_usage_and_settings():
    ticks = iter(range(100))
    c = FakeClient(["<think>x</think>Fee is 2% [S1]."])
    r = stream_call(c, "m", "prompt", 0.1, 1024, clock=lambda: next(ticks) * 0.1)
    assert r.answer == "Fee is 2% [S1]." and r.prompt_tokens == 100 and r.completion_tokens == 20
    assert r.finish_reason == "stop" and r.reasoning_chars == len("<think>x</think>")
    kw = c.calls[0]
    assert kw["stream"] is True and kw["temperature"] == 0.1 and kw["max_completion_tokens"] == 1024
    assert kw["messages"] == [{"role": "user", "content": "prompt"}]


# ---- metrics -----------------------------------------------------------------------------------
def test_citations_parse_and_ignore_non_ids():
    assert GM.citations("A [S1][S3]. B [S1, S2]. See [Reserved] and [S9].") == ["S1", "S3", "S2", "S9"]


@pytest.mark.parametrize("text,refused", [
    ("Not found in the document.", True),
    ("The documents do not state the actual ratio.", True),
    ("The ratio is not specified in the context.", True),
    ("The fee is 0.25% [S1]. The agreement does not state the payment date.", False),
    ("The fee is 0.25% [S1].", False),
])
def test_refusal(text, refused):
    assert GM.is_refusal(text) is refused


@pytest.mark.parametrize("answer,value", [
    ("Steps... [S1]\nFinal answer: 3.75% per annum", 3.75),
    ("Final answer: $500 million", 500e6),
    ("Final answer: $1,250,000,000.00", 1.25e9),
    ("The increase is $8,000,000 [S2].", 8e6),               # no final line: last number
    ("A ratio of 90% [S1]. So 171,000 / 190,000 = 0.9", 0.9),
])
def test_final_value(answer, value):
    assert GM.final_value(answer)[0] == pytest.approx(value)


def test_numeric_em_tolerance():
    assert GM.numeric_em("Final answer: 87.5%", 87.29, 0.44)
    assert not GM.numeric_em("Final answer: 88%", 87.29, 0.44)
    assert GM.numeric_em("Final answer: 0.50", 0.5, 0)


Q = {"qid": "q1", "type": "lookup", "unanswerable": False, "superseded_values": [], "entities": []}


def test_score_answer_citations_and_format():
    c = ctx([0], [], [1])                                  # S1, S3 cover evidence
    m = GM.score_answer(Q, "Fee is 2% [S1][S2].", c, 1.0)
    assert m["citation_precision"] == 0.5 and m["citation_recall"] == 0.5
    assert m["format_ok"] and m["false_refusal"] is False
    m = GM.score_answer(Q, "Fee is 2% [S7].", c, 1.0)       # unknown id = wrong citation, bad format
    assert m["citation_precision"] == 0 and m["invalid_citations"] == 1 and not m["format_ok"]
    m = GM.score_answer(Q, "Fee is 2%.", c, 0.5)            # no citation; partial evidence: no false-refusal check
    assert m["citation_precision"] is None and not m["format_ok"] and m["false_refusal"] is None


def test_score_answer_refusals():
    un = {**Q, "type": "unanswerable", "unanswerable": True}
    m = GM.score_answer(un, REFUSAL, ctx([]), None)
    assert m["refusal_correct"] and m["format_ok"] and m["citation_precision"] is None
    m = GM.score_answer(un, "The ratio is 3.75 [S1].", ctx([]), None)
    assert m["refusal_correct"] is False
    m = GM.score_answer(Q, REFUSAL, ctx([0]), 1.0)
    assert m["false_refusal"] is True and m["citation_recall"] is None


def test_stale_and_calculation_format():
    am = {**Q, "superseded_values": ["June 30, 2024"], "entities": ["March 31, 2025"]}
    assert GM.score_answer(am, "Matures June 30, 2024 [S1].", ctx([0]), 1.0)["stale"] is True
    assert GM.score_answer(am, "Was June 30, 2024, now March 31, 2025 [S1].", ctx([0]), 1.0)["stale"] is False
    calc = {**Q, "type": "calculation", "answer_numeric": 3.75, "numeric_tolerance": 0}
    m = GM.score_answer(calc, "1.25 + 2.50 [S1]\nFinal answer: 3.75%", ctx([0]), 1.0)
    assert m["numeric_em"] and m["final_line"] and m["format_ok"]
    m = GM.score_answer(calc, "It is 3.75% [S1].", ctx([0]), 1.0)
    assert m["numeric_em"] and not m["final_line"] and not m["format_ok"]


# ---- runner ------------------------------------------------------------------------------------
class FakeStore:
    def chunks(self, chunker_id, params, doc_id, drop):
        return {f"{doc_id}-C2-{i}": NS(chunk_id=f"{doc_id}-C2-{i}", doc_id=doc_id, page_start=1,
                                      text=f"chunk {i} " * 10) for i in range(5)}


def write_retrieval(root, stage, run, qids):
    d = root / stage.lower() / "test" / run
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"run": {"chunker": "C2", "chunker_params": {}}, "limit": None}))
    rows = [{"qid": q, "part": "edgar", "view": [["d", []]], "metrics": {"recall@budget": 1.0},
             "timings_s": {"total": 0.05},
             "ranked": [{"chunk_id": "d-C2-0", "doc_id": "d", "covers": [0], "in_budget": True},
                        {"chunk_id": "d-C2-1", "doc_id": "d", "covers": [], "in_budget": True},
                        {"chunk_id": "d-C2-2", "doc_id": "d", "covers": [], "in_budget": False}]}
            for q in qids]
    (d / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def setup(tmp_path):
    rroot, oroot = tmp_path / "r4", tmp_path / "r6"
    write_retrieval(rroot, "S2", "A", ["q1", "q2"])
    write_retrieval(rroot, "S1", "B", ["q1", "q2"])
    cfg = GR.GenConfig("S4", "qwen/qwen3.8-27b", 0.1, 1024, 10**9,
                       [GR.GenRun("A", {"stage": "S2", "run": "A"}), GR.GenRun("B", {"stage": "S1", "run": "B"})])
    qs = {q: {**Q, "qid": q, "question": f"question {q}?"} for q in ("q1", "q2")}
    return cfg, qs, rroot, oroot, Ledger(path=tmp_path / "ledger.jsonl")


def test_dry_run_estimates_without_calling(setup):
    cfg, qs, rroot, oroot, led = setup
    res = GR.generate(cfg, "test", qs, None, led, FakeStore(), retrieval_root=rroot, out_root=oroot)
    assert res["calls_todo"] == 4 and res["called"] == 0 and res["est_total_tokens"] > 0
    assert not oroot.exists()


def test_generate_interleaves_resumes_and_summarises(setup):
    cfg, qs, rroot, oroot, led = setup
    client = FakeClient(["Fee [S1]."] * 2)
    res = GR.generate(cfg, "test", qs, client, led, FakeStore(), limit=1, confirm=True,
                      retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    assert res["called"] == 2                                    # q1 for A and B
    a = [json.loads(l) for l in (oroot / "s4/test/A/answers.jsonl").read_text().splitlines()]
    assert a[0]["qid"] == "q1" and [c["sid"] for c in a[0]["context"]] == ["S1", "S2"]   # budget only
    assert a[0]["cited_chunk_ids"] == ["d-C2-0"] and a[0]["metrics"]["citation_precision"] == 1.0
    assert a[0]["timings_s"]["e2e_total"] == pytest.approx(a[0]["timings_s"]["e2e"] + 0.05)
    assert led.tokens_today() == 2 * 120
    client = FakeClient(["Fee [S2]."] * 2)                     # resume: only q2 left
    res = GR.generate(cfg, "test", qs, client, led, FakeStore(), confirm=True,
                      retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    assert res["called"] == 2 and res["est_prompt_tokens"] > 0
    s = GR.summarise(cfg, "test", out_root=oroot)
    assert s["A"]["answered"] == 2 and s["A"]["overall"]["citation_precision"]["mean"] == 0.5
    assert s["A"]["cost"]["per_1k_queries_usd"] > 0


def test_budget_and_rate_limit_stop(setup):
    cfg, qs, rroot, oroot, led = setup
    dry = GR.generate(cfg, "test", qs, None, led, FakeStore(), retrieval_root=rroot, out_root=oroot)
    led.daily_budget = dry["est_total_tokens"] // 4 + 60        # room for one call (120 real) only
    client = FakeClient(["Fee [S1]."] * 4)
    res = GR.generate(cfg, "test", qs, client, led, FakeStore(), confirm=True,
                      retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    assert res["called"] == 1 and "daily token budget" in res["stopped"]

    class RateLimitError(Exception):
        pass
    led.daily_budget = 10**9
    res = GR.generate(cfg, "test", qs, FakeClient([RateLimitError("429")]), led, FakeStore(), confirm=True,
                      retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    assert res["called"] == 0 and "RateLimitError" in res["stopped"]
    rows = (oroot / "s4/test/B/answers.jsonl")
    assert not rows.exists()                                     # a 429 writes no error row


def test_answer_entity_recall():
    q = {**Q, "entities": ["$500,000", "$1,000,000"]}
    assert GM.score_answer(q, "Multiples of $500,000, at least $1,000,000 [S1].", ctx([0]), 1.0)["answer_entity_recall"] == 1.0
    assert GM.score_answer(q, "Multiples of $500,000 [S1].", ctx([0]), 1.0)["answer_entity_recall"] == 0.5
    assert GM.score_answer(q, REFUSAL, ctx([0]), 1.0)["answer_entity_recall"] == 0.0
    assert GM.score_answer({**q, "unanswerable": True}, REFUSAL, ctx([]), None)["answer_entity_recall"] is None


def test_report_renders_and_flags_partial(setup):
    from evals.generation import report as RP
    cfg, qs, rroot, oroot, led = setup
    GR.generate(cfg, "test", qs, FakeClient(["Fee [S1].", REFUSAL, "Fee [S2].", "Fee [S1]."]), led, FakeStore(),
                confirm=True, retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    cfg.runs[1].name = RP.BASELINE                               # B plays the baseline
    (oroot / "s4/test/B").rename(oroot / f"s4/test/{RP.BASELINE}")
    retr, gen = RP.load(cfg, "test", rroot, oroot)
    md = RP.render(cfg, "test", retr, gen, questions_n=2, provenance={"generator": "m"})
    assert "PARTIAL" not in md and "Paired question set: **2**" in md and "## Latency" in md
    assert "| Answered (answerable Qs) | **1.000** [1.000, 1.000] (n 2) | **0.500**" in md   # A both, baseline refused q1
    md = RP.render(cfg, "test", retr, gen, questions_n=3, provenance={})
    assert "PARTIAL — not quotable" in md
