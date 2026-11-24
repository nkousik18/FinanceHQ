"""R6 judge: JSON validation, scores, rule grades, resumable runner. Fake client, no network."""
import json
from types import SimpleNamespace as NS

import pytest

from evals.generation import judge as J
from evals.generation import run as GR
from evals.generation.prompt import REFUSAL
from evals.llm.ledger import Ledger
from evals.tests.test_generation import FakeClient, FakeStore, Q, ctx, write_retrieval

GOOD = {"claims": [{"claim": "fee is 2%", "supported": True}, {"claim": "due monthly", "supported": False}],
        "correctness": "partial", "key_points": [True, False], "relevance": 4, "rationale": "ok"}


def test_parse_valid_and_scores():
    g = J.parse("noise " + json.dumps(GOOD) + " trailing", 2)
    s = J.scores(g)
    assert s == {"faithfulness": 0.5, "hallucinations": 1, "correctness": 0.5, "completeness": 0.5, "relevance": 4}


@pytest.mark.parametrize("bad,why", [
    ({**GOOD, "correctness": "mostly"}, "correctness"),
    ({**GOOD, "relevance": 6}, "relevance"),
    ({**GOOD, "relevance": True}, "relevance"),
    ({**GOOD, "key_points": [True]}, "key_points"),
    ({**GOOD, "claims": [{"claim": "x", "supported": "yes"}]}, "claims"),
])
def test_parse_rejects_off_schema(bad, why):
    with pytest.raises(J.JudgeParseError, match=why):
        J.parse(json.dumps(bad), 2)
    with pytest.raises(J.JudgeParseError):
        J.parse("no json here", 0)


def test_no_claims_faithfulness_undefined():
    s = J.scores({"claims": [], "correctness": "correct", "key_points": [], "relevance": 5})
    assert s["faithfulness"] is None and s["completeness"] is None and s["hallucinations"] == 0


def test_rule_grades():
    un = {**Q, "unanswerable": True, "key_points": ["x"]}
    g = J.rule_grade(un, {"metrics": {"refused": True}})
    assert g["correctness"] == "correct" and J._scores_rule(g)["relevance"] == 5
    g = J.rule_grade({**Q, "key_points": ["a", "b"]}, {"metrics": {"refused": True}})
    s = J._scores_rule(g)
    assert g["correctness"] == "wrong" and s["completeness"] == 0 and s["relevance"] is None
    assert J.rule_grade(Q, {"metrics": {"refused": False}}) is None


def test_messages_carry_context_reference_and_answer():
    q = {**Q, "question": "What fee?", "reference_answer": "2% per annum.", "key_points": ["2%"]}
    m = J.messages(q, "Fee is 2% [S1].", ctx([0]))
    u = m[1]["content"]
    assert "[S1 | d | Page 1]" in u and "2% per annum." in u and "- 2%" in u and "Fee is 2% [S1]." in u


class FakeJudge:
    def __init__(self, contents):
        self.contents, self.calls = list(contents), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        c = self.contents.pop(0)
        return NS(model=J.JUDGE_MODEL, choices=[NS(message=NS(content=c), finish_reason="stop")],
                  usage=NS(prompt_tokens=1000, completion_tokens=300,
                           completion_tokens_details=NS(reasoning_tokens=200)))


@pytest.fixture
def answered(tmp_path):
    rroot, oroot = tmp_path / "r4", tmp_path / "r6"
    write_retrieval(rroot, "S2", "A", ["q1", "q2", "q3"])
    cfg = GR.GenConfig("S4", "qwen/qwen3.8-27b", 0.1, 1024, 10**9, [GR.GenRun("A", {"stage": "S2", "run": "A"})])
    qs = {q: {**Q, "qid": q, "question": f"{q}?", "reference_answer": "2%.", "key_points": ["2%", "monthly"]}
          for q in ("q1", "q2", "q3")}
    led = Ledger(path=tmp_path / "ledger.jsonl")
    GR.generate(cfg, "test", qs, FakeClient(["Fee 2% [S1].", REFUSAL, "Fee 3% [S1]."]), led, FakeStore(),
                confirm=True, retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    return cfg, qs, rroot, oroot, led


def test_judge_run_rules_calls_parse_errors_and_resume(answered):
    cfg, qs, rroot, oroot, led = answered
    kw = dict(retrieval_root=rroot, out_root=oroot, sleep=lambda s: None)
    dry = J.judge_run(cfg, "test", qs, None, led, FakeStore(), **kw)
    assert dry["calls_todo"] == 2 and dry["rule_graded"] == 1 and dry["called"] == 0
    client = FakeJudge([json.dumps(GOOD), "not json"])
    res = J.judge_run(cfg, "test", qs, client, led, FakeStore(), confirm=True, **kw)
    assert res["called"] == 2 and res["parse_failures"] == 1
    kw_call = client.calls[0]
    assert kw_call["model"] == J.JUDGE_MODEL and kw_call["temperature"] == 0 and kw_call["reasoning_effort"] == "low"
    rows = GR.read_jsonl(oroot / "s4/test/A/judgments.jsonl")
    by = {(r["qid"], r["judged_by"]): r for r in rows}
    assert by[("q2", "rule")]["scores"]["correctness"] == 0.0           # answerable refusal = wrong
    assert by[("q1", "llm")]["scores"]["faithfulness"] == 0.5 and by[("q1", "llm")]["reasoning_tokens"] == 200
    assert by[("q3", "llm")]["grade"] is None and "parse_error" in by[("q3", "llm")]
    res = J.judge_run(cfg, "test", qs, FakeJudge([json.dumps(GOOD)]), led, FakeStore(), confirm=True, **kw)
    assert res["called"] == 1 and res["rule_graded"] == 0               # only the parse failure retried
    assert res["est_completion_tokens"] == 300                          # measured, not the 900 guess
