"""
Native Groq streaming with client-side timings (scope v2 §8.5, §9.5).

Every received stream chunk is timestamped with `time.perf_counter()`. Timings (seconds, from the
moment the request is sent):
- ttft:        first non-empty *answer* content (text inside a `<think>` block or in a separate
               `reasoning` delta is not answer content; `ttft_any` is the first token of either kind)
- ttfs:        first sentence end in the answer: `.`, `?`, `!` followed by whitespace or end of
               stream; a `.` between two digits (a decimal point) doesn't count
- inter_token: mean gap between answer content chunks after the first (Groq sends ~1 token per
               chunk; the chunk count is recorded so this can be checked)
- tokens_per_s: provider completion tokens ÷ (last content time − first content time)
- e2e:         request sent → stream finished (generation only; the runner adds retrieval time)
Token counts come from the provider's usage block on the final chunk (`x_groq.usage` or `usage`).
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_SENT_END = re.compile(r"[.?!](?=\s)")      # needs the following whitespace: "3." + "75" is not an end


def strip_think(text: str) -> str:
    """Remove `<think>…</think>` blocks (an unclosed block runs to the end)."""
    return _THINK.sub("", text).strip()


def first_sentence_end(text: str, final: bool = False) -> int | None:
    """Index just past the first sentence-ending mark, or None. `3.75` is not a sentence end. A mark at
    the very end of the text counts only when `final` (end of stream): mid-stream it may be a decimal
    point whose digits haven't arrived yet."""
    m = _SENT_END.search(text)
    if m:
        return m.end()
    s = text.rstrip()
    return len(s) if final and s and s[-1] in ".?!" else None


@dataclass
class StreamResult:
    raw: str                                  # all content as streamed (may include <think>)
    answer: str                               # raw without think blocks
    reasoning_chars: int                      # chars of hidden reasoning (think block / reasoning delta)
    events: list[tuple[float, str, str]] = field(default_factory=list)   # (t, kind, text), kind content|reasoning
    returned_model: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    timings: dict = field(default_factory=dict)

    def to_dict(self, keep_events: bool = False) -> dict:
        d = asdict(self)
        if not keep_events:
            d.pop("events")
        return d


def compute_timings(events: list[tuple[float, str, str]], t_end: float, completion_tokens: int | None) -> dict:
    """Timings from (seconds since send, kind, text) events. Pure function (tested without a network)."""
    answer_times: list[float] = []
    ttft_any = ttfs = None
    in_think = False
    answer_so_far = ""
    for t, kind, text in events:
        if not text:
            continue
        if ttft_any is None:
            ttft_any = t
        if kind != "content":
            continue
        # track <think> state across chunk boundaries; keep only answer text
        visible = ""
        rest = text
        while rest:
            if in_think:
                j = rest.find("</think>")
                if j < 0:
                    rest = ""
                else:
                    in_think, rest = False, rest[j + len("</think>"):]
            else:
                j = rest.find("<think>")
                if j < 0:
                    visible, rest = visible + rest, ""
                else:
                    visible, in_think, rest = visible + rest[:j], True, rest[j + len("<think>"):]
        if visible.strip() or (visible and answer_times):
            answer_times.append(t)
            answer_so_far += visible
            if ttfs is None and first_sentence_end(answer_so_far) is not None:
                ttfs = t
    if ttfs is None and answer_times and first_sentence_end(answer_so_far, final=True) is not None:
        ttfs = answer_times[-1]          # the only sentence end is the end of the stream
    gaps = [b - a for a, b in zip(answer_times, answer_times[1:])]
    span = (answer_times[-1] - answer_times[0]) if len(answer_times) > 1 else None
    return {
        "ttft": answer_times[0] if answer_times else None,
        "ttft_any": ttft_any,
        "ttfs": ttfs,
        "inter_token": sum(gaps) / len(gaps) if gaps else None,
        "tokens_per_s": (completion_tokens / span) if (completion_tokens and span) else None,
        "e2e": t_end,
        "content_chunks": len(answer_times),
    }


def _usage(chunk):
    xg = getattr(chunk, "x_groq", None)
    u = getattr(xg, "usage", None) if xg is not None else None
    return u or getattr(chunk, "usage", None)


def stream_call(client, model: str, prompt: str, temperature: float, max_tokens: int,
                clock=time.perf_counter, extra: dict | None = None) -> StreamResult:
    """One streamed chat completion, single user message (as production `stream_tokens`)."""
    t0 = clock()
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
        max_completion_tokens=max_tokens, stream=True, **(extra or {}))
    events: list[tuple[float, str, str]] = []
    raw, reasoning = [], []
    res = StreamResult("", "", 0)
    for chunk in stream:
        t = clock() - t0
        if getattr(chunk, "model", None):
            res.returned_model = chunk.model
        u = _usage(chunk)
        if u is not None:
            res.prompt_tokens = getattr(u, "prompt_tokens", None)
            res.completion_tokens = getattr(u, "completion_tokens", None)
            det = getattr(u, "completion_tokens_details", None)
            res.reasoning_tokens = getattr(det, "reasoning_tokens", None) if det else None
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            res.finish_reason = ch.finish_reason
        delta = ch.delta
        r = getattr(delta, "reasoning", None)
        if r:
            reasoning.append(r)
            events.append((t, "reasoning", r))
        c = getattr(delta, "content", None)
        if c:
            raw.append(c)
            events.append((t, "content", c))
    t_end = clock() - t0
    res.raw = "".join(raw)
    res.answer = strip_think(res.raw)
    think = "".join(m.group(0) for m in _THINK.finditer(res.raw))
    res.reasoning_chars = len("".join(reasoning)) + len(think)
    res.events = events
    res.timings = compute_timings(events, t_end, res.completion_tokens)
    return res
