"""
LLM spend and token ledger (scope v2 D7), shared by every evals step that calls an LLM.

One JSON line per call in `data/spend_ledger.jsonl` (gitignored, F9): time (UTC), purpose, model
asked for / returned, prompt / completion / reasoning tokens, cost in $ from published prices
(computed even on the free tier, D7).

Guards, checked before each call with the caller's estimate of the call's tokens:
- **D7 spend caps:** $5 per runner invocation, $40 total for the initiative (override flag to
  exceed the total).
- **Groq free tier:** 200,000 tokens per UTC day per account (scope §14 Open Question 3); the
  guard stops at `daily_budget` (default 190,000, a margin for estimate error).
Prices: Groq list prices per million tokens, checked 2026-11-13 (aggregators quoting Groq's
model docs; launch price for gpt-oss-120b output was $0.75, now $0.60).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "data" / "spend_ledger.jsonl"

PRICES = {   # $ per 1M tokens: (input, output). Reasoning tokens are billed as output.
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.8-27b": (0.80, 4.00),   # R6 generator; Groq preview-model list price, checked 2026-11-24
}                                       # via price trackers (pi.dev, computeprices.com) — confirm on groq.com/pricing
RUN_CAP_USD = 5.0
TOTAL_CAP_USD = 40.0
DAILY_TOKENS = 200_000
DAILY_BUDGET = 190_000


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICES[model]
    return (prompt_tokens * pin + completion_tokens * pout) / 1e6


def utc_day(ts: str) -> str:
    return ts[:10]


@dataclass
class Ledger:
    path: Path = LEDGER
    daily_budget: int = DAILY_BUDGET
    run_cap: float = RUN_CAP_USD
    total_cap: float = TOTAL_CAP_USD
    allow_over_total: bool = False
    run_cost: float = 0.0
    run_tokens: int = 0
    _rows: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.path.exists():
            self._rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def total_cost(self) -> float:
        return sum(r["cost_usd"] for r in self._rows)

    def tokens_today(self, day: str | None = None) -> int:
        day = day or utc_day(self.now())
        return sum(r["total_tokens"] for r in self._rows if utc_day(r["ts"]) == day)

    def check(self, model: str, est_prompt: int, est_completion: int, day: str | None = None) -> None:
        """Raise BudgetExceeded if a call of this size would break a cap."""
        est_cost = cost_usd(model, est_prompt, est_completion)
        if self.run_cost + est_cost > self.run_cap:
            raise BudgetExceeded(f"run cap ${self.run_cap:.2f} (spent ${self.run_cost:.4f})")
        if not self.allow_over_total and self.total_cost() + est_cost > self.total_cap:
            raise BudgetExceeded(f"total cap ${self.total_cap:.2f} (spent ${self.total_cost():.4f})")
        if self.tokens_today(day) + est_prompt + est_completion > self.daily_budget:
            raise BudgetExceeded(f"daily token budget {self.daily_budget:,} "
                                 f"(used {self.tokens_today(day):,} today, UTC)")

    def record(self, purpose: str, model: str, returned_model: str, prompt_tokens: int,
               completion_tokens: int, reasoning_tokens: int | None = None, ts: str | None = None) -> dict:
        row = {"ts": ts or self.now(), "purpose": purpose, "model": model, "returned_model": returned_model,
               "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
               "reasoning_tokens": reasoning_tokens, "total_tokens": prompt_tokens + completion_tokens,
               "cost_usd": round(cost_usd(model, prompt_tokens, completion_tokens), 6)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self._rows.append(row)
        self.run_cost += row["cost_usd"]
        self.run_tokens += row["total_tokens"]
        return row
