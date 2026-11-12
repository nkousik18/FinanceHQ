"""
Minimal SEC EDGAR client that follows the SEC fair-access rules (scope v2 F11).

- User-Agent from the SEC_USER_AGENT environment variable (or the repo .env);
  refuses to run without one. Never hard-coded.
- At most ~8 requests/second (SEC limit is 10).
- Retries on 429/5xx with backoff.
- Optional on-disk cache under data/sec_cache/ so re-runs don't re-download.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("evals.corpus.sec")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "data" / "sec_cache"
MIN_INTERVAL_S = 0.125            # 8 req/s
RETRY_STATUS = {429, 500, 502, 503, 504}
FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"


class SECConfigError(RuntimeError):
    pass


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip().strip('"')
    if not ua:
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("SEC_USER_AGENT="):
                    ua = line.split("=", 1)[1].strip().strip('"')
    if not ua or "@" not in ua:
        raise SECConfigError(
            "Set SEC_USER_AGENT='Your Name you@example.com' in the environment or .env "
            "(SEC fair-access policy requires a contact in the User-Agent)."
        )
    return ua


class SECClient:
    def __init__(self, cache_dir: Optional[Path] = DEFAULT_CACHE, max_retries: int = 4):
        self.ua = _user_agent()
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self._last = 0.0
        self.requests_made = 0

    # -- low level -------------------------------------------------------
    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _cache_path(self, url: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self.cache_dir / h[:2] / h

    def get_bytes(self, url: str, use_cache: bool = True) -> bytes:
        cp = self._cache_path(url) if use_cache else None
        if cp and cp.exists():
            return cp.read_bytes()

        for attempt in range(self.max_retries + 1):
            self._throttle()
            req = urllib.request.Request(url, headers={
                "User-Agent": self.ua,
                "Accept-Encoding": "identity",
            })
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                self.requests_made += 1
                break
            except urllib.error.HTTPError as e:
                self.requests_made += 1
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    delay = 2 ** attempt
                    log.warning("sec_retry status=%s attempt=%d delay=%ss url=%s", e.code, attempt, delay, url)
                    time.sleep(delay)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise

        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(data)
        return data

    def get_json(self, url: str, use_cache: bool = True) -> Any:
        return json.loads(self.get_bytes(url, use_cache=use_cache))

    # -- EDGAR helpers ---------------------------------------------------
    def full_text_search(self, query: str, *, forms: str = "8-K,10-Q,10-K",
                         startdt: str = "2019-01-01", enddt: str = "2025-12-31",
                         ciks: Optional[str] = None, offset: int = 0) -> dict:
        params = {"q": query, "dateRange": "custom", "startdt": startdt,
                  "enddt": enddt, "forms": forms, "from": offset}
        if ciks:
            params["ciks"] = ciks
        return self.get_json(f"{FULL_TEXT_SEARCH}?{urllib.parse.urlencode(params)}")

    @staticmethod
    def archive_url(cik: str, adsh: str, filename: str) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/{filename}"

    def filing_index(self, cik: str, adsh: str) -> dict:
        return self.get_json(
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{adsh.replace('-', '')}/index.json"
        )
