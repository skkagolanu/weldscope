"""Bounded quota-aware HTTP transport; no credentials or request bodies are logged."""

import asyncio
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import re
from time import monotonic

import httpx


def retry_seconds(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(
                0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            )
        except (TypeError, ValueError, OverflowError):
            return 0.0


def duration_seconds(value):
    return sum(
        float(number) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
        for number, unit in re.findall(r"([0-9.]+)(ms|s|m|h)", value or "")
    )


class QuotaDeferred(RuntimeError):
    pass


class RequestPacer:
    def __init__(self, interval, tokens_per_minute=6000, token_budget=100000):
        self.interval = interval
        self.tokens_per_minute = tokens_per_minute
        self.token_budget = token_budget
        self.last = 0.0
        self.lock = asyncio.Lock()
        self.window = deque()
        self.reservation = None
        self.requests = 0
        self.usage_reports = 0
        self.remaining_tokens = None
        self.token_reset_at = 0.0
        self.status_counts = {}
        self.observed_token_limit = None
        self.deferred_reason = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._encoder = None

    def estimate(self, request):
        import tiktoken

        if self._encoder is None:
            self._encoder = tiktoken.get_encoding("o200k_base")
        try:
            payload = json.loads(request.content)
            prompt = json.dumps(payload.get("messages", []), ensure_ascii=False)
        except (ValueError, UnicodeError):
            prompt = request.content.decode(errors="replace")
        return len(self._encoder.encode(prompt)) + 512

    async def request(self, request):
        async with self.lock:
            estimate = self.estimate(request)
            if self.usage["total_tokens"] + estimate > self.token_budget:
                self.deferred_reason = "per_run_token_budget"
                raise QuotaDeferred("Assessment token budget reached; resume the checkpoint later")
            if estimate > self.tokens_per_minute:
                self.deferred_reason = "request_exceeds_conservative_minute_budget"
                raise QuotaDeferred("One request exceeds the configured minute budget")
            while True:
                now = monotonic()
                while self.window and self.window[0][0] <= now - 60:
                    self.window.popleft()
                delay = max(0, self.interval - (now - self.last))
                if sum(item[1] for item in self.window) + estimate > self.tokens_per_minute:
                    delay = max(delay, self.window[0][0] + 60 - now + 0.1)
                if self.remaining_tokens is not None and self.remaining_tokens < estimate:
                    delay = max(delay, self.token_reset_at - now + 0.1)
                if delay <= 0.01:
                    break
                await asyncio.sleep(delay)
            self.last = monotonic()
            self.requests += 1
            self.reservation = [self.last, estimate]
            self.window.append(self.reservation)

    async def response(self, response):
        await response.aread()
        code = str(response.status_code)
        self.status_counts[code] = self.status_counts.get(code, 0) + 1
        for header, attribute in [
            ("x-ratelimit-remaining-tokens", "remaining_tokens"),
            ("x-ratelimit-limit-tokens", "observed_token_limit"),
        ]:
            value = response.headers.get(header)
            if value and value.isdigit():
                setattr(self, attribute, int(value))
        self.token_reset_at = monotonic() + duration_seconds(
            response.headers.get("x-ratelimit-reset-tokens")
        )
        if response.status_code == 429 and self.reservation:
            self.reservation[1] = 0
        if response.is_success:
            try:
                usage = response.json().get("usage") or {}
            except (ValueError, AttributeError):
                usage = {}
            if isinstance(usage.get("total_tokens"), int):
                self.usage_reports += 1
                if self.reservation:
                    self.reservation[1] = usage["total_tokens"]
            for key in self.usage:
                if isinstance(usage.get(key), int):
                    self.usage[key] += usage[key]


class QuotaTransport(httpx.AsyncBaseTransport):
    def __init__(self, pacer, inner=None, max_attempts=6, max_retry_wait=90):
        self.pacer = pacer
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.max_attempts = max_attempts
        self.max_retry_wait = max_retry_wait

    async def handle_async_request(self, request):
        body = await request.aread()
        for attempt in range(self.max_attempts):
            fresh = httpx.Request(
                request.method,
                request.url,
                headers=request.headers,
                content=body,
                extensions=request.extensions,
            )
            await self.pacer.request(fresh)
            response = await self.inner.handle_async_request(fresh)
            await self.pacer.response(response)
            if response.status_code != 429 or attempt + 1 == self.max_attempts:
                return response
            wait = max(retry_seconds(response.headers.get("retry-after")), 1) + 0.25
            if wait > self.max_retry_wait:
                self.pacer.deferred_reason = "quota_retry_wait_exceeds_budget"
                return response
            await response.aclose()
            await asyncio.sleep(wait)
        raise AssertionError("Unreachable retry loop")

    async def aclose(self):
        await self.inner.aclose()
