import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
import tiktoken

from config import GROQ_API_KEY, GROQ_BASE_URL, MODEL_NAME, STANDIN_BASE_URL
from llm.governor import RateLimitGovernor, parse_groq_duration_to_seconds


@dataclass
class LLMCallResult:
    content: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    queue_wait_ms: float = 0.0
    work_ms: float = 0.0
    first_token_offset_ms: Optional[float] = None  # Time from call start to 1st SSE token
    retry_pre_ttft_ms: float = 0.0
    retry_post_ttft_ms: float = 0.0
    retries_used: int = 0
    status: str = "ok"  # "ok", "rate_limited", "timeout", "error"
    error_detail: str = ""
    raw_headers: Dict[str, str] = field(default_factory=dict)


class UnifiedLLMClient:
    """
    Speaks OpenAI-compatible HTTP/SSE to BOTH Real Groq and the Local Stand-In Server.
    Only `base_url` changes between 'real' and 'stand-in'.
    """
    def __init__(
        self,
        source: str = "real",
        governor: Optional[RateLimitGovernor] = None,
        max_retries: int = 2,
        naive_retry_mode: bool = False,  # Used in Section 6 to demonstrate retry storms!
    ):
        self.source = source
        self.base_url = GROQ_BASE_URL if source == "real" else STANDIN_BASE_URL
        self.api_key = GROQ_API_KEY if source == "real" else "standin-local-key"
        self.governor = governor or RateLimitGovernor()
        self.max_retries = max_retries
        self.naive_retry_mode = naive_retry_mode
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )

    def estimate_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))

    async def close(self):
        await self._client.aclose()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        step_num: int,
        remaining_budget_ms: float,
        max_tokens: int = 350,
        temperature: float = 0.0,
        stream: bool = True,
    ) -> LLMCallResult:
        res = LLMCallResult()
        prompt_text = " ".join(m.get("content", "") for m in messages)
        est_in_tokens = self.estimate_tokens(prompt_text) + 12

        # Fast-fail if daily quota is already known to be exhausted
        if self.governor.is_daily_exhausted:
            res.status = "rate_limited"
            res.error_detail = "daily_budget_exhausted"
            return res

        # 1. Acquire concurrency & rate-limit slot
        try:
            wait_ms = await self.governor.acquire_slot(remaining_budget_ms, step_num=step_num)
            res.queue_wait_ms += wait_ms
        except PermissionError as e:
            res.status = "rate_limited"
            res.error_detail = str(e)
            return res
        except asyncio.TimeoutError:
            res.status = "timeout"
            res.error_detail = "queue_wait_exceeded_patience"
            return res

        t_work_start = time.perf_counter()
        attempt = 0

        try:
            while attempt <= self.max_retries:
                elapsed_so_far_ms = res.queue_wait_ms + (time.perf_counter() - t_work_start) * 1000.0
                time_left_sec = (remaining_budget_ms - elapsed_so_far_ms) / 1000.0
                if time_left_sec <= 0.15:
                    res.status = "timeout"
                    res.error_detail = "deadline_expired_before_attempt"
                    break

                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": stream,
                }

                got_first_token_this_attempt = False
                t_attempt_start = time.perf_counter()

                try:
                    if not stream:
                        resp = await self._client.post(
                            "/chat/completions", json=payload, timeout=time_left_sec
                        )
                        res.raw_headers = dict(resp.headers)
                        self.governor.update_from_headers(
                            res.raw_headers, resp.status_code, resp.text if resp.status_code != 200 else ""
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            res.content = data["choices"][0]["message"]["content"]
                            usage = data.get("usage", {})
                            res.tokens_in += usage.get("prompt_tokens", est_in_tokens)
                            res.tokens_out += usage.get("completion_tokens", self.estimate_tokens(res.content))
                            res.first_token_offset_ms = (time.perf_counter() - t_attempt_start) * 1000.0
                            res.status = "ok"
                            break
                        else:
                            err_status = resp.status_code
                            err_body = resp.text
                    else:
                        # Streaming SSE mode for exact first_token_ms measurement
                        chunks_out = []
                        async with self._client.stream(
                            "POST", "/chat/completions", json=payload, timeout=time_left_sec
                        ) as resp:
                            res.raw_headers = dict(resp.headers)
                            if resp.status_code != 200:
                                err_bytes = await resp.aread()
                                err_body = err_bytes.decode("utf-8", errors="ignore")
                                err_status = resp.status_code
                                self.governor.update_from_headers(res.raw_headers, err_status, err_body)
                            else:
                                self.governor.update_from_headers(res.raw_headers, 200)
                                async for line in resp.aiter_lines():
                                    if not line or not line.startswith("data: "):
                                        continue
                                    data_str = line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        packet = json.loads(data_str)
                                    except json.JSONDecodeError:
                                        continue

                                    # Capture token usage if sent in final x_groq or usage chunk
                                    if "x_groq" in packet and "usage" in packet["x_groq"]:
                                        u = packet["x_groq"]["usage"]
                                        res.tokens_in += u.get("prompt_tokens", 0)
                                        res.tokens_out += u.get("completion_tokens", 0)
                                    elif "usage" in packet and packet["usage"]:
                                        u = packet["usage"]
                                        res.tokens_in += u.get("prompt_tokens", 0)
                                        res.tokens_out += u.get("completion_tokens", 0)

                                    choices = packet.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {}).get("content")
                                        if delta:
                                            if not got_first_token_this_attempt:
                                                got_first_token_this_attempt = True
                                                res.first_token_offset_ms = (
                                                    time.perf_counter() - t_attempt_start
                                                ) * 1000.0
                                            chunks_out.append(delta)

                                res.content = "".join(chunks_out)
                                if res.tokens_in == 0:
                                    res.tokens_in += est_in_tokens
                                if res.tokens_out == 0:
                                    res.tokens_out += self.estimate_tokens(res.content)
                                res.status = "ok"
                                break

                except (httpx.TimeoutException, asyncio.TimeoutError):
                    err_status = 408
                    err_body = "socket_timeout"
                except httpx.HTTPError as e:
                    err_status = 503
                    err_body = str(e)

                # --- Handle non-200 or Timeout (Retry Policy for Section 6) ---
                if err_status == 429:
                    res.status = "rate_limited"
                    res.error_detail = err_body[:200]
                    if self.governor.is_daily_exhausted and not self.naive_retry_mode:
                        # Never retry a daily quota exhaustion!
                        break
                elif err_status == 408:
                    res.status = "timeout"
                    res.error_detail = "http_timeout"
                else:
                    res.status = "error"
                    res.error_detail = f"HTTP_{err_status}: {err_body[:120]}"

                if attempt >= self.max_retries:
                    break

                # Compute retry sleep
                if self.naive_retry_mode:
                    # Naive immediate retry (0.1s) - used to demonstrate retry storm in Section 6
                    backoff_sec = 0.1
                else:
                    retry_hdr = res.raw_headers.get("retry-after", "")
                    hdr_wait = parse_groq_duration_to_seconds(retry_hdr)
                    base_backoff = hdr_wait if hdr_wait > 0 else (0.6 * (2 ** attempt))
                    backoff_sec = base_backoff + random.uniform(0.05, 0.25)

                    # Budget-aware check: do not sleep & retry if backoff + 1.0s work > remaining patience!
                    elapsed_ms = res.queue_wait_ms + (time.perf_counter() - t_work_start) * 1000.0
                    if (elapsed_ms + (backoff_sec + 0.8) * 1000.0) >= remaining_budget_ms:
                        res.status = "timeout"
                        res.error_detail = "retry_backoff_exceeds_patience"
                        break

                attempt += 1
                res.retries_used += 1
                t_sleep0 = time.perf_counter()
                await asyncio.sleep(backoff_sec)
                slept_ms = (time.perf_counter() - t_sleep0) * 1000.0

                if got_first_token_this_attempt:
                    res.retry_post_ttft_ms += slept_ms
                else:
                    res.retry_pre_ttft_ms += slept_ms

        finally:
            total_loop_ms = (time.perf_counter() - t_work_start) * 1000.0
            res.work_ms = max(0.0, total_loop_ms - res.retry_pre_ttft_ms - res.retry_post_ttft_ms)
            self.governor.release_slot()

        return res