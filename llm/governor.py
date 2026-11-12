import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import LOW_TPM_THRESHOLD_RATIO, USER_PATIENCE_BUDGET_MS


def parse_groq_duration_to_seconds(val: str) -> float:
    """
    Parses Groq's reset strings like '12.34s', '1m20.5s', '450ms', or '2h10m' into seconds.
    """
    if not val:
        return 0.0
    val = str(val).strip()
    try:
        return float(val)
    except ValueError:
        pass

    total_sec = 0.0
    # Match hours, minutes, milliseconds, seconds
    hr = re.search(r"(\d+(?:\.\d+)?)h", val)
    mn = re.search(r"(\d+(?:\.\d+)?)m(?!s)", val)
    ms = re.search(r"(\d+(?:\.\d+)?)ms", val)
    sc = re.search(r"(\d+(?:\.\d+)?)s", val)

    if hr:
        total_sec += float(hr.group(1)) * 3600.0
    if mn:
        total_sec += float(mn.group(1)) * 60.0
    if ms:
        total_sec += float(ms.group(1)) / 1000.0
    if sc:
        total_sec += float(sc.group(1))
    return total_sec


@dataclass
class RateLimitGovernor:
    """
    Tracks real-time rate-limit headers from Groq / Stand-in and enforces
    priority queuing, adaptive degradation, and fast-shedding.
    """
    max_concurrency: int = 6
    limit_rpd: int = 14400
    remaining_rpd: int = 14400
    limit_tpm: int = 6000
    remaining_tpm: int = 6000
    reset_tpm_sec: float = 0.0
    reset_rpd_sec: float = 0.0
    retry_after_until: float = 0.0  # monotonic timestamp until which 429 cooldown holds
    daily_exhausted_flag: bool = False

    # Queue metrics for Table 2b ("Peak queue length")
    current_queue_length: int = 0
    peak_queue_length: int = 0

    _sem: asyncio.Semaphore = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self):
        self._sem = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()

    def reset_stats(self):
        self.current_queue_length = 0
        self.peak_queue_length = 0
        self.retry_after_until = 0.0
        self.daily_exhausted_flag = False
        self.remaining_tpm = self.limit_tpm
        self.remaining_rpd = self.limit_rpd
        self._sem = asyncio.Semaphore(self.max_concurrency)

    def update_from_headers(self, headers: Dict[str, str], status_code: int = 200, error_body: str = "") -> None:
        h = {k.lower(): v for k, v in headers.items()}

        if "x-ratelimit-limit-requests" in h:
            self.limit_rpd = int(float(h["x-ratelimit-limit-requests"]))
        if "x-ratelimit-remaining-requests" in h:
            self.remaining_rpd = int(float(h["x-ratelimit-remaining-requests"]))
        if "x-ratelimit-limit-tokens" in h:
            self.limit_tpm = int(float(h["x-ratelimit-limit-tokens"]))
        if "x-ratelimit-remaining-tokens" in h:
            self.remaining_tpm = int(float(h["x-ratelimit-remaining-tokens"]))
        if "x-ratelimit-reset-tokens" in h:
            self.reset_tpm_sec = parse_groq_duration_to_seconds(h["x-ratelimit-reset-tokens"])
        if "x-ratelimit-reset-requests" in h:
            self.reset_rpd_sec = parse_groq_duration_to_seconds(h["x-ratelimit-reset-requests"])

        if status_code == 429:
            retry_sec = parse_groq_duration_to_seconds(h.get("retry-after", "2.0"))
            if retry_sec <= 0:
                retry_sec = max(self.reset_tpm_sec, 1.5)
            self.retry_after_until = max(self.retry_after_until, time.monotonic() + retry_sec)

            # Check if daily limit (RPD / TPD) was hit
            if self.remaining_rpd <= 0 or "per day" in error_body.lower() or "rpd" in error_body.lower() or "tpd" in error_body.lower():
                self.daily_exhausted_flag = True

    @property
    def is_daily_exhausted(self) -> bool:
        return self.daily_exhausted_flag or self.remaining_rpd <= 0

    @property
    def should_degrade(self) -> bool:
        """
        Triggers low-token degradation mode when < 20% of per-minute tokens remain
        or when queue depth builds up.
        """
        if self.limit_tpm > 0 and (self.remaining_tpm / self.limit_tpm) < LOW_TPM_THRESHOLD_RATIO:
            return True
        if self.current_queue_length >= 4:
            return True
        return False

    async def acquire_slot(self, remaining_budget_ms: float, step_num: int) -> float:
        """
        Waits for any active 429 cooldown and concurrency slot.
        Returns queue_wait_ms. Raises TimeoutError if wait exceeds remaining_budget_ms.
        """
        t0 = time.perf_counter()
        self.current_queue_length += 1
        if self.current_queue_length > self.peak_queue_length:
            self.peak_queue_length = self.current_queue_length

        try:
            # 1. Admission check: if a global 429 cooldown is active and longer than our patience budget, shed immediately!
            now = time.monotonic()
            cooldown_sec = max(0.0, self.retry_after_until - now)
            time_left_sec = remaining_budget_ms / 1000.0

            # If required cooldown exceeds patience budget, wait until patience budget expires, then time out
            if cooldown_sec > time_left_sec:
                if time_left_sec > 0:
                    await asyncio.sleep(time_left_sec)
                raise asyncio.TimeoutError(
                    f"patience_exceeded: Cooldown ({cooldown_sec:.2f}s) exceeds budget ({time_left_sec:.2f}s)"
                )

            # If cooldown is active but within budget, wait for cooldown
            if cooldown_sec > 0:
                await asyncio.sleep(cooldown_sec)

            # 2. Wait for concurrency semaphore within remaining patience budget
            elapsed_sec = time.perf_counter() - t0
            sem_timeout = max(0.01, (remaining_budget_ms / 1000.0) - elapsed_sec)
            await asyncio.wait_for(self._sem.acquire(), timeout=sem_timeout)

            wait_ms = (time.perf_counter() - t0) * 1000.0
            return wait_ms
        finally:
            self.current_queue_length -= 1

    def release_slot(self) -> None:
        self._sem.release()