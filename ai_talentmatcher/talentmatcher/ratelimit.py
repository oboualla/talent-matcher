"""A process-wide, thread-safe requests-per-minute limiter.

Every LLM call (background extraction and interactive matching) goes through the
same limiter, because the provider's quota applies to the API key, not to a job.
"""

import threading
import time
from collections import deque
from typing import Callable, Optional

WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = requests_per_minute  # <= 0 means unlimited
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()  # timestamps of requests in the last minute
        self._paused_until = 0.0  # global cooldown requested by the provider (HTTP 429)

    def _wait_needed(self, now: float) -> float:
        """Seconds to wait before a request may be sent (0 = go). Caller holds the lock."""
        while self._sent and self._sent[0] <= now - WINDOW_SECONDS:
            self._sent.popleft()
        if now < self._paused_until:
            return self._paused_until - now
        if self.rpm > 0 and len(self._sent) >= self.rpm:
            return self._sent[0] + WINDOW_SECONDS - now
        return 0.0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Block until a request slot is free. Returns False if `timeout` expires first."""
        deadline = None if timeout is None else self._clock() + timeout
        while True:
            with self._lock:
                now = self._clock()
                wait = self._wait_needed(now)
                if wait <= 0:
                    self._sent.append(now)
                    return True
            if deadline is not None and now + wait > deadline:
                return False
            # Sleep in small steps so a pause() or a freed slot is noticed quickly.
            self._sleep(min(wait, 1.0))

    def pause(self, seconds: float) -> None:
        """Stop all requests for `seconds` (used when the provider says 'rate limited')."""
        with self._lock:
            self._paused_until = max(self._paused_until, self._clock() + seconds)

    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._paused_until - self._clock())

    def snapshot(self) -> dict:
        with self._lock:
            now = self._clock()
            self._wait_needed(now)  # drops expired timestamps
            return {
                "requests_per_minute": self.rpm,
                "used_last_minute": len(self._sent),
                "cooldown_seconds": round(max(0.0, self._paused_until - now), 1),
            }
