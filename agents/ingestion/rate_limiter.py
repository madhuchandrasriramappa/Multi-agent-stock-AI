from __future__ import annotations

import threading
import time


class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Enforces a minimum interval between calls so a client never exceeds
    `calls_per_minute` requests.  When the limit would be exceeded the
    calling thread sleeps for the remaining interval.

    Example:
        limiter = RateLimiter(calls_per_minute=15)
        for coin in coins:
            limiter.acquire()          # blocks if needed
            response = requests.get(url)
    """

    def __init__(self, calls_per_minute: int) -> None:
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be > 0")
        self._min_interval: float = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0

    def acquire(self) -> None:
        """Block the calling thread until it is safe to make the next call."""
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.monotonic()

    @property
    def min_interval_seconds(self) -> float:
        return self._min_interval
