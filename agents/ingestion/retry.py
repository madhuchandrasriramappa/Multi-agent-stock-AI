from __future__ import annotations

import functools
import time
from typing import Any, Callable, Tuple, Type, TypeVar

import structlog

T = TypeVar("T")
_logger = structlog.get_logger("retry")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    reraise_on: Tuple[Type[Exception], ...] = (),
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts:    total call attempts including the first
        base_delay:      seconds to wait after the first failure
        backoff_factor:  multiplier applied to delay on each subsequent retry
        reraise_on:      exception types that skip retry and propagate immediately
                         (e.g. ValueError for bad input that will never succeed)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except reraise_on:
                    # Caller said: don't retry these, just propagate
                    raise

                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        _logger.warning(
                            "retrying",
                            func=func.__name__,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            delay_seconds=round(delay, 2),
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )
                        time.sleep(delay)
                    else:
                        _logger.error(
                            "all_retries_exhausted",
                            func=func.__name__,
                            max_attempts=max_attempts,
                            error=str(exc),
                            error_type=type(exc).__name__,
                        )

            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
