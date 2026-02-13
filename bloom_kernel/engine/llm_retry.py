"""LLM call retry with exponential backoff.

Differentiates transient errors (429/502/503/timeout) from permanent
errors (401/400) and retries only transient failures.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

import httpx
from loguru import logger

T = TypeVar("T")

# HTTP status codes that are transient and worth retrying
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that are permanent failures — no retry
_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}


def _is_transient(exc: Exception) -> bool:
    """Check if an exception represents a transient (retryable) error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS_CODES
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError)):
        return True
    return False


async def retry_llm_call(
    fn: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    **kwargs,
) -> T:
    """Call an async function with exponential backoff retry on transient errors.

    Args:
        fn: Async callable to retry
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay cap in seconds
        *args, **kwargs: Passed to fn

    Returns:
        The result of fn(*args, **kwargs)

    Raises:
        The last exception if all retries are exhausted, or immediately
        for permanent errors.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not _is_transient(exc):
                logger.warning(f"LLM call permanent error (no retry): {exc}")
                raise

            if attempt >= max_retries:
                logger.error(
                    f"LLM call failed after {max_retries + 1} attempts: {exc}"
                )
                raise

            # Exponential backoff with 10% jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * 0.1 * random.random()
            wait = delay + jitter

            logger.warning(
                f"LLM call transient error (attempt {attempt + 1}/{max_retries + 1}), "
                f"retrying in {wait:.1f}s: {exc}"
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]
