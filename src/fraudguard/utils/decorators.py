"""Reusable function decorators.

`@timed` logs execution time. `@retry` retries on failure with backoff.
Both are used across the API and data-loading layers.
"""

import functools
import time
from typing import Any, Callable, Type


def timed(func: Callable) -> Callable:
    """Log how long the decorated function takes to run.

    Usage:
        @timed
        def load_data(): ...
    """

    # functools.wraps copies the original function's name and docstring
    # onto the wrapper. Without it, `load_data.__name__` would be 'wrapper'.
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()  # high-resolution timer
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timed] {func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 0.5,
    exceptions: tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry the decorated function on failure.

    Args:
        max_attempts: total tries before giving up (including the first)
        delay_seconds: seconds to wait between attempts
        exceptions: tuple of exception classes that trigger a retry;
            anything else propagates immediately

    Usage:
        @retry(max_attempts=5, delay_seconds=1.0)
        def flaky_call(): ...
    """

    # This is a "decorator factory" — retry() returns the actual decorator.
    # That's why we use it as @retry(...) with parens, not bare @retry.
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    print(
                        f"[retry] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {exc}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay_seconds)
            # All attempts failed — re-raise the last exception
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
