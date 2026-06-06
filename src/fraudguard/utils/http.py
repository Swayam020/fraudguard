"""Async HTTP client wrapper using httpx.

Centralizes outbound HTTP calls so we get consistent timeouts, retries,
and logging. Used by any future module that hits external services.
"""

from typing import Any

import httpx


# Default timeout in seconds. httpx default is 5s for everything; we set
# it explicitly so behavior is predictable.
DEFAULT_TIMEOUT = 10.0


async def fetch_json(
    url: str,
    *,  # everything after this MUST be passed by keyword (e.g. timeout=5.0)
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """GET a URL and return its JSON body as a dict.

    Args:
        url: full URL to fetch
        timeout: seconds before giving up
        headers: optional HTTP headers (e.g. auth tokens)

    Raises:
        httpx.HTTPStatusError: if the response is 4xx or 5xx
        httpx.RequestError: on network failure
    """

    # `async with` is the async version of `with`. It guarantees the client
    # is closed (releasing connections) even if an exception is raised.
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, headers=headers)
        # raise_for_status() raises HTTPStatusError on 4xx/5xx so callers
        # don't have to check response.status_code manually.
        response.raise_for_status()
        return response.json()


async def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST a JSON payload and return the JSON response.

    Args:
        url: full URL to post to
        payload: dict that will be serialized to JSON in the request body
        timeout: seconds before giving up
        headers: optional HTTP headers
    """

    async with httpx.AsyncClient(timeout=timeout) as client:
        # `json=payload` tells httpx to serialize the dict and set
        # Content-Type: application/json automatically.
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
