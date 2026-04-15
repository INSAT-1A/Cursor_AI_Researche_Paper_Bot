from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests


def parse_response_json_dict(
    response: requests.Response,
    logger: logging.Logger,
    context: str = "response",
) -> Optional[dict[str, Any]]:
    """Parse JSON body; return a dict or None if invalid or not an object."""
    try:
        data = response.json()
    except ValueError as exc:
        logger.warning("%s: invalid JSON (%s)", context, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("%s: expected JSON object, got %s", context, type(data).__name__)
        return None
    return data


def request_with_retries(
    *,
    logger: logging.Logger,
    method: str,
    url: str,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout: int = 15,
    retries: int = 2,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    context: str = "request",
) -> requests.Response:
    """
    Perform HTTP request with simple retry/backoff logic.
    Raises requests.RequestException on final failure.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
            if resp.status_code in retry_statuses and attempt < retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = float(retry_after)
                else:
                    sleep_s = float(1 + attempt)
                logger.warning(
                    "%s: status=%s attempt=%s/%s retrying in %.1fs",
                    context,
                    resp.status_code,
                    attempt + 1,
                    retries + 1,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = float(1 + attempt)
            logger.warning(
                "%s: error=%s attempt=%s/%s retrying in %.1fs",
                context,
                exc,
                attempt + 1,
                retries + 1,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise requests.RequestException(f"{context} failed after retries: {last_exc}") from last_exc
