"""OpenRouter LLM client — thin HTTP wrapper with retry logic.

Implements retry strategies for rate limits (429), server errors (5xx),
and network timeouts.  Holds a single :class:`httpx.Client` per instance.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Unrecoverable error from the OpenRouter LLM API."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin wrapper around OpenRouter's ``/chat/completions`` endpoint.

    Parameters
    ----------
    base_url:
        Base URL for the OpenRouter API (e.g. ``https://openrouter.ai/api/v1``).
    api_key:
        OpenRouter API key sent as ``Authorization: Bearer``.
    timeout:
        Request timeout in seconds applied to connect, read, write, and pool.
        Defaults to 120 as specified in ``pipeline.llm_request_timeout_seconds``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(float(timeout)),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ai-news-pipeline",
                "X-Title": "AI News Pipeline",
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(
        self,
        model_id: str,
        temperature: float,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send a chat completion request and return the assistant's response text.

        Implements per-error-type retry:

        * HTTP **429** — exponential backoff (base 10 s, capped at 60 s, up to **5**
          retries).
        * HTTP **5xx** — exponential backoff (base 5 s, capped at 60 s, up to **3**
          retries).
        * Network **timeout** — immediate retry, up to **3** retries.
        * HTTP **4xx** (other than 429) — no retry, raises :exc:`LLMClientError`
          immediately.

        Parameters
        ----------
        model_id:
            OpenRouter model identifier (e.g. ``deepseek/deepseek-v4-pro``).
        temperature:
            Sampling temperature in ``[0.0, 2.0]``.
        system_prompt:
            The system message sent to the model.
        user_prompt:
            The user message sent to the model.

        Returns
        -------
        str
            The text content from ``choices[0].message.content``.

        Raises
        ------
        LLMClientError
            If all retries are exhausted, the server returns an unrecoverable error,
            or the response cannot be parsed.
        """
        payload = {
            "model": model_id,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        retry_429: int = 0
        retry_5xx: int = 0
        retry_timeout: int = 0

        MAX_RETRIES_429 = 5
        MAX_RETRIES_5XX = 3
        MAX_RETRIES_TIMEOUT = 3

        while True:
            try:
                start = time.monotonic()
                resp = self._client.post("/chat/completions", json=payload)
                elapsed = time.monotonic() - start
            except httpx.TimeoutException:
                retry_timeout += 1
                if retry_timeout > MAX_RETRIES_TIMEOUT:
                    raise LLMClientError("Request timed out after 3 retries")
                logger.warning(
                    "LLM request timeout — retrying (attempt %d/%d)",
                    retry_timeout,
                    MAX_RETRIES_TIMEOUT,
                )
                continue
            except Exception as exc:
                logger.error("Unexpected error during LLM request: %s", exc)
                raise LLMClientError(f"Unexpected error: {exc}") from exc

            # --- Success -------------------------------------------------------
            if resp.status_code == 200:
                return self._parse_response(resp, model_id, elapsed)

            # --- Rate limit (retryable) ----------------------------------------
            if resp.status_code == 429:
                retry_429 += 1
                if retry_429 > MAX_RETRIES_429:
                    raise LLMClientError("Rate limit (429) exceeded after 5 retries")
                delay = min(10 * (2 ** (retry_429 - 1)), 60)
                logger.warning(
                    "LLM rate limit (429) — retrying in %ds (attempt %d/%d)",
                    delay,
                    retry_429,
                    MAX_RETRIES_429,
                )
                time.sleep(delay)
                continue

            # --- Server error (retryable) --------------------------------------
            if 500 <= resp.status_code <= 599:
                retry_5xx += 1
                if retry_5xx > MAX_RETRIES_5XX:
                    raise LLMClientError(
                        f"Server error ({resp.status_code}) after 3 retries"
                    )
                delay = min(5 * (2 ** (retry_5xx - 1)), 60)
                logger.warning(
                    "LLM server error (%d) — retrying in %ds (attempt %d/%d)",
                    resp.status_code,
                    delay,
                    retry_5xx,
                    MAX_RETRIES_5XX,
                )
                time.sleep(delay)
                continue

            # --- Client error (non-429) — not retryable ------------------------
            # 4xx other than 429
            raise LLMClientError(
                f"HTTP {resp.status_code}: {resp.text[:500]}"
            )

    def close(self) -> None:
        """Close the underlying ``httpx.Client``."""
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        resp: httpx.Response,
        model_id: str,
        elapsed: float,
    ) -> str:
        """Extract ``choices[0].message.content`` from a successful response.

        Logs token usage and latency on success.
        """
        try:
            data = resp.json()
        except ValueError:
            logger.error("LLM response was not valid JSON")
            raise LLMClientError("Non-JSON response received from API")

        choices: list[dict] = data.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise LLMClientError("Response missing 'choices' array")

        message: Optional[dict] = choices[0].get("message") if choices else None
        if not message or not isinstance(message, dict):
            raise LLMClientError("Response missing choices[0].message")

        content: Optional[str] = message.get("content")
        if content is None:
            raise LLMClientError("Response missing choices[0].message.content")

        usage = data.get("usage", {})
        logger.info(
            "LLM API call — model=%s tokens[p=%d c=%d t=%d] latency=%.3fs",
            model_id,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
            elapsed,
        )

        return content
