"""Tests for the OpenRouter LLM client — retry logic, response parsing, init.

Covers all error-handling paths in ``src.llm.LLMClient`` including rate-limit
retry, server-error retry, timeout retry, immediate-raise for non-429 4xx,
malformed responses, and client lifecycle.
"""

import json
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm import LLMClient, LLMClientError


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def client():
    """Return an ``LLMClient`` with ``httpx.Client`` replaced by a mock.

    The mock is injected **after** ``__init__`` so that ``__init__`` can still
    exercise the real header/constructor logic.  Tests that only care about
    the constructed state (headers, timeout) should use this fixture directly.
    """
    with patch("src.llm.httpx.Client") as mock_http_cls:
        c = LLMClient("https://openrouter.ai/api/v1", "sk-test-key-123")
        mock_client = MagicMock()
        c._client = mock_client
        yield c


@pytest.fixture
def mock_sleep():
    """Patch ``time.sleep`` to a no-op so retry tests run instantly."""
    with patch("src.llm.time.sleep") as m:
        yield m


# ===================================================================
# 1. Initialization
# ===================================================================


class TestLLMClientInit:
    """Verify that ``LLMClient.__init__`` configures the underlying HTTP client."""

    def test_base_url_stripped(self):
        """Trailing slash on base_url must be removed."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1/", "key")
        _call_kwargs = mock_http_cls.call_args.kwargs
        assert _call_kwargs["base_url"] == "https://example.com/v1"

    def test_base_url_preserved_without_slash(self):
        """base_url without trailing slash must stay unchanged."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key")
        _call_kwargs = mock_http_cls.call_args.kwargs
        assert _call_kwargs["base_url"] == "https://example.com/v1"

    def test_authorization_header(self):
        """Authorization header must be ``Bearer <api_key>``."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "my-secret-key")
        headers = mock_http_cls.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    def test_content_type_header(self):
        """Content-Type header must be ``application/json``."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key")
        headers = mock_http_cls.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_referer_header(self):
        """HTTP-Referer header must point to the GitHub repo."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key")
        headers = mock_http_cls.call_args.kwargs["headers"]
        assert headers["HTTP-Referer"] == "https://github.com/ai-news-pipeline"

    def test_title_header(self):
        """X-Title header must be ``AI News Pipeline``."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key")
        headers = mock_http_cls.call_args.kwargs["headers"]
        assert headers["X-Title"] == "AI News Pipeline"

    def test_custom_timeout(self):
        """User-supplied timeout (seconds) must be passed to httpx."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key", timeout=60)
        timeout = mock_http_cls.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 60.0
        assert timeout.read == 60.0
        assert timeout.write == 60.0
        assert timeout.pool == 60.0

    def test_default_timeout(self):
        """Default timeout must be 120 seconds."""
        with patch("src.llm.httpx.Client") as mock_http_cls:
            LLMClient("https://example.com/v1", "key")
        timeout = mock_http_cls.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.connect == 120.0
        assert timeout.read == 120.0
        assert timeout.write == 120.0
        assert timeout.pool == 120.0


# ===================================================================
# 2. Successful completion
# ===================================================================


class TestSuccessfulComplete:
    """Happy-path: 200 response with valid JSON."""

    def test_returns_content_string(self, client):
        """A 200 with content ``"hello"`` must return ``"hello"``."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        client._client.post.return_value = mock_resp

        result = client.complete("test-model", 0.7, "sys", "user")
        assert result == "hello"

    def test_returns_empty_string(self, client):
        """A 200 with empty content ``""`` must return ``""``."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": ""}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp

        result = client.complete("test-model", 0.7, "sys", "user")
        assert result == ""

    def test_posts_to_correct_endpoint(self, client):
        """The POST request must go to ``/chat/completions``."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp
        client.complete("m", 0.5, "s", "u")
        client._client.post.assert_called_once()
        args, _ = client._client.post.call_args
        assert args[0] == "/chat/completions"

    def test_accepts_zero_temperature(self, client):
        """temperature=0.0 must be accepted."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp
        result = client.complete("m", 0.0, "s", "u")
        assert result == "ok"

    def test_accepts_max_temperature(self, client):
        """temperature=2.0 must be accepted."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp
        result = client.complete("m", 2.0, "s", "u")
        assert result == "ok"


# ===================================================================
# 3. Request body format
# ===================================================================


class TestRequestFormat:
    """Verify the JSON payload sent to the API."""

    def test_payload_includes_model_and_temperature(self, client):
        """The request body must contain ``model`` and ``temperature``."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp

        client.complete("deepseek/deepseek-v4-pro", 0.7, "Be helpful", "Hello")

        _call = client._client.post.call_args
        payload = _call[1]["json"]
        assert payload["model"] == "deepseek/deepseek-v4-pro"
        assert payload["temperature"] == 0.7

    def test_payload_includes_messages(self, client):
        """The request body must contain system and user messages."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }
        client._client.post.return_value = mock_resp

        client.complete("m", 0.5, "You are a helpful assistant", "What is AI?")

        _call = client._client.post.call_args
        messages = _call[1]["json"]["messages"]
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is AI?"},
        ]


# ===================================================================
# 4. Retry — HTTP 429 (Rate limit)
# ===================================================================


class TestRetryRateLimit:
    """HTTP 429 responses are retried up to 5 times with exponential backoff."""

    def test_succeeds_after_one_retry(self, client, mock_sleep):
        """A 429 followed by a 200 must succeed on the second attempt."""
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": "finally"}}],
            "usage": {},
        }

        rate_limited = MagicMock(spec=httpx.Response)
        rate_limited.status_code = 429

        client._client.post.side_effect = [rate_limited, ok_resp]

        result = client.complete("m", 0.5, "s", "u")
        assert result == "finally"
        assert client._client.post.call_count == 2

    def test_raises_after_five_retries(self, client, mock_sleep):
        """Five consecutive 429s must raise ``LLMClientError``."""
        rate_limited = MagicMock(spec=httpx.Response)
        rate_limited.status_code = 429

        client._client.post.return_value = rate_limited

        with pytest.raises(LLMClientError, match="Rate limit \\(429\\) exceeded after 5 retries"):
            client.complete("m", 0.5, "s", "u")

        # 1 original + 5 retries = 6 calls total
        assert client._client.post.call_count == 6

    def test_exponential_backoff_base_10(self, client, mock_sleep):
        """Retry delays for 429 must be: 10, 20, 40, 60, 60 (capped at 60)."""
        rate_limited = MagicMock(spec=httpx.Response)
        rate_limited.status_code = 429

        client._client.post.return_value = rate_limited

        with pytest.raises(LLMClientError):
            client.complete("m", 0.5, "s", "u")

        expected_delays = [10, 20, 40, 60, 60]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays


# ===================================================================
# 5. Retry — HTTP 5xx (Server error)
# ===================================================================


class TestRetryServerError:
    """HTTP 5xx responses are retried up to 3 times with exponential backoff."""

    def test_succeeds_after_one_retry(self, client, mock_sleep):
        """A 503 followed by a 200 must succeed on the second attempt."""
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }

        server_err = MagicMock(spec=httpx.Response)
        server_err.status_code = 503

        client._client.post.side_effect = [server_err, ok_resp]

        result = client.complete("m", 0.5, "s", "u")
        assert result == "ok"
        assert client._client.post.call_count == 2

    def test_raises_after_three_retries(self, client, mock_sleep):
        """Three consecutive 5xx must raise ``LLMClientError``."""
        server_err = MagicMock(spec=httpx.Response)
        server_err.status_code = 502

        client._client.post.return_value = server_err

        with pytest.raises(LLMClientError, match="Server error \\(502\\) after 3 retries"):
            client.complete("m", 0.5, "s", "u")

        # 1 original + 3 retries = 4 calls
        assert client._client.post.call_count == 4

    def test_exponential_backoff_base_5(self, client, mock_sleep):
        """Retry delays for 5xx must be: 5, 10, 20 (capped at 60)."""
        server_err = MagicMock(spec=httpx.Response)
        server_err.status_code = 500

        client._client.post.return_value = server_err

        with pytest.raises(LLMClientError):
            client.complete("m", 0.5, "s", "u")

        expected_delays = [5, 10, 20]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    def test_retries_various_5xx_codes(self, client, mock_sleep):
        """500, 502, 503 must all be retried identically."""
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }

        for code in (500, 502, 503):
            client._client.post.reset_mock()
            mock_sleep.reset_mock()

            err_resp = MagicMock(spec=httpx.Response)
            err_resp.status_code = code
            client._client.post.side_effect = [err_resp, ok_resp]

            result = client.complete("m", 0.5, "s", "u")
            assert result == "ok"
            assert client._client.post.call_count == 2


# ===================================================================
# 6. No retry — HTTP 4xx (non-429)
# ===================================================================


class TestNoRetryClientError:
    """Non-429 4xx responses raise immediately without retry."""

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 405, 422, 451])
    def test_raises_immediately_on_4xx(self, client, status_code):
        """Any 4xx other than 429 must raise ``LLMClientError`` on first attempt."""
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.status_code = status_code
        err_resp.text = f"Error {status_code}"

        client._client.post.return_value = err_resp

        with pytest.raises(LLMClientError, match=f"HTTP {status_code}: Error {status_code}"):
            client.complete("m", 0.5, "s", "u")

        # Must only have been called once (no retry)
        assert client._client.post.call_count == 1

    def test_truncates_long_response_text(self, client):
        """Response text longer than 500 chars must be truncated in the error."""
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.status_code = 400
        err_resp.text = "x" * 1000

        client._client.post.return_value = err_resp

        with pytest.raises(LLMClientError, match="x" * 500):
            client.complete("m", 0.5, "s", "u")


# ===================================================================
# 7. Retry — Timeout
# ===================================================================


class TestRetryTimeout:
    """``httpx.TimeoutException`` is retried up to 3 times."""

    def test_succeeds_after_one_retry(self, client, mock_sleep):
        """A timeout followed by a 200 must succeed on the second attempt."""
        ok_resp = MagicMock(spec=httpx.Response)
        ok_resp.status_code = 200
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
        }

        client._client.post.side_effect = [httpx.TimeoutException("timed out"), ok_resp]

        result = client.complete("m", 0.5, "s", "u")
        assert result == "ok"
        assert client._client.post.call_count == 2

    def test_raises_after_three_retries(self, client, mock_sleep):
        """Three consecutive timeouts must raise ``LLMClientError``."""
        client._client.post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(LLMClientError, match="Request timed out after 3 retries"):
            client.complete("m", 0.5, "s", "u")

        # 1 original + 3 retries = 4 calls
        assert client._client.post.call_count == 4


# ===================================================================
# 8. Malformed / unexpected responses
# ===================================================================


class TestMalformedResponse:
    """Non-JSON, missing fields, and unexpected errors."""

    def test_non_json_response(self, client):
        """A 200 with non-JSON body must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.side_effect = ValueError("No JSON")

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="Non-JSON response received from API"):
            client.complete("m", 0.5, "s", "u")

    def test_missing_choices_key(self, client):
        """Response without ``choices`` key must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing .choices."):
            client.complete("m", 0.5, "s", "u")

    def test_empty_choices_array(self, client):
        """Response with empty ``choices`` array must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": []}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing .choices."):
            client.complete("m", 0.5, "s", "u")

    def test_choices_not_a_list(self, client):
        """Response where ``choices`` is not a list must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": "not-a-list"}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing .choices."):
            client.complete("m", 0.5, "s", "u")

    def test_missing_message_in_choice(self, client):
        """Choice without ``message`` key must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"not_message": {}}]}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing choices\\[0\\].message"):
            client.complete("m", 0.5, "s", "u")

    def test_message_is_not_dict(self, client):
        """Choice where ``message`` is not a dict must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": "string"}]}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing choices\\[0\\].message"):
            client.complete("m", 0.5, "s", "u")

    def test_missing_content_in_message(self, client):
        """Message without ``content`` key must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"not_content": "x"}}]}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing choices\\[0\\].message.content"):
            client.complete("m", 0.5, "s", "u")

    def test_content_is_null(self, client):
        """Message with ``content: null`` must raise ``LLMClientError``."""
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": None}}]}

        client._client.post.return_value = resp

        with pytest.raises(LLMClientError, match="missing choices\\[0\\].message.content"):
            client.complete("m", 0.5, "s", "u")


# ===================================================================
# 9. Unexpected exception
# ===================================================================


class TestUnexpectedException:
    """Non-TimeoutException exceptions from the HTTP layer are wrapped."""

    def test_unexpected_exception_wrapped(self, client):
        """A generic ``ConnectionError`` must be wrapped in ``LLMClientError``."""
        client._client.post.side_effect = ConnectionError("connection refused")

        with pytest.raises(LLMClientError, match="Unexpected error: connection refused"):
            client.complete("m", 0.5, "s", "u")

        # Must not retry — only one attempt
        assert client._client.post.call_count == 1


# ===================================================================
# 10. Close
# ===================================================================


class TestClose:
    """``LLMClient.close()`` must delegate to the underlying HTTP client."""

    def test_close_delegates_to_httpx_client(self, client):
        """Calling ``close()`` must invoke ``_client.close()``."""
        client.close()
        client._client.close.assert_called_once_with()

    def test_close_is_idempotent(self, client):
        """Calling ``close()`` twice must not raise (httpx allows this)."""
        client.close()
        client.close()  # second call — should be a no-op
        assert client._client.close.call_count == 2


# ===================================================================
# 11. Error hierarchy
# ===================================================================


class TestErrorHierarchy:
    """``LLMClientError`` must be a subclass of ``Exception``."""

    def test_is_subclass_of_exception(self):
        """``LLMClientError`` must inherit from ``Exception``."""
        assert issubclass(LLMClientError, Exception)

    def test_can_be_raised_and_caught_as_exception(self):
        """``raise LLMClientError(...)`` must be catchable as ``Exception``."""
        try:
            raise LLMClientError("test error")
        except Exception as exc:
            assert isinstance(exc, LLMClientError)
            assert str(exc) == "test error"
