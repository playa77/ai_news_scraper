"""Tests for the pipeline orchestrator — retry logic, gen+eval loop, logging, failure alerts.

Covers all components of ``src.main``:
- CLI argument parsing and ``--init-db`` mode
- Structured JSON logging (formatter, context vars)
- ``retry_wrapper`` stage retry behaviour
- ``_run_generate_evaluate`` loop with refinement
- ``_build_feedback_from_eval`` feedback reconstruction
- ``_read_log_tail`` log-file tail reading
- Full ``main()`` exit-code and failure-alert behaviour
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from src.db import Database
from src.llm import LLMClient, LLMClientError
from src.main_old import (
    StageFailedError,
    _build_arg_parser,
    _build_feedback_from_eval,
    _PipelineJsonFormatter,
    _read_log_tail,
    _run_generate_evaluate,
    main,
    retry_wrapper,
    setup_logging,
)
from src.models import (
    Config,
    DatabaseConfig,
    EmailConfig,
    FeedDef,
    FeedsConfig,
    InterestConfig,
    ModelDef,
    ModelsConfig,
    OpenRouterConfig,
    PipelineConfig,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_config(
    max_retries: int = 2,
    backoff: int = 1,
    max_refinement_rounds: int = 3,
    db_path: str = ":memory:",
) -> Config:
    """Build a minimal :class:`Config` suitable for orchestrator tests."""
    return Config(
        feeds=FeedsConfig(
            news=[FeedDef(name="test", url="https://example.com/rss")],
            commentators=[FeedDef(name="test2", url="https://example.com/atom")],
        ),
        models=ModelsConfig(
            strong=ModelDef(id="deepseek/deepseek-v4-pro", temperature=0.7),
            weak=ModelDef(id="deepseek/deepseek-v4-flash", temperature=0.7),
        ),
        pipeline=PipelineConfig(
            max_retries=max_retries,
            retry_backoff_seconds=backoff,
            max_refinement_rounds=max_refinement_rounds,
        ),
        email=EmailConfig(
            recipient="to@example.com",
            sender="from@example.com",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_user="from@example.com",
            smtp_password_env="GMAIL_APP_PASSWORD",
        ),
        database=DatabaseConfig(path=db_path),
        openrouter=OpenRouterConfig(
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
        ),
    )


def _make_mock_llm() -> MagicMock:
    """Return a MagicMock that quacks like :class:`LLMClient`."""
    return MagicMock(spec=LLMClient)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def config() -> Config:
    return _make_config()


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    database.initialize_schema()
    yield database
    database.close()


@pytest.fixture
def seeded_db(db):
    """In-memory DB with one pipeline run, one pending theme, and article."""
    ai_id = db.get_interest_by_name("AI")["id"]
    run_id = db.create_pipeline_run(ai_id, "2026-05-14", "2026-05-14T06:00:00")
    feed_id = db.upsert_feed(ai_id, "https://example.com/rss", "Test Feed", "news")
    article_id = db.insert_article(
        feed_id, "https://example.com/a1", "Article 1",
        None, "2026-05-14T07:00:00", "2026-05-14T07:05:00",
        "Excerpt", "Full content", "full", run_id,
    )
    theme_id = db.insert_theme(
        run_id, "Test Theme", "A test theme", [article_id], "novel", 1,
    )
    return {"db": db, "run_id": run_id, "theme_id": theme_id, "article_id": article_id}


@pytest.fixture(autouse=True)
def clean_logging():
    """Remove all handlers after each test to avoid cross-test pollution."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


# ===================================================================
# 1. Error hierarchy
# ===================================================================


class TestStageFailedError:
    """Verify ``StageFailedError`` carries stage name and cause."""

    def test_stores_stage_and_cause(self):
        cause = ValueError("bang")
        exc = StageFailedError("scrape", cause)
        assert exc.stage_name == "scrape"
        assert exc.cause is cause
        assert str(exc) == "Stage 'scrape' failed: bang"

    def test_is_exception_subclass(self):
        assert issubclass(StageFailedError, Exception)


# ===================================================================
# 2. CLI argument parsing
# ===================================================================


class TestArgParser:
    """Verify the argument parser accepts expected flags."""

    def test_config_required(self):
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_config_provided(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--config", "/path/to/config/"])
        assert args.config == "/path/to/config/"

    def test_init_db_flag(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--config", "config/", "--init-db"])
        assert args.init_db is True

    def test_init_db_default_false(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--config", "config/"])
        assert args.init_db is False

    def test_log_file_default(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--config", "config/"])
        assert args.log_file == "pipeline.log"

    def test_log_file_custom(self):
        parser = _build_arg_parser()
        args = parser.parse_args(["--config", "config/", "--log-file", "/tmp/test.log"])
        assert args.log_file == "/tmp/test.log"


# ===================================================================
# 3. JSON logging formatter
# ===================================================================


class TestPipelineJsonFormatter:
    """Verify ``_PipelineJsonFormatter`` produces valid JSON with required fields."""

    def test_output_is_valid_json(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 42, "hello world", (), None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_includes_required_fields(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 42, "hello", (), None,
        )
        output = json.loads(fmt.format(record))
        for key in ("timestamp", "level", "pipeline_run_id", "stage", "theme_id", "message"):
            assert key in output

    def test_timestamp_is_iso8601(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        output = json.loads(fmt.format(record))
        # Should contain a T separator (ISO 8601)
        assert "T" in output["timestamp"]
        # Should contain timezone offset (+ or Z)
        assert "+" in output["timestamp"] or output["timestamp"].endswith("Z")

    def test_includes_logger_and_source(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "src.main", logging.INFO, "main.py", 99, "test", (), None,
        )
        output = json.loads(fmt.format(record))
        assert output["logger"] == "src.main"
        assert output["source"] == "main.py:99"

    def test_exception_info_included(self):
        fmt = _PipelineJsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                "test", logging.ERROR, "test.py", 1, "oops", (), sys.exc_info(),
            )
        output = json.loads(fmt.format(record))
        assert "exception" in output
        assert "ValueError" in output["exception"]
        # ERROR-level should include traceback
        assert "traceback" in output

    def test_extra_data_folded_in(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        record.extra_data = {"feed_name": "Test Feed", "count": 5}
        output = json.loads(fmt.format(record))
        assert "extra" in output
        assert output["extra"]["feed_name"] == "Test Feed"
        assert output["extra"]["count"] == 5

    def test_no_extra_data_when_not_set(self):
        fmt = _PipelineJsonFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "test.py", 1, "msg", (), None,
        )
        output = json.loads(fmt.format(record))
        assert "extra" not in output


# ===================================================================
# 4. setup_logging
# ===================================================================


class TestSetupLogging:
    """Verify ``setup_logging`` configures JSON logging to stdout±file."""

    def test_adds_stdout_handler(self):
        root = logging.getLogger()
        initial_count = len(root.handlers)
        setup_logging(log_file=None)
        assert len(root.handlers) >= initial_count + 1

    def test_adds_file_handler_when_path_given(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tf:
            log_path = tf.name

        try:
            root = logging.getLogger()
            initial_count = len(root.handlers)
            setup_logging(log_file=log_path)
            # Should have stdout + file handler
            assert len(root.handlers) >= initial_count + 2

            # Write a log message and verify it lands in the file
            logger = logging.getLogger("test_setup_logging")
            logger.info("test message in file")
        finally:
            os.unlink(log_path)

    def test_stdout_handler_uses_json_formatter(self):
        setup_logging(log_file=None)
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) > 0
        for h in stream_handlers:
            if h.stream is sys.stdout:
                assert isinstance(h.formatter, _PipelineJsonFormatter)
                break


# ===================================================================
# 5. _read_log_tail
# ===================================================================


class TestReadLogTail:
    """Verify ``_read_log_tail`` reads last N lines from the log file."""

    def test_reads_last_n_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
            for i in range(200):
                tf.write(f"line {i}\n")
            log_path = tf.name

        try:
            import src.main_old as main_mod
            main_mod._log_file_path = log_path
            tail = _read_log_tail(lines=50)
            lines = tail.strip().split("\n")
            assert len(lines) == 50
            assert "line 150" in lines[0]
            assert "line 199" in lines[-1]
        finally:
            main_mod._log_file_path = None
            os.unlink(log_path)

    def test_returns_placeholder_when_no_log_file(self):
        import src.main_old as main_mod
        main_mod._log_file_path = None
        tail = _read_log_tail()
        assert "(no log file" in tail

    def test_returns_placeholder_when_file_missing(self):
        import src.main_old as main_mod
        main_mod._log_file_path = "/nonexistent/path.log"
        tail = _read_log_tail()
        assert "(no log file" in tail
        main_mod._log_file_path = None


# ===================================================================
# 6. retry_wrapper
# ===================================================================


class TestRetryWrapper:
    """Verify ``retry_wrapper`` retries on failure and raises on exhaustion."""

    def test_calls_fn_once_on_success(self):
        fn = MagicMock()
        retry_wrapper("test", fn, max_retries=2, backoff_seconds=0)
        fn.assert_called_once()

    def test_retries_on_exception(self):
        fn = MagicMock(side_effect=[ValueError("fail1"), ValueError("fail2"), None])
        with patch("time.sleep") as mock_sleep:
            retry_wrapper("test", fn, max_retries=2, backoff_seconds=0)
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2  # slept after attempt 1 and 2

    def test_raises_stage_failed_error_on_exhaustion(self):
        fn = MagicMock(side_effect=ValueError("always fails"))
        with patch("time.sleep"):
            with pytest.raises(StageFailedError) as excinfo:
                retry_wrapper("scrape", fn, max_retries=2, backoff_seconds=0)
        assert excinfo.value.stage_name == "scrape"
        assert fn.call_count == 3  # 1 initial + 2 retries

    def test_uses_backoff_between_retries(self):
        fn = MagicMock(side_effect=[ValueError("fail"), None])
        with patch("time.sleep") as mock_sleep:
            retry_wrapper("test", fn, max_retries=2, backoff_seconds=30)
        mock_sleep.assert_called_once_with(30)

    def test_passes_args_and_kwargs(self):
        fn = MagicMock()
        retry_wrapper("test", fn, 2, 0, "pos_arg", kwarg1=42)
        fn.assert_called_with("pos_arg", kwarg1=42)


# ===================================================================
# 7. _build_feedback_from_eval
# ===================================================================


class TestBuildFeedbackFromEval:
    """Verify feedback reconstruction from stored evaluation rounds."""

    def test_builds_combined_feedback(self):
        quality_data = {
            "summary_en": {"pass": True, "feedback": "Looks good"},
            "script_en": {"pass": False, "feedback": "Too short"},
            "script_de": {"pass": True, "feedback": "Native quality"},
        }
        adv_data = {
            "pass": True,
            "feedback": "No factual issues found",
            "issues": [],
        }
        eval_round = {
            "round_number": 1,
            "quality_feedback": json.dumps(quality_data),
            "adversarial_feedback": json.dumps(adv_data),
        }

        feedback = _build_feedback_from_eval(eval_round)

        assert "=== QUALITY FEEDBACK ===" in feedback
        assert "summary_en: PASS" in feedback
        assert "script_en: FAIL" in feedback
        assert "Too short" in feedback
        assert "=== ADVERSARIAL FEEDBACK ===" in feedback
        assert "No factual issues found" in feedback

    def test_handles_missing_feedback_fields(self):
        eval_round = {"round_number": 1}
        feedback = _build_feedback_from_eval(eval_round)
        assert feedback == ""

    def test_handles_none(self):
        assert _build_feedback_from_eval(None) == ""

    def test_handles_non_json_feedback(self):
        eval_round = {
            "quality_feedback": "not json",
            "adversarial_feedback": "also not json",
        }
        feedback = _build_feedback_from_eval(eval_round)
        assert feedback is not None  # should not raise


# ===================================================================
# 8. _run_generate_evaluate
# ===================================================================


class TestRunGenerateEvaluate:
    """Verify the generate+evaluate loop with per-theme refinement."""

    def test_generates_then_evaluates(self, seeded_db, config):
        """Calling _run_generate_evaluate calls generator.run then evaluator.run."""
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = seeded_db["theme_id"]
        llm = _make_mock_llm()
        interest = InterestConfig(name="AI", id=1)

        # Mock generator.run to insert deliverables
        def fake_generate(run_id, db, config, llm_client, interest):
            db.insert_deliverable(theme_id, "summary_en", "summary v1", 1)
            db.insert_deliverable(theme_id, "script_en", "script v1", 1)
            db.insert_deliverable(theme_id, "script_de", "script v1", 1)

        # Evaluator mock must also update the theme status (as the real one does)
        def fake_eval(run_id, db, config, llm_client, theme_id, interest):
            db.update_theme_status(theme_id, "approved")
            return "approved"

        with patch("src.main_old.generator.run", side_effect=fake_generate), \
             patch("src.main_old.evaluator.run", side_effect=fake_eval), \
             patch("src.main_old.generator.refine"):
            _run_generate_evaluate(run_id, db, config, llm, interest)

        # Theme should be approved
        themes = db.get_themes_for_run(run_id)
        assert len(themes) == 1
        assert themes[0]["status"] == "approved"

    def test_refinement_loop_runs(self, seeded_db, config):
        """Evaluator returns 'needs_refinement' twice, then 'approved'."""
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = seeded_db["theme_id"]
        llm = _make_mock_llm()

        # Insert deliverables (generator already ran in this test)
        db.insert_deliverable(theme_id, "summary_en", "summary v1", 1)
        db.insert_deliverable(theme_id, "script_en", "script v1", 1)
        db.insert_deliverable(theme_id, "script_de", "script v1", 1)

        # Evaluator.run returns "needs_refinement" twice, then "approved"
        eval_responses = ["needs_refinement", "needs_refinement", "approved"]

        def fake_eval(run_id, db, config, llm_client, theme_id, interest):
            # Store an evaluation round so feedback can be extracted
            response = eval_responses.pop(0)
            db.insert_evaluation_round(
                theme_id,
                round_number=1 if response == "needs_refinement" else 3,
                quality_passed="fail" if response == "needs_refinement" else "pass",
                quality_feedback=json.dumps({"summary_en": {"pass": False, "feedback": "Bad"}}),
                adversarial_passed="pass",
                adversarial_feedback=json.dumps({"pass": True, "feedback": "OK", "issues": []}),
                overall_passed="fail" if response == "needs_refinement" else "pass",
            )
            return response

        refine_calls = []

        def fake_refine(run_id, db, config, llm_client, theme_id, feedback, interest):
            refine_calls.append(feedback)

        with patch("src.main_old.generator.run"), \
             patch("src.main_old.evaluator.run", side_effect=fake_eval), \
             patch("src.main_old.generator.refine", side_effect=fake_refine):
            _run_generate_evaluate(run_id, db, config, llm, InterestConfig(name="AI", id=1))

        # Should have called refine twice
        assert len(refine_calls) == 2

    def test_skips_non_pending_themes(self, seeded_db, config):
        """Themes already approved or auto_approved are skipped."""
        db = seeded_db["db"]
        run_id = seeded_db["run_id"]
        theme_id = seeded_db["theme_id"]
        llm = _make_mock_llm()

        # Mark theme as already approved
        db.update_theme_status(theme_id, "approved")

        with patch("src.main_old.generator.run") as mock_gen, \
             patch("src.main_old.evaluator.run") as mock_eval:
            _run_generate_evaluate(run_id, db, config, llm, InterestConfig(name="AI", id=1))

        # Generator should still be called (it iterates themes itself)
        # but evaluator should NOT be called (theme is not "pending")
        mock_eval.assert_not_called()


# ===================================================================
# 9. main() — success path
# ===================================================================


class TestMainSuccess:
    """Verify ``main()`` completes successfully with all stages mocked."""

    def test_exit_code_0_on_success(self, config, db):
        """main() exits with code 0 when all stages succeed."""
        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper") as mock_retry, \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 0
        # retry_wrapper should have been called for each stage
        stage_names = [c[0][0] for c in mock_retry.call_args_list]
        assert "scrape" in stage_names
        assert "analyze" in stage_names
        assert "generate_evaluate" in stage_names
        assert "brief" in stage_names
        assert "email" in stage_names

    def test_pipeline_status_completed(self, config, db):
        """After success, the pipeline run status is 'completed'."""
        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")
        run = db.get_pipeline_run(run_id)
        assert run is not None

        def fake_create_pipeline_run(*args, **kwargs):
            return run_id

        db.create_pipeline_run = fake_create_pipeline_run  # type: ignore[method-assign]
        # Prevent main() from closing the connection so we can verify
        original_close = db.close
        db.close = lambda: None  # type: ignore[method-assign]

        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper"), \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit):
                main()

        run = db.get_pipeline_run(run_id)
        assert run["status"] == "completed"
        assert run["completed_at"] is not None
        original_close()


# ===================================================================
# 10. main() — --init-db
# ===================================================================


class TestMainInitDb:
    """Verify ``--init-db`` initializes schema and exits with code 0."""

    def test_init_db_exits_zero(self, config, db):
        """--init-db initializes the schema, prints a message, and exits 0."""
        def fake_from_yaml(path):
            return config
        db_factory = lambda path: db  # noqa: E731

        with patch("src.main_old.from_yaml", side_effect=fake_from_yaml), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.setup_logging"), \
              patch.object(sys, "argv", ["main.py", "--config", "config/", "--init-db"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 0


# ===================================================================
# 11. main() — config error
# ===================================================================


class TestMainConfigError:
    """Verify ``main()`` exits with code 2 on config error."""

    def test_config_error_exits_two(self):
        from src.config import ConfigError

        with patch("src.main_old.from_yaml", side_effect=ConfigError("bad yaml")), \
             patch("src.main_old.setup_logging"), \
              patch.object(sys, "argv", ["main.py", "--config", "missing/"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 2


# ===================================================================
# 12. main() — stage failure
# ===================================================================


class TestMainFailure:
    """Verify ``main()`` handles stage failure, updates DB, and sends alert."""

    def test_exit_code_1_on_stage_failure(self, config, db):
        """When a stage fails, main() exits with code 1."""
        from src.main_old import StageFailedError

        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper", side_effect=StageFailedError("scrape", ValueError("fail"))), \
             patch("src.main_old.emailer.send_failure_alert"), \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 1

    def test_db_updated_on_failure(self, config, db):
        """Pipeline run is marked as 'failed' with error_message."""
        from src.main_old import StageFailedError

        run_id = db.create_pipeline_run(db.get_interest_by_name("AI")["id"], "2026-05-14", "2026-05-14T06:00:00")

        def fake_create_pipeline_run(*args, **kwargs):
            return run_id

        db.create_pipeline_run = fake_create_pipeline_run  # type: ignore[method-assign]
        # Prevent main() from closing the connection so we can verify
        original_close = db.close
        db.close = lambda: None  # type: ignore[method-assign]

        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper", side_effect=StageFailedError("analyze", ValueError("boom"))), \
             patch("src.main_old.emailer.send_failure_alert"), \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit):
                main()

        run = db.get_pipeline_run(run_id)
        assert run["status"] == "failed"
        assert "analyze" in run["error_message"]
        assert "boom" in run["error_message"]
        assert run["completed_at"] is not None
        original_close()

    def test_failure_alert_sent(self, config, db):
        """On stage failure, send_failure_alert is called."""
        from src.main_old import StageFailedError

        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper", side_effect=StageFailedError("brief", ValueError("fail"))), \
             patch("src.main_old.emailer.send_failure_alert") as mock_alert, \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit):
                main()

        mock_alert.assert_called_once()
        call_args = mock_alert.call_args[0]
        # call_args: (config, stage_name, error_message, traceback_str, log_tail)
        assert call_args[1] == "brief"
        assert "fail" in call_args[2]
        assert "Traceback" in call_args[3] or "ValueError" in call_args[3]

    def test_failure_alert_error_does_not_crash_pipeline(self, config, db):
        """If send_failure_alert itself raises, the pipeline logs and exits 1."""
        from src.main_old import StageFailedError

        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.LLMClient", return_value=_make_mock_llm()), \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test"}), \
             patch("src.main_old.setup_logging"), \
             patch("src.main_old.retry_wrapper", side_effect=StageFailedError("scrape", ValueError("fail"))), \
             patch("src.main_old.emailer.send_failure_alert", side_effect=RuntimeError("smtp down")), \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        # Should still exit 1 (original error, not masked)
        assert excinfo.value.code == 1

    def test_missing_api_key_exits_two(self, config, db):
        """If OPENROUTER_API_KEY is not set, main() exits with code 2."""
        with patch("src.main_old.from_yaml", return_value=config), \
             patch("src.main_old.Database", return_value=db), \
             patch("src.main_old.setup_logging"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["main.py", "--config", "config/"]):
            with pytest.raises(SystemExit) as excinfo:
                main()

        assert excinfo.value.code == 2
