"""
Tests for structured logging system
"""

import pytest
import json
import logging
from io import StringIO
from src.logging import (
    LogLevel,
    LogFormat,
    StructuredFormatter,
    PerformanceLogger,
    LoggerAdapter,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    setup_logger,
    log_request,
    log_response,
    log_agent_execution,
    log_llm_call
)


class TestCorrelationID:
    """Test correlation ID management"""

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID"""
        test_id = "test-correlation-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
        clear_correlation_id()

    def test_generate_correlation_id(self):
        """Test auto-generation of correlation ID"""
        generated_id = set_correlation_id()
        assert generated_id is not None
        assert len(generated_id) > 0
        assert get_correlation_id() == generated_id
        clear_correlation_id()

    def test_clear_correlation_id(self):
        """Test clearing correlation ID"""
        set_correlation_id("test-123")
        clear_correlation_id()
        assert get_correlation_id() is None


class TestStructuredFormatter:
    """Test structured log formatting"""

    def test_json_format(self):
        """Test JSON log formatting"""
        formatter = StructuredFormatter(LogFormat.JSON)

        # Create log record
        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Format the record
        output = formatter.format(record)

        # Parse JSON
        log_entry = json.loads(output)

        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
        assert log_entry["logger"] == "test"
        assert "timestamp" in log_entry

    def test_json_format_with_correlation_id(self):
        """Test JSON format includes correlation ID"""
        formatter = StructuredFormatter(LogFormat.JSON)
        set_correlation_id("test-correlation-123")

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert log_entry["correlation_id"] == "test-correlation-123"
        clear_correlation_id()

    def test_json_format_with_extra_fields(self):
        """Test JSON format includes extra fields"""
        formatter = StructuredFormatter(LogFormat.JSON)

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.custom_field = "custom_value"
        record.user_id = 123

        output = formatter.format(record)
        log_entry = json.loads(output)

        assert "extra" in log_entry
        assert log_entry["extra"]["custom_field"] == "custom_value"
        assert log_entry["extra"]["user_id"] == 123

    def test_human_format(self):
        """Test human-readable log formatting"""
        formatter = StructuredFormatter(LogFormat.HUMAN)

        logger = logging.getLogger("test")
        record = logger.makeRecord(
            name="test",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)

        assert "[INFO]" in output
        assert "test:" in output
        assert "Test message" in output


class TestPerformanceLogger:
    """Test performance logging"""

    def test_performance_logger_success(self):
        """Test performance logger for successful operations"""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(LogFormat.JSON))

        logger = logging.getLogger("test_perf")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Use performance logger
        with PerformanceLogger("test_operation", logger):
            pass  # Simulate work

        # Check logs
        output = stream.getvalue()
        assert "Starting operation" in output
        assert "Operation completed" in output
        assert "duration_ms" in output

    def test_performance_logger_error(self):
        """Test performance logger handles errors"""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(LogFormat.JSON))

        logger = logging.getLogger("test_perf_error")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            with PerformanceLogger("failing_operation", logger):
                raise ValueError("Test error")
        except ValueError:
            pass

        output = stream.getvalue()
        assert "Operation failed" in output
        assert "duration_ms" in output


class TestSetupLogger:
    """Test logger setup"""

    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        logger = setup_logger("test_basic", level=LogLevel.DEBUG)

        assert logger.name == "test_basic"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logger_with_file(self, tmp_path):
        """Test logger setup with file output"""
        log_file = tmp_path / "test.log"
        logger = setup_logger(
            "test_file",
            log_file=str(log_file),
            console=False
        )

        logger.info("Test message")

        # Check file was created and contains log
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


class TestLoggerAdapter:
    """Test logger adapter"""

    def test_adapter_adds_context(self):
        """Test adapter adds default context"""
        base_logger = logging.getLogger("test_adapter")
        adapter = LoggerAdapter(base_logger, {"service": "api", "version": "1.0"})

        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(LogFormat.JSON))
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        adapter.info("Test message")

        output = stream.getvalue()
        log_entry = json.loads(output)

        assert log_entry["extra"]["service"] == "api"
        assert log_entry["extra"]["version"] == "1.0"


class TestConvenienceFunctions:
    """Test convenience logging functions"""

    def setup_method(self):
        """Set up logger for each test"""
        self.stream = StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(StructuredFormatter(LogFormat.JSON))

        self.logger = logging.getLogger("test_convenience")
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def test_log_request(self):
        """Test HTTP request logging"""
        log_request(
            self.logger,
            method="POST",
            path="/api/v1/collaborate",
            client_host="127.0.0.1"
        )

        output = self.stream.getvalue()
        log_entry = json.loads(output)

        assert log_entry["extra"]["event"] == "http_request"
        assert log_entry["extra"]["http_method"] == "POST"
        assert log_entry["extra"]["http_path"] == "/api/v1/collaborate"

    def test_log_response(self):
        """Test HTTP response logging"""
        log_response(
            self.logger,
            status_code=200,
            duration_ms=123.45
        )

        output = self.stream.getvalue()
        log_entry = json.loads(output)

        assert log_entry["extra"]["event"] == "http_response"
        assert log_entry["extra"]["http_status"] == 200
        assert log_entry["extra"]["duration_ms"] == 123.45

    def test_log_agent_execution(self):
        """Test agent execution logging"""
        log_agent_execution(
            self.logger,
            agent_name="architect",
            stage="design",
            status="success",
            duration_ms=500.0
        )

        output = self.stream.getvalue()
        log_entry = json.loads(output)

        assert log_entry["extra"]["event"] == "agent_execution"
        assert log_entry["extra"]["agent"] == "architect"
        assert log_entry["extra"]["stage"] == "design"
        assert log_entry["extra"]["status"] == "success"
        assert log_entry["extra"]["duration_ms"] == 500.0

    def test_log_llm_call(self):
        """Test LLM call logging"""
        log_llm_call(
            self.logger,
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration_ms=1500.0,
            cost_usd=0.003
        )

        output = self.stream.getvalue()
        log_entry = json.loads(output)

        assert log_entry["extra"]["event"] == "llm_call"
        assert log_entry["extra"]["model"] == "gpt-4"
        assert log_entry["extra"]["prompt_tokens"] == 100
        assert log_entry["extra"]["completion_tokens"] == 50
        assert log_entry["extra"]["total_tokens"] == 150
        assert log_entry["extra"]["cost_usd"] == 0.003


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
