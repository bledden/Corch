"""
Structured Logging System with Correlation IDs and Performance Metrics

Provides:
- Structured JSON logging for production
- Correlation IDs for request tracing
- Performance metrics logging
- Context-aware logging with automatic enrichment
- Multiple output formats (JSON, human-readable)
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, Optional
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
import traceback


# Context variable for correlation ID (thread-safe across async)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats"""
    JSON = "json"
    HUMAN = "human"


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging

    Outputs logs as JSON with:
    - Timestamp (ISO 8601)
    - Log level
    - Message
    - Correlation ID (if present)
    - Module/function info
    - Extra context fields
    """

    def __init__(self, format_type: LogFormat = LogFormat.JSON):
        super().__init__()
        self.format_type = format_type

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON or human-readable"""

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id

        # Add extra fields from record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info', 'taskName'
            ]
        }

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Format output
        if self.format_type == LogFormat.JSON:
            return json.dumps(log_entry)
        else:
            # Human-readable format
            parts = [
                f"[{log_entry['timestamp']}]",
                f"[{log_entry['level']}]",
            ]

            if correlation_id:
                parts.append(f"[{correlation_id[:8]}]")

            parts.append(f"{log_entry['logger']}:")
            parts.append(log_entry['message'])

            if extra_fields:
                parts.append(f"({json.dumps(extra_fields)})")

            return " ".join(parts)


class CorrelationIDFilter(logging.Filter):
    """Filter that adds correlation ID to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class PerformanceLogger:
    """
    Context manager for logging performance metrics

    Usage:
        with PerformanceLogger("database_query", logger):
            result = db.query()
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        level: int = logging.INFO,
        extra: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.logger = logger
        self.level = level
        self.extra = extra or {}
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(
            self.level,
            f"Starting operation: {self.operation}",
            extra={**self.extra, "operation": self.operation, "event": "start"}
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        extra = {
            **self.extra,
            "operation": self.operation,
            "duration_ms": round(duration_ms, 2),
            "event": "complete"
        }

        if exc_type:
            extra["error"] = str(exc_val)
            self.logger.log(
                logging.ERROR,
                f"Operation failed: {self.operation} ({duration_ms:.2f}ms)",
                extra=extra,
                exc_info=True
            )
        else:
            self.logger.log(
                self.level,
                f"Operation completed: {self.operation} ({duration_ms:.2f}ms)",
                extra=extra
            )


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context

    Args:
        correlation_id: ID to set, or None to generate new UUID

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id():
    """Clear correlation ID from current context"""
    correlation_id_var.set(None)


def setup_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.JSON,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a structured logger

    Args:
        name: Logger name
        level: Minimum log level
        format_type: Output format (JSON or human-readable)
        log_file: Optional log file path
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.value))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = StructuredFormatter(format_type)

    # Add correlation ID filter
    correlation_filter = CorrelationIDFilter()

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(correlation_filter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation_filter)
        logger.addHandler(file_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds default context to all log messages

    Usage:
        logger = LoggerAdapter(base_logger, {"service": "api", "version": "1.0"})
        logger.info("Request received")  # Automatically includes service and version
    """

    def process(self, msg, kwargs):
        """Add extra context to log records"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        # Merge adapter context with call-specific extra
        kwargs['extra'] = {**self.extra, **kwargs['extra']}

        return msg, kwargs


def log_function_call(logger: logging.Logger, level: LogLevel = LogLevel.DEBUG):
    """
    Decorator to log function calls with arguments and return values

    Usage:
        @log_function_call(logger)
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            logger.log(
                getattr(logging, level.value),
                f"Calling {func_name}",
                extra={
                    "function": func_name,
                    "args": str(args)[:200],  # Truncate long args
                    "kwargs": str(kwargs)[:200],
                    "event": "function_call"
                }
            )

            try:
                result = func(*args, **kwargs)

                logger.log(
                    getattr(logging, level.value),
                    f"Completed {func_name}",
                    extra={
                        "function": func_name,
                        "event": "function_complete"
                    }
                )

                return result
            except Exception as e:
                logger.error(
                    f"Error in {func_name}: {e}",
                    extra={
                        "function": func_name,
                        "error": str(e),
                        "event": "function_error"
                    },
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


# Convenience functions for common logging patterns

def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    correlation_id: Optional[str] = None,
    **extra
):
    """Log HTTP request"""
    if correlation_id:
        set_correlation_id(correlation_id)

    logger.info(
        f"{method} {path}",
        extra={
            "event": "http_request",
            "http_method": method,
            "http_path": path,
            **extra
        }
    )


def log_response(
    logger: logging.Logger,
    status_code: int,
    duration_ms: float,
    **extra
):
    """Log HTTP response"""
    logger.info(
        f"Response {status_code} ({duration_ms:.2f}ms)",
        extra={
            "event": "http_response",
            "http_status": status_code,
            "duration_ms": round(duration_ms, 2),
            **extra
        }
    )


def log_agent_execution(
    logger: logging.Logger,
    agent_name: str,
    stage: str,
    status: str,
    duration_ms: Optional[float] = None,
    **extra
):
    """Log agent execution"""
    message = f"Agent {agent_name} [{stage}]: {status}"

    log_extra = {
        "event": "agent_execution",
        "agent": agent_name,
        "stage": stage,
        "status": status,
        **extra
    }

    if duration_ms is not None:
        log_extra["duration_ms"] = round(duration_ms, 2)
        message += f" ({duration_ms:.2f}ms)"

    level = logging.INFO if status == "success" else logging.WARNING
    logger.log(level, message, extra=log_extra)


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    cost_usd: Optional[float] = None,
    **extra
):
    """Log LLM API call"""
    log_extra = {
        "event": "llm_call",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration_ms": round(duration_ms, 2),
        **extra
    }

    if cost_usd is not None:
        log_extra["cost_usd"] = round(cost_usd, 6)

    logger.info(
        f"LLM call: {model} ({prompt_tokens + completion_tokens} tokens, {duration_ms:.2f}ms)",
        extra=log_extra
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Dict[str, Any],
    message: Optional[str] = None
):
    """Log error with rich context"""
    if message is None:
        message = f"Error: {type(error).__name__}: {str(error)}"

    logger.error(
        message,
        extra={
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        },
        exc_info=True
    )
