"""
Structured logging module for Facilitair
"""

from .structured_logger import (
    LogLevel,
    LogFormat,
    StructuredFormatter,
    PerformanceLogger,
    LoggerAdapter,
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
    setup_logger,
    log_function_call,
    log_request,
    log_response,
    log_agent_execution,
    log_llm_call,
    log_error_with_context
)

__all__ = [
    'LogLevel',
    'LogFormat',
    'StructuredFormatter',
    'PerformanceLogger',
    'LoggerAdapter',
    'get_correlation_id',
    'set_correlation_id',
    'clear_correlation_id',
    'setup_logger',
    'log_function_call',
    'log_request',
    'log_response',
    'log_agent_execution',
    'log_llm_call',
    'log_error_with_context'
]
