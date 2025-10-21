"""
Error handling and user-friendly error messages for Facilitair
"""

from .error_messages import (
    ErrorCategory,
    FacilitairError,
    ValidationError,
    LLMError,
    TimeoutError,
    ConfigurationError,
    ResourceError,
    format_error,
    format_validation_errors,
    get_troubleshooting_hint
)

__all__ = [
    'ErrorCategory',
    'FacilitairError',
    'ValidationError',
    'LLMError',
    'TimeoutError',
    'ConfigurationError',
    'ResourceError',
    'format_error',
    'format_validation_errors',
    'get_troubleshooting_hint'
]
