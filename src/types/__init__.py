"""
Type definitions for the Facilitair orchestration system
"""

from .result_types import (
    ErrorType,
    LLMError,
    LLMSuccess,
    LLMResult,
    StageError,
    StageSuccess,
    StageResult,
    is_success,
    is_error,
    create_timeout_error,
    create_api_error,
    create_rate_limit_error,
    create_validation_error,
)

__all__ = [
    'ErrorType',
    'LLMError',
    'LLMSuccess',
    'LLMResult',
    'StageError',
    'StageSuccess',
    'StageResult',
    'is_success',
    'is_error',
    'create_timeout_error',
    'create_api_error',
    'create_rate_limit_error',
    'create_validation_error',
]
