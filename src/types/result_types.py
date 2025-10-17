"""
Result Types for Type-Safe Error Handling

This module defines Result types to replace string-based error handling
throughout the codebase. Instead of checking `output.startswith("[ERROR]")`,
we use proper typed Result objects.
"""

from dataclasses import dataclass
from typing import Union, Optional, Any, Dict
from enum import Enum


class ErrorType(str, Enum):
    """Categories of errors that can occur"""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    INTERNAL_ERROR = "internal_error"


@dataclass
class LLMError:
    """
    Represents an error from an LLM operation

    Attributes:
        error_type: Category of error (timeout, API error, etc.)
        message: Human-readable error message
        retryable: Whether this error can be retried
        original_exception: The underlying exception if available
        context: Additional context about the error
    """
    error_type: ErrorType
    message: str
    retryable: bool
    original_exception: Optional[Exception] = None
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}

    def to_string(self) -> str:
        """Convert error to string representation for backward compatibility"""
        return f"[ERROR:{self.error_type.value}] {self.message}"


@dataclass
class LLMSuccess:
    """
    Represents a successful LLM operation

    Attributes:
        content: The generated content
        model: Model used for generation
        tokens_used: Number of tokens consumed
        latency_ms: Latency in milliseconds
        metadata: Additional metadata (temperature, etc.)
    """
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Main Result type - can be either Success or Error
LLMResult = Union[LLMSuccess, LLMError]


def is_success(result: LLMResult) -> bool:
    """Type guard to check if result is a success"""
    return isinstance(result, LLMSuccess)


def is_error(result: LLMResult) -> bool:
    """Type guard to check if result is an error"""
    return isinstance(result, LLMError)


@dataclass
class StageError:
    """
    Represents an error from an orchestration stage

    Similar to LLMError but specific to stage-level operations.
    """
    stage: str
    error_type: ErrorType
    message: str
    retryable: bool
    original_exception: Optional[Exception] = None

    def to_string(self) -> str:
        """Convert error to string representation"""
        return f"[ERROR:{self.stage}:{self.error_type.value}] {self.message}"


@dataclass
class StageSuccess:
    """
    Represents a successful stage execution

    Contains the stage output and metadata.
    """
    stage: str
    output: str
    duration_seconds: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# Stage Result type
StageResult = Union[StageSuccess, StageError]


def create_timeout_error(message: str, context: Dict[str, Any] = None) -> LLMError:
    """Helper to create a timeout error"""
    return LLMError(
        error_type=ErrorType.TIMEOUT,
        message=message,
        retryable=True,
        context=context or {}
    )


def create_api_error(message: str, exception: Exception = None, retryable: bool = True) -> LLMError:
    """Helper to create an API error"""
    return LLMError(
        error_type=ErrorType.API_ERROR,
        message=message,
        retryable=retryable,
        original_exception=exception
    )


def create_rate_limit_error(message: str) -> LLMError:
    """Helper to create a rate limit error"""
    return LLMError(
        error_type=ErrorType.RATE_LIMIT,
        message=message,
        retryable=True  # Rate limits are retryable with backoff
    )


def create_validation_error(message: str) -> LLMError:
    """Helper to create a validation error"""
    return LLMError(
        error_type=ErrorType.VALIDATION_ERROR,
        message=message,
        retryable=False  # Validation errors shouldn't be retried
    )
