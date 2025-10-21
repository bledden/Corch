"""
User-friendly error messages and troubleshooting hints

This module provides comprehensive error handling with:
- Clear, actionable error messages
- Troubleshooting hints for common issues
- Error categorization for better handling
- Consistent error formatting
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for better organization"""
    VALIDATION = "validation"
    LLM = "llm"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    INTERNAL = "internal"


class FacilitairError(Exception):
    """
    Base exception for all Facilitair errors

    Provides structured error information with troubleshooting hints.
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        details: Optional[Dict[str, Any]] = None,
        hint: Optional[str] = None,
        user_message: Optional[str] = None
    ):
        """
        Initialize error with detailed information

        Args:
            message: Technical error message (for logs)
            category: Error category for handling
            details: Additional error context
            hint: Troubleshooting hint for users
            user_message: User-friendly message (defaults to message if not provided)
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.hint = hint or self._get_default_hint()
        self.user_message = user_message or message

    def _get_default_hint(self) -> str:
        """Get default troubleshooting hint based on category"""
        return CATEGORY_HINTS.get(self.category, "Please try again or contact support.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error": self.user_message,
            "category": self.category,
            "hint": self.hint,
            "details": self.details
        }


class ValidationError(FacilitairError):
    """Raised when input validation fails"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)[:100]  # Truncate long values

        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            details=details,
            **kwargs
        )


class LLMError(FacilitairError):
    """Raised when LLM operations fail"""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        stage: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if model:
            details['model'] = model
        if stage:
            details['stage'] = stage

        super().__init__(
            message=message,
            category=ErrorCategory.LLM,
            details=details,
            **kwargs
        )


class TimeoutError(FacilitairError):
    """Raised when operations timeout"""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if operation:
            details['operation'] = operation
        if timeout_seconds:
            details['timeout_seconds'] = timeout_seconds

        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            details=details,
            **kwargs
        )


class ConfigurationError(FacilitairError):
    """Raised when configuration is invalid or missing"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if config_key:
            details['config_key'] = config_key

        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            details=details,
            **kwargs
        )


class ResourceError(FacilitairError):
    """Raised when resource limits are exceeded"""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        limit: Optional[Any] = None,
        current: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        if resource_type:
            details['resource_type'] = resource_type
        if limit is not None:
            details['limit'] = limit
        if current is not None:
            details['current'] = current

        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            details=details,
            **kwargs
        )


# Default troubleshooting hints by category
CATEGORY_HINTS = {
    ErrorCategory.VALIDATION: (
        "Check your input parameters and ensure they meet the requirements. "
        "See API documentation for valid formats."
    ),
    ErrorCategory.LLM: (
        "The AI model encountered an issue. This could be due to rate limits, "
        "invalid API keys, or temporary service issues. Please try again in a moment."
    ),
    ErrorCategory.TIMEOUT: (
        "The operation took longer than expected. Try breaking down your task into "
        "smaller steps or increasing timeout limits in configuration."
    ),
    ErrorCategory.CONFIGURATION: (
        "Configuration error detected. Check your config files and environment variables. "
        "Ensure all required settings are present."
    ),
    ErrorCategory.RESOURCE: (
        "Resource limit exceeded. Consider reducing the scope of your request or "
        "upgrading your resource limits."
    ),
    ErrorCategory.NETWORK: (
        "Network connection issue. Check your internet connection and firewall settings. "
        "If the problem persists, the service may be temporarily unavailable."
    ),
    ErrorCategory.AUTHENTICATION: (
        "Authentication failed. Verify your API keys are valid and have the necessary "
        "permissions. Check environment variables or config files."
    ),
    ErrorCategory.INTERNAL: (
        "An internal error occurred. This has been logged for investigation. "
        "Please try again or contact support if the issue persists."
    )
}


# Specific troubleshooting hints for common error patterns
ERROR_PATTERN_HINTS = {
    "rate limit": (
        "Rate limit exceeded. Wait a few moments before retrying. "
        "Consider reducing the frequency of requests or upgrading your API plan."
    ),
    "api key": (
        "API key issue detected. Verify:\n"
        "1. OPENROUTER_API_KEY is set in .env file\n"
        "2. The key is valid and active\n"
        "3. The key has necessary permissions"
    ),
    "timeout": (
        "Operation timed out. Try:\n"
        "1. Breaking task into smaller steps\n"
        "2. Increasing timeout in config/evaluation.yaml\n"
        "3. Simplifying the task description"
    ),
    "invalid json": (
        "JSON parsing failed. This may indicate:\n"
        "1. Malformed response from LLM\n"
        "2. Configuration file syntax error\n"
        "Check logs for the specific JSON that failed to parse."
    ),
    "connection": (
        "Connection error. Verify:\n"
        "1. Internet connection is stable\n"
        "2. Firewall allows outbound HTTPS\n"
        "3. No proxy issues\n"
        "4. OpenRouter service is operational"
    ),
    "model not found": (
        "Model not available. Check:\n"
        "1. Model ID is correct in config/model_strategy_config.yaml\n"
        "2. Model is available in your OpenRouter plan\n"
        "3. Model ID spelling and formatting"
    ),
    "insufficient quota": (
        "Quota exceeded. You've reached your usage limit. "
        "Upgrade your OpenRouter plan or wait for quota reset."
    ),
    "token limit": (
        "Token limit exceeded. The input or output is too large. Try:\n"
        "1. Reducing task description length\n"
        "2. Using a model with larger context window\n"
        "3. Breaking task into smaller subtasks"
    )
}


def format_error(
    error: Exception,
    include_traceback: bool = False,
    user_facing: bool = True
) -> Dict[str, Any]:
    """
    Format any exception into a user-friendly error response

    Args:
        error: The exception to format
        include_traceback: Whether to include technical traceback
        user_facing: Whether this is for end users (affects detail level)

    Returns:
        Dictionary with formatted error information
    """
    # If it's already a FacilitairError, use its built-in formatting
    if isinstance(error, FacilitairError):
        result = error.to_dict()
    else:
        # Convert standard exceptions to FacilitairError format
        message = str(error)
        hint = get_troubleshooting_hint(message)

        result = {
            "error": message if user_facing else f"{error.__class__.__name__}: {message}",
            "category": _infer_category(error),
            "hint": hint,
            "details": {}
        }

    # Add traceback for internal debugging (not user-facing)
    if include_traceback and not user_facing:
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


def format_validation_errors(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format Pydantic validation errors into user-friendly messages

    Args:
        errors: List of validation errors from Pydantic

    Returns:
        Formatted error response with hints
    """
    formatted_errors = []

    for err in errors:
        field = ' -> '.join(str(loc) for loc in err.get('loc', []))
        msg = err.get('msg', 'Invalid value')
        error_type = err.get('type', 'validation_error')

        formatted_errors.append({
            'field': field,
            'message': msg,
            'type': error_type
        })

    return {
        "error": "Input validation failed",
        "category": ErrorCategory.VALIDATION,
        "hint": "Review the field errors below and correct your input.",
        "details": {
            "validation_errors": formatted_errors
        }
    }


def get_troubleshooting_hint(error_message: str) -> str:
    """
    Get troubleshooting hint based on error message content

    Args:
        error_message: The error message to analyze

    Returns:
        Appropriate troubleshooting hint
    """
    error_lower = error_message.lower()

    # Check for specific error patterns
    for pattern, hint in ERROR_PATTERN_HINTS.items():
        if pattern in error_lower:
            return hint

    # Default hint
    return "An error occurred. Check the logs for more details or contact support."


def _infer_category(error: Exception) -> ErrorCategory:
    """Infer error category from exception type and message"""
    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()

    # Check exception type
    if 'validation' in error_type or 'value' in error_type:
        return ErrorCategory.VALIDATION
    if 'timeout' in error_type:
        return ErrorCategory.TIMEOUT
    if 'connection' in error_type or 'network' in error_type:
        return ErrorCategory.NETWORK
    if 'auth' in error_type or 'permission' in error_type:
        return ErrorCategory.AUTHENTICATION

    # Check error message
    if 'timeout' in error_msg:
        return ErrorCategory.TIMEOUT
    if 'api key' in error_msg or 'authentication' in error_msg:
        return ErrorCategory.AUTHENTICATION
    if 'config' in error_msg:
        return ErrorCategory.CONFIGURATION
    if 'rate limit' in error_msg or 'quota' in error_msg:
        return ErrorCategory.RESOURCE
    if 'llm' in error_msg or 'model' in error_msg:
        return ErrorCategory.LLM

    return ErrorCategory.INTERNAL
