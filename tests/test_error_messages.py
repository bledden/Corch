"""
Tests for user-friendly error messages and troubleshooting hints
"""

import pytest
from src.errors import (
    FacilitairError,
    ValidationError,
    LLMError,
    TimeoutError,
    ConfigurationError,
    ResourceError,
    ErrorCategory,
    format_error,
    format_validation_errors,
    get_troubleshooting_hint
)


class TestFacilitairError:
    """Test base FacilitairError class"""

    def test_basic_error_creation(self):
        """Test creating a basic error"""
        error = FacilitairError("Test error message")
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.INTERNAL
        assert error.hint is not None

    def test_error_with_details(self):
        """Test error with additional details"""
        error = FacilitairError(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            details={"field": "task", "value": "invalid"},
            hint="Check your input"
        )
        assert error.details["field"] == "task"
        assert error.hint == "Check your input"

    def test_error_to_dict(self):
        """Test converting error to dictionary"""
        error = FacilitairError("Test", category=ErrorCategory.LLM)
        result = error.to_dict()
        assert "error" in result
        assert "category" in result
        assert "hint" in result
        assert "details" in result


class TestSpecificErrors:
    """Test specific error subclasses"""

    def test_validation_error(self):
        """Test ValidationError with field information"""
        error = ValidationError(
            message="Invalid task",
            field="task",
            value="short"
        )
        assert error.category == ErrorCategory.VALIDATION
        assert error.details["field"] == "task"
        assert "short" in error.details["value"]

    def test_llm_error(self):
        """Test LLMError with model and stage"""
        error = LLMError(
            message="Model timeout",
            model="gpt-4",
            stage="architecture"
        )
        assert error.category == ErrorCategory.LLM
        assert error.details["model"] == "gpt-4"
        assert error.details["stage"] == "architecture"

    def test_timeout_error(self):
        """Test TimeoutError with operation details"""
        error = TimeoutError(
            message="Operation timed out",
            operation="code_generation",
            timeout_seconds=180.0
        )
        assert error.category == ErrorCategory.TIMEOUT
        assert error.details["operation"] == "code_generation"
        assert error.details["timeout_seconds"] == 180.0

    def test_configuration_error(self):
        """Test ConfigurationError with config key"""
        error = ConfigurationError(
            message="Missing API key",
            config_key="OPENROUTER_API_KEY"
        )
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.details["config_key"] == "OPENROUTER_API_KEY"

    def test_resource_error(self):
        """Test ResourceError with limits"""
        error = ResourceError(
            message="Token limit exceeded",
            resource_type="tokens",
            limit=4096,
            current=5000
        )
        assert error.category == ErrorCategory.RESOURCE
        assert error.details["limit"] == 4096
        assert error.details["current"] == 5000


class TestErrorFormatting:
    """Test error formatting functions"""

    def test_format_facilitair_error(self):
        """Test formatting FacilitairError"""
        error = ValidationError("Invalid input", field="task")
        result = format_error(error, user_facing=True)
        assert "error" in result
        assert "hint" in result
        assert result["category"] == ErrorCategory.VALIDATION

    def test_format_standard_exception(self):
        """Test formatting standard Python exceptions"""
        error = ValueError("Invalid value")
        result = format_error(error, user_facing=True)
        assert "error" in result
        assert "hint" in result
        assert result["category"] == ErrorCategory.VALIDATION

    def test_format_with_traceback(self):
        """Test formatting with traceback (for internal use)"""
        error = Exception("Test error")
        result = format_error(error, include_traceback=True, user_facing=False)
        assert "traceback" in result

    def test_format_validation_errors(self):
        """Test formatting Pydantic-style validation errors"""
        errors = [
            {
                "loc": ["body", "task"],
                "msg": "field required",
                "type": "value_error.missing"
            },
            {
                "loc": ["body", "temperature"],
                "msg": "ensure this value is less than or equal to 2.0",
                "type": "value_error.number.not_le"
            }
        ]
        result = format_validation_errors(errors)
        assert result["category"] == ErrorCategory.VALIDATION
        assert len(result["details"]["validation_errors"]) == 2
        assert "body -> task" in result["details"]["validation_errors"][0]["field"]


class TestTroubleshootingHints:
    """Test troubleshooting hint generation"""

    def test_rate_limit_hint(self):
        """Test hint for rate limit errors"""
        hint = get_troubleshooting_hint("Error: rate limit exceeded")
        assert "rate limit" in hint.lower()
        assert "wait" in hint.lower() or "retry" in hint.lower()

    def test_api_key_hint(self):
        """Test hint for API key errors"""
        hint = get_troubleshooting_hint("Invalid API key provided")
        assert "api key" in hint.lower()
        assert "OPENROUTER_API_KEY" in hint

    def test_timeout_hint(self):
        """Test hint for timeout errors"""
        hint = get_troubleshooting_hint("Request timeout after 180 seconds")
        assert "timeout" in hint.lower()
        assert "breaking" in hint.lower() or "smaller" in hint.lower()

    def test_json_parsing_hint(self):
        """Test hint for JSON parsing errors"""
        hint = get_troubleshooting_hint("Failed to parse invalid JSON")
        assert "json" in hint.lower()

    def test_connection_hint(self):
        """Test hint for connection errors"""
        hint = get_troubleshooting_hint("Connection refused to api.openrouter.ai")
        assert "connection" in hint.lower()
        assert "internet" in hint.lower() or "firewall" in hint.lower()

    def test_model_not_found_hint(self):
        """Test hint for model not found errors"""
        hint = get_troubleshooting_hint("Model not found: invalid-model-id")
        assert "model" in hint.lower()
        assert "config" in hint.lower()

    def test_quota_hint(self):
        """Test hint for quota exceeded errors"""
        hint = get_troubleshooting_hint("Insufficient quota remaining")
        assert "quota" in hint.lower()

    def test_token_limit_hint(self):
        """Test hint for token limit errors"""
        hint = get_troubleshooting_hint("Token limit exceeded: 5000/4096")
        assert "token" in hint.lower()
        assert "context" in hint.lower() or "limit" in hint.lower()

    def test_default_hint(self):
        """Test default hint for unknown errors"""
        hint = get_troubleshooting_hint("Some random error message")
        assert "error occurred" in hint.lower()


class TestCategoryInference:
    """Test automatic error category inference"""

    def test_infer_validation_error(self):
        """Test inferring validation error category"""
        error = ValueError("Invalid input")
        result = format_error(error)
        assert result["category"] == ErrorCategory.VALIDATION

    def test_infer_timeout_error(self):
        """Test inferring timeout error category"""
        import asyncio
        error = asyncio.TimeoutError("Operation timed out")
        result = format_error(error)
        assert result["category"] == ErrorCategory.TIMEOUT

    def test_infer_from_message_content(self):
        """Test inferring category from error message"""
        error = Exception("API key is invalid")
        result = format_error(error)
        assert result["category"] == ErrorCategory.AUTHENTICATION


class TestUserFriendlyMessages:
    """Test that error messages are user-friendly"""

    def test_validation_error_message(self):
        """Test validation error has user-friendly message"""
        error = ValidationError(
            "Task must be at least 10 characters",
            field="task",
            value="short"
        )
        result = error.to_dict()
        # Should not contain stack traces or technical jargon
        assert "[" not in result["error"]
        assert "traceback" not in str(result).lower()

    def test_llm_error_message(self):
        """Test LLM error has helpful context"""
        error = LLMError(
            "Model request failed",
            model="claude-3-sonnet",
            stage="refinement"
        )
        result = error.to_dict()
        assert "hint" in result
        assert len(result["hint"]) > 20  # Should have substantial hint

    def test_error_includes_actionable_hint(self):
        """Test all errors include actionable hints"""
        errors = [
            ValidationError("Invalid input"),
            LLMError("Model failed"),
            TimeoutError("Operation timed out"),
            ConfigurationError("Missing config"),
            ResourceError("Limit exceeded")
        ]
        for error in errors:
            result = error.to_dict()
            assert "hint" in result
            assert len(result["hint"]) > 0
            # Hint should contain actionable words
            hint_lower = result["hint"].lower()
            actionable_words = ["check", "try", "ensure", "verify", "consider", "see", "upgrade", "wait"]
            assert any(word in hint_lower for word in actionable_words)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
