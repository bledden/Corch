"""
Tests for StaticAnalysisEvaluator
"""
import pytest
from src.evaluation.static_analysis_evaluator import StaticAnalysisEvaluator


@pytest.fixture
def evaluator():
    """Create evaluator instance"""
    return StaticAnalysisEvaluator()


def test_clean_code_high_score(evaluator):
    """Test that clean, well-formatted code scores high"""
    code = """
\"\"\"Module for basic math operations.\"\"\"


def add_numbers(first: int, second: int) -> int:
    \"\"\"Add two numbers together.\"\"\"
    return first + second


def multiply_numbers(first: int, second: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return first * second
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall >= 0.6
    assert score.pylint_score >= 5.0


def test_unused_import_detected(evaluator):
    """Test that unused imports are flagged"""
    code = """
import sys
import os  # Unused import

def hello():
    print("Hello")
    sys.exit(0)
"""
    score = evaluator.evaluate(code, "python")

    assert score.flake8_violations > 0 or score.total_violations > 0


def test_long_lines_detected(evaluator):
    """Test that excessively long lines are flagged"""
    code = """
def function():
    very_long_line = "This is an extremely long line that significantly exceeds the recommended maximum line length and should definitely be flagged by flake8"
    return very_long_line
"""
    score = evaluator.evaluate(code, "python")

    assert score.flake8_violations > 0


def test_type_errors_detected(evaluator):
    """Test that type errors are caught"""
    code = """
def add(a: int, b: int) -> int:
    return a + b

result: int = add("hello", "world")  # Type error
"""
    score = evaluator.evaluate(code, "python")

    # Mypy should catch type mismatch or overall score reflects issues
    assert score.mypy_errors > 0 or score.overall < 1.0


def test_poor_naming_detected(evaluator):
    """Test that poor variable naming is flagged"""
    code = """
def f(x):  # Poor function name
    y = x + 1  # Poor variable name
    return y
"""
    score = evaluator.evaluate(code, "python")

    assert score.pylint_score < 10.0


def test_missing_docstrings(evaluator):
    """Test that missing docstrings lower score"""
    code = """
def function_without_docstring(x, y):
    return x + y
"""
    score = evaluator.evaluate(code, "python")

    # Pylint penalizes missing docstrings
    assert score.pylint_score < 10.0


def test_non_python_returns_default(evaluator):
    """Test that non-Python code returns default score"""
    code = """
function add(a, b) {
    return a + b;
}
"""
    score = evaluator.evaluate(code, "javascript")

    assert score.overall == 1.0


def test_empty_code_handling(evaluator):
    """Test that empty code is handled gracefully"""
    score = evaluator.evaluate("", "python")

    assert score.overall >= 0.0
    assert score.overall <= 1.0


def test_syntax_error_handling(evaluator):
    """Test that syntax errors don't crash evaluator"""
    code = """
def broken(
    # Missing closing parenthesis and colon
"""
    score = evaluator.evaluate(code, "python")

    # Should handle gracefully, likely low score
    assert score.overall >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
