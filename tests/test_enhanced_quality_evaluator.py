"""
Tests for enhanced quality evaluator with AST analysis
"""

import pytest
from src.evaluation.enhanced_quality_evaluator import (
    EnhancedCodeQualityEvaluator,
    ASTAnalyzer,
    StaticAnalyzer,
    QualityDimension
)


class TestASTAnalyzer:
    """Test AST-based analysis"""

    def test_valid_python_code(self):
        """Test analyzing valid Python code"""
        code = """
def factorial(n: int) -> int:
    '''Calculate factorial'''
    if n < 0:
        raise ValueError("Negative numbers not allowed")
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        assert result["valid"] is True
        assert "complexity" in result
        assert "structure" in result
        assert "patterns" in result

    def test_syntax_error(self):
        """Test handling syntax errors"""
        code = "def broken(:\n    pass"
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        assert result["valid"] is False
        assert "error" in result

    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation"""
        code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            return "big"
        else:
            return "small"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        complexity = result["complexity"]
        assert complexity["max_complexity"] > 1  # Has multiple branches
        assert len(complexity["functions"]) == 1

    def test_detect_security_issues(self):
        """Test security vulnerability detection"""
        code = """
def dangerous(user_input):
    result = eval(user_input)  # Security risk!
    return result
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        security_issues = result["security"]
        assert len(security_issues) > 0
        assert any(issue["type"] == "dangerous_function" for issue in security_issues)

    def test_detect_code_smells(self):
        """Test code smell detection"""
        code = """
def too_many_params(a, b, c, d, e, f, g):  # Too many parameters
    pass

def with_mutable_default(items=[]):  # Mutable default
    items.append(1)
    return items
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        smells = result["smells"]
        assert len(smells) > 0
        assert any(smell["type"] == "too_many_parameters" for smell in smells)
        assert any(smell["type"] == "mutable_default" for smell in smells)

    def test_pattern_detection(self):
        """Test good pattern detection"""
        code = """
def typed_function(x: int) -> str:
    '''Documented function'''
    with open('file.txt') as f:  # Context manager
        data = f.read()
    return [str(i) for i in range(x)]  # List comprehension
"""
        analyzer = ASTAnalyzer()
        result = analyzer.analyze(code)

        patterns = result["patterns"]
        assert patterns["context_managers"] > 0
        assert patterns["list_comprehensions"] > 0
        assert patterns["type_hints"] > 0
        assert patterns["docstrings"] > 0


class TestEnhancedEvaluator:
    """Test enhanced quality evaluator"""

    def test_high_quality_code(self):
        """Test evaluation of high-quality code"""
        code = """
def calculate_factorial(n: int) -> int:
    '''
    Calculate factorial of a number.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n

    Raises:
        ValueError: If n is negative
    '''
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    return n * calculate_factorial(n - 1)


def test_calculate_factorial():
    '''Test factorial calculation'''
    assert calculate_factorial(0) == 1
    assert calculate_factorial(1) == 1
    assert calculate_factorial(5) == 120
"""
        evaluator = EnhancedCodeQualityEvaluator(use_static_analysis=False)
        result = evaluator.evaluate(code, "Write a function to calculate factorial")

        assert result.passed is True
        assert result.overall > 0.7
        assert result.dimensions[QualityDimension.CORRECTNESS.value] > 0.7
        assert result.dimensions[QualityDimension.DOCUMENTATION.value] >= 0.6  # Adjusted threshold
        assert result.dimensions[QualityDimension.ERROR_HANDLING.value] >= 0.3  # Has error handling
        assert result.dimensions[QualityDimension.TESTING.value] >= 0.6  # Has tests

    def test_low_quality_code(self):
        """Test evaluation of low-quality code"""
        code = """
def f(x):
    return x*2
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(code, "Write a function to double a number")

        # Should pass basic requirements but score lower on quality
        assert result.dimensions[QualityDimension.DOCUMENTATION.value] < 0.3
        assert result.dimensions[QualityDimension.ERROR_HANDLING.value] < 0.3

    def test_syntax_error_fails(self):
        """Test that syntax errors result in failure"""
        code = "def broken(\n    pass"
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(code, "Write a function")

        assert result.passed is False
        assert result.overall == 0.0
        assert len(result.issues) > 0
        assert result.issues[0]["type"] == "syntax_error"

    def test_security_issues_lower_score(self):
        """Test that security issues lower the score"""
        code = """
def run_code(user_input):
    return eval(user_input)  # Critical security issue
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(code, "Write a function to evaluate input")

        assert result.dimensions[QualityDimension.SECURITY.value] < 1.0
        assert any(issue["severity"] == "critical" for issue in result.issues)

    def test_complexity_affects_score(self):
        """Test that high complexity affects correctness score"""
        # Very complex function with many branches (complexity > 20)
        code = """
def very_complex(x, y, z):
    if x > 0 and y > 0:
        if x > 10 or y > 10:
            if z > 5 and x > 5:
                if y > 15 or z > 15:
                    if x > 20 and y > 20:
                        if z > 25 or x > 25:
                            if y > 30 and z > 30:
                                return "case1"
                            elif y < 30 or z < 30:
                                return "case2"
                        return "case3"
                    return "case4"
                return "case5"
            return "case6"
        return "case7"
    elif x < 0 and y < 0:
        if z > 0 or x < -10:
            if y < -15 and z < 5:
                return "case8"
            return "case9"
        return "case10"
    else:
        return "case11"
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(code, "Write a function")

        # High complexity (> 20) should lower correctness score to 0.4
        assert result.details["complexity"]["max_complexity"] > 20
        assert result.dimensions[QualityDimension.CORRECTNESS.value] <= 0.5

    def test_code_smells_reported(self):
        """Test that code smells are reported in issues"""
        code = """
def function_with_issues(a, b, c, d, e, f):  # Too many params
    try:
        result = a + b
    except:  # Bare except
        result = 0
    return result
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(code, "Add numbers")

        assert len(result.issues) > 0
        issue_types = [issue["type"] for issue in result.issues]
        assert "too_many_parameters" in issue_types or "bare_except" in issue_types

    def test_good_patterns_increase_score(self):
        """Test that good patterns increase quality score"""
        code_without_patterns = """
def process():
    f = open('file.txt')
    data = f.read()
    f.close()
    return data
"""
        code_with_patterns = """
def process():
    with open('file.txt') as f:  # Context manager
        data = f.read()
    return data
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result_without = evaluator.evaluate(code_without_patterns, "Read a file")
        result_with = evaluator.evaluate(code_with_patterns, "Read a file")

        # Code with context manager should score higher
        assert result_with.dimensions[QualityDimension.CODE_QUALITY.value] >= \
               result_without.dimensions[QualityDimension.CODE_QUALITY.value]

    def test_completeness_scoring(self):
        """Test completeness scoring based on task"""
        code = """
import requests

def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result = evaluator.evaluate(
            code,
            "Write a function to fetch JSON data from an HTTP API"
        )

        # Should recognize appropriate imports for HTTP task
        assert result.dimensions[QualityDimension.COMPLETENESS.value] > 0.5

    def test_documentation_scoring(self):
        """Test documentation scoring"""
        undocumented = """
def add(a, b):
    return a + b
"""
        documented = """
def add(a: int, b: int) -> int:
    '''
    Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    '''
    return a + b
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result_undoc = evaluator.evaluate(undocumented, "Add two numbers")
        result_doc = evaluator.evaluate(documented, "Add two numbers")

        assert result_doc.dimensions[QualityDimension.DOCUMENTATION.value] > \
               result_undoc.dimensions[QualityDimension.DOCUMENTATION.value]

    def test_error_handling_scoring(self):
        """Test error handling scoring"""
        no_handling = """
def divide(a, b):
    return a / b
"""
        with_handling = """
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
"""
        evaluator = EnhancedCodeQualityEvaluator()
        result_no = evaluator.evaluate(no_handling, "Divide two numbers")
        result_with = evaluator.evaluate(with_handling, "Divide two numbers")

        assert result_with.dimensions[QualityDimension.ERROR_HANDLING.value] > \
               result_no.dimensions[QualityDimension.ERROR_HANDLING.value]


class TestStaticAnalyzer:
    """Test static analysis integration"""

    def test_tool_detection(self):
        """Test detection of available static analysis tools"""
        analyzer = StaticAnalyzer()
        # Just verify it doesn't crash - tools may or may not be installed
        assert isinstance(analyzer.has_pylint, bool)
        assert isinstance(analyzer.has_mypy, bool)

    def test_analyze_returns_results(self):
        """Test that analyze returns results structure"""
        code = """
def test_function():
    return 42
"""
        analyzer = StaticAnalyzer()
        result = analyzer.analyze(code, "python")

        assert "available_tools" in result
        assert isinstance(result["available_tools"], list)

    def test_non_python_returns_unavailable(self):
        """Test that non-Python returns unavailable"""
        analyzer = StaticAnalyzer()
        result = analyzer.analyze("const x = 1;", "javascript")

        assert result["available"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
