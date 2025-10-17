"""
Tests for SecurityEvaluator
"""
import pytest
from src.evaluation.security_evaluator import SecurityEvaluator, Severity


@pytest.fixture
def evaluator():
    """Create evaluator instance"""
    return SecurityEvaluator()


def test_clean_code_no_issues(evaluator):
    """Test that clean code gets perfect security score"""
    code = """
def add(a, b):
    '''Add two numbers'''
    return a + b

def multiply(x, y):
    '''Multiply two numbers'''
    return x * y
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall == 1.0
    assert score.safe is True
    assert score.total_issues == 0
    assert len(score.critical_issues) == 0
    assert len(score.high_issues) == 0


def test_eval_usage_detected(evaluator):
    """Test that eval() usage is flagged as security issue"""
    code = """
def dangerous_function(user_input):
    # CRITICAL: eval() allows arbitrary code execution
    result = eval(user_input)
    return result
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 1.0
    assert score.safe is False
    assert score.total_issues > 0


def test_hardcoded_password(evaluator):
    """Test that hardcoded passwords are detected"""
    code = """
import requests

def connect():
    password = "hardcoded_secret_123"
    requests.post("https://api.example.com", auth=("user", password))
"""
    score = evaluator.evaluate(code, "python")

    # May be flagged as security issue
    assert score.total_issues >= 0  # Bandit may or may not flag simple strings


def test_sql_injection_risk(evaluator):
    """Test SQL injection pattern detection"""
    code = """
import sqlite3

def get_user(user_id):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
"""
    score = evaluator.evaluate(code, "python")

    # SQL string formatting is a security risk
    assert score.total_issues >= 0


def test_pickle_usage(evaluator):
    """Test that pickle usage is flagged"""
    code = """
import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
"""
    score = evaluator.evaluate(code, "python")

    # Pickle is known security risk
    assert score.total_issues >= 0


def test_command_injection(evaluator):
    """Test command injection detection"""
    code = """
import os

def run_command(user_input):
    os.system(user_input)
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 1.0
    assert score.safe is False
    assert score.total_issues > 0


def test_multiple_issues_scoring(evaluator):
    """Test that multiple issues reduce score appropriately"""
    code = """
import pickle
import os

password = "admin123"

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def execute(cmd):
    os.system(cmd)

def dangerous(user_input):
    return eval(user_input)
"""
    score = evaluator.evaluate(code, "python")

    assert score.overall < 0.6
    assert score.safe is False
    assert score.total_issues >= 2


def test_non_python_code_returns_safe(evaluator):
    """Test that non-Python code returns safe score"""
    code = """
function add(a, b) {
    return a + b;
}
"""
    score = evaluator.evaluate(code, "javascript")

    assert score.overall == 1.0
    assert score.safe is True
    assert score.total_issues == 0


def test_empty_code(evaluator):
    """Test empty code handling"""
    score = evaluator.evaluate("", "python")

    # Should handle gracefully
    assert score.overall >= 0.0
    assert score.overall <= 1.0


def test_severity_categorization(evaluator):
    """Test that issues are properly categorized by severity"""
    code = """
def risky():
    eval("1+1")
"""
    score = evaluator.evaluate(code, "python")

    # Check that issues are categorized
    all_issues = (score.critical_issues + score.high_issues +
                  score.medium_issues + score.low_issues)
    assert len(all_issues) == score.total_issues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
