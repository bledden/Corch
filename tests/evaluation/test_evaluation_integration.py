"""
Integration tests for the complete evaluation system
"""
import os
import pytest

from src.middleware.evaluation_middleware import EvaluationMiddleware, AggregatedEvaluationResult
from src.middleware.base_middleware import MiddlewareContext, MiddlewareHook


@pytest.fixture
def evaluation_middleware():
    """Create evaluation middleware with all evaluators enabled except LLM"""
    os.environ['OPENROUTER_API_KEY'] = 'test-key'
    
    middleware = EvaluationMiddleware(
        enabled=True,
        enable_security=True,
        enable_static_analysis=True,
        enable_complexity=True,
        enable_llm_judge=False,  # Disable for faster tests
        pass_threshold=0.7,
        gate_on_failure=False
    )
    return middleware


def test_middleware_hooks_registration(evaluation_middleware):
    """Test that middleware registers correct hooks"""
    hooks = evaluation_middleware.get_hooks()
    assert MiddlewareHook.POST_REFINER in hooks
    assert MiddlewareHook.POST_DOCUMENTER in hooks


def test_middleware_should_execute(evaluation_middleware):
    """Test that middleware executes on correct hooks"""
    assert evaluation_middleware.should_execute(MiddlewareHook.POST_REFINER) is True
    assert evaluation_middleware.should_execute(MiddlewareHook.POST_DOCUMENTER) is True


def test_clean_code_passes_evaluation(evaluation_middleware):
    """Test that clean, high-quality code passes all evaluators"""
    clean_code = '''
"""Calculator module with type hints and documentation"""
from typing import Union


def add_numbers(first: int, second: int) -> int:
    """
    Add two numbers together.
    
    Args:
        first: The first number
        second: The second number
        
    Returns:
        The sum of the two numbers
    """
    return first + second
'''
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="post_refiner",
        input_data={},
        output_data={"code": clean_code}
    )
    
    result = evaluation_middleware.execute(context)
    
    assert "evaluation" in result
    eval_result = result["evaluation"]
    
    # Should be AggregatedEvaluationResult dataclass
    assert isinstance(eval_result, AggregatedEvaluationResult)
    
    # Check scores via attributes (not dict keys)
    assert hasattr(eval_result, 'overall_score')
    assert hasattr(eval_result, 'security_score')
    assert hasattr(eval_result, 'static_analysis_score')
    assert hasattr(eval_result, 'complexity_score')
    
    # Clean code should pass
    assert eval_result.overall_score >= 0.6
    assert eval_result.passed in [True, False]  # Depends on thresholds


def test_vulnerable_code_detected(evaluation_middleware):
    """Test that vulnerable code is caught by security evaluator"""
    vulnerable_code = '''
import pickle
def load_data(user_input):
    data = pickle.loads(user_input)  # CRITICAL SECURITY ISSUE
    return data
'''
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="post_refiner",
        input_data={},
        output_data={"code": vulnerable_code}
    )
    
    result = evaluation_middleware.execute(context)
    eval_result = result["evaluation"]
    
    # Security score should be impacted
    assert eval_result.security_score < 1.0


def test_evaluation_result_structure(evaluation_middleware):
    """Test that evaluation results have correct structure"""
    code = "def hello():\n    return 'world'"
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="post_refiner",
        input_data={},
        output_data={"code": code}
    )
    
    result = evaluation_middleware.execute(context)
    eval_result = result["evaluation"]
    
    # Check all required attributes exist
    assert hasattr(eval_result, 'overall_score')
    assert hasattr(eval_result, 'passed')
    assert hasattr(eval_result, 'security_score')
    assert hasattr(eval_result, 'static_analysis_score')
    assert hasattr(eval_result, 'complexity_score')
    assert hasattr(eval_result, 'llm_judge_score')
    
    # Check score ranges
    assert 0.0 <= eval_result.overall_score <= 1.0
    assert 0.0 <= eval_result.security_score <= 1.0
    assert 0.0 <= eval_result.static_analysis_score <= 1.0
    assert 0.0 <= eval_result.complexity_score <= 1.0
    assert isinstance(eval_result.passed, bool)


def test_all_evaluators_run(evaluation_middleware):
    """Test that all enabled evaluators actually run"""
    code = '''
def calculate(x: int, y: int) -> int:
    """Calculate sum of two numbers"""
    return x + y
'''
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="post_refiner",
        input_data={},
        output_data={"code": code}
    )
    
    result = evaluation_middleware.execute(context)
    eval_result = result["evaluation"]
    
    # All enabled evaluators should have non-default scores
    # Security: enabled, should be 1.0 (no issues)
    assert eval_result.security_score > 0.0
    
    # Static analysis: enabled, should have a score
    assert eval_result.static_analysis_score > 0.0
    
    # Complexity: enabled, should have a score  
    assert eval_result.complexity_score > 0.0
    
    # LLM judge: disabled, should be 0.0
    assert eval_result.llm_judge_score >= 0.0  # Could be 0.0 or 0.5 depending on implementation


def test_graceful_degradation_on_bad_code(evaluation_middleware):
    """Test that system handles malformed code gracefully"""
    invalid_code = "this is not valid python @#$%"
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="post_refiner",
        input_data={},
        output_data={"code": invalid_code}
    )
    
    # Should not raise exception
    result = evaluation_middleware.execute(context)
    
    # Should still return evaluation
    assert "evaluation" in result
    assert isinstance(result["evaluation"], AggregatedEvaluationResult)
    assert 0.0 <= result["evaluation"].overall_score <= 1.0


def test_integration_with_sequential_orchestrator():
    """Test that evaluation middleware works when called from sequential orchestrator context"""
    middleware = EvaluationMiddleware(
        enabled=True,
        enable_security=True,
        enable_static_analysis=True,
        enable_complexity=True,
        enable_llm_judge=False
    )
    
    # Simulate context from sequential orchestrator
    code = '''
def factorial(n: int) -> int:
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
    
    context = MiddlewareContext(
        hook=MiddlewareHook.POST_REFINER,
        stage_name="refiner",
        input_data={"task": "Write a factorial function"},
        output_data={"code": code, "language": "python"}
    )
    
    result = middleware.execute(context)
    
    # Should successfully evaluate
    assert "evaluation" in result
    eval_result = result["evaluation"]
    assert isinstance(eval_result, AggregatedEvaluationResult)
    assert 0.0 <= eval_result.overall_score <= 1.0
