"""
Quick import test to verify evaluation system works
"""

print("Testing evaluation system imports...")

# Test evaluator imports
print("1. Importing evaluators...")
from src.evaluation import (
    SecurityEvaluator,
    StaticAnalysisEvaluator,
    ComplexityEvaluator,
    LLMJudgeEvaluator
)
print("✓ All evaluators imported successfully")

# Test middleware imports
print("2. Importing middleware...")
from src.middleware import (
    BaseMiddleware,
    MiddlewareHook,
    EvaluationMiddleware
)
print("✓ Middleware imported successfully")

# Test instantiation
print("3. Testing evaluator instantiation...")
sec_eval = SecurityEvaluator()
static_eval = StaticAnalysisEvaluator()
complexity_eval = ComplexityEvaluator()
llm_eval = LLMJudgeEvaluator()
print("✓ All evaluators instantiated")

# Test middleware instantiation
print("4. Testing middleware instantiation...")
eval_middleware = EvaluationMiddleware(
    enabled=True,
    enable_llm_judge=False  # Skip LLM to avoid API calls
)
print("✓ Middleware instantiated")

# Test basic evaluation
print("5. Testing basic code evaluation...")
test_code = """
def add(a, b):
    '''Add two numbers'''
    return a + b

def multiply(x, y):
    '''Multiply two numbers'''
    return x * y
"""

sec_result = sec_eval.evaluate(test_code, "python")
print(f"  Security score: {sec_result.overall:.3f}")

static_result = static_eval.evaluate(test_code, "python")
print(f"  Static analysis score: {static_result.overall:.3f}")

complexity_result = complexity_eval.evaluate(test_code, "python")
print(f"  Complexity score: {complexity_result.overall:.3f}")

print("✓ All evaluations completed successfully")

print("\n" + "="*60)
print("SUCCESS! Evaluation system is working correctly.")
print("="*60)
print(f"\nSample Results for Test Code:")
print(f"  Security: {sec_result.overall:.3f} ({'SAFE' if sec_result.safe else 'UNSAFE'})")
print(f"  Static Analysis: {static_result.overall:.3f}")
print(f"    - Pylint: {static_result.pylint_score}/10")
print(f"    - Flake8 violations: {static_result.flake8_violations}")
print(f"    - Mypy errors: {static_result.mypy_errors}")
print(f"  Complexity: {complexity_result.overall:.3f}")
print(f"    - Avg complexity: {complexity_result.average_complexity}")
print(f"    - Maintainability Index: {complexity_result.maintainability_index}")
print(f"    - Functions analyzed: {complexity_result.total_functions}")
