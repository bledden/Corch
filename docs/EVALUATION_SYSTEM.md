# Enhanced Evaluation System Documentation

## Overview

The Corch evaluation system provides comprehensive, multi-dimensional code quality assessment that goes beyond basic AST analysis. It integrates four complementary evaluators that work together to detect issues including security vulnerabilities, code quality problems, complexity issues, and semantic hallucinations.

## Table of Contents

1. [Architecture](#architecture)
2. [Evaluators](#evaluators)
3. [Configuration](#configuration)
4. [Integration](#integration)
5. [Scoring System](#scoring-system)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)

---

## Architecture

### System Overview

```
Sequential Collaboration Workflow
┌─────────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐
│  Architect  │ ──>│  Coder │ ──>│ Reviewer │ ──>│ Refiner │
└─────────────┘    └────────┘    └──────────┘    └─────────┘
                                                       │
                                    ┌──────────────────┘
                                    ▼
                        ┌───────────────────────────┐
                        │  POST_REFINER HOOK       │
                        │  Evaluation Middleware    │
                        └───────────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
    ┌────────────────┐   ┌────────────────────┐  ┌──────────────────┐
    │Security (30%)  │   │Static Analysis(30%)│  │Complexity (20%)  │
    │  - Bandit      │   │  - Pylint          │  │  - Radon         │
    │  - CVE scan    │   │  - Flake8          │  │  - Cyclomatic    │
    │               │   │  - Mypy            │  │  - Maintainability│
    └────────────────┘   └────────────────────┘  └──────────────────┘
             │                      │                      │
             │          ┌───────────────────────┐         │
             └────────> │  LLM Judge (20%)      │ <───────┘
                        │  - Claude Sonnet 4.5  │
                        │  - Semantic Analysis  │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │  Weighted Aggregation     │
                        │  Overall Score + Pass/Fail│
                        └───────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────────┐
                        │  W&B Weave Logging        │
                        │  Documenter Stage         │
                        └───────────────────────────┘
```

### Key Components

1. **Evaluators** - Four independent code quality analyzers
2. **Middleware** - Orchestrates evaluator execution at hook points
3. **Configuration** - YAML-based settings for all evaluators
4. **Integration** - Seamless connection to Sequential Orchestrator

---

## Evaluators

### 1. SecurityEvaluator (30% weight)

**Purpose:** Detect security vulnerabilities and dangerous code patterns

**Tool:** Bandit (Python security linting)

**Detected Issues:**
- `eval()` and `exec()` usage
- Hardcoded passwords/secrets
- SQL injection vulnerabilities
- Command injection risks
- Insecure pickle/yaml usage
- Weak cryptography
- 250+ security test patterns

**Scoring Algorithm:**
```python
score = 1.0
score -= len(critical_issues) * 0.4    # -0.4 per critical
score -= len(high_issues) * 0.2        # -0.2 per high
score -= len(medium_issues) * 0.1      # -0.1 per medium
score -= len(low_issues) * 0.05        # -0.05 per low
score = max(0.0, score)                # Floor at 0.0
```

**Example:**
```python
from src.evaluation.security_evaluator import SecurityEvaluator

evaluator = SecurityEvaluator()
code = """
def load_config(filepath):
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)  # CRITICAL: Pickle vulnerability
"""

result = evaluator.evaluate(code, "python")
print(f"Security Score: {result.overall}")  # 0.6 (1.0 - 0.4)
print(f"Critical Issues: {len(result.critical_issues)}")  # 1
```

---

### 2. StaticAnalysisEvaluator (30% weight)

**Purpose:** Validate code quality, style, and type correctness

**Tools:**
- **Pylint** (50% weight) - Code quality analyzer (0-10 scale)
- **Flake8** (25% weight) - PEP 8 style checker
- **Mypy** (25% weight) - Static type checker

**Detected Issues:**
- Syntax errors
- Undefined variables
- Unused imports
- Type errors
- PEP 8 violations
- Code smells
- Documentation gaps

**Scoring Algorithm:**
```python
pylint_normalized = pylint_score / 10.0     # 0-1 scale
flake8_score = max(0.0, 1.0 - (violations * 0.05))
mypy_score = max(0.0, 1.0 - (errors * 0.1))

overall = (pylint_normalized * 0.5 + 
           flake8_score * 0.25 + 
           mypy_score * 0.25)
```

**Example:**
```python
from src.evaluation.static_analysis_evaluator import StaticAnalysisEvaluator

evaluator = StaticAnalysisEvaluator()
code = '''
"""Calculator module"""
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
'''

result = evaluator.evaluate(code, "python")
print(f"Static Analysis Score: {result.overall}")  # ~0.85
print(f"Pylint Score: {result.pylint_score}/10")   # 8.5
print(f"Flake8 Violations: {result.flake8_violations}")  # 0
```

---

### 3. ComplexityEvaluator (20% weight)

**Purpose:** Measure code maintainability and complexity

**Tool:** Radon (Complexity metrics)

**Metrics:**
- **Cyclomatic Complexity** - Control flow complexity
- **Maintainability Index** - Overall maintainability (0-100)
- **Halstead Metrics** - Program difficulty
- **Function Ranks** - A (best) to F (worst)

**Scoring Algorithm:**
```python
# Maintainability Index (0-100 scale)
mi_score = maintainability_index / 100.0

# Average Complexity (lower is better)
avg_complexity_score = max(0.0, 1.0 - (avg_complexity - 1) * 0.1)

# Max Complexity (penalty for complex functions)
max_complexity_score = max(0.0, 1.0 - (max_complexity - 5) * 0.05)

# Weighted average
overall = (mi_score * 0.60 + 
           avg_complexity_score * 0.25 + 
           max_complexity_score * 0.15)
```

**Complexity Ranks:**
- **A** (1-5): Simple, easy to maintain
- **B** (6-10): More complex but manageable
- **C** (11-20): Complex, consider refactoring
- **D** (21-30): Very complex, needs refactoring
- **E** (31-40): Extremely complex
- **F** (41+): Unmaintainable

**Example:**
```python
from src.evaluation.complexity_evaluator import ComplexityEvaluator

evaluator = ComplexityEvaluator()
code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

result = evaluator.evaluate(code, "python")
print(f"Complexity Score: {result.overall}")  # ~0.95
print(f"Maintainability Index: {result.maintainability_index}")  # 95.2
print(f"Average Complexity: {result.average_complexity}")  # 2.0
```

---

### 4. LLMJudgeEvaluator (20% weight)

**Purpose:** Semantic code quality analysis and hallucination detection

**Tool:** Claude Sonnet 4.5 (via OpenRouter)

**Analysis Dimensions:**
1. **CORRECTNESS** (40%) - Does code solve the problem?
2. **BEST_PRACTICES** (25%) - Follows industry standards?
3. **READABILITY** (15%) - Is code understandable?
4. **EDGE_CASES** (15%) - Handles boundary conditions?
5. **DESIGN_PATTERNS** (5%) - Uses appropriate patterns?

**Hallucination Detection:**
- Fabricated APIs that don't exist
- Incorrect library usage
- False assumptions about behavior
- Logic errors not caught by static analysis
- Semantic correctness issues

**Scoring:**
```python
overall = (
    correctness * 0.40 +
    best_practices * 0.25 +
    readability * 0.15 +
    edge_cases * 0.15 +
    design_patterns * 0.05
)
```

**Example:**
```python
from src.evaluation.llm_judge_evaluator import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator(
    openrouter_api_key="your-key",
    judge_model="anthropic/claude-sonnet-4.5"
)

code = '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
'''

result = evaluator.evaluate(
    code=code,
    language="python",
    task_description="Check if number is prime"
)

print(f"LLM Judge Score: {result.overall}")  # 0.87
print(f"Correctness: {result.correctness}")  # 0.95
print(f"Best Practices: {result.best_practices}")  # 0.85
```

---

## Configuration

### config/evaluation.yaml

Complete configuration file for all evaluators:

```yaml
evaluation:
  # Global settings
  enabled: true
  gate_on_failure: false    # Block output if evaluation fails
  pass_threshold: 0.7       # Minimum overall score to pass
  
  # Hook points
  hooks:
    post_refiner: true      # Run after refiner stage
    post_documenter: false  # Run after documenter stage
  
  # Evaluator configurations
  evaluators:
    security:
      enabled: true
      weight: 0.30
      min_score: 0.6
      timeout: 30
      
    static_analysis:
      enabled: true
      weight: 0.30
      min_score: 0.6
      timeout: 60
      pylint_threshold: 7.0  # Minimum pylint score
      
    complexity:
      enabled: true
      weight: 0.20
      min_score: 0.6
      timeout: 30
      max_complexity_threshold: 10
      min_maintainability: 65.0
      
    llm_judge:
      enabled: true
      weight: 0.20
      min_score: 0.6
      timeout: 60
      model: "anthropic/claude-sonnet-4.5"
      temperature: 0.3
```

### Python Configuration API

```python
from src.config.evaluation_config import EvaluationConfig

# Load configuration
config = EvaluationConfig("config/evaluation.yaml")

# Check if evaluator is enabled
if config.is_evaluator_enabled("security"):
    print("Security evaluator is enabled")

# Get evaluator weight
weight = config.get_evaluator_weight("static_analysis")
print(f"Static analysis weight: {weight}")  # 0.30

# Get minimum score threshold
min_score = config.get_evaluator_min_score("complexity")
print(f"Min complexity score: {min_score}")  # 0.6

# Disable an evaluator programmatically
config.set_evaluator_enabled("llm_judge", False)
```

---

## Integration

### With Sequential Orchestrator

```python
from src.orchestrators.sequential_orchestrator import SequentialOrchestrator
from src.middleware.evaluation_middleware import EvaluationMiddleware
from src.config.evaluation_config import EvaluationConfig

# Load configuration
eval_config = EvaluationConfig()

# Create evaluation middleware
middleware = EvaluationMiddleware(
    enabled=eval_config.is_enabled(),
    enable_security=eval_config.is_evaluator_enabled("security"),
    enable_static_analysis=eval_config.is_evaluator_enabled("static_analysis"),
    enable_complexity=eval_config.is_evaluator_enabled("complexity"),
    enable_llm_judge=eval_config.is_evaluator_enabled("llm_judge"),
    pass_threshold=eval_config.get_pass_threshold(),
    gate_on_failure=eval_config.get_gate_on_failure()
)

# Create orchestrator with middleware
orchestrator = SequentialOrchestrator(
    model_manager=model_manager,
    middleware=[middleware]
)

# Run collaboration
result = orchestrator.run_collaboration(
    task="Write a secure login function",
    language="python"
)

# Access evaluation results
if hasattr(result, 'evaluation'):
    eval_result = result.evaluation
    print(f"Overall Score: {eval_result.overall_score}")
    print(f"Passed: {eval_result.passed}")
    print(f"Security: {eval_result.security_score}")
    print(f"Static Analysis: {eval_result.static_analysis_score}")
    print(f"Complexity: {eval_result.complexity_score}")
    print(f"LLM Judge: {eval_result.llm_judge_score}")
```

### Standalone Usage

```python
from src.middleware.evaluation_middleware import EvaluationMiddleware
from src.middleware.base_middleware import MiddlewareContext, MiddlewareHook

# Create middleware
middleware = EvaluationMiddleware(enabled=True)

# Create context
context = MiddlewareContext(
    hook=MiddlewareHook.POST_REFINER,
    stage_name="refiner",
    input_data={"task": "Write a sorting algorithm"},
    output_data={"code": "def bubble_sort(arr): ..."}
)

# Execute evaluation
result = middleware.execute(context)

# Access results
evaluation = result["evaluation"]
print(f"Score: {evaluation.overall_score}")
print(f"Strengths: {evaluation.strengths}")
print(f"Weaknesses: {evaluation.weaknesses}")
print(f"Recommendations: {evaluation.recommendations}")
```

---

## Scoring System

### Weighted Aggregation

```python
overall_score = (
    security_score * 0.30 +
    static_analysis_score * 0.30 +
    complexity_score * 0.20 +
    llm_judge_score * 0.20
)
```

### Pass/Fail Logic

Code passes evaluation if:
1. **Overall score** >= `pass_threshold` (default: 0.7)
2. **All individual scores** >= `min_score` (default: 0.6)

```python
passed = (
    overall_score >= pass_threshold and
    security_score >= min_scores['security'] and
    static_analysis_score >= min_scores['static_analysis'] and
    complexity_score >= min_scores['complexity'] and
    llm_judge_score >= min_scores['llm_judge']
)
```

### Score Interpretation

| Overall Score | Quality Level | Recommendation |
|--------------|---------------|----------------|
| 0.90 - 1.00  | Excellent     | Ready for production |
| 0.75 - 0.89  | Good          | Minor improvements recommended |
| 0.60 - 0.74  | Acceptable    | Needs improvement |
| 0.40 - 0.59  | Poor          | Significant refactoring required |
| 0.00 - 0.39  | Critical      | Major issues, do not deploy |

---

## Usage Examples

### Example 1: Basic Evaluation

```python
from src.middleware.evaluation_middleware import EvaluationMiddleware

middleware = EvaluationMiddleware(enabled=True)

code = '''
def add(a, b):
    return a + b
'''

result = middleware._run_evaluations(
    code=code,
    task_description="Add two numbers",
    language="python"
)

print(f"Overall: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")
```

### Example 2: Configuration-Based Evaluation

```python
from src.config.evaluation_config import EvaluationConfig
from src.middleware.evaluation_middleware import EvaluationMiddleware

# Load custom config
config = EvaluationConfig("my_evaluation.yaml")

# Create middleware from config
middleware = EvaluationMiddleware(
    enabled=config.is_enabled(),
    enable_security=config.is_evaluator_enabled("security"),
    enable_static_analysis=config.is_evaluator_enabled("static_analysis"),
    enable_complexity=config.is_evaluator_enabled("complexity"),
    enable_llm_judge=config.is_evaluator_enabled("llm_judge")
)

# Evaluate code
code = "..."  # Your code here
result = middleware._run_evaluations(code, "Task description", "python")
```

### Example 3: Handling Failed Evaluations

```python
middleware = EvaluationMiddleware(
    enabled=True,
    gate_on_failure=True  # Block output on failure
)

result = middleware._run_evaluations(code, task, language)

if not result.passed:
    print("Evaluation failed!")
    print(f"Weaknesses: {result.weaknesses}")
    print(f"Recommendations: {result.recommendations}")
    
    # Log to monitoring system
    logger.error(f"Code quality check failed: {result.overall_score}")
else:
    print("Evaluation passed!")
    deploy_to_production(code)
```

---

## API Reference

### EvaluationMiddleware

```python
class EvaluationMiddleware(BaseMiddleware):
    def __init__(
        self,
        enabled: bool = True,
        enable_security: bool = True,
        enable_static_analysis: bool = True,
        enable_complexity: bool = True,
        enable_llm_judge: bool = True,
        pass_threshold: float = 0.7,
        gate_on_failure: bool = False
    )
```

**Methods:**
- `execute(context: MiddlewareContext) -> Dict`: Execute evaluation
- `get_hooks() -> List[MiddlewareHook]`: Get registered hooks
- `should_execute(hook: MiddlewareHook) -> bool`: Check if hook active

### AggregatedEvaluationResult

```python
@dataclass
class AggregatedEvaluationResult:
    overall_score: float           # 0.0 - 1.0
    passed: bool                   # Pass/fail status
    security_score: float          # Individual scores
    static_analysis_score: float
    complexity_score: float
    llm_judge_score: float
    security_details: Optional[SecurityScore]
    static_analysis_details: Optional[StaticAnalysisScore]
    complexity_details: Optional[ComplexityScore]
    llm_judge_details: Optional[LLMJudgeScore]
    strengths: List[str]           # Positive feedback
    weaknesses: List[str]          # Issues found
    recommendations: List[str]     # Improvement suggestions
    evaluators_run: int            # Number of evaluators executed
    evaluators_failed: int         # Number of failures
```

### SecurityScore

```python
@dataclass
class SecurityScore:
    overall: float                           # 0.0 - 1.0
    safe: bool                              # True if no critical/high issues
    critical_issues: List[SecurityIssue]
    high_issues: List[SecurityIssue]
    medium_issues: List[SecurityIssue]
    low_issues: List[SecurityIssue]
    total_issues: int
    scanned_files: int
```

### StaticAnalysisScore

```python
@dataclass
class StaticAnalysisScore:
    overall: float                    # 0.0 - 1.0
    pylint_score: float              # 0.0 - 10.0
    flake8_violations: int
    mypy_errors: int
    errors: List[AnalysisIssue]
    warnings: List[AnalysisIssue]
    info: List[AnalysisIssue]
    hints: List[AnalysisIssue]
    total_issues: int
    passed: bool
    pylint_details: Dict
    flake8_details: Dict
    mypy_details: Dict
```

### ComplexityScore

```python
@dataclass
class ComplexityScore:
    overall: float                              # 0.0 - 1.0
    average_complexity: float
    max_complexity: int
    maintainability_index: float               # 0.0 - 100.0
    total_functions: int
    high_complexity_functions: List[FunctionComplexity]
    all_functions: List[FunctionComplexity]
    rank_a_count: int                          # Function rank distribution
    rank_b_count: int
    rank_c_count: int
    rank_d_count: int
    rank_e_count: int
    rank_f_count: int
    maintainability_rank: MaintainabilityRank
    passed: bool
```

### LLMJudgeScore

```python
@dataclass
class LLMJudgeScore:
    overall: float                    # 0.0 - 1.0
    correctness: float               # Individual dimension scores
    best_practices: float
    readability: float
    edge_cases: float
    design_patterns: float
    feedback: str                    # Detailed LLM feedback
    reasoning: str                   # Chain-of-thought reasoning
    passed: bool
```

---

## Testing

### Running Tests

```bash
# Run all evaluation tests
python3 -m pytest tests/evaluation/ -v

# Run specific test suite
python3 -m pytest tests/evaluation/test_security_evaluator.py -v
python3 -m pytest tests/evaluation/test_static_analysis_evaluator.py -v
python3 -m pytest tests/evaluation/test_evaluation_integration.py -v

# Run with coverage
python3 -m pytest tests/evaluation/ --cov=src/evaluation --cov-report=html
```

### Test Coverage

- SecurityEvaluator: 10 tests
- StaticAnalysisEvaluator: 9 tests
- Integration: 8 tests
- **Total: 27 tests (100% passing)**

---

## Performance Considerations

### Timeouts

All evaluators run with configurable timeouts:
- Security: 30s (default)
- Static Analysis: 60s (default)
- Complexity: 30s (default)
- LLM Judge: 60s (default)

### Parallel Execution

Evaluators run in **parallel** using isolated subprocesses for maximum performance:
```python
# All 4 evaluators run simultaneously
# Total time = max(evaluator_times), not sum
```

### Graceful Degradation

If an evaluator fails or times out:
- Returns safe default score (0.5)
- System continues with other evaluators
- Failure logged but doesn't block workflow

---

## Troubleshooting

### Common Issues

**1. LLM Judge Failing**
```
Error: OPENROUTER_API_KEY not set
Solution: export OPENROUTER_API_KEY=your-key
```

**2. Bandit Not Found**
```
Error: bandit: command not found
Solution: pip3 install bandit==1.7.7
```

**3. Low Scores**
```
Issue: All scores below 0.5
Solution: Check that evaluator tools are installed and in PATH
```

**4. Timeouts**
```
Issue: Evaluators timing out
Solution: Increase timeout in config/evaluation.yaml
```

---

## Best Practices

1. **Enable all evaluators** for comprehensive analysis
2. **Tune weights** based on your priorities
3. **Set appropriate thresholds** for your quality standards
4. **Monitor W&B Weave** for trends and patterns
5. **Review failed evaluations** to improve code generation
6. **Use gate_on_failure** for critical production code
7. **Test configuration** before deploying

---

## Future Enhancements

- [ ] Support for additional languages (JavaScript, TypeScript, Go, Rust)
- [ ] Custom evaluator plugins
- [ ] Real-time evaluation streaming
- [ ] Historical score tracking and trends
- [ ] Automated remediation suggestions
- [ ] Integration with CI/CD pipelines
- [ ] Performance profiling evaluator
- [ ] License compliance checker

---

## References

- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pylint Documentation](https://pylint.readthedocs.io/)
- [Radon Documentation](https://radon.readthedocs.io/)
- [Claude API Documentation](https://docs.anthropic.com/)
- [W&B Weave Documentation](https://docs.wandb.ai/guides/weave)

---

**Version:** 1.0.0  
**Last Updated:** 2025-10-17  
**Maintained By:** Facilitair Team
