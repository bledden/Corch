# Enhanced Evaluation System Implementation Plan
## Corch (weavehacks-collaborative)

**Date**: 2025-10-17
**Priority**: HIGH
**Status**: IMPLEMENTATION READY

---

## Executive Summary

This document outlines the implementation of advanced code evaluation tools beyond AST analysis for the Corch collaborative coding platform. The enhancements will provide multi-dimensional quality assessment including security, performance, maintainability, and semantic correctness.

---

## Current State

### Existing Evaluation (AST-Based)
- **Correctness** (30%): Syntax validation
- **Completeness** (25%): Has functions/classes/logic
- **Code Quality** (20%): Line length, naming, spacing
- **Documentation** (10%): Docstrings, comments
- **Error Handling** (10%): Try/except, validation
- **Testing** (5%): Test functions, assertions

**Limitations**:
- No security vulnerability detection
- No performance/complexity analysis
- No semantic correctness validation
- No runtime testing
- No static analysis tool integration

---

## Enhancement Goals

### 1. **Security Analysis** (CRITICAL)
**Tools**: bandit (Python), eslint-plugin-security (JS/TS)
**What it detects**:
- SQL injection risks
- Command injection
- Insecure deserialization
- Hardcoded secrets
- Unsafe eval/exec usage
- OWASP Top 10 violations

**Implementation**:
```python
# src/evaluation/security_evaluator.py
class SecurityEvaluator:
    def evaluate(self, code: str, language: str) -> SecurityScore:
        if language == "python":
            return self._run_bandit(code)
        elif language in ["javascript", "typescript"]:
            return self._run_eslint_security(code)
        # ... other languages
```

**When to run**:
- After Refiner stage (before Documenter)
- Optional: After every Coder iteration

**Latency**: ~200-500ms per scan

---

### 2. **Static Analysis Tools**
**Tools**: pylint, flake8, mypy (Python), ESLint (JS/TS)
**What it detects**:
- Code quality violations
- Type errors
- Style inconsistencies
- Unused imports/variables
- Cyclomatic complexity

**Implementation**:
```python
# src/evaluation/static_analysis_evaluator.py
class StaticAnalysisEvaluator:
    def evaluate(self, code: str, language: str) -> StaticAnalysisScore:
        scores = {}
        if language == "python":
            scores["pylint"] = self._run_pylint(code)
            scores["flake8"] = self._run_flake8(code)
            scores["mypy"] = self._run_mypy(code)
        return StaticAnalysisScore(scores)
```

**When to run**:
- After Refiner stage
- Report findings to Refiner for auto-fix

**Latency**: ~300-800ms per analysis

---

### 3. **Performance & Complexity Analysis**
**Tools**: radon (Python), complexity-report (JS)
**What it measures**:
- Cyclomatic complexity
- Cognitive complexity
- Maintainability index
- Lines of code metrics

**Implementation**:
```python
# src/evaluation/complexity_evaluator.py
class ComplexityEvaluator:
    def evaluate(self, code: str, language: str) -> ComplexityScore:
        if language == "python":
            cc = self._radon_cyclomatic_complexity(code)
            mi = self._radon_maintainability_index(code)
            return ComplexityScore(cyclomatic=cc, maintainability=mi)
```

**When to run**:
- After final Refiner output
- Flag high-complexity functions for review

**Latency**: ~100-300ms

---

### 4. **LLM-as-Judge Evaluation**
**Purpose**: Semantic correctness that static tools miss
**What it evaluates**:
- Does code actually solve the task?
- Is the logic correct?
- Are edge cases handled?
- Is the approach efficient?
- Code elegance and best practices

**Implementation**:
```python
# src/evaluation/llm_judge_evaluator.py
class LLMJudgeEvaluator:
    def __init__(self, judge_model="meta-llama/llama-3.3-70b-instruct"):
        self.judge_model = judge_model

    async def evaluate(self, code: str, task: str, architecture: str) -> JudgeScore:
        prompt = f'''You are an expert code reviewer. Evaluate this code:

Task: {task}

Architecture: {architecture}

Code:
{code}

Rate on scale 0.0-1.0:
1. Correctness: Does it solve the task?
2. Logic: Is the approach sound?
3. Edge cases: Are they handled?
4. Efficiency: Is it performant?
5. Best practices: Does it follow standards?

Return JSON with scores and explanation.'''

        response = await self.llm.complete(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return self._parse_judge_response(response)
```

**When to run**:
- After final Documenter output
- Use for final quality gate decision

**Latency**: ~2-5 seconds (LLM call)

**Note**: Judge should NOT be a current team member to avoid bias

---

## Question 2: Middleware System for Tool Injection

### Design: Configurable Stage Interceptors

```python
# src/orchestrators/middleware.py
from enum import Enum
from typing import List, Callable, Optional

class StageHook(Enum):
    PRE_ARCHITECT = "pre_architect"
    POST_ARCHITECT = "post_architect"
    PRE_CODER = "pre_coder"
    POST_CODER = "post_coder"
    PRE_REVIEWER = "pre_reviewer"
    POST_REVIEWER = "post_reviewer"
    PRE_REFINER = "pre_refiner"
    POST_REFINER = "post_refiner"
    PRE_DOCUMENTER = "pre_documenter"
    POST_DOCUMENTER = "post_documenter"

class ToolMiddleware:
    """Intercepts stage outputs to run additional tools"""

    def __init__(self):
        self.hooks: Dict[StageHook, List[Callable]] = {}

    def register_hook(self, stage: StageHook, tool: Callable):
        """Register a tool to run at a specific stage"""
        if stage not in self.hooks:
            self.hooks[stage] = []
        self.hooks[stage].append(tool)

    async def run_hooks(self, stage: StageHook, context: Dict) -> Dict:
        """Run all registered hooks for a stage"""
        if stage not in self.hooks:
            return context

        for tool in self.hooks[stage]:
            try:
                result = await tool(context)
                # Merge tool results back into context
                context["tool_results"] = context.get("tool_results", {})
                context["tool_results"][tool.__name__] = result

                # Optionally modify the code if tool suggests fixes
                if result.get("auto_fix"):
                    context["code"] = result["fixed_code"]
                    context["modifications"].append({
                        "tool": tool.__name__,
                        "changes": result["changes"]
                    })
            except Exception as e:
                context["tool_errors"] = context.get("tool_errors", [])
                context["tool_errors"].append({
                    "tool": tool.__name__,
                    "error": str(e)
                })

        return context
```

### Usage Example:

```python
# In sequential_orchestrator.py
class SequentialCollaborativeOrchestrator:
    def __init__(self, config):
        self.middleware = ToolMiddleware()

        # Register tools at specific stages
        if config.get("enable_security_scan"):
            self.middleware.register_hook(
                StageHook.POST_REFINER,
                self._run_security_scan
            )

        if config.get("enable_static_analysis"):
            self.middleware.register_hook(
                StageHook.POST_REFINER,
                self._run_static_analysis
            )

        if config.get("enable_llm_judge"):
            self.middleware.register_hook(
                StageHook.POST_DOCUMENTER,
                self._run_llm_judge
            )

    async def _refiner_stage(self, context):
        # ... existing refiner logic ...

        # Run post-refiner hooks
        context = await self.middleware.run_hooks(
            StageHook.POST_REFINER,
            context
        )

        # If security issues found, send back to refiner
        if context.get("tool_results", {}).get("security_scan"):
            security_issues = context["tool_results"]["security_scan"]
            if security_issues["severity"] == "HIGH":
                print(f"[SECURITY] High severity issues found, re-refining...")
                context["review"]["critical_issues"].extend(
                    security_issues["issues"]
                )
                # Trigger another refiner iteration
                return await self._refiner_stage(context)

        return context
```

---

## Answer to Your Questions

### **Q1: Tools at each step vs final result only?**

**Recommendation**: Hybrid approach
- **POST_REFINER**: Security scan, static analysis (allows auto-fix before documenter)
- **POST_DOCUMENTER**: LLM-as-judge, complexity analysis (final quality gate)

**Why**:
- Running security/static analysis AFTER Refiner allows one more iteration to fix issues
- Running LLM-judge at the END avoids contaminating the team's thinking
- Latency is acceptable (~1-2s added to POST_REFINER, ~2-5s to POST_DOCUMENTER)

### **Q2: Does this add too much latency?**

**Latency Analysis**:
```
Current sequential flow: ~30-40s per task
With enhancements:
  - POST_REFINER tools: +1-2s (security + static analysis)
  - POST_DOCUMENTER tools: +2-5s (LLM judge + complexity)
Total: ~33-47s per task (+10-17% increase)
```

**Verdict**: Acceptable latency increase for 4-5x more evaluation dimensions

### **Q3: Risky to modify code between stages?**

**Risk Mitigation**:
- Only apply AUTO-FIX for non-breaking changes (formatting, unused imports)
- For security issues, ADD to review feedback (don't auto-fix)
- Keep audit trail of all modifications
- Allow user to disable auto-fix via config

```python
context["modifications"] = [
    {
        "stage": "post_refiner",
        "tool": "flake8",
        "type": "auto_fix",
        "changes": "Removed 3 unused imports"
    },
    {
        "stage": "post_refiner",
        "tool": "bandit",
        "type": "flagged",
        "issues": ["HIGH: eval() usage detected"]
    }
]
```

### **Q4: LLM-as-Judge when judge is also a teammate?**

**Solution**: Use SEPARATE model as judge
- **Team models**: DeepSeek (coder), Llama-3.3-70b (reviewer), etc.
- **Judge model**: **Use different model** (e.g., GPT-4o, Claude-3.5, or different Llama instance)
- Judge evaluates FINAL output only, doesn't participate in creation
- This avoids "judging your own work" bias

**Implementation**:
```python
class LLMJudgeEvaluator:
    def __init__(self, judge_model="openai/gpt-4o"):
        # Use premium model as judge (not part of team)
        assert judge_model not in self.orchestrator.team_models
        self.judge_model = judge_model
```

---

## Implementation Priority

### Phase 1: Core Tool Integration (Week 1)
1. ✅ Security evaluator (bandit for Python)
2. ✅ Static analysis evaluator (pylint, flake8)
3. ✅ Complexity evaluator (radon)
4. ✅ Middleware system for stage hooks

### Phase 2: LLM Judge (Week 1-2)
5. ✅ LLM-as-judge evaluator
6. ✅ Language-specific model routing

### Phase 3: Language-Specific Security (Week 2)
7. ✅ JavaScript/TypeScript security (eslint-plugin-security)
8. ✅ Multi-language static analysis routing

### Phase 4: Configuration & Testing (Week 2-3)
9. ✅ User-configurable tool enablement
10. ✅ Comprehensive test suite
11. ✅ Benchmark with/without enhancements

---

## Configuration Schema

```yaml
# config/evaluation.yaml
evaluation:
  security:
    enabled: true
    stage: post_refiner
    auto_fix: false  # Only flag, don't fix
    severity_threshold: "MEDIUM"

  static_analysis:
    enabled: true
    stage: post_refiner
    tools:
      - pylint
      - flake8
      - mypy
    auto_fix: true  # Auto-fix formatting/imports

  complexity:
    enabled: true
    stage: post_documenter
    max_cyclomatic_complexity: 10
    min_maintainability_index: 20

  llm_judge:
    enabled: true
    stage: post_documenter
    model: "openai/gpt-4o"  # Premium judge model
    temperature: 0.0
    criteria:
      - correctness
      - logic
      - edge_cases
      - efficiency
      - best_practices
```

---

## Expected Outcomes

### Evaluation Dimensions (Before → After)
- **Before**: 6 dimensions (AST-based)
- **After**: 11+ dimensions (AST + security + static analysis + complexity + LLM judge)

### Quality Gate Improvements
- **Security**: Block high-severity vulnerabilities
- **Maintainability**: Flag overly complex code
- **Semantic Correctness**: LLM judge catches logic errors AST misses

### Benchmark Targets
- **Pass@1 improvement**: +5-10% (better code quality)
- **Hallucination reduction**: -20% (LLM judge catches nonsense)
- **Security score**: 95%+ (no critical vulnerabilities)

---

## Next Steps for Implementation

1. **Create evaluator modules** in `src/evaluation/`:
   - `security_evaluator.py`
   - `static_analysis_evaluator.py`
   - `complexity_evaluator.py`
   - `llm_judge_evaluator.py`

2. **Implement middleware system** in `src/orchestrators/middleware.py`

3. **Integrate into sequential_orchestrator.py**:
   - Add hook registration
   - Add hook execution at stage boundaries
   - Handle tool results and auto-fixes

4. **Update evaluation metrics** in `collaborative_orchestrator.py`:
   - Add new dimension scores
   - Weight security/complexity appropriately

5. **Create configuration system** for user control

6. **Test and benchmark** with/without enhancements

---

## Dependencies to Install

```bash
# Python tools
pip install bandit pylint flake8 mypy radon

# JavaScript tools (via npm)
npm install -g eslint eslint-plugin-security complexity-report
```

---

## Conclusion

This enhanced evaluation system will provide **production-grade code quality assessment** beyond simple AST analysis. The middleware architecture allows flexible tool injection without major refactoring, and the configurable approach lets users balance quality vs latency.

**Ready to implement**: All design decisions finalized, implementation can begin immediately.
