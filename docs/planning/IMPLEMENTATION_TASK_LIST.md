# Enhanced Evaluation System - Detailed Implementation Task List
## Corch (weavehacks-collaborative)

**Project**: Production-grade code evaluation beyond AST
**Status**: READY TO BEGIN
**Estimated Timeline**: 2-3 weeks
**Date**: 2025-10-17

---

## Task Ownership Legend
- 🤖 **Claude** - AI can implement autonomously
- 👤 **User** - Requires user decision/approval/testing
- 🤝 **Both** - Collaborative (Claude implements, User reviews/approves)

---

## Phase 1: Foundation & Core Evaluators (Week 1)

### Task 1.1: Install Required Dependencies
**Owner**: 🤖 Claude
**Estimate**: 5 minutes
**Status**: PENDING

**Steps**:
1. Create `requirements-evaluation.txt` with:
   ```
   bandit>=1.7.5
   pylint>=3.0.0
   flake8>=6.1.0
   mypy>=1.7.0
   radon>=6.0.0
   ```
2. Update main `requirements.txt` to include evaluation deps
3. Test installation in dev environment

**Deliverable**: Working installation of all evaluation tools

**Dependencies**: None

**Approval Required**: 👤 User confirms tools install correctly

---

### Task 1.2: Create Security Evaluator (Python - Bandit)
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Create `src/evaluation/security_evaluator.py`
2. Implement `SecurityEvaluator` class:
   ```python
   class SecurityEvaluator:
       def evaluate(self, code: str, language: str) -> SecurityScore
       def _run_bandit(self, code: str) -> Dict
       def _categorize_by_severity(self, issues: List) -> Dict
   ```
3. Define `SecurityScore` dataclass with:
   - `safe: bool`
   - `critical_issues: List[Dict]`
   - `medium_issues: List[Dict]`
   - `low_issues: List[Dict]`
   - `overall_score: float`
4. Write unit tests in `tests/test_security_evaluator.py`

**Test Cases**:
- ✅ Code with no security issues → score 1.0
- ✅ Code with `eval()` usage → critical issue flagged
- ✅ Code with hardcoded password → medium issue flagged
- ✅ Code with `assert` in production → low issue flagged

**Deliverable**: Working security evaluator for Python

**Dependencies**: None

**Approval Required**: 🤝 User reviews test results, approves approach

---

### Task 1.3: Create Static Analysis Evaluator (Pylint/Flake8)
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Create `src/evaluation/static_analysis_evaluator.py`
2. Implement `StaticAnalysisEvaluator` class:
   ```python
   class StaticAnalysisEvaluator:
       def evaluate(self, code: str, language: str) -> StaticAnalysisScore
       def _run_pylint(self, code: str) -> Dict
       def _run_flake8(self, code: str) -> Dict
       def _run_mypy(self, code: str) -> Dict
       def _aggregate_scores(self, results: Dict) -> float
   ```
3. Define `StaticAnalysisScore` dataclass with:
   - `overall: float`
   - `pylint_score: float`
   - `flake8_violations: int`
   - `mypy_errors: int`
   - `violations: List[Dict]`
   - `auto_fixable: List[Dict]`
4. Write unit tests

**Test Cases**:
- ✅ Clean code → high scores
- ✅ Code with unused imports → flake8 catches
- ✅ Code with type errors → mypy catches
- ✅ Code with complexity issues → pylint catches

**Deliverable**: Working static analysis evaluator

**Dependencies**: Task 1.1

**Approval Required**: 🤝 User reviews scoring thresholds, auto-fix rules

---

### Task 1.4: Create Complexity Evaluator (Radon)
**Owner**: 🤖 Claude
**Estimate**: 1-2 hours
**Status**: PENDING

**Steps**:
1. Create `src/evaluation/complexity_evaluator.py`
2. Implement `ComplexityEvaluator` class:
   ```python
   class ComplexityEvaluator:
       def evaluate(self, code: str, language: str) -> ComplexityScore
       def _radon_cyclomatic_complexity(self, code: str) -> float
       def _radon_maintainability_index(self, code: str) -> float
       def _identify_complex_functions(self, code: str) -> List[Dict]
   ```
3. Define `ComplexityScore` dataclass with:
   - `overall: float`
   - `average_complexity: float`
   - `maintainability_index: float`
   - `complex_functions: List[Dict]`  # Functions with CC > threshold
4. Write unit tests

**Test Cases**:
- ✅ Simple function → low complexity
- ✅ Nested conditionals → high complexity flagged
- ✅ Long function → low maintainability flagged

**Deliverable**: Working complexity evaluator

**Dependencies**: Task 1.1

**Approval Required**: 🤝 User sets complexity thresholds (default: CC > 10 is flagged)

---

### Task 1.5: Create LLM-as-Judge Evaluator
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Create `src/evaluation/llm_judge_evaluator.py`
2. Implement `LLMJudgeEvaluator` class:
   ```python
   class LLMJudgeEvaluator:
       def __init__(self, judge_model: str = "openai/gpt-4o")
       async def evaluate(self, code: str, task: str, architecture: str) -> JudgeScore
       def _create_judge_prompt(self, code: str, task: str, architecture: str) -> str
       def _parse_judge_response(self, response: str) -> JudgeScore
   ```
3. Define `JudgeScore` dataclass with:
   - `overall: float`
   - `correctness: float`
   - `logic_quality: float`
   - `edge_case_handling: float`
   - `efficiency: float`
   - `best_practices: float`
   - `explanation: str`
4. Design judge prompt template
5. Write unit tests (mocked LLM responses)

**Test Cases**:
- ✅ Judge evaluates simple correct code → high scores
- ✅ Judge catches logical error → low correctness score
- ✅ Judge identifies missing edge cases → low edge_case_handling
- ✅ JSON parsing handles malformed responses

**Deliverable**: Working LLM judge evaluator

**Dependencies**: None (uses existing LLM client)

**Approval Required**: 👤 User reviews judge prompt template, selects judge model

---

## Phase 2: Middleware System (Week 1-2)

### Task 2.1: Design Middleware Architecture
**Owner**: 🤝 Both (Claude designs, User approves)
**Estimate**: 1 hour
**Status**: PENDING

**Steps**:
1. Create `src/orchestrators/middleware.py`
2. Define `StageHook` enum with all hook points:
   ```python
   class StageHook(Enum):
       POST_ARCHITECT = "post_architect"
       POST_CODER = "post_coder"
       POST_REVIEWER = "post_reviewer"
       POST_REFINER = "post_refiner"
       POST_DOCUMENTER = "post_documenter"
   ```
3. Design `ToolMiddleware` class interface
4. Document hook execution flow

**Deliverable**: Middleware architecture document + skeleton code

**Dependencies**: None

**Approval Required**: 👤 User approves hook points and execution flow

---

### Task 2.2: Implement ToolMiddleware Class
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Implement `ToolMiddleware` class:
   ```python
   class ToolMiddleware:
       def __init__(self)
       def register_hook(self, stage: StageHook, tool: Callable)
       async def run_hooks(self, stage: StageHook, context: Dict) -> Dict
       def _apply_auto_fixes(self, context: Dict, fixes: List) -> Dict
       def _merge_tool_results(self, context: Dict, results: Dict) -> Dict
   ```
2. Implement context modification logic
3. Implement audit trail tracking
4. Add error handling for tool failures
5. Write unit tests

**Test Cases**:
- ✅ Hook registration and execution
- ✅ Multiple hooks at same stage execute in order
- ✅ Tool failures don't crash pipeline
- ✅ Auto-fix modifications are tracked
- ✅ Context properly merges tool results

**Deliverable**: Working middleware system

**Dependencies**: Task 2.1

**Approval Required**: 🤝 User reviews error handling strategy

---

### Task 2.3: Integrate Middleware into Sequential Orchestrator
**Owner**: 🤖 Claude
**Estimate**: 3-4 hours
**Status**: PENDING

**Steps**:
1. Update `src/orchestrators/sequential_orchestrator.py`:
   ```python
   class SequentialCollaborativeOrchestrator:
       def __init__(self, config):
           self.middleware = ToolMiddleware()
           self._register_evaluation_hooks()

       def _register_evaluation_hooks(self):
           # Register based on config

       async def _refiner_stage(self, context):
           # ... existing logic ...
           context = await self.middleware.run_hooks(
               StageHook.POST_REFINER, context
           )
           # Handle security issues found

       async def _documenter_stage(self, context):
           # ... existing logic ...
           context = await self.middleware.run_hooks(
               StageHook.POST_DOCUMENTER, context
           )
   ```
2. Add hook execution at POST_REFINER
3. Add hook execution at POST_DOCUMENTER
4. Implement feedback loop for security issues
5. Update tests

**Test Cases**:
- ✅ POST_REFINER hooks execute correctly
- ✅ Security issues trigger re-refinement
- ✅ POST_DOCUMENTER hooks execute correctly
- ✅ Pipeline completes without hooks (backward compatible)

**Deliverable**: Middleware integrated into orchestrator

**Dependencies**: Task 2.2, all Phase 1 evaluators

**Approval Required**: 🤝 User reviews integration points, tests end-to-end

---

## Phase 3: Configuration & Tool Registration (Week 2)

### Task 3.1: Create Configuration Schema
**Owner**: 🤖 Claude
**Estimate**: 1-2 hours
**Status**: PENDING

**Steps**:
1. Create `config/evaluation.yaml`:
   ```yaml
   evaluation:
     security:
       enabled: true
       stage: post_refiner
       auto_fix: false
       severity_threshold: "MEDIUM"
       block_critical: true

     static_analysis:
       enabled: true
       stage: post_refiner
       tools: [pylint, flake8, mypy]
       auto_fix: true
       auto_fix_types: [unused_imports, formatting]

     complexity:
       enabled: true
       stage: post_documenter
       max_cyclomatic_complexity: 10
       min_maintainability_index: 20

     llm_judge:
       enabled: true
       stage: post_documenter
       model: "openai/gpt-4o"
       temperature: 0.0
   ```
2. Create `EvaluationConfig` dataclass to load config
3. Add validation for config values
4. Write config loading tests

**Deliverable**: Configuration schema and loader

**Dependencies**: None

**Approval Required**: 👤 User reviews default values and config structure

---

### Task 3.2: Implement Tool Registration System
**Owner**: 🤖 Claude
**Estimate**: 2 hours
**Status**: PENDING

**Steps**:
1. Create `src/orchestrators/evaluation_tools.py`:
   ```python
   class EvaluationToolRegistry:
       def __init__(self, config: EvaluationConfig, middleware: ToolMiddleware)
       def register_all_tools(self)
       async def run_security_scan(self, context: Dict) -> Dict
       async def run_static_analysis(self, context: Dict) -> Dict
       async def run_complexity_analysis(self, context: Dict) -> Dict
       async def run_llm_judge(self, context: Dict) -> Dict
   ```
2. Implement conditional registration based on config
3. Wire up evaluator instances
4. Add tool wrapper functions
5. Write tests

**Test Cases**:
- ✅ Tools register only when enabled in config
- ✅ Tool wrappers call correct evaluators
- ✅ Tool results properly formatted
- ✅ Disabled tools don't execute

**Deliverable**: Tool registration system

**Dependencies**: Task 3.1, all Phase 1 evaluators

**Approval Required**: 🤝 User validates tool execution flow

---

### Task 3.3: Update Collaborative Orchestrator Metrics
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Update `src/orchestrators/collaborative_orchestrator.py`:
   ```python
   # Add new metric dimensions
   metrics = {
       "quality": quality_score,  # Existing AST score
       "security": context.get("tool_results", {}).get("security_scan", {}).get("overall_score", 1.0),
       "static_analysis": context.get("tool_results", {}).get("static_analysis", {}).get("overall", 1.0),
       "complexity": context.get("tool_results", {}).get("complexity_analysis", {}).get("overall", 1.0),
       "llm_judge": context.get("tool_results", {}).get("llm_judge", {}).get("overall", None),
       "overall": self._calculate_overall_score(...)
   }
   ```
2. Implement weighted overall score calculation
3. Update result object to include all dimension scores
4. Update logging to show new dimensions
5. Write tests

**Test Cases**:
- ✅ All dimensions included in metrics
- ✅ Overall score correctly weighted
- ✅ Missing dimensions handled gracefully
- ✅ Backward compatible with old code

**Deliverable**: Enhanced metrics in orchestrator

**Dependencies**: Task 2.3, Task 3.2

**Approval Required**: 🤝 User reviews weighting scheme for overall score

---

## Phase 4: Testing & Validation (Week 2-3)

### Task 4.1: Create Comprehensive Test Suite
**Owner**: 🤖 Claude
**Estimate**: 4-5 hours
**Status**: PENDING

**Steps**:
1. Create `tests/evaluation/` directory
2. Write integration tests:
   - `test_security_integration.py`
   - `test_static_analysis_integration.py`
   - `test_complexity_integration.py`
   - `test_llm_judge_integration.py`
   - `test_middleware_integration.py`
   - `test_end_to_end_evaluation.py`
3. Create test fixtures with known code samples:
   - Clean code (should pass all)
   - Insecure code (should fail security)
   - Complex code (should fail complexity)
   - Incorrect code (should fail LLM judge)
4. Run full test suite

**Test Cases**:
- ✅ Clean code passes all evaluators
- ✅ Vulnerable code caught by security
- ✅ Poor quality code caught by static analysis
- ✅ Complex code caught by radon
- ✅ Logically incorrect code caught by judge
- ✅ End-to-end pipeline with all tools enabled

**Deliverable**: Comprehensive test suite with >90% coverage

**Dependencies**: All previous tasks

**Approval Required**: 👤 User runs tests, validates results

---

### Task 4.2: Run Smoke Test with Enhanced Evaluation
**Owner**: 🤝 Both
**Estimate**: 30 minutes
**Status**: PENDING

**Steps**:
1. Update `scripts/testing/run_smoke_test.py` to enable new evaluation
2. Run 10-task smoke test with all evaluators enabled
3. Review results:
   - Check that all dimensions are populated
   - Verify security issues are flagged
   - Confirm complexity scores make sense
   - Review LLM judge feedback
4. Compare pass rates: with vs without enhanced evaluation

**Expected Outcomes**:
- Pass@1 may decrease initially (more strict evaluation)
- Security score should be 95%+ (few vulnerabilities)
- Complexity scores should identify overly complex solutions
- LLM judge should catch logic errors missed by AST

**Deliverable**: Smoke test results with enhanced evaluation

**Dependencies**: Task 4.1, all previous tasks

**Approval Required**: 👤 User reviews results, approves metrics

---

### Task 4.3: Benchmark Latency Impact
**Owner**: 🤖 Claude
**Estimate**: 1 hour
**Status**: PENDING

**Steps**:
1. Create `scripts/benchmarking/measure_evaluation_latency.py`
2. Run 50 tasks with evaluation tools disabled (baseline)
3. Run 50 tasks with POST_REFINER tools only
4. Run 50 tasks with POST_DOCUMENTER tools only
5. Run 50 tasks with all tools enabled
6. Generate latency comparison report

**Metrics to Measure**:
- Average task duration
- P50, P95, P99 latencies
- Latency breakdown by tool
- Overall throughput impact

**Expected Results**:
- Baseline: ~30-40s per task
- POST_REFINER tools: +1-2s (+3-5%)
- POST_DOCUMENTER tools: +2-5s (+5-12%)
- All tools: +3-7s (+10-17%)

**Deliverable**: Latency benchmark report

**Dependencies**: Task 4.2

**Approval Required**: 👤 User confirms latency is acceptable

---

## Phase 5: Documentation & Deployment (Week 3)

### Task 5.1: Write User Documentation
**Owner**: 🤖 Claude
**Estimate**: 2-3 hours
**Status**: PENDING

**Steps**:
1. Create `docs/ENHANCED_EVALUATION_GUIDE.md`:
   - Overview of new evaluation dimensions
   - Configuration guide
   - How to enable/disable tools
   - Interpreting results
   - Troubleshooting
2. Update main `README.md` with evaluation features
3. Add code examples
4. Create configuration templates for common scenarios

**Deliverable**: Complete user documentation

**Dependencies**: All previous tasks

**Approval Required**: 👤 User reviews docs for clarity

---

### Task 5.2: Create Migration Guide
**Owner**: 🤖 Claude
**Estimate**: 1 hour
**Status**: PENDING

**Steps**:
1. Create `docs/EVALUATION_MIGRATION_GUIDE.md`:
   - Backward compatibility notes
   - How to upgrade from old evaluation
   - Breaking changes (if any)
   - Configuration migration steps
   - FAQ
2. Test migration path on existing installation

**Deliverable**: Migration guide

**Dependencies**: Task 5.1

**Approval Required**: 👤 User validates migration steps

---

### Task 5.3: Update CI/CD Pipeline
**Owner**: 🤝 Both
**Estimate**: 1-2 hours
**Status**: PENDING

**Steps**:
1. Update GitHub Actions workflow (if exists)
2. Add evaluation tool installation to CI
3. Run full test suite in CI
4. Add smoke test with evaluation to CI
5. Configure failure thresholds

**Deliverable**: CI/CD runs with new evaluation

**Dependencies**: All previous tasks

**Approval Required**: 👤 User configures CI secrets/settings

---

### Task 5.4: Final Review & Release
**Owner**: 🤝 Both
**Estimate**: 2 hours
**Status**: PENDING

**Steps**:
1. Final code review of all changes
2. Run full test suite one more time
3. Update CHANGELOG.md
4. Tag release version
5. Deploy to production (if applicable)
6. Monitor for issues

**Deliverable**: Production-ready enhanced evaluation system

**Dependencies**: All previous tasks

**Approval Required**: 👤 User approves release

---

## Summary

### Total Estimated Time: 2-3 weeks

**Phase Breakdown**:
- Phase 1 (Evaluators): 8-12 hours
- Phase 2 (Middleware): 6-8 hours
- Phase 3 (Configuration): 5-7 hours
- Phase 4 (Testing): 5-6 hours
- Phase 5 (Documentation): 6-8 hours

**Total**: 30-41 hours of development work

### Task Distribution:
- 🤖 **Claude tasks**: 18 tasks (implementation)
- 👤 **User tasks**: 4 tasks (decisions/approvals)
- 🤝 **Collaborative tasks**: 10 tasks (review/approval)

### Critical Path:
1. Task 1.1 → Task 1.2-1.5 (can run in parallel)
2. Task 2.1 → Task 2.2 → Task 2.3
3. Task 3.1 → Task 3.2 → Task 3.3
4. Task 4.1 → Task 4.2 → Task 4.3
5. Task 5.1 → Task 5.2 → Task 5.3 → Task 5.4

---

## Next Steps

**User Actions Required**:
1. ✅ Review this task list
2. ✅ Approve approach and priorities
3. ✅ Confirm we should start with Task 1.1 (Dependencies)
4. ✅ Set any preferences for:
   - Judge model selection (default: openai/gpt-4o)
   - Complexity thresholds (default: CC > 10)
   - Security severity threshold (default: MEDIUM)
   - Auto-fix preferences (default: formatting/imports only)

**Claude's Next Action**:
Once approved, begin with **Task 1.1: Install Required Dependencies**

---

## Notes

- All code will be committed incrementally with clear commit messages
- Each task will be tested before moving to next
- User can pause/reprioritize at any phase boundary
- Breaking changes will be clearly documented
- Backward compatibility maintained throughout
