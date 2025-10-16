# Test Evaluation Investigation Findings

## Date: 2025-10-12

## Executive Summary
Analyzed 10-task test results (test IDs: 6ef037, 2de595, d0a12f). Found 4 critical issues preventing accurate evaluation.

## Test Results Overview

| Test ID | Strategy | Sequential | Baseline | Status |
|---------|----------|------------|----------|--------|
| 6ef037  | BALANCED | 0% (0/10)  | 60% (6/10) | ❌ FAILED - metadata bug |
| 2de595  | CLOSED   | 0% (0/10)  | 40% (4/10) | ❌ FAILED - metadata bug |
| d0a12f  | OPEN     | 0% (0/10)  | 100% (10/10) | ❌ FAILED - metadata bug |

## Critical Issues Found

### 1. CollaborationResult.metadata Bug (BLOCKING)

**Location**: `run_smoke_test.py:140`

**Error**: `'CollaborationResult' object has no attribute 'metadata'`

**Code**:
```python
models_used = result.metadata.get("models_used", {})  # ❌ FAILS
primary_model = models_used.get("coder", "unknown")
```

**Impact**: ALL 20/20 sequential tasks failed before evaluation could complete

**Root Cause**: The `CollaborativeOrchestrator.collaborate()` method returns an object without a `.metadata` attribute

**Fix Required**:
- Option A: Add `.metadata` attribute to the return object
- Option B: Use a different method to get models_used (e.g., from orchestrator state)
- Option C: Remove this requirement and use configured model instead

### 2. False Passes in Baseline Evaluation (ACCURACY ISSUE)

**Location**: `run_smoke_test.py:215-224`

**Issue**: Too lenient pass criteria

**Current Logic**:
```python
has_code = any(marker in output for marker in ["```", "def ", "function ", "class "])
has_substantial_output = len(output.strip()) > 100

quality_estimate = (
    0.8 if (has_code and has_substantial_output) else
    0.5 if has_code else 0.2
)

pass_at_1 = (quality_estimate >= 0.7 and ...)
```

**Impact**:
- d0a12f (OPEN source): 100% pass rate (10/10) - likely inflated
- Simple outputs with code blocks automatically get 0.8 quality score
- No validation of actual code correctness or completeness

**Evidence**:
- All 10 OPEN source baseline tasks passed with 0.8 quality
- CLOSED source tasks 6-10 returned only architecture docs (no code) and failed
- Quality scoring doesn't distinguish between complete vs incomplete implementations

**Recommendations**:
1. Add code completeness check (e.g., function definitions, imports, etc.)
2. Validate code structure (not just markers)
3. Consider LLM-based quality evaluation for accuracy
4. Add minimum line count for code blocks

### 3. Invalid DeepSeek Model ID (OPERATIONAL)

**Location**: `config.yaml` (lines 42-43)

**Invalid IDs**:
```yaml
- deepseek-ai/deepseek-r1        # ❓ Verify
- deepseek-ai/deepseek-v3        # ❌ INVALID
- deepseek-ai/deepseek-v2.5      # ❓ Verify
- deepseek-ai/deepseek-coder     # ❓ Verify
```

**Valid IDs on OpenRouter**:
```
deepseek/deepseek-v3.2-exp
deepseek/deepseek-chat-v3.1
deepseek/deepseek-chat-v3.1:free
```

**Impact**:
- Tests 6d54b4, 3adb17, c9f053 all killed due to repeated invalid model errors
- Error: `deepseek-ai/deepseek-v3 is not a valid model ID`

**Fix Required**: Update config.yaml with valid OpenRouter model IDs

### 4. GPT-5 Incomplete Code Generation (QUALITY ISSUE)

**Location**: Test 2de595 (CLOSED source), Tasks 6-10

**Observation**: GPT-5 returned architecture documents but no actual code implementation

**Evidence from test output**:
```
Task 6: [LLM] coder using openai/gpt-5: ...
[LLM] reviewer using anthropic/claude-sonnet-4.5:
  "critical_issues": ["No actual code implementation provided..."]

Task 7-10: Same pattern repeated
```

**Impact**:
- CLOSED source baseline scored only 40% (4/10)
- Tasks 1-5: Passed with code
- Tasks 6-10: Failed - only architecture docs, no implementation

**Analysis**:
- GPT-5 may have hit context limits or followed a different prompt interpretation
- More complex web-search tasks failed while simple tasks succeeded
- Reviewer correctly identified missing implementations

**Recommendations**:
1. Investigate GPT-5 prompt engineering for consistent code output
2. Add explicit "provide complete implementation" instruction
3. Consider shorter, more focused prompts for complex tasks
4. Verify token limits and context window usage

## Comparison: OPEN vs CLOSED Source

### OPEN Source Models (d0a12f)
- **Models**: Qwen/qwen3-coder-plus, DeepSeek/deepseek-chat, Llama 3.3, Mistral Codestral
- **Baseline Pass Rate**: 100% (10/10)
- **Average Quality**: 0.80
- **Average Duration**: 32.03s per task
- **Verdict**: ⚠️ Likely inflated due to lenient evaluation

### CLOSED Source Models (2de595)
- **Models**: GPT-5, GPT-5-Pro, Claude Sonnet 4.5
- **Baseline Pass Rate**: 40% (4/10)
- **Average Quality**: 0.44
- **Average Duration**: 22.86s per task
- **Verdict**: ⚠️ GPT-5 incomplete code issue

## Sequential Method Analysis

**Current Status**: 0% pass rate across ALL tests (0/30 tasks)

**Root Cause**: CollaborationResult.metadata bug prevents evaluation from completing

**Expected Behavior**:
- Once bug is fixed, sequential method should be properly evaluated
- Sequential uses full architect → coder → reviewer → documenter → final coder workflow
- Should theoretically produce higher quality than baseline single-model approach

## Recommended Actions (Priority Order)

### P0 - CRITICAL (Blocks all testing)
1. ✅ **Fix CollaborationResult.metadata bug**
   - Prevents all sequential evaluations
   - Must be fixed before any further testing

### P1 - HIGH (Affects accuracy)
2. ✅ **Fix invalid DeepSeek model IDs**
   - Update config.yaml with valid OpenRouter IDs
   - Enables OPEN source model testing

3. ✅ **Improve baseline evaluation criteria**
   - Add code completeness validation
   - Prevent false positives from simple outputs

### P2 - MEDIUM (Quality improvements)
4. ⏳ **Investigate GPT-5 incomplete code generation**
   - Debug why GPT-5 returns architecture docs without code
   - Improve prompt engineering for consistent output

5. ⏳ **Re-run all tests after fixes**
   - Validate sequential method works correctly
   - Get accurate OPEN vs CLOSED comparison
   - Verify 100% OPEN source pass rate is legitimate

## Test Environment Details

- **Python Version**: 3.9 (deprecation warning for Weave)
- **LLM Router**: OpenRouter via LiteLLM
- **Tracing**: Weights & Biases Weave
- **Test Framework**: run_smoke_test.py (10-task evaluation)
- **Evaluation Method**: Pass@1 with quality threshold (>0.7)

## Files Modified/To Modify

1. `run_smoke_test.py` - Fix metadata access (line 140)
2. `config.yaml` - Update DeepSeek model IDs (lines 42-45)
3. `run_smoke_test.py` - Improve baseline evaluation (lines 215-224)
4. TBD - Investigate GPT-5 prompt/context issues

## Conclusion

The test infrastructure has critical bugs preventing accurate evaluation. The apparent "OPEN source wins 100% vs CLOSED source 40%" result is unreliable due to:
1. Sequential method completely broken (metadata bug)
2. Baseline evaluation too lenient (false positives)
3. GPT-5 not generating complete code (unexpected behavior)

Once these issues are resolved, we can get a true comparison of sequential collaboration vs baseline approaches, and OPEN vs CLOSED source models.
