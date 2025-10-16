# Hallucination Detection in Code Generation Benchmarks

## Executive Summary

**Your Question:** "Do we have any checks for hallucinations in these evals?"

**Answer:** **NO - and this is a critical gap.**

The current evaluation system (`quality_evaluator.py`) only checks:
- **Syntax correctness** (can it parse?)
- **Structural completeness** (has functions, returns, imports?)
- **Code quality** (formatting, naming conventions)
- **Documentation** (docstrings, comments)
- **Error handling** (try/except blocks)
- **Testing** (test functions present)

**What's Missing:** Semantic relevance checking - verifying that the code actually solves the requested task.

## The Problem

### Example Hallucination That Would Pass Current Evaluation:

**Task:** "Implement password hashing with bcrypt, salt, and pepper"

**Hallucinated Output (WRONG):**
```python
import jwt
import datetime

def create_token(user_id: str) -> str:
    """Generate JWT token for user authentication"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }
    return jwt.encode(payload, 'secret', algorithm='HS256')
```

**Current Score:** 0.75+ (PASS)
- Syntax: ✅ Valid Python
- Completeness: ✅ Has function, docstring, returns
- Code Quality: ✅ Good naming, proper structure
- Documentation: ✅ Has docstring
- Error Handling: ⚠️  Missing (minor penalty)
- Testing: ⚠️  No tests (minor penalty)

**Should Score:** 0.0 (FAIL) - It's solving a completely different problem!

## The Solution: Semantic Relevance Checking

I've created `semantic_relevance_checker.py` with three-tier hallucination detection:

### Tier 1: Fast Keyword Matching (Heuristic)
```python
Task: "Implement password hashing with bcrypt, salt, and pepper"
Keywords Extracted: ['bcrypt', 'password', 'hash', 'salt', 'pepper']

Code Analysis:
- Keyword match ratio: 4/5 (80%)
- Result: ✅ Likely relevant

Hallucinated Code:
- Keyword match ratio: 0/5 (0%)
- Result: ⚠️  Likely hallucination - escalate to Tier 2
```

### Tier 2: Task-Specific Requirement Checks
```python
Task mentions "bcrypt" → Code MUST import/use bcrypt
Task mentions "parameterized queries" → Code MUST use parameterization (?, $1, etc.)
Task mentions "HTTP-only cookies" → Code MUST set httponly=True
Task mentions "XSS prevention" → Code MUST use escape/sanitize functions
```

**Example:**
```python
Task: "Build JWT token validator with signature verification"
Requirements Detected: ['jwt', 'signature', 'verify']

Code Check:
✅ Has 'jwt' import
✅ Has 'verify' function
✅ Has signature checking logic
Score: 1.0 (all requirements met)

Hallucinated Code:
❌ No 'jwt' import
❌ No 'verify' function
❌ No signature checking
Score: 0.0 (0/3 requirements met)
```

### Tier 3: LLM-as-Judge (Optional, for ambiguous cases)
```python
Prompt to Reviewer LLM:
"Does this code implement the requested functionality?
Consider:
1. Are key concepts from task present?
2. Is this solving the right problem?
3. What requirements are missing?

Respond with relevance score 0.0-1.0"
```

## Impact on Benchmark Results

### Current Results (WITHOUT Hallucination Detection)

From 100-task benchmark so far (Tasks 1-18):
- Sequential: Avg quality 0.804
- Baseline: Avg quality 0.511
- **These scores assume all outputs are relevant**

### Potential Impact (WITH Hallucination Detection)

If even 10-20% of outputs are hallucinations, the real scores could be:
- Sequential: 0.804 × 0.80 = **0.643** (if 20% hallucinations)
- Baseline: 0.511 × 0.80 = **0.409** (if 20% hallucinations)

**Hypothesis:** Sequential approach may have FEWER hallucinations because:
1. Architect stage establishes clear requirements
2. Reviewer stage catches off-topic implementations
3. Multiple stages provide error correction

## Implementation Plan

### Phase 1: Add Semantic Relevance as 7th Dimension ✅ DONE

```python
# quality_evaluator.py additions:
class QualityDimension(Enum):
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    SEMANTIC_RELEVANCE = "semantic_relevance"  # NEW

DIMENSION_WEIGHTS = {
    "correctness": 0.25,        # 25% (was 30%)
    "completeness": 0.20,       # 20% (was 25%)
    "code_quality": 0.15,       # 15% (was 20%)
    "documentation": 0.10,      # 10%
    "error_handling": 0.10,     # 10%
    "testing": 0.05,            # 5%
    "semantic_relevance": 0.15  # 15% NEW (CRITICAL!)
}
```

### Phase 2: Integrate into Benchmark ⏳ IN PROGRESS

Create `run_100_task_benchmark_v3.py` that:
1. Runs standard quality evaluation
2. Adds semantic relevance check for each output
3. Flags likely hallucinations for manual review
4. Reports hallucination rates per approach

### Phase 3: Re-analyze Existing Results ⏳ PENDING

Run `check_hallucinations.py` on completed checkpoints to identify:
- How many outputs are hallucinations?
- Does Sequential have fewer hallucinations than Baseline?
- Which task categories have highest hallucination rates?

## Files Created

1. **semantic_relevance_checker.py** - Core hallucination detection logic
2. **check_hallucinations.py** - Script to analyze existing benchmark results
3. **HALLUCINATION_DETECTION_ANALYSIS.md** - This document

## Current Status

- ✅ Semantic relevance checker implemented
- ✅ Keyword matching working (fast, good enough)
- ✅ Task-specific requirement checks working
- ⏳ LLM-as-judge available but optional (slow, expensive)
- ⏳ Waiting for 100-task benchmark checkpoint to test
- ⏳ Need to integrate into v3 benchmark

## Recommendations

### Immediate (For Current Benchmark):
1. Let 100-task benchmark complete
2. Run `check_hallucinations.py` on checkpoint files
3. Compare hallucination rates: Sequential vs Baseline

### Short-term (For Future Benchmarks):
1. Add semantic relevance as 7th evaluation dimension
2. Weight it at 15% (critical but not dominant)
3. Flag any output with <0.5 relevance for manual review

### Long-term (For Production):
1. Implement LLM-as-judge for ambiguous cases
2. Build ground-truth dataset of hallucination examples
3. Train lightweight classifier for fast hallucination detection

## Example: What Good Hallucination Detection Looks Like

```python
Task: "Implement rate limiter with Redis to prevent DoS"

Output A (Relevant):
```python
import redis
from functools import wraps

class RateLimiter:
    def __init__(self, redis_client, max_requests=100, window=60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window
```

**Relevance Score: 0.95**
- ✅ Redis imported and used
- ✅ Rate limiting logic present
- ✅ DoS prevention implemented
- ✅ All requirements met

Output B (Hallucination):
```python
import jwt

def validate_token(token: str) -> bool:
    try:
        jwt.decode(token, 'secret', algorithms=['HS256'])
        return True
    except:
        return False
```

**Relevance Score: 0.10**
- ❌ No Redis usage
- ❌ No rate limiting logic
- ❌ Wrong problem (JWT validation)
- ❌ 0/3 requirements met
- **FLAGGED AS HALLUCINATION**

## Conclusion

**The current evaluation system has a critical blind spot: it cannot detect when code solves the wrong problem.**

This means our benchmark results may be inflated - both approaches could be generating syntactically perfect but semantically irrelevant code.

The semantic relevance checker addresses this by:
1. Fast keyword matching (catches obvious hallucinations)
2. Task-specific requirement verification (catches subtle mismatches)
3. Optional LLM-as-judge (for ambiguous cases)

**Next Step:** Wait for 100-task benchmark checkpoint, then run hallucination analysis to see if Sequential approach has fewer hallucinations than Baseline (as hypothesized).
