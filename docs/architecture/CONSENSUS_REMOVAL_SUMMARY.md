# Consensus Removed - Sequential Only

## Changes Made

### 1. Removed Consensus Fallback
File: [collaborative_orchestrator.py](collaborative_orchestrator.py:217-261)

**Before:**
```python
if self.use_sequential and self.sequential_orchestrator:
    try:
        # Sequential workflow
    except Exception as e:
        print(f"Falling back to consensus")  # ← BAD
        # Consensus code...
```

**After:**
```python
if not self.use_sequential or not self.sequential_orchestrator:
    raise RuntimeError("Consensus has been removed.")  # ← GOOD

# Only sequential workflow
workflow_result = await self.sequential_orchestrator.execute_workflow(...)
```

### 2. Simplified Sequential Workflow

**Issue:** Sequential orchestrator was trying to use agents that don't exist:
- ❌ `TESTER` - Not in config.yaml
- ❌ `REFINER` - Not in config.yaml (should reuse CODER)

**Agents that DO exist in config:**
- ✅ `architect`
- ✅ `coder`
- ✅ `reviewer`
- ✅ `documenter`
- ✅ `researcher` (unused)

**Solution:** Simplified workflow to 4 stages:
1. **ARCHITECT** → Design
2. **CODER** → Implement
3. **REVIEWER** → Review (can iterate with coder)
4. **DOCUMENTER** → Document

## Real Baseline: Single Model

You're right - the real baseline is **single model requests**, not consensus.

**Facilitair_v2 experience:**
- Consensus testing was misleading
- Collaborative (sequential) was the effective method
- Need to compare against single model to show actual improvement

## Next Steps

1. ✅ Remove consensus completely
2. ⏭️ Fix sequential orchestrator to only use existing agents
3. ⏭️ Run proper evaluation: Sequential vs Single-Model
4. ⏭️ Measure hallucination rates for both
5. ⏭️ Generate comparison report

## Hallucination Detection

Created `HallucinationDetector` class that checks for:
- Non-existent imports/modules
- Impossible claims ("O(0) complexity", "100% accuracy")
- Contradictions
- Made-up syntax
- Excessive confidence without substance

Both sequential and single-model will be tested for hallucinations.
