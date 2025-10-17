# Phase 2: Naming Convention Cleanup

## Primary Renames

### Classes
1. `SelfImprovingCollaborativeOrchestrator` → `CollaborativeOrchestrator`
   - Reason: "Self-improving" is marketing fluff, not descriptive
   - Files affected: 33 references across codebase

### Variables/Methods (examples to address)
- `self_improving` → `orchestrator`
- `advance_generation()` → Remove (unused learning concept)
- `task_type_patterns` → `workflow_cache` (if used)

## Files to Update (in order)
1. collaborative_orchestrator.py (source)
2. cli.py (imports it)
3. api.py (imports it)
4. run_*.py (evaluation scripts)
5. tests/*.py
6. All other importers

## Progress Tracking
- [x] Update class definition
- [x] Update all imports
- [x] Update all instantiations
- [x] Update documentation
- [x] Run smoke tests
- [x] Commit changes

## Completed Changes

### Primary Rename
[OK] `SelfImprovingCollaborativeOrchestrator` → `CollaborativeOrchestrator`
- Updated in 14 files
- All imports fixed
- All instantiations updated
- Smoke tests passing
- Committed: refactor commit 3e48c98

### Notes on Deferred Items
- `advance_generation()` and `task_type_patterns` are only used in demo/training scripts (demo.py, train.py, execute.py)
- These are NOT used by main system (CLI, API, evaluations)
- Can be addressed later if needed, but not breaking main functionality
