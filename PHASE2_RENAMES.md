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
- [ ] Update class definition
- [ ] Update all imports
- [ ] Update all instantiations  
- [ ] Update documentation
- [ ] Run smoke tests
- [ ] Commit changes
