# Codebase Cleanup & Reorganization Plan

## Phase 1: Kill Background Processes & Assess Current State

### Tasks:
- [ ] Kill all background evaluation/API processes
- [ ] Audit current file structure
- [ ] Identify naming issues (AI-heavy terms, unclear names)
- [ ] Identify excessive comments
- [ ] Map out ideal file structure

---

## Phase 2: Naming Convention Cleanup

### Issues to Fix:
1. **AI-Heavy Names** - Remove buzzwords, use clear technical terms
2. **Unclear Variable Names** - Make intent obvious
3. **Inconsistent Naming** - Standardize across codebase

### Files to Update:
- [ ] `collaborative_orchestrator.py` - Remove "self-improving" buzzwords
- [ ] `sequential_orchestrator.py` - Simplify naming
- [ ] `agents/llm_client.py` - Clear, technical names only
- [ ] `agents/strategy_selector.py` - Remove marketing language
- [ ] All evaluation scripts - Professional naming

### Tracking Context:
```
RENAME_MAP.json will track:
{
  "old_name": "new_name",
  "reason": "why changed",
  "files_affected": ["list", "of", "files"]
}
```

---

## Phase 3: Comment Reduction

### Strategy:
- Keep: Docstrings, complex algorithm explanations, API contracts
- Remove: Obvious comments, cheerleader comments, redundant explanations

### Files with Excessive Comments:
- [ ] `cli.py` - Remove verbose comments
- [ ] `api.py` - Keep only API contract docs
- [ ] All orchestrator files - Essential logic only
- [ ] Evaluation scripts - Minimal comments

---

## Phase 4: File & Folder Reorganization

### Current Structure Issues:
```
weavehacks-collaborative/
+-- Many files in root (cluttered)
+-- agents/ (good)
+-- integrations/ (good)
+-- tests/ (good)
+-- utils/ (good)
+-- Lots of .md files in root (should be docs/)
```

### Target Structure:
```
facilitair/
+-- src/
|   +-- core/
|   |   +-- orchestrator.py (was: collaborative_orchestrator.py)
|   |   +-- sequential.py (was: sequential_orchestrator.py)
|   |   +-- workflow.py (new: workflow definitions)
|   +-- agents/
|   |   +-- client.py (was: llm_client.py)
|   |   +-- selector.py (was: strategy_selector.py)
|   |   +-- profiles.py (new: agent configurations)
|   +-- api/
|   |   +-- server.py (was: api.py)
|   |   +-- routes/ (new: endpoint modules)
|   |   +-- models.py (new: request/response models)
|   +-- cli/
|   |   +-- commands.py (was: cli.py)
|   |   +-- utils.py (new: CLI helpers)
|   +-- evaluation/
|   |   +-- runner.py (was: run_sequential_vs_baseline_eval.py)
|   |   +-- tasks.py (new: task definitions)
|   |   +-- metrics.py (new: scoring logic)
|   +-- utils/
|       +-- validators.py (was: api_key_validator.py)
|       +-- config.py (new: configuration management)
+-- tests/
|   +-- unit/
|   +-- integration/
|   +-- fixtures/
+-- docs/
|   +-- README.md (main)
|   +-- api/ (API documentation)
|   +-- architecture/ (design docs)
|   +-- guides/ (user guides)
+-- config/
|   +-- model_strategy.yaml
|   +-- agents.yaml
+-- scripts/
|   +-- setup.sh
|   +-- evaluate.sh
+-- .github/
|   +-- workflows/ (CI/CD)
+-- requirements.txt
+-- setup.py
+-- pyproject.toml
```

### File Move Tracking:

**MOVE_CONTEXT.json** will track every move:
```json
{
  "moves": [
    {
      "from": "collaborative_orchestrator.py",
      "to": "src/core/orchestrator.py",
      "imports_to_update": ["cli.py", "api.py", "tests/"],
      "status": "pending"
    }
  ]
}
```

---

## Phase 5: Documentation Organization

### Current Docs (Root Directory Clutter):
- README.md [OK] (keep in root)
- ARCHITECTURE_EXPLANATION.md → docs/architecture/overview.md
- INTERFACES_README.md → docs/guides/interfaces.md
- SEQUENTIAL_COLLABORATION_DESIGN.md → docs/architecture/sequential.md
- SUBMISSION.md → docs/project/submission.md
- SECURITY_NOTICE.md → docs/security.md
- SPONSOR_INTEGRATIONS.md → docs/integrations/sponsors.md
- FINAL_SPONSOR_SUMMARY.md → docs/project/sponsors.md
- CLEANUP_PLAN.md → docs/project/cleanup.md
- All *_STATUS.md files → docs/project/status/

### Target Docs Structure:
```
docs/
+-- README.md (index, links to all docs)
+-- guides/
|   +-- quickstart.md
|   +-- cli.md
|   +-- api.md
|   +-- evaluation.md
+-- architecture/
|   +-- overview.md
|   +-- sequential.md
|   +-- agents.md
|   +-- workflow.md
+-- api/
|   +-- endpoints.md
|   +-- models.md
|   +-- authentication.md
+-- integrations/
|   +-- weave.md
|   +-- openrouter.md
|   +-- sponsors.md
+-- project/
|   +-- submission.md
|   +-- sponsors.md
|   +-- status/
+-- security.md
```

---

## Phase 6: Import Path Updates

### After Each Move:
1. Update all imports in affected files
2. Run tests to verify nothing broke
3. Update MOVE_CONTEXT.json with status
4. Commit with descriptive message

### Import Update Pattern:
```python
# Before
from collaborative_orchestrator import SelfImprovingCollaborativeOrchestrator

# After
from src.core.orchestrator import Orchestrator
```

---

## Phase 7: Testing & Validation

### After All Changes:
- [ ] Run all unit tests
- [ ] Run integration tests
- [ ] Test CLI commands
- [ ] Test API endpoints
- [ ] Run small evaluation (10 tasks)
- [ ] Verify W&B Weave tracking still works

---

## Execution Order:

1. **Phase 1**: Kill processes, create context files [OK]
2. **Phase 2**: Rename variables/functions (no file moves yet)
3. **Phase 3**: Remove excessive comments
4. **Phase 4**: Move files (track in MOVE_CONTEXT.json)
5. **Phase 5**: Organize documentation
6. **Phase 6**: Update all imports iteratively
7. **Phase 7**: Test everything
8. **Final**: Commit, run 100-task eval

---

## Context Files to Maintain:

### MOVE_CONTEXT.json
Tracks every file move and import updates needed

### RENAME_MAP.json
Tracks variable/function/class renames

### TEST_STATUS.md
Tracks what's been tested after each change

### IMPORT_GRAPH.md
Visual map of import dependencies
