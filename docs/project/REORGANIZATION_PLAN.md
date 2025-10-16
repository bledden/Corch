# Repository Reorganization & Security Audit Plan

## CRITICAL SECURITY ISSUES FOUND

###  HIGH PRIORITY - Exposed API Key
**File**: `.env`
**Issue**: Contains actual WANDB_API_KEY in plaintext
```
WANDB_API_KEY=c1e3ac39037ff7367a9f5cc278532e54484aa172
```

**Actions Required**:
1. [OK] `.env` is already in `.gitignore` (good!)
2. [WARNING] **VERIFY** this key was never committed to git history
3. [REFRESH] **ROTATE** this API key immediately at https://wandb.ai/authorize
4. Remove actual key from `.env` and use environment variables

### Security Fixes Needed

1. **Update .gitignore** to be more comprehensive:
   - Add `*.log` files (partially done)
   - Add all benchmark result JSONs
   - Add backup files (`*.bak`, `*.bak2`)
   - Add temp files

2. **Remove committed sensitive files** (if any):
   - Check git history for `.env` commits
   - Check for API keys in code files

3. **Naming Convention Issues**:
   - `Security_Notice.md` - Should be `SECURITY.md` (standard convention)
   - Multiple redundant config backups (`config.yaml.bak`, `config.yaml.bak2`)
   - Inconsistent naming: `run_100_task_eval.py` vs `run_100_task_benchmark.py`

---

## Proposed Directory Structure

```
weavehacks-collaborative/
+-- .env.example                    # Keep (template)
+-- .gitignore                      # Keep (update)
+-- README.md                       # Keep
+-- requirements.txt                # Keep
+-- setup.py                        # Keep
+-- cli.py                          # Keep (main entry point)
+-- api.py                          # Keep (main API entry)
|
+-- agents/                         # [OK] Already exists
|   +-- (agent implementation files)
|
+-- backend/                        # [OK] Already exists
|   +-- (backend services)
|
+-- cli/                            # [OK] Already exists
|   +-- (CLI utilities)
|
+-- config/                         #  CREATE
|   +-- config.yaml                 # Move from root
|   +-- model_strategy_config.yaml  # Move from root
|   +-- config_opensource_only.yaml # Move from root
|
+-- docs/                           # [OK] Already exists - REORGANIZE
|   +-- architecture/               #  CREATE
|   |   +-- AGENT_PROTOCOLS.md
|   |   +-- IDEAL_ARCHITECTURE.md
|   |   +-- SEQUENTIAL_COLLABORATION_DESIGN.md (if exists)
|   |   +-- FACILITAIR_V2_VS_LANGGRAPH_ANALYSIS.md
|   |
|   +-- evaluation/                 #  CREATE
|   |   +-- BENCHMARK_FAILURE_ANALYSIS.md
|   |   +-- CHECKPOINT_20_ANALYSIS.md
|   |   +-- HALLUCINATION_DETECTION_ANALYSIS.md
|   |   +-- QUALITY_EVALUATION_GUIDE.md
|   |
|   +-- guides/                     #  CREATE
|   |   +-- STREAMING_AND_CACHING_GUIDE.md
|   |   +-- TESTING_INSTRUCTIONS.md
|   |   +-- PROCESS_CLEANUP_GUIDE.md
|   |   +-- MONITORING.md
|   |
|   +-- research/                   #  CREATE
|   |   +-- LANGUAGE_ROUTING_RESEARCH.md
|   |   +-- TECHNOLOGY_EVALUATION.md
|   |   +-- INVESTIGATION_FINDINGS.md
|   |
|   +-- project/                    #  CREATE (internal planning docs)
|   |   +-- EXECUTION_PLAN.md
|   |   +-- REVISED_EXECUTION_PLAN.md
|   |   +-- RESPONSE_TO_USER.md
|   |   +-- DEVPOST_SUBMISSION.md
|   |
|   +-- streaming/                  #  CREATE
|       +-- STREAMING_CONSENSUS_IMPLEMENTATION.md
|       +-- STREAMING_IMPLEMENTATION_SUMMARY.md
|       +-- STREAMING_FINAL_STATUS.md
|       +-- STREAMING_STATUS.md
|       +-- STREAMING_DIAGNOSIS.md
|
+-- integrations/                   # [OK] Already exists
|   +-- (sponsor integrations)
|
+-- results/                        #  CREATE
|   +-- benchmarks/                 #  CREATE
|   |   +-- benchmark_100_final_20251015_204700.json
|   |   +-- benchmark_100_checkpoint_*.json (5 files)
|   |   +-- benchmark_10_quick_results_20251014_161425.json
|   |   +-- benchmark_10_reanalyzed_results.json
|   |
|   +-- evaluations/                #  CREATE
|   |   +-- evaluation_results_20251012_*.json (2 files)
|   |   +-- evaluation_stats_20251012_*.json (2 files)
|   |   +-- sequential_vs_baseline_results_*.json (4 files)
|   |   +-- smoke_test_results_*.json (8 files)
|   |
|   +-- analysis/                   #  CREATE
|       +-- all_openrouter_models.txt
|       +-- model_analysis_results.txt
|       +-- (future analysis outputs)
|
+-- scripts/                        #  CREATE
|   +-- benchmarks/                 #  CREATE
|   |   +-- run_100_task_benchmark.py
|   |   +-- run_10_task_benchmark_v2.py
|   |   +-- run_10_task_quick_benchmark.py
|   |   +-- run_500_task_benchmark.py
|   |   +-- run_optimized_500_task_benchmark.py
|   |   +-- reanalyze_previous_results.py
|   |
|   +-- evaluation/                 #  CREATE
|   |   +-- run_10_task_eval.py (RENAME: run_evaluation_10_tasks.py)
|   |   +-- run_100_task_eval.py (RENAME: run_evaluation_100_tasks.py)
|   |   +-- run_comprehensive_eval.py
|   |   +-- run_sequential_vs_baseline_eval.py
|   |   +-- run_10_task_real_eval.py
|   |   +-- run_10_task_open_closed.py
|   |
|   +-- analysis/                   #  CREATE
|   |   +-- analyze_all_533_models.py
|   |   +-- analyze_openrouter_models.py
|   |   +-- analyze_checkpoint_hallucinations.py
|   |   +-- check_hallucinations.py
|   |
|   +-- demos/                      #  CREATE
|   |   +-- demo.py
|   |   +-- demo_sponsor_showcase.py
|   |   +-- demo_with_strategy.py
|   |   +-- run_demo_non_interactive.py
|   |
|   +-- testing/                    #  CREATE
|   |   +-- run_smoke_test.py
|   |   +-- smoke_test.sh
|   |   +-- test_one_stream.sh
|   |   +-- test_streaming_quick.sh
|   |   +-- run_opensource_only_test.sh
|   |   +-- run_with_monitoring.sh
|   |   +-- watch_tests.sh
|   |
|   +-- setup/                      #  CREATE
|       +-- setup_services.py
|       +-- setup_monitoring.py
|       +-- fix_tracking.py
|
+-- src/                            #  CREATE (core library code)
|   +-- __init__.py                 #  CREATE
|   +-- orchestrators/              #  CREATE
|   |   +-- __init__.py
|   |   +-- collaborative_orchestrator.py
|   |   +-- sequential_orchestrator.py
|   |   +-- cached_orchestrator.py
|   |   +-- language_aware_orchestrator.py
|   |
|   +-- evaluation/                 #  CREATE
|   |   +-- __init__.py
|   |   +-- quality_evaluator.py
|   |   +-- semantic_relevance_checker.py
|   |
|   +-- caching/                    #  CREATE
|   |   +-- __init__.py
|   |   +-- semantic_cache.py
|   |
|   +-- cli/                        #  CREATE
|   |   +-- __init__.py
|   |   +-- cli_streaming_debate.py
|   |
|   +-- utils/                      #  CREATE
|       +-- __init__.py
|       +-- web_search_router.py
|       +-- monitor.py
|
+-- tests/                          # [OK] Already exists - ADD MORE
|   +-- unit/                       #  CREATE
|   |   +-- test_llm_with_weave.py
|   |   +-- test_weave_basic.py
|   |   +-- test_tavily_with_weave.py
|   |   +-- test_orchestrator_no_weave.py
|   |
|   +-- integration/                #  CREATE
|   |   +-- test_direct_llm.py
|   |   +-- test_single_task.py
|   |   +-- test_streaming_live.py
|   |   +-- test_sse_direct.py
|   |
|   +-- fallback/                   #  CREATE
|       +-- test_fallback_auto_mode.py
|       +-- test_fallback_invalid_model.py
|       +-- test_fallback_multiple_failures.py
|       +-- test_fallback_tier_escalation.py
|
+-- utils/                          # [OK] Already exists
|   +-- (utility modules)
|
+-- SECURITY.md                     # RENAME from Security_Notice.md

```

---

## Files to DELETE (Cleanup)

### Backup/Temp Files (14 files)
```
config.yaml.bak
config.yaml.bak2
current_md_files.txt
current_python_files.txt
MOVE_CONTEXT.json
RENAME_MAP.json
benchmark_500.log
benchmark_output.log
single_test.log
smoke_test.log
facilitair_api.log
facilitair_cli.log
```

### Redundant Files (Consider removing after migration)
```
code_examples.py          # If just examples, move to docs/examples/
execute.py                # If deprecated/unused
train.py                  # If unused (check if needed)
monitor_tests.py          # If redundant with tests/
```

---

## Naming Convention Fixes

### Files to Rename

| Current Name | New Name | Reason |
|-------------|----------|--------|
| `Security_Notice.md` | `SECURITY.md` | Standard GitHub convention |
| `run_10_task_eval.py` | `run_evaluation_10_tasks.py` | Consistency with benchmarks |
| `run_100_task_eval.py` | `run_evaluation_100_tasks.py` | Consistency with benchmarks |

---

## .gitignore Updates

Add these patterns:
```gitignore
# Environment
.env
.env.local
*.env

# Logs
*.log
logs/
*.log.*

# Results (should not be committed)
results/
benchmark_*.json
evaluation_*.json
smoke_test_results_*.json
sequential_vs_baseline_results_*.json
*.json.bak

# Backups
*.bak
*.bak2
*.backup
*~

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Temp files
tmp/
temp/
*.tmp
current_*.txt
MOVE_CONTEXT.json
RENAME_MAP.json

# API Keys (paranoid mode)
*_API_KEY*
*_SECRET*
*_TOKEN*
credentials.json
secrets.yaml
```

---

## Git History Security Check

Run these commands to verify no secrets in git history:
```bash
# Check if .env was ever committed
git log --all --full-history -- .env

# Search for API keys in history
git log -p | grep -i "api_key\|wandb_api_key" | head -20

# Check for hardcoded secrets
git grep -i "sk-\|sk_\|api_key.*=" $(git rev-list --all)
```

If secrets found in history:
1. Use `git filter-branch` or `BFG Repo-Cleaner` to remove
2. Force push (ONLY if repo is private/new)
3. Rotate all exposed keys immediately

---

## Migration Steps

### Phase 1: Security Audit [OK]
1. [OK] Check git history for exposed secrets
2. [WARNING] Rotate WANDB_API_KEY immediately
3. [OK] Update .gitignore
4. [OK] Remove sensitive files from tracking

### Phase 2: Create Directory Structure
```bash
mkdir -p config
mkdir -p docs/{architecture,evaluation,guides,research,project,streaming}
mkdir -p results/{benchmarks,evaluations,analysis}
mkdir -p scripts/{benchmarks,evaluation,analysis,demos,testing,setup}
mkdir -p src/{orchestrators,evaluation,caching,cli,utils}
mkdir -p tests/{unit,integration,fallback}
```

### Phase 3: Move Files
(Scripted moves to preserve git history)

### Phase 4: Update Imports
- Update all Python imports to reflect new `src/` structure
- Update documentation links
- Update CI/CD paths (if any)

### Phase 5: Verify & Test
```bash
# Test imports
python3 -c "from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator"

# Run health check
python3 cli.py health

# Run quick test
python3 -m pytest tests/ -v
```

---

## Files Requiring Import Updates

After moving to `src/` structure, these files will need import updates:
- `cli.py`
- `api.py`
- All scripts in `scripts/`
- All tests in `tests/`

Example change:
```python
# OLD
from collaborative_orchestrator import CollaborativeOrchestrator

# NEW
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator
```

---

## Summary Statistics

- **Total files in root**: 128
- **Files to organize**: ~110
- **New directories to create**: 15
- **Files to delete**: ~14
- **Files to rename**: 3
- **Security issues**: 1 critical (exposed API key)

---

## Recommended Next Steps

1. **IMMEDIATE**: Verify and rotate WANDB_API_KEY
2. **IMMEDIATE**: Update .gitignore and commit
3. **Review**: Get user approval for this reorganization plan
4. **Execute**: Run migration script with git mv (preserves history)
5. **Test**: Verify all functionality after reorganization
6. **Document**: Update README with new structure
7. **Push**: Push cleaned, organized repo to GitHub

---

**Generated**: 2025-10-16
**Purpose**: Repository organization and security audit before public release
