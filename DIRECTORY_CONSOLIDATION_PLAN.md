# Directory Consolidation Plan - Weavehacks Collaborative

## Identified Duplicates

### 1. **Evaluation Documentation**
- `docs/evaluation/` - 2 files (analysis, guide)
- `docs/evaluations/` - 1 file (results)
- **Action**: Merge into `docs/evaluation/`

### 2. **CLI Directories**
- `cli/` - 3 streaming client files + README
- `src/cli/` - 1 debate file + __init__.py
- **Action**: Consolidate into `src/cli/` (keep code in src tree)

### 3. **Config Directories**
- `config/` - 6 YAML configuration files
- `src/config/` - 3 Python config modules
- **Action**: Keep separate (YAML in config/, Python in src/config/) - these serve different purposes

### 4. **Results Directories**
- `results/` - Active results with analysis/, benchmarks/, evaluations/ subdirs + recent files
- `test_results/` - Empty benchmarks/, evaluations/ dirs + logs/ with some files
- **Action**: Merge into `results/` with clear subdirectory structure

### 5. **Root-Level Files**
- `task1_comparison.html` - Move to results/analysis/
- `benchmark_10_v2_results_20251017_160421.json` - Move to results/benchmarks/
- `cli.py` - Keep in root (entry point)
- `api.py` - Keep in root (entry point)

## Proposed Final Structure

```
weavehacks-collaborative/
├── agents/                    # Agent implementations
├── api.py                     # FastAPI entry point (root)
├── backend/                   # Backend services
│   ├── routers/
│   ├── services/
│   └── streaming/
├── cli.py                     # CLI entry point (root)
├── config/                    # YAML configuration files
│   ├── config.yaml
│   ├── config_opensource_only.yaml
│   ├── config_premium.yaml
│   ├── evaluation.yaml
│   ├── model_selector.yaml
│   └── model_strategy_config.yaml
├── docs/                      # All documentation
│   ├── README.md
│   ├── EVALUATION_CONFIGURATION.md
│   ├── EVALUATION_SYSTEM.md
│   ├── MODEL_SELECTOR_STRATEGIES.md
│   ├── USER_WORKFLOW.md
│   ├── Old_Review/           # Facilitair artifacts
│   ├── architecture/
│   ├── archive/
│   ├── evaluation/           # ← Merged (evaluation + evaluations)
│   ├── guides/
│   ├── integrations/
│   ├── planning/
│   └── project/
├── integrations/              # External integrations
├── results/                   # All test/benchmark results
│   ├── analysis/             # Analysis files + HTML reports
│   ├── benchmarks/           # Benchmark JSON files
│   ├── evaluations/          # Evaluation results
│   └── logs/                 # Test logs
├── scripts/                   # Utility scripts
│   ├── analysis/
│   ├── benchmarks/
│   ├── demos/
│   ├── evaluation/
│   ├── setup/
│   └── testing/
├── src/                       # Source code
│   ├── caching/
│   ├── cli/                  # ← Consolidated CLI code
│   ├── config/               # Python config modules
│   ├── evaluation/           # Evaluation code
│   ├── middleware/
│   ├── orchestrators/
│   ├── security/
│   └── utils/
├── tests/                     # Test suites
│   ├── evaluation/
│   ├── fallback/
│   ├── integration/
│   ├── security/
│   └── unit/
└── utils/                     # Utility modules
```

## Consolidation Steps

### Step 1: Merge Evaluation Docs
```bash
mv docs/evaluations/*.md docs/evaluation/
rmdir docs/evaluations
```

### Step 2: Consolidate CLI
```bash
# Move root cli files to src/cli/
mv cli/streaming_client.py src/cli/
mv cli/streaming_client_simple.py src/cli/
mv cli/README.md src/cli/
rmdir cli/
```

### Step 3: Merge Results
```bash
# Move test_results content to results
mv test_results/logs/*.log results/logs/ 2>/dev/null || true
rmdir test_results/benchmarks test_results/evaluations
mv test_results/README.md results/
rmdir test_results
```

### Step 4: Organize Root Files
```bash
# Move analysis/benchmark files to results
mv task1_comparison.html results/analysis/
mv benchmark_10_v2_results_20251017_160421.json results/benchmarks/
```

## Import Updates Required

### Files importing from `cli/`:
- None found (root cli/ contains only standalone scripts)

### Files importing from `utils/`:
- Need to verify utils/ contents and update if needed

### Files that may reference old paths:
- docs/guides/ may reference old directory structure
- README.md may reference old paths

## Benefits

1. **Single Source of Truth**: No duplicate directories
2. **Clear Separation**:
   - Code in `src/`
   - Config YAML in `config/`
   - Config Python in `src/config/`
   - Results in `results/`
   - Docs in `docs/`
3. **Easier Navigation**: Logical grouping
4. **Cleaner Root**: Only entry points (api.py, cli.py) and essential files
