# Evaluation System Configuration Guide

## Quick Start

The evaluation system is configured via `config/evaluation.yaml`. This guide explains all configuration options and common use cases.

## Configuration File Structure

```yaml
evaluation:
  # Global settings
  enabled: true
  gate_on_failure: false
  pass_threshold: 0.7
  
  # Hook points
  hooks:
    post_refiner: true
    post_documenter: false
  
  # Evaluator configurations
  evaluators:
    security:
      enabled: true
      weight: 0.30
      min_score: 0.6
      timeout: 30
      
    static_analysis:
      enabled: true
      weight: 0.30
      min_score: 0.6
      timeout: 60
      pylint_threshold: 7.0
      
    complexity:
      enabled: true
      weight: 0.20
      min_score: 0.6
      timeout: 30
      max_complexity_threshold: 10
      min_maintainability: 65.0
      
    llm_judge:
      enabled: true
      weight: 0.20
      min_score: 0.6
      timeout: 60
      model: "anthropic/claude-sonnet-4.5"
      temperature: 0.3
```

## Global Settings

### `enabled`
- **Type:** boolean
- **Default:** `true`
- **Description:** Master switch for entire evaluation system

```yaml
evaluation:
  enabled: false  # Disable all evaluation
```

### `gate_on_failure`
- **Type:** boolean
- **Default:** `false`
- **Description:** Block code output if evaluation fails

```yaml
evaluation:
  gate_on_failure: true  # Prevent deployment of low-quality code
```

Use cases:
- **Production environments:** Set to `true`
- **Development/testing:** Set to `false`

### `pass_threshold`
- **Type:** float (0.0 - 1.0)
- **Default:** `0.7`
- **Description:** Minimum overall score to pass

```yaml
evaluation:
  pass_threshold: 0.8  # Stricter quality requirements
```

Recommended values:
- **Critical systems:** 0.85 - 0.90
- **Production code:** 0.75 - 0.85
- **Development:** 0.60 - 0.75
- **Prototypes:** 0.50 - 0.60

## Hook Configuration

### `hooks.post_refiner`
- **Type:** boolean
- **Default:** `true`
- **Description:** Run evaluation after refiner stage

```yaml
evaluation:
  hooks:
    post_refiner: true  # Evaluate after code refinement
```

###hooks.post_documenter`
- **Type:** boolean
- **Default:** `false`
- **Description:** Run evaluation after documenter stage

```yaml
evaluation:
  hooks:
    post_documenter: true  # Evaluate after documentation
```

**When to use each hook:**
- **POST_REFINER:** Evaluate code quality before documentation
- **POST_DOCUMENTER:** Final quality check including docs

## Evaluator Settings

### Common Options (All Evaluators)

#### `enabled`
Enable/disable individual evaluator

```yaml
evaluators:
  security:
    enabled: false  # Disable security evaluator
```

#### `weight`
Contribution to overall score (must sum to 1.0)

```yaml
evaluators:
  security:
    weight: 0.40  # 40% of overall score
  static_analysis:
    weight: 0.30  # 30%
  complexity:
    weight: 0.20  # 20%
  llm_judge:
    weight: 0.10  # 10%
```

#### `min_score`
Minimum score required to pass (0.0 - 1.0)

```yaml
evaluators:
  security:
    min_score: 0.8  # Security must score >= 0.8
```

#### `timeout`
Maximum execution time in seconds

```yaml
evaluators:
  static_analysis:
    timeout: 120  # Allow 2 minutes
```

### Security Evaluator

```yaml
evaluators:
  security:
    enabled: true
    weight: 0.30
    min_score: 0.6
    timeout: 30
```

**No additional options** - uses Bandit with default configuration.

### Static Analysis Evaluator

```yaml
evaluators:
  static_analysis:
    enabled: true
    weight: 0.30
    min_score: 0.6
    timeout: 60
    pylint_threshold: 7.0  # Minimum Pylint score (0-10)
```

**pylint_threshold:**
- Minimum acceptable Pylint score
- Range: 0.0 - 10.0
- Recommended: 7.0 - 8.0

### Complexity Evaluator

```yaml
evaluators:
  complexity:
    enabled: true
    weight: 0.20
    min_score: 0.6
    timeout: 30
    max_complexity_threshold: 10      # Maximum cyclomatic complexity
    min_maintainability: 65.0          # Minimum maintainability index
```

**max_complexity_threshold:**
- Maximum allowed cyclomatic complexity for any function
- Recommended values:
  - **Simple code:** 5
  - **Standard code:** 10
  - **Complex code:** 15

**min_maintainability:**
- Minimum maintainability index (0-100)
- Recommended values:
  - **High quality:** 80+
  - **Good quality:** 65+
  - **Acceptable:** 50+

### LLM Judge Evaluator

```yaml
evaluators:
  llm_judge:
    enabled: true
    weight: 0.20
    min_score: 0.6
    timeout: 60
    model: "anthropic/claude-sonnet-4.5"
    temperature: 0.3
```

**model:**
- OpenRouter model identifier
- Recommended: `anthropic/claude-sonnet-4.5`
- Alternatives:
  - `anthropic/claude-3-opus`
  - `openai/gpt-4-turbo`

**temperature:**
- LLM sampling temperature (0.0 - 1.0)
- Lower = more consistent
- Higher = more creative
- Recommended: 0.3

## Common Configuration Scenarios

### Scenario 1: Maximum Security

```yaml
evaluation:
  enabled: true
  gate_on_failure: true
  pass_threshold: 0.85
  
  evaluators:
    security:
      weight: 0.50      # Prioritize security
      min_score: 0.90
    static_analysis:
      weight: 0.30
      min_score: 0.75
    complexity:
      weight: 0.10
      min_score: 0.60
    llm_judge:
      weight: 0.10
      min_score: 0.60
```

### Scenario 2: Fast Development

```yaml
evaluation:
  enabled: true
  gate_on_failure: false
  pass_threshold: 0.60
  
  evaluators:
    security:
      enabled: true
      weight: 0.40
      timeout: 15       # Faster timeout
    static_analysis:
      enabled: false    # Disable for speed
    complexity:
      enabled: true
      weight: 0.30
      timeout: 15
    llm_judge:
      enabled: false    # Disable for speed
      weight: 0.30
```

### Scenario 3: Balanced Production

```yaml
evaluation:
  enabled: true
  gate_on_failure: true
  pass_threshold: 0.75
  
  evaluators:
    security:
      weight: 0.30
      min_score: 0.70
    static_analysis:
      weight: 0.30
      min_score: 0.70
      pylint_threshold: 7.5
    complexity:
      weight: 0.20
      min_score: 0.65
      max_complexity_threshold: 8
      min_maintainability: 70.0
    llm_judge:
      weight: 0.20
      min_score: 0.70
```

### Scenario 4: Code Quality Focus

```yaml
evaluation:
  enabled: true
  gate_on_failure: false
  pass_threshold: 0.80
  
  evaluators:
    security:
      weight: 0.20
      min_score: 0.70
    static_analysis:
      weight: 0.40      # Prioritize static analysis
      min_score: 0.80
      pylint_threshold: 8.0
    complexity:
      weight: 0.25
      min_score: 0.75
      max_complexity_threshold: 6
      min_maintainability: 75.0
    llm_judge:
      weight: 0.15
      min_score: 0.70
```

## Environment Variables

Required environment variables:

```bash
# Required for LLM Judge
export OPENROUTER_API_KEY=your-key-here

# Required for W&B Weave logging
export WANDB_API_KEY=your-key-here

# Optional: Custom config path
export EVALUATION_CONFIG_PATH=/path/to/custom/evaluation.yaml
```

## Loading Custom Configuration

### Python API

```python
from src.config.evaluation_config import EvaluationConfig

# Load from custom path
config = EvaluationConfig("my_config.yaml")

# Or use environment variable
config = EvaluationConfig()  # Reads EVALUATION_CONFIG_PATH
```

### CLI

```bash
# Set config path
export EVALUATION_CONFIG_PATH=config/production_evaluation.yaml

# Run with custom config
python3 cli.py process "task" --config $EVALUATION_CONFIG_PATH
```

## Validation

Validate your configuration:

```python
from src.config.evaluation_config import EvaluationConfig

config = EvaluationConfig("config/evaluation.yaml")

# Check weights sum to 1.0
total_weight = (
    config.get_evaluator_weight("security") +
    config.get_evaluator_weight("static_analysis") +
    config.get_evaluator_weight("complexity") +
    config.get_evaluator_weight("llm_judge")
)
assert abs(total_weight - 1.0) < 0.01, "Weights must sum to 1.0"

# Check all enabled evaluators have valid config
for name in ["security", "static_analysis", "complexity", "llm_judge"]:
    if config.is_evaluator_enabled(name):
        assert config.get_evaluator_weight(name) > 0, f"{name} weight must be > 0"
        assert config.get_evaluator_min_score(name) >= 0, f"{name} min_score invalid"
```

## Troubleshooting

### Issue: Weights don't sum to 1.0

```yaml
# ❌ Bad - sums to 0.9
evaluators:
  security:
    weight: 0.30
  static_analysis:
    weight: 0.30
  complexity:
    weight: 0.20
  llm_judge:
    weight: 0.10

# ✅ Good - sums to 1.0
evaluators:
  security:
    weight: 0.30
  static_analysis:
    weight: 0.30
  complexity:
    weight: 0.20
  llm_judge:
    weight: 0.20
```

### Issue: Timeouts too aggressive

```yaml
# If evaluators time out frequently:
evaluators:
  static_analysis:
    timeout: 120  # Increase from 60
  llm_judge:
    timeout: 90   # Increase from 60
```

### Issue: Scores too low

```yaml
# Lower thresholds temporarily:
evaluation:
  pass_threshold: 0.65  # Lower from 0.75

evaluators:
  security:
    min_score: 0.55  # Lower from 0.70
```

## Best Practices

1. **Start with defaults** - Adjust only when needed
2. **Test configuration changes** - Run smoke tests after changes
3. **Monitor W&B Weave** - Track score trends over time
4. **Balance speed vs quality** - Adjust timeouts based on needs
5. **Document custom configs** - Add comments explaining choices
6. **Version control** - Commit config changes with code
7. **Environment-specific configs** - Use different configs per environment

## Migration Guide

### From No Evaluation

```yaml
# Phase 1: Enable with lenient thresholds
evaluation:
  enabled: true
  gate_on_failure: false
  pass_threshold: 0.60

# Phase 2: Tighten thresholds gradually
evaluation:
  pass_threshold: 0.70  # After 1 week
  
# Phase 3: Enable gating
evaluation:
  gate_on_failure: true  # After 2 weeks
```

### From Legacy AST Evaluation

```yaml
# Keep AST alongside new system
evaluation:
  enabled: true
  gate_on_failure: false  # Run in parallel first
  
# Monitor for 1-2 weeks, then switch:
evaluation:
  gate_on_failure: true   # Replace AST system
```

---

**Pro Tip:** Copy `config/evaluation.yaml` to `config/evaluation.development.yaml` and `config/evaluation.production.yaml` for environment-specific settings.
