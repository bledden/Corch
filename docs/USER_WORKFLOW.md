# User Workflow - How to Run Facilitair Manually

## Quick Start (3 Steps)

### 1. Set Up Environment

```bash
# Clone and navigate
git clone https://github.com/bledden/weavehacks-collaborative.git
cd weavehacks-collaborative

# Install dependencies
pip3 install -r requirements.txt

# Set API keys
export WANDB_API_KEY="your-wandb-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

###2. Run a Task

```bash
# Basic usage
python3 cli.py collaborate "Write a function to check if a number is prime"

# With sequential mode (uses evaluation system)
python3 cli.py collaborate "Create REST API endpoint" --sequential

# Save output to file
python3 cli.py collaborate "Binary search tree" --save output.py

# JSON format
python3 cli.py collaborate "Calculator class" --format json
```

### 3. View Results

- **Terminal**: See live output with evaluation scores
- **W&B Weave**: Visit the URL shown in output for full trace

---

## CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `health` | Check system status | `python3 cli.py health` |
| `collaborate` | Generate code | `python3 cli.py collaborate "task"` |
| `evaluate` | Run benchmarks | `python3 cli.py evaluate --tasks 10` |
| `serve` | Start API server | `python3 cli.py serve` |
| `init` | Create config file | `python3 cli.py init` |

---

## CLI Options

### `--sequential` / `--consensus`
- **Sequential**: 5-stage collaboration with evaluation (default, recommended)
- **Consensus**: Alternative orchestration mode

```bash
python3 cli.py collaborate "task" --sequential
```

### `--format [text|json|markdown]`
- **text**: Human-readable (default)
- **json**: Machine-readable
- **markdown**: Documentation-ready

```bash
python3 cli.py collaborate "task" --format json
```

### `--save FILE`
Save output to file

```bash
python3 cli.py collaborate "task" --save output.py
```

### `--stream` / `--no-stream`
Enable/disable streaming output

```bash
python3 cli.py collaborate "task" --stream
```

---

## Workflow Stages

When you run a task, Facilitair executes **5 sequential stages**:

```
1. ARCHITECT  → Designs solution architecture
2. CODER      → Implements the code
3. REVIEWER   → Reviews for issues
4. REFINER    → Improves code quality (3 iterations)
5. DOCUMENTER → Creates documentation

[EVALUATION] → 4-layer quality assessment runs after REFINER:
  - SecurityEvaluator (30%)
  - StaticAnalysisEvaluator (30%)
  - ComplexityEvaluator (20%)
  - LLMJudgeEvaluator (20%)
```

---

## Example Session

```bash
$ python3 cli.py collaborate "Write a function to validate email addresses" --sequential

[INFO] Facilitair CLI initialized
[GOAL] Sequential collaboration enabled

[STAGE] Architect (qwen/qwen3-coder-plus)
Designing email validation system...

[STAGE] Coder (deepseek/deepseek-chat)
Implementing regex-based validator...

[STAGE] Reviewer (meta-llama/llama-3.3-70b-instruct)
Reviewing code quality...

[STAGE] Refiner (deepseek/deepseek-chat)
Applying improvements...

[EVALUATION] Running quality assessment...
  Security Score: 1.00 ✅ (0 vulnerabilities)
  Static Analysis: 0.88 ✅ (Pylint 8.8/10)
  Complexity: 0.95 ✅ (MI 91.2, CC 2.0)
  LLM Judge: 0.90 ✅ (Strong correctness)

  Overall Score: 0.93 ✅ PASS

Strengths:
  - No security vulnerabilities detected
  - Highly maintainable code (MI 91.2)
  - Comprehensive edge case handling

[STAGE] Documenter (mistralai/codestral-2501)
Creating documentation...

✅ Task completed successfully!
📊 View trace: https://wandb.ai/your-username/self-improving-collaboration/weave/calls/...
```

---

## Understanding Evaluation Scores

### Score Breakdown

After the REFINER stage, you'll see:

```
[EVALUATION] Running quality assessment...
  Security: 1.00 ✅
  Static Analysis: 0.85 ✅
  Complexity: 0.92 ✅
  LLM Judge: 0.88 ✅

  Overall: 0.91 ✅ PASS
```

**What each evaluator checks:**

1. **Security** (30%) - Bandit scan for vulnerabilities
2. **Static Analysis** (30%) - Pylint, Flake8, Mypy for code quality
3. **Complexity** (20%) - Radon for maintainability
4. **LLM Judge** (20%) - Claude Sonnet 4.5 for semantic correctness

**Pass criteria**: Overall >= 0.7 AND all individual >= 0.6

### Score Interpretation

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.90-1.00 | Excellent | Production-ready |
| 0.75-0.89 | Good | Minor improvements needed |
| 0.60-0.74 | Acceptable | Needs work |
| Below 0.60 | Poor | Significant refactoring required |

---

## Common Use Cases

### 1. Quick Function
```bash
python3 cli.py collaborate "Write a factorial function"
```

### 2. REST API
```bash
python3 cli.py collaborate "Create FastAPI endpoint for user authentication with JWT"
```

### 3. Algorithm
```bash
python3 cli.py collaborate "Implement merge sort with O(n log n) complexity"
```

### 4. Class Design
```bash
python3 cli.py collaborate "Design a binary search tree with insert, delete, search"
```

### 5. Save to File
```bash
python3 cli.py collaborate "Prime number checker" --save prime.py
```

### 6. JSON Output
```bash
python3 cli.py collaborate "Calculator class" --format json > result.json
```

---

## Benchmarking

### Quick 10-Task Benchmark (~30 mins)
```bash
python3 run_10_task_benchmark_v2.py
```

### Full 100-Task Benchmark (~17 hours)
```bash
python3 run_100_task_benchmark.py
```

Results saved to `benchmark_*.json` with quality scores and analysis.

---

## REST API Usage

### Start Server
```bash
python3 cli.py serve --port 8000
```

### Make Request
```bash
curl -X POST http://localhost:8000/api/v1/collaborate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a prime checker",
    "language": "python"
  }'
```

### View API Docs
```
http://localhost:8000/docs
```

---

## Configuration

### Custom Config File

Create `config.yaml`:

```yaml
strategy: BALANCED  # OPEN, CLOSED, or BALANCED

evaluation:
  enabled: true
  pass_threshold: 0.75

  evaluators:
    security: {enabled: true, weight: 0.30}
    static_analysis: {enabled: true, weight: 0.30}
    complexity: {enabled: true, weight: 0.20}
    llm_judge: {enabled: true, weight: 0.20}
```

Use it:
```bash
python3 cli.py collaborate "task" --config config.yaml
```

---

## Troubleshooting

### "OPENROUTER_API_KEY not set"
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "WANDB_API_KEY not set"
```bash
export WANDB_API_KEY="your-key-here"
```

### Low Evaluation Scores
Review the feedback and rerun with more specific instructions:
```bash
python3 cli.py collaborate "Write a SECURE login function with proper error handling and type hints"
```

### Check System Health
```bash
python3 cli.py health
```

---

## Getting Help

```bash
# Command help
python3 cli.py --help
python3 cli.py collaborate --help

# Check logs
tail -f facilitair_cli.log

# View W&B traces
# Visit URL shown in command output
```

---

## Next Steps

- **Read full docs**: [docs/EVALUATION_SYSTEM.md](EVALUATION_SYSTEM.md)
- **Configure evaluation**: [docs/EVALUATION_CONFIGURATION.md](EVALUATION_CONFIGURATION.md)
- **Run benchmarks**: Test the system at scale
- **Start API server**: For programmatic access

---

**Happy coding with Facilitair!** 🚀
