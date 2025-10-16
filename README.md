# Facilitair - Collaborative AI Orchestration

[![WeaveHacks 2](https://img.shields.io/badge/WeaveHacks-2-blue)](https://wandb.ai/site/weavehacks-2) [![W&B Weave](https://img.shields.io/badge/W%26B-Weave-orange)](https://wandb.ai/facilitair/) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Sequential AI collaboration: 73% pass rate, +36.8% quality improvement vs baseline, validated across 100 production tasks.**

Built for WeaveHacks 2 using W&B Weave, OpenRouter, and 200+ LLMs.

## Benchmark Results: 100-Task Evaluation (Completed)

**Comprehensive 100-task evaluation across 5 categories:**

| Metric | Sequential (5-Stage) | Baseline (Single-Pass) | Advantage |
|--------|---------------------|----------------------|-----------|
| **Average Quality** | **0.726** | 0.531 | **+0.195 (+36.8%)** |
| **Pass Rate (≥0.7)** | **73%** | 19% | **+54 tasks** |
| **Task Wins** | **78/100** | 22/100 | **78% win rate** |
| **Runtime** | ~17 hours | ~17 hours | Parallel execution |

### Quality by Category (All favor Sequential)

| Category | Sequential | Baseline | Advantage |
|----------|-----------|----------|-----------|
| **Security-Critical** | 0.763 | 0.534 | **+0.228** ⭐ |
| **Complex Algorithms** | 0.720 | 0.490 | **+0.230** ⭐ |
| **Data Validation** | 0.762 | 0.550 | **+0.212** |
| **Production Features** | 0.676 | 0.497 | **+0.179** |
| **Error Handling** | 0.710 | 0.582 | **+0.128** |

> **Quality Score** = Weighted average across 6 dimensions: Correctness (30%), Completeness (25%), Code Quality (20%), Documentation (10%), Error Handling (10%), Testing (5%). Pass threshold: ≥0.70.

**Live Tracking:** [W&B Weave 100-Task Benchmark](https://wandb.ai/facilitair/100-task-benchmark/weave)

### Evaluation Methodology

**Quality Evaluation System** (6 dimensions):
- **Correctness** (30%): Syntax validation via AST parsing
- **Completeness** (25%): Has functions, logic, returns, imports
- **Code Quality** (20%): Line length, naming conventions, formatting
- **Documentation** (10%): Docstrings, comments, module docs
- **Error Handling** (10%): try/except, validation, finally blocks
- **Testing** (5%): Test functions, assertions, test frameworks

**Task Categories** (100 tasks total):
1. **Security-Critical** (20 tasks): Password hashing, SQL injection prevention, JWT validation, XSS/CSRF protection
2. **Complex Algorithms** (20 tasks): BST with rotation, LRU cache, topological sort, Dijkstra, segment trees
3. **Production Features** (20 tasks): REST API clients, connection pools, job queues, circuit breakers
4. **Data Validation** (20 tasks): JSON/XML parsers, email/phone validators, CSV parsers, regex validators
5. **Error Handling** (20 tasks): Retry decorators, error boundaries, timeout wrappers, graceful degradation

## Architecture

```
User Request
     ↓
ARCHITECT  → Designs solution
     ↓
CODER      → Implements code
     ↓
REVIEWER   → Reviews quality
     ↓
REFINER    → Fixes issues (iterates 3x)
     ↓
DOCUMENTER → Creates docs
     ↓
Result + Metrics
```

## Quick Start

### Setup
```bash
pip3 install -r requirements.txt
export WANDB_API_KEY="your_key"
export OPENROUTER_API_KEY="your_key"
```

### CLI
```bash
# Health check
python3 cli.py health

# Collaborate
python3 cli.py collaborate "Write a factorial function"

# Evaluate
python3 cli.py evaluate --tasks 10

# Streaming CLI (real-time output)
python3 cli/streaming_client_simple.py "Write a factorial function"
```

### REST API
```bash
# Start server
python3 cli.py serve

# API docs
open http://localhost:8000/docs

# Example request
curl -X POST http://localhost:8000/api/v1/collaborate \
  -H "Content-Type: application/json" \
  -d '{"task": "Implement LRU cache"}'
```

## Features

- **5-Stage Sequential Collaboration** - Specialized pipeline
- **73% Pass Rate** - 73/100 tasks pass quality threshold (≥0.7)
- **+36.8% Quality Improvement** - Validated across 100 production tasks
- **Objective Quality Metrics** - 6-dimension evaluation system
- **2 Interfaces** - CLI + REST API
- **Complete W&B Weave Integration** - Experiment tracking
- **200+ Models** - Via OpenRouter
- **5x Higher Latency** - 5 LLM calls vs 1 (quality over speed)

## Run Your Own Evaluation

```bash
# Run 100-task benchmark (takes ~17 hours)
python3 run_100_task_benchmark.py

# Run quick 10-task benchmark (takes ~30 mins)
python3 run_10_task_benchmark_v2.py

# Analyze results for hallucinations
python3 check_hallucinations.py benchmark_100_final_*.json
```

Results saved to `benchmark_100_final_{timestamp}.json` with complete quality breakdown by category.

## Sequential vs Single-Model: Trade-offs

### Sequential Advantages
- **+36.8% Quality**: 0.726 vs 0.531 average quality score
- **+54 More Passes**: 73% vs 19% pass rate (≥0.7 threshold)
- **78% Win Rate**: Outperforms baseline on 78/100 tasks
- **Consistent Across Categories**: Wins in all 5 task categories
- **Complete Lineage**: Full traceability via W&B Weave
- **5 Specialized Stages**: Each agent optimized for specific role

### Sequential Disadvantages
- **5x Higher Latency**: 5 sequential LLM calls vs 1
- **5x Higher Cost**: 5x more API calls per task
- **Additional Complexity**: Orchestration overhead

### When to Use Each

**Use Sequential When:**
- **Multi-category tasks** (architecture + coding + review + docs)
- **High complexity** (LRU cache, N-Queens, system design)
- **Production-critical** code where quality matters most
- **Zero tolerance** for hallucinations
- Audit trail and explainability required

**Use Single-Model When:**
- **Single-category tasks** (just coding, just review, just documentation)
- **Low-medium complexity** (factorial, string reverse, simple algorithms)
- Speed and cost are priorities
- Prototyping and experimentation

**Rule of Thumb:** Multi-category or high-complexity → Sequential. Single-category, focused tasks → Single-model.

## CLI Commands

```bash
python3 cli.py health          # Check system status
python3 cli.py collaborate     # Execute task
python3 cli.py evaluate        # Run evaluation
python3 cli.py serve           # Start API server
python3 cli.py init            # Create config
```

## API Endpoints

### Standard Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/collaborate` - Execute task
- `GET /api/v1/agents` - List agents
- `GET /api/v1/tasks` - List tasks
- `POST /api/v1/evaluate` - Run evaluation
- `GET /api/v1/metrics` - Get metrics

### Streaming Endpoints (SSE)
- `POST /api/stream/task` - Create streaming task
- `GET /api/stream/events/{stream_id}` - Stream real-time events
- `GET /api/stream/status/{stream_id}` - Get stream status

## Logging & Telemetry

- **CLI:** `facilitair_cli.log`
- **API:** `facilitair_api.log`
- **W&B Weave:** https://wandb.ai/facilitair/

## Testing

```bash
python3 -m pytest tests/ -v
```

## Documentation

- [INTERFACES_README.md](INTERFACES_README.md) - Complete interface guide
- [SEQUENTIAL_COLLABORATION_DESIGN.md](SEQUENTIAL_COLLABORATION_DESIGN.md) - Architecture
- API Docs: http://localhost:8000/docs

## WeaveHacks 2

### Tech Stack
- **W&B Weave** - Experiment tracking & observability
- **OpenRouter** - 200+ LLM models
- **FastAPI** - Production REST API
- **Pydantic** - Type-safe validation

### Key Innovations
1. **Sequential > Single-Model**: 73% vs 19% pass rate on 100-task benchmark
2. **Objective Quality Metrics**: 6-dimension evaluation system with AST parsing
3. **+36.8% Quality Improvement**: 0.726 vs 0.531 average quality score
4. **Complete W&B Weave Observability**: Full experiment tracking and lineage

## Statistics

- **Lines of Code**: 10,000+
- **Test Coverage**: 85%+
- **API Endpoints**: 8
- **CLI Commands**: 6
- **Available Models**: 200+ (via OpenRouter)
- **Sequential Pass Rate**: 73% (73/100 tasks ≥0.7 quality threshold)
- **Baseline Pass Rate**: 19% (19/100 tasks ≥0.7 quality threshold)
- **Sequential Win Rate**: 78% (78/100 tasks favor sequential)
- **Average Quality**: 0.726 (sequential) vs 0.531 (baseline)
- **Latency Multiplier**: 5x slower (5 LLM calls vs 1)
- **Cost Multiplier**: 5x more expensive (5 API calls vs 1)

## Links

- [W&B Weave Dashboard](https://wandb.ai/facilitair/)
- [Evaluation Results](https://wandb.ai/facilitair/sequential-vs-baseline-20251012_130636/weave)
- [API Docs](http://localhost:8000/docs)
- [WeaveHacks 2](https://wandb.ai/site/weavehacks-2)
- [GitHub Repository](https://github.com/bledden/weavehacks-collaborative)

**Built for WeaveHacks 2**
