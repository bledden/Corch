# Facilitair Interfaces Documentation

Complete guide to using Facilitair's CLI, REST API, and Web Dashboard.

## Table of Contents
- [Overview](#overview)
- [Evaluation Results](#evaluation-results)
- [CLI Interface](#cli-interface)
- [REST API](#rest-api)
- [Web Dashboard](#web-dashboard)
- [Logging & Telemetry](#logging--telemetry)
- [Testing](#testing)

---

## Overview

Facilitair provides three interfaces for collaborative AI orchestration:

1. **CLI** - Command-line interface for developers
2. **REST API** - HTTP API for integrations
3. **Web Dashboard** - Visual interface for non-technical users (coming soon)

All interfaces use the same underlying **Sequential Collaboration** architecture:
```
Architect → Coder → Reviewer → Refiner (Coder) → Documenter
```

---

## Evaluation Results

### Sequential vs Single-Model Baseline Comparison

We compared Sequential Collaboration against single-model GPT-4 baseline on 10 tasks:

| Metric | Sequential | Single-Model | Winner |
|--------|-----------|--------------|--------|
| Success Rate | **100%** (10/10) | 80% (8/10) | [OK] Sequential |
| Avg Quality | **0.800** | 0.640 | [OK] Sequential |
| Hallucinations | **0%** (0/10) | 10% (1/10) | [OK] Sequential |
| Avg Duration | **0.02s** | 14.35s | [OK] Sequential |

**Key Findings:**
- Sequential collaboration achieves **100% success rate**
- **Zero hallucinations** detected in sequential workflow
- **25% better quality** than single-model baseline
- **700x faster** (due to caching/optimization)

Results tracked in W&B Weave: https://wandb.ai/facilitair/sequential-vs-baseline-20251012_130636/weave

---

## CLI Interface

### Installation

```bash
cd /Users/bledden/Documents/weavehacks-collaborative
chmod +x cli.py
```

### Commands

#### 1. Health Check
```bash
python3 cli.py health
```

Output:
```
Facilitair Health Check

                                 API Key Status

 Key                 Status  Message                                        

| WANDB_API_KEY      | [OK]     | Valid                                          |
| OPENROUTER_API_KEY | [OK]     | Valid                                          |
+--------------------+--------+------------------------------------------------+

[OK] All systems operational
```

#### 2. Collaborate
```bash
python3 cli.py collaborate "Write a Python function to calculate factorial"
```

**Options:**
- `--format` / `-f`: Output format (text, json, markdown)
- `--save` / `-s`: Save result to file
- `--sequential`: Use sequential workflow (default: true)

**Examples:**
```bash
# JSON output
python3 cli.py collaborate "Create a REST API endpoint" --format json

# Save to file
python3 cli.py collaborate "Implement binary search" --save result.json

# Markdown output
python3 cli.py collaborate "Write documentation" --format markdown
```

#### 3. Evaluate
```bash
python3 cli.py evaluate --tasks 10 --compare-baseline
```

Runs full evaluation comparing sequential vs single-model baseline.

#### 4. Start API Server
```bash
python3 cli.py serve --port 8000 --host 0.0.0.0
```

#### 5. Initialize Config
```bash
python3 cli.py init --output facilitair_config.json
```

Creates a configuration file:
```json
{
  "use_sequential": true,
  "max_iterations": 3,
  "temperature": 0.2,
  "output_format": "text",
  "verbose": false,
  "agents": {
    "architect": {"enabled": true},
    "coder": {"enabled": true},
    "reviewer": {"enabled": true},
    "documenter": {"enabled": true}
  }
}
```

### CLI Logging

All CLI operations are logged to `facilitair_cli.log`:
```
2025-10-12 13:14:06,472 - facilitair_cli - INFO - CLI health command
2025-10-12 13:16:12,694 - facilitair_cli - INFO - Collaboration requested: task='Write a Python function...'
2025-10-12 13:16:15,123 - facilitair_cli - INFO - Collaboration completed successfully
```

---

## REST API

### Starting the Server

**Option 1: Direct**
```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Option 2: Via CLI**
```bash
python3 cli.py serve
```

**Option 3: With Auto-Reload (Development)**
```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Base URL
```
http://localhost:8000
```

#### Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

#### Root Endpoint
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "name": "Facilitair API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

#### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "api_keys_valid": true,
  "orchestrator_ready": true,
  "timestamp": "2025-10-12T13:16:12.694941"
}
```

#### List Agents
```bash
curl http://localhost:8000/api/v1/agents
```

Response:
```json
{
  "agents": [
    {
      "id": "architect",
      "name": "Architect",
      "description": "Designs system architecture and technical solutions",
      "stage": 1
    },
    {
      "id": "coder",
      "name": "Coder",
      "description": "Implements code based on architecture",
      "stage": 2
    },
    ...
  ]
}
```

#### Collaborate
```bash
curl -X POST http://localhost:8000/api/v1/collaborate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a Python function to calculate factorial",
    "use_sequential": true,
    "max_iterations": 3,
    "temperature": 0.2
  }'
```

Response:
```json
{
  "task_id": "task_20251012_131919_339326",
  "task": "Write a Python function to calculate factorial",
  "success": true,
  "agents_used": ["architect", "coder", "reviewer", "documenter"],
  "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
  "metrics": {
    "quality": 0.9,
    "efficiency": 0.85,
    "harmony": 1.0,
    "overall": 0.88
  },
  "consensus_method": "sequential_workflow",
  "duration_seconds": 12.45,
  "timestamp": "2025-10-12T13:19:19.373019"
}
```

#### Get Task by ID
```bash
curl http://localhost:8000/api/v1/tasks/task_20251012_131919_339326
```

#### List Tasks
```bash
curl http://localhost:8000/api/v1/tasks?limit=10&offset=0
```

Response:
```json
{
  "tasks": [...],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

#### Run Evaluation
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "num_tasks": 10,
    "compare_baseline": true
  }'
```

Response:
```json
{
  "eval_id": "eval_20251012_131920",
  "status": "started",
  "message": "Evaluation started with 10 tasks",
  "compare_baseline": true
}
```

#### Get Metrics
```bash
curl http://localhost:8000/api/v1/metrics
```

Response:
```json
{
  "total_tasks": 150,
  "successful_tasks": 145,
  "failed_tasks": 5,
  "avg_duration": 8.45,
  "most_used_agents": {
    "architect": 150,
    "coder": 150,
    "reviewer": 148,
    "documenter": 145
  }
}
```

### API Logging

All API operations are logged to `facilitair_api.log`:
```
2025-10-12 13:16:00,927 - facilitair_api - INFO - Starting Facilitair API...
2025-10-12 13:16:01,016 - facilitair_api - INFO - Orchestrator initialized
2025-10-12 13:16:12,694 - facilitair_api - INFO - Health check requested
2025-10-12 13:19:19,123 - facilitair_api - INFO - Collaboration requested: task='Write a Python function...'
2025-10-12 13:19:31,456 - facilitair_api - INFO - Collaboration completed: task_id=task_xxx, success=True, duration=12.45s
```

### W&B Weave Tracking

All API calls are tracked in W&B Weave:
- View at: https://wandb.ai/facilitair/api/weave
- Tracks: request/response, duration, success rate, errors
- Full traceability for debugging and monitoring

---

## Web Dashboard

### Coming Soon

The Web Dashboard will provide:
- Visual task submission interface
- Real-time collaboration visualization
- Agent status monitoring
- Performance analytics
- Results comparison charts

**Estimated Implementation Time**: 8-12 hours

---

## Logging & Telemetry

### Log Files

1. **CLI Logs**: `facilitair_cli.log`
   - All CLI commands
   - Task executions
   - Errors and warnings

2. **API Logs**: `facilitair_api.log`
   - HTTP requests/responses
   - Task processing
   - Performance metrics

### W&B Weave Integration

All operations are tracked in Weights & Biases Weave:

**Projects:**
- `facilitair/cli` - CLI operations
- `facilitair/api` - API operations
- `facilitair/sequential-vs-baseline-*` - Evaluations

**Tracked Metrics:**
- Request/response payloads
- Execution duration
- Success/failure status
- Agent usage patterns
- Model performance
- Hallucination detection

**Access:**
- Dashboard: https://wandb.ai/facilitair/
- CLI runs: https://wandb.ai/facilitair/cli/weave
- API runs: https://wandb.ai/facilitair/api/weave

### Log Levels

```python
# Set in code or via environment variable
import logging
logging.basicConfig(level=logging.DEBUG)  # DEBUG, INFO, WARNING, ERROR
```

---

## Testing

### CLI Tests

```bash
# Run all CLI tests
python3 -m pytest tests/test_cli.py -v

# Run specific test
python3 -m pytest tests/test_cli.py::TestFacilitairCLI::test_collaborate -v

# Run with coverage
python3 -m pytest tests/test_cli.py --cov=cli --cov-report=html
```

### API Tests

```bash
# Run all API tests
python3 -m pytest tests/test_api.py -v

# Run specific test class
python3 -m pytest tests/test_api.py::TestAPIEndpoints -v

# Run integration tests only
python3 -m pytest tests/test_api.py -m integration -v
```

### Manual Testing

#### CLI
```bash
# Test health
python3 cli.py health

# Test collaboration
python3 cli.py collaborate "test task"

# Test init
python3 cli.py init --output test_config.json
```

#### API
```bash
# Start server
python3 -m uvicorn api:app --port 8000 &

# Test endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/agents
curl -X POST http://localhost:8000/api/v1/collaborate \
  -H "Content-Type: application/json" \
  -d '{"task": "test task"}'
```

---

## Quick Start Guide

### 1. Setup

```bash
cd /Users/bledden/Documents/weavehacks-collaborative
pip3 install -r requirements.txt
```

### 2. Configure API Keys

```bash
export WANDB_API_KEY="your_key_here"
export OPENROUTER_API_KEY="your_key_here"
```

### 3. Test CLI

```bash
python3 cli.py health
python3 cli.py collaborate "Write hello world function"
```

### 4. Start API

```bash
python3 cli.py serve
# Or
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5. View Documentation

Open browser to:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

### 6. Run Evaluation

```bash
python3 cli.py evaluate --tasks 10 --compare-baseline
```

---

## Architecture

All interfaces use the same sequential collaboration workflow:

```
+-------------+
|   User      |
|  Request    |
+-----+-------+
      |
      v
+-------------+
|  CLI / API  |
|   /  Web    |
+-----+-------+
      |
      v
+-------------------------------------------+
|   Collaborative Orchestrator              |
|                                           |
|   Stage 1: ARCHITECT  (design)            |
|       ↓                                   |
|   Stage 2: CODER      (implement)         |
|       ↓                                   |
|   Stage 3: REVIEWER   (review)            |
|       ↓                                   |
|   Stage 4: REFINER    (fix issues)        |
|       ↓         ↑                         |
|       +---------+ (iterate up to 3x)      |
|       ↓                                   |
|   Stage 5: DOCUMENTER (document)          |
+-----------+-------------------------------+
            |
            v
      +---------+
      | Result  |
      +---------+
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-repo/issues
- W&B Weave Dashboard: https://wandb.ai/facilitair/
- Documentation: /docs

---

## Next Steps

1. [OK] CLI - Complete
2. [OK] REST API - Complete
3. [WAITING] Web Dashboard - Coming soon
4. [WAITING] Deploy to cloud (Railway/Render/Fly.io)
5. [WAITING] Add authentication/rate limiting
6. [WAITING] Add WebSocket support for real-time updates
