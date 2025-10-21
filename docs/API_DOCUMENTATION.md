# Facilitair API Documentation

## Overview

Facilitair provides a REST API for collaborative AI task execution using multiple specialized agents working together sequentially.

**Base URL**: `http://localhost:8000` (development)

**API Version**: `v1`

**OpenAPI Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc)

## Quick Start

### 1. Health Check

Verify the API is running:

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
  "timestamp": "2025-10-21T10:30:00"
}
```

### 2. Execute a Task

Submit a collaborative task:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/collaborate",
    json={
        "task": "Write a Python function to validate email addresses using regex",
        "temperature": 0.2,
        "max_iterations": 3
    }
)

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Success: {result['success']}")
print(f"Output:\n{result['output']}")
```

## API Endpoints

### POST /api/v1/collaborate

Execute a collaborative task with multiple AI agents.

**Request Body**:
```json
{
  "task": "string (10-10000 chars, required)",
  "use_sequential": true,
  "max_iterations": 3,
  "temperature": 0.2,
  "force_agents": ["architect", "coder", "reviewer"]  // optional
}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | - | Task description (10-10000 characters) |
| `use_sequential` | boolean | No | true | Always use sequential workflow (consensus deprecated) |
| `max_iterations` | integer | No | 3 | Refinement iterations (1-10) |
| `temperature` | float | No | 0.2 | LLM creativity (0.0-2.0) |
| `force_agents` | array[string] | No | null | Specific agents to use |

**Valid Agents**:
- `architect` - Designs solution structure
- `coder` - Implements the code
- `reviewer` - Reviews for issues
- `refiner` - Applies improvements
- `tester` - Generates and runs tests
- `documenter` - Adds documentation

**Success Response (200)**:
```json
{
  "task_id": "task_20251021_103000_123456",
  "task": "Write a Python function to validate email addresses",
  "success": true,
  "agents_used": ["architect", "coder", "reviewer", "refiner", "documenter"],
  "output": "def validate_email(email: str) -> bool:\n    \"\"\"Validate email using regex\"\"\"\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
  "metrics": {
    "total_tokens": 1500,
    "total_cost_usd": 0.045,
    "quality_score": 0.92,
    "total_latency_ms": 12500
  },
  "consensus_method": "sequential",
  "duration_seconds": 12.5,
  "timestamp": "2025-10-21T10:30:00"
}
```

**Error Response (400 - Validation Error)**:
```json
{
  "error": "Input validation failed",
  "category": "validation",
  "hint": "Review the field errors below and correct your input.",
  "details": {
    "validation_errors": [
      {
        "field": "task",
        "message": "ensure this value has at least 10 characters",
        "type": "value_error.any_str.min_length"
      }
    ]
  }
}
```

**Error Response (500 - LLM Error)**:
```json
{
  "error": "The AI model encountered an issue",
  "category": "llm",
  "hint": "The AI model encountered an issue. This could be due to rate limits, invalid API keys, or temporary service issues. Please try again in a moment.",
  "details": {
    "model": "anthropic/claude-3.5-sonnet",
    "stage": "architecture"
  }
}
```

### GET /api/v1/tasks/{task_id}

Retrieve a previously executed task.

**Path Parameters**:
- `task_id` (string, required): The unique task identifier

**Success Response (200)**:
```json
{
  "task_id": "task_20251021_103000_123456",
  "task": "Write a Python function to validate email addresses",
  "success": true,
  "agents_used": ["architect", "coder", "reviewer"],
  "output": "...",
  "metrics": {...},
  "consensus_method": "sequential",
  "duration_seconds": 12.5,
  "timestamp": "2025-10-21T10:30:00"
}
```

**Error Response (404)**:
```json
{
  "detail": "Task task_xyz not found"
}
```

### GET /api/v1/tasks

List all executed tasks with pagination.

**Query Parameters**:
- `limit` (integer, optional): Maximum number of tasks to return (default: 10, max: 100)
- `offset` (integer, optional): Number of tasks to skip (default: 0)

**Success Response (200)**:
```json
{
  "tasks": [
    {
      "task_id": "task_20251021_103000_123456",
      "task": "...",
      "success": true,
      "agents_used": ["architect", "coder"],
      "output": "...",
      "metrics": {...},
      "consensus_method": "sequential",
      "duration_seconds": 12.5,
      "timestamp": "2025-10-21T10:30:00"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

### GET /api/v1/health

Check API health and readiness.

**Success Response (200)**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "api_keys_valid": true,
  "orchestrator_ready": true,
  "timestamp": "2025-10-21T10:30:00"
}
```

**Status Values**:
- `healthy`: All systems operational
- `degraded`: Some issues detected (e.g., invalid API keys)
- `unhealthy`: Critical issues preventing operation

### GET /

Root endpoint - returns available API endpoints.

**Success Response (200)**:
```json
{
  "message": "Facilitair API - Collaborative AI Orchestration",
  "version": "1.0.0",
  "docs": "/docs",
  "collaborate": "/api/v1/collaborate",
  "tasks": "/api/v1/tasks",
  "health": "/api/v1/health"
}
```

## Error Handling

All errors follow a consistent format with troubleshooting hints.

### Error Categories

| Category | HTTP Code | Description |
|----------|-----------|-------------|
| `validation` | 400 | Invalid input parameters |
| `authentication` | 401 | Authentication failed |
| `llm` | 500 | LLM/model error |
| `timeout` | 500 | Operation timed out |
| `configuration` | 500 | Configuration error |
| `resource` | 500 | Resource limit exceeded |
| `network` | 500 | Network connection issue |
| `internal` | 500 | Internal server error |

### Common Error Messages and Troubleshooting

#### Rate Limit Exceeded
```json
{
  "error": "Rate limit exceeded",
  "category": "resource",
  "hint": "Rate limit exceeded. Wait a few moments before retrying. Consider reducing the frequency of requests or upgrading your API plan."
}
```

**Solution**: Wait 60 seconds and retry, or upgrade your OpenRouter plan.

#### Invalid API Key
```json
{
  "error": "Invalid API key",
  "category": "authentication",
  "hint": "API key issue detected. Verify:\n1. OPENROUTER_API_KEY is set in .env file\n2. The key is valid and active\n3. The key has necessary permissions"
}
```

**Solution**: Check `.env` file for `OPENROUTER_API_KEY` and verify the key is valid.

#### Timeout
```json
{
  "error": "Operation timed out",
  "category": "timeout",
  "hint": "Operation timed out. Try:\n1. Breaking task into smaller steps\n2. Increasing timeout in config/evaluation.yaml\n3. Simplifying the task description"
}
```

**Solution**: Simplify your task or break it into smaller subtasks.

#### Token Limit Exceeded
```json
{
  "error": "Token limit exceeded",
  "category": "resource",
  "hint": "Token limit exceeded. The input or output is too large. Try:\n1. Reducing task description length\n2. Using a model with larger context window\n3. Breaking task into smaller subtasks"
}
```

**Solution**: Reduce task description length or use a different model with larger context.

## Usage Examples

### Python Requests

```python
import requests

# Execute a task
response = requests.post(
    "http://localhost:8000/api/v1/collaborate",
    json={
        "task": "Create a Python class for managing a todo list with add, remove, and list methods",
        "temperature": 0.3,
        "max_iterations": 2,
        "force_agents": ["architect", "coder", "documenter"]
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Success! Task ID: {result['task_id']}")
    print(f"Output:\n{result['output']}")
    print(f"Cost: ${result['metrics']['total_cost_usd']:.4f}")
else:
    error = response.json()
    print(f"Error: {error['error']}")
    print(f"Hint: {error['hint']}")
```

### cURL

```bash
# Execute a task
curl -X POST http://localhost:8000/api/v1/collaborate \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a function to check if a string is a palindrome",
    "temperature": 0.2,
    "max_iterations": 3
  }'

# Retrieve a task
curl http://localhost:8000/api/v1/tasks/task_20251021_103000_123456

# List all tasks
curl http://localhost:8000/api/v1/tasks?limit=20&offset=0

# Health check
curl http://localhost:8000/api/v1/health
```

### JavaScript (Fetch)

```javascript
async function collaborateTask(taskDescription) {
  const response = await fetch('http://localhost:8000/api/v1/collaborate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      task: taskDescription,
      temperature: 0.2,
      max_iterations: 3
    })
  });

  if (!response.ok) {
    const error = await response.json();
    console.error(`Error: ${error.error}`);
    console.error(`Hint: ${error.hint}`);
    throw new Error(error.error);
  }

  const result = await response.json();
  return result;
}

// Usage
collaborateTask("Create a function to find the longest word in a sentence")
  .then(result => {
    console.log("Task ID:", result.task_id);
    console.log("Output:", result.output);
    console.log("Agents used:", result.agents_used);
  })
  .catch(error => console.error("Failed:", error));
```

## Authentication

Currently, authentication is handled via API keys configured in environment variables:

1. Set `OPENROUTER_API_KEY` in `.env` file
2. The API validates keys on startup via `/health` endpoint
3. Invalid keys result in `degraded` health status

Future versions will support API key headers for per-request authentication.

## Rate Limits

Rate limits are enforced by the underlying LLM providers (OpenRouter):

- **Free tier**: ~10 requests/minute
- **Paid tier**: Higher limits based on plan

Monitor the `total_cost_usd` in metrics to track usage.

## Best Practices

### 1. Task Description

✅ **Good**:
```json
{
  "task": "Write a Python function called 'calculate_discount' that takes a price and discount percentage as input, validates that both are positive numbers, and returns the discounted price rounded to 2 decimal places."
}
```

❌ **Bad**:
```json
{
  "task": "make a discount calculator"
}
```

### 2. Agent Selection

- Use all agents for complete, production-ready code
- Use `["architect", "coder"]` for quick prototypes
- Use `["reviewer", "refiner"]` to improve existing code (include code in task description)

### 3. Temperature Selection

- **0.0-0.3**: Deterministic, consistent output (recommended for code generation)
- **0.4-0.7**: Balanced creativity and consistency
- **0.8-2.0**: High creativity (use for brainstorming, not production code)

### 4. Error Handling

Always handle errors with retry logic:

```python
import time

def collaborate_with_retry(task, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={"task": task})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

## Monitoring and Observability

All tasks are logged to W&B Weave for monitoring:

1. View task traces at https://wandb.ai/facilitair/api
2. Track metrics: tokens, cost, latency, quality scores
3. Debug failures with full execution traces

## Support

- **Documentation**: `/docs` (Swagger UI)
- **GitHub Issues**: https://github.com/yourusername/facilitair/issues
- **API Status**: Check `/api/v1/health` endpoint

## Changelog

### v1.0.0 (2025-10-21)

- Initial API release
- Sequential collaboration workflow
- Comprehensive error handling with troubleshooting hints
- Input sanitization for security
- OpenAPI documentation
- Health check endpoint
- Task retrieval and listing
- W&B Weave integration
