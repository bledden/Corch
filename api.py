"""
Facilitair REST API - FastAPI server for collaborative AI orchestration
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import weave
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
log_dir = Path('test_results/logs')
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'facilitair_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('facilitair_api')

from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator
from utils.api_key_validator import APIKeyValidator
from backend.routers import streaming
from src.security import InputSanitizer
from src.errors import format_error, format_validation_errors, FacilitairError

# Initialize W&B Weave
weave.init("facilitair/api")

# Initialize FastAPI
app = FastAPI(
    title="Facilitair API",
    description="Collaborative AI Orchestration API with Sequential Workflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include streaming router
app.include_router(streaming.router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers
@app.exception_handler(FacilitairError)
async def facilitair_error_handler(request, exc: FacilitairError):
    """Handle Facilitair-specific errors with user-friendly messages"""
    error_response = exc.to_dict()
    status_code = 400 if exc.category in ["validation", "authentication"] else 500
    logger.error(f"FacilitairError: {exc.message}", exc_info=True, extra={"category": exc.category})
    return JSONResponse(status_code=status_code, content=error_response)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected errors with helpful troubleshooting hints"""
    error_response = format_error(exc, include_traceback=False, user_facing=True)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content=error_response)


# Global orchestrator instance (process-wide only)
orchestrator: Optional[CollaborativeOrchestrator] = None
orchestrator_lock = threading.Lock()

# In-memory task storage (use Redis/DB in production)
task_results = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class CollaborateRequest(BaseModel):
    """Request model for collaboration endpoint"""
    task: str = Field(
        ...,
        description="Task description for agents to collaborate on",
        min_length=10,
        max_length=10000
    )
    use_sequential: bool = Field(True, description="Use sequential workflow (recommended)")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum refinement iterations")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)")
    force_agents: Optional[List[str]] = Field(None, description="Force specific agents (e.g., ['architect', 'coder'])")

    @validator('task')
    def validate_task_content(cls, v):
        """Validate and sanitize task content for safety and quality"""
        # Use InputSanitizer for comprehensive security validation
        v = InputSanitizer.sanitize_task_description(v)

        # Check for minimum meaningful content (at least one word of 3+ chars)
        words = v.split()
        if not any(len(word) >= 3 for word in words):
            raise ValueError("Task must contain meaningful content")

        return v

    @validator('force_agents')
    def validate_force_agents(cls, v):
        """Validate and sanitize force_agents against known agent roles"""
        if v is None:
            return v

        # Sanitize the agent list first (security check)
        v = InputSanitizer.sanitize_agent_list(v)

        # Known agent roles from sequential orchestrator
        valid_agents = {
            'architect', 'coder', 'reviewer', 'refiner',
            'tester', 'documenter'
        }

        invalid_agents = [agent for agent in v if agent not in valid_agents]
        if invalid_agents:
            raise ValueError(
                f"Invalid agent(s): {invalid_agents}. "
                f"Valid agents are: {sorted(valid_agents)}"
            )

        return v

    class Config:
        schema_extra = {
            "example": {
                "task": "Write a Python function to calculate factorial with error handling",
                "use_sequential": True,
                "max_iterations": 3,
                "temperature": 0.2
            }
        }


class CollaborateResponse(BaseModel):
    """
    Response model for collaboration endpoint

    Contains the complete results of a collaborative task execution including
    which agents were used, the final output, performance metrics, and metadata.
    """
    task_id: str = Field(..., description="Unique identifier for this task", example="task-uuid-1234")
    task: str = Field(..., description="The original task description", example="Write a Python function to calculate factorial")
    success: bool = Field(..., description="Whether the task completed successfully without errors")
    agents_used: List[str] = Field(..., description="List of agents that participated in this task", example=["architect", "coder", "reviewer"])
    output: str = Field(..., description="Final output/result from the collaborative agents")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics including token usage, costs, and quality scores")
    consensus_method: str = Field(..., description="Method used for consensus (e.g., 'sequential', 'voting')", example="sequential")
    duration_seconds: float = Field(..., description="Total execution time in seconds", example=45.7)
    timestamp: str = Field(..., description="ISO 8601 timestamp when task completed", example="2025-10-21T10:30:00")


class TaskStatus(BaseModel):
    """
    Task status model for tracking long-running tasks

    Provides real-time progress updates for tasks being executed asynchronously.
    """
    task_id: str = Field(..., description="Unique identifier for the task", example="task-uuid-1234")
    status: str = Field(..., description="Current status: pending, running, completed, or failed", example="running")
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress from 0.0 (not started) to 1.0 (complete)", example=0.65)
    message: str = Field(..., description="Human-readable status message", example="Reviewing code...")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    api_keys_valid: bool
    orchestrator_ready: bool
    timestamp: str


class TaskListResponse(BaseModel):
    """Response model for task listing"""
    tasks: List[CollaborateResponse]
    total: int
    limit: int
    offset: int


class EvaluationRequest(BaseModel):
    """Evaluation request model"""
    num_tasks: int = Field(10, ge=1, le=100, description="Number of tasks to evaluate")
    compare_baseline: bool = Field(True, description="Compare against single-model baseline")


class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint"""
    eval_id: str
    status: str
    message: str
    num_tasks: int
    compare_baseline: bool
    timestamp: str


# ============================================================================
# Dependency Functions
# ============================================================================

def get_orchestrator() -> CollaborativeOrchestrator:
    """Get or create orchestrator instance (process-wide singleton).

    Thread-safe initialization using double-checked locking pattern.
    Note: Each FastAPI worker process will have its own instance.
    """
    global orchestrator
    if orchestrator is None:
        with orchestrator_lock:
            if orchestrator is None:  # Double-check inside lock
                logger.info("Initializing orchestrator...")
                orchestrator = CollaborativeOrchestrator(use_sequential=True)
                logger.info("Orchestrator initialized")
    return orchestrator


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "Facilitair API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check API health status and readiness",
    responses={
        200: {
            "description": "API is healthy and ready to accept requests",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "1.0.0",
                        "api_keys_valid": True,
                        "orchestrator_ready": True,
                        "timestamp": "2025-10-21T10:30:00"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Check the health and readiness of the Facilitair API.

    Returns information about:
    - Overall API status (healthy/degraded/unhealthy)
    - API version
    - API key validation status
    - Orchestrator readiness
    - Current timestamp

    This endpoint should be used for health checks, monitoring, and startup probes.
    """
    logger.info("Health check requested")

    # Validate API keys
    validator = APIKeyValidator()
    all_valid, results = validator.validate_all()

    # Check orchestrator
    orchestrator_ready = orchestrator is not None

    return HealthResponse(
        status="healthy" if all_valid else "degraded",
        version="1.0.0",
        api_keys_valid=all_valid,
        orchestrator_ready=orchestrator_ready,
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/api/v1/collaborate",
    response_model=CollaborateResponse,
    tags=["Collaboration"],
    summary="Execute Collaborative Task",
    description="Submit a task for collaborative execution by multiple AI agents",
    status_code=200,
    responses={
        200: {
            "description": "Task successfully completed",
            "content": {
                "application/json": {
                    "example": {
                        "task_id": "task-abc123",
                        "task": "Write a Python function to calculate factorial",
                        "success": True,
                        "agents_used": ["architect", "coder", "reviewer", "refiner"],
                        "output": "def factorial(n):\\n    if n < 0:\\n        raise ValueError('Factorial not defined for negative numbers')\\n    if n == 0:\\n        return 1\\n    return n * factorial(n - 1)",
                        "metrics": {
                            "total_tokens": 1500,
                            "total_cost_usd": 0.045,
                            "quality_score": 0.92
                        },
                        "consensus_method": "sequential",
                        "duration_seconds": 12.5,
                        "timestamp": "2025-10-21T10:30:00"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input - validation failed",
            "content": {
                "application/json": {
                    "example": {
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
                }
            }
        },
        500: {
            "description": "Internal server error or task execution failed",
            "content": {
                "application/json": {
                    "example": {
                        "error": "The AI model encountered an issue",
                        "category": "llm",
                        "hint": "The AI model encountered an issue. This could be due to rate limits, invalid API keys, or temporary service issues. Please try again in a moment.",
                        "details": {
                            "model": "anthropic/claude-3.5-sonnet",
                            "stage": "architecture"
                        }
                    }
                }
            }
        }
    }
)
@weave.op()
async def collaborate(
    request: CollaborateRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a collaborative task with AI agents using sequential workflow.

    ## Workflow
    Tasks are executed using a sequential collaboration pipeline where agents work in a chain:
    1. **Architect**: Designs the solution structure
    2. **Coder**: Implements the code
    3. **Reviewer**: Reviews for issues and suggests improvements
    4. **Refiner**: Applies improvements (iterative up to max_iterations)
    5. **Documenter**: Adds documentation and comments
    6. **Tester**: (Optional) Generates and runs tests

    ## Agent Selection
    - By default, all agents execute in sequence
    - Use `force_agents` to run only specific agents (e.g., ["architect", "coder"])
    - Valid agents: architect, coder, reviewer, refiner, tester, documenter

    ## Parameters
    - **task**: Detailed description of what you want to build (10-10000 chars)
    - **use_sequential**: Always True (consensus mode deprecated)
    - **max_iterations**: How many refinement cycles (1-10, default: 3)
    - **temperature**: LLM creativity (0.0=deterministic, 2.0=very creative, default: 0.2)
    - **force_agents**: Optional list of specific agents to use

    ## Response
    Returns a complete task result including:
    - Unique task ID for retrieval
    - Success status
    - List of agents that participated
    - Final output (code, documentation, etc.)
    - Performance metrics (tokens, cost, quality score)
    - Execution duration

    ## Error Handling
    All errors include:
    - Clear error message
    - Error category (validation, llm, timeout, etc.)
    - Troubleshooting hint with actionable steps
    - Detailed context for debugging

    ## Example
    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/api/v1/collaborate",
        json={
            "task": "Write a Python function to validate email addresses with regex",
            "temperature": 0.3,
            "max_iterations": 2
        }
    )
    result = response.json()
    print(result["output"])
    ```
    """
    logger.info(f"Collaboration requested: task='{request.task[:50]}...'")

    try:
        # Get orchestrator
        orch = get_orchestrator()

        # Generate task ID
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Update orchestrator config if needed
        if not request.use_sequential:
            logger.warning("Consensus mode deprecated, forcing sequential")
            request.use_sequential = True

        # Execute collaboration
        start_time = datetime.now()
        result = await orch.collaborate(
            task=request.task,
            force_agents=request.force_agents
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Build response
        # Determine success based on output (no errors)
        success = not result.final_output.startswith("[ERROR]") if result.final_output else False

        response = CollaborateResponse(
            task_id=task_id,
            task=request.task,
            success=success,
            agents_used=result.agents_used,
            output=result.final_output,
            metrics=result.metrics,
            consensus_method=result.consensus_method,
            duration_seconds=duration,
            timestamp=datetime.now().isoformat()
        )

        # Store result
        task_results[task_id] = response.dict()

        logger.info(f"Collaboration completed: task_id={task_id}, success={success}, duration={duration:.2f}s")

        return response

    except Exception as e:
        logger.error(f"Collaboration failed: {str(e)}", exc_info=True)
        # The global exception handler will format this with troubleshooting hints
        raise


@app.get("/api/v1/tasks/{task_id}", response_model=CollaborateResponse, tags=["Tasks"])
async def get_task(task_id: str):
    """Get task result by ID"""
    logger.info(f"Task result requested: task_id={task_id}")

    if task_id not in task_results:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return task_results[task_id]


@app.get("/api/v1/tasks", response_model=TaskListResponse, tags=["Tasks"])
async def list_tasks(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip")
):
    """List all tasks with pagination"""
    logger.info(f"Task list requested: limit={limit}, offset={offset}")

    tasks = list(task_results.values())
    total = len(tasks)

    # Paginate
    paginated = tasks[offset:offset + limit]

    return TaskListResponse(
        tasks=paginated,
        total=total,
        limit=limit,
        offset=offset
    )


@app.post("/api/v1/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def run_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Run evaluation comparing sequential vs single-model baseline

    This endpoint triggers a background evaluation task that compares:
    - Sequential collaboration (Architect → Coder → Reviewer → Refiner → Documenter)
    - Single-model baseline (direct GPT-4 call)

    Includes hallucination detection and quality metrics.
    """
    logger.info(f"Evaluation requested: num_tasks={request.num_tasks}, compare_baseline={request.compare_baseline}")

    eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start evaluation in background
    def run_eval():
        import subprocess
        if request.compare_baseline:
            subprocess.run(["python3", "run_sequential_vs_baseline_eval.py"], cwd=Path(__file__).parent)
        else:
            subprocess.run(["python3", "run_comprehensive_eval.py"], cwd=Path(__file__).parent)

    background_tasks.add_task(run_eval)

    return EvaluationResponse(
        eval_id=eval_id,
        status="started",
        message=f"Evaluation started with {request.num_tasks} tasks",
        num_tasks=request.num_tasks,
        compare_baseline=request.compare_baseline,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/agents", tags=["Agents"])
async def list_agents():
    """List available agents"""
    logger.info("Agent list requested")

    return {
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
            {
                "id": "reviewer",
                "name": "Reviewer",
                "description": "Reviews code for quality, security, and best practices",
                "stage": 3
            },
            {
                "id": "refiner",
                "name": "Refiner",
                "description": "Refines code based on review feedback (reuses Coder agent)",
                "stage": 4
            },
            {
                "id": "documenter",
                "name": "Documenter",
                "description": "Creates comprehensive documentation",
                "stage": 5
            }
        ]
    }


@app.get("/api/v1/metrics", tags=["Metrics"])
async def get_metrics():
    """Get API metrics"""
    logger.info("Metrics requested")

    return {
        "total_tasks": len(task_results),
        "successful_tasks": sum(1 for t in task_results.values() if t["success"]),
        "failed_tasks": sum(1 for t in task_results.values() if not t["success"]),
        "avg_duration": sum(t["duration_seconds"] for t in task_results.values()) / len(task_results) if task_results else 0,
        "most_used_agents": _get_most_used_agents()
    }


def _get_most_used_agents() -> Dict[str, int]:
    """Calculate most used agents"""
    agent_counts = {}
    for task in task_results.values():
        for agent in task["agents_used"]:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
    return agent_counts


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Facilitair API...")

    # Validate API keys
    validator = APIKeyValidator()
    all_valid, results = validator.validate_all()

    if not all_valid:
        logger.warning("Some API keys are invalid!")
        for result in results:
            if not result.is_valid and result.is_required:
                logger.error(f"Required key {result.key_name} is invalid: {result.error_message}")

    # Initialize orchestrator
    get_orchestrator()

    logger.info("Facilitair API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Facilitair API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
