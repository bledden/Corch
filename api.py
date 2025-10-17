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
        """Validate task content for safety and quality"""
        if not v or not v.strip():
            raise ValueError("Task cannot be empty or whitespace-only")

        # Check for minimum meaningful content (at least one word of 3+ chars)
        words = v.split()
        if not any(len(word) >= 3 for word in words):
            raise ValueError("Task must contain meaningful content")

        return v.strip()

    @validator('force_agents')
    def validate_force_agents(cls, v):
        """Validate force_agents against known agent roles"""
        if v is None:
            return v

        # Known agent roles from sequential orchestrator
        valid_agents = {
            'architect', 'coder', 'reviewer', 'refiner',
            'tester', 'documenter'
        }

        invalid_agents = [agent for agent in v if agent.lower() not in valid_agents]
        if invalid_agents:
            raise ValueError(
                f"Invalid agent(s): {invalid_agents}. "
                f"Valid agents are: {sorted(valid_agents)}"
            )

        # Normalize to lowercase
        return [agent.lower() for agent in v]

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
    """Response model for collaboration endpoint"""
    task_id: str
    task: str
    success: bool
    agents_used: List[str]
    output: str
    metrics: Dict[str, Any]
    consensus_method: str
    duration_seconds: float
    timestamp: str


class TaskStatus(BaseModel):
    """Task status model"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str


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


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
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


@app.post("/api/v1/collaborate", response_model=CollaborateResponse, tags=["Collaboration"])
@weave.op()
async def collaborate(
    request: CollaborateRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a collaborative task with AI agents

    This endpoint uses sequential collaboration where agents work in a chain:
    Architect → Coder → Reviewer → Refiner → Documenter

    Returns detailed results including output, metrics, and agent usage.
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
        raise HTTPException(status_code=500, detail=f"Collaboration failed: {str(e)}")


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
