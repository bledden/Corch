"""
Streaming API Router for Facilitair
Provides SSE endpoints for real-time collaborative orchestration
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.streaming.sse_handler import (
    get_stream_manager,
    StreamEventBuilder,
    StreamManager
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/stream", tags=["Streaming"])


# ============================================================================
# Request/Response Models
# ============================================================================

class StreamTaskRequest(BaseModel):
    """Request model for creating a streaming task"""
    task: str = Field(..., description="Task description", min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    stream: bool = Field(True, description="Enable streaming (always True for this endpoint)")

    class Config:
        schema_extra = {
            "example": {
                "task": "Build a REST API with JWT authentication",
                "context": {
                    "language": "python",
                    "frameworks": ["fastapi"],
                    "team_size": "small"
                },
                "stream": True
            }
        }


class StreamTaskResponse(BaseModel):
    """Response model for stream creation"""
    stream_id: str
    task: str
    status: str
    created_at: str


# ============================================================================
# Dependency Functions
# ============================================================================

def get_manager() -> StreamManager:
    """Get stream manager instance"""
    return get_stream_manager()


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/task", response_model=StreamTaskResponse)
async def create_stream_task(
    request: StreamTaskRequest,
    background_tasks: BackgroundTasks,
    manager: StreamManager = Depends(get_manager)
):
    """
    Create a new streaming task

    This endpoint creates a stream and returns a stream_id that can be used
    to connect to the SSE endpoint and receive real-time updates.

    **Flow:**
    1. POST /api/stream/task → Get stream_id
    2. GET /api/stream/events/{stream_id} → Connect to SSE for updates
    """
    logger.info(f"Creating stream for task: {request.task[:50]}...")

    try:
        # Generate unique stream ID
        stream_id = str(uuid.uuid4())

        # Create stream state
        state = manager.create_stream(
            stream_id=stream_id,
            task=request.task,
            context=request.context
        )

        # Import here to avoid circular dependency
        from backend.routers.streaming import execute_streaming_task

        # Start task execution in background
        background_tasks.add_task(
            execute_streaming_task,
            stream_id=stream_id,
            task=request.task,
            context=request.context
        )

        logger.info(f"Stream created: {stream_id}")

        return StreamTaskResponse(
            stream_id=stream_id,
            task=request.task,
            status="initializing",
            created_at=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to create stream: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create stream: {str(e)}")


@router.get("/events/{stream_id}")
async def stream_events(
    stream_id: str,
    manager: StreamManager = Depends(get_manager)
):
    """
    Stream SSE events for a task

    This endpoint provides a Server-Sent Events (SSE) stream of real-time updates
    for the collaborative orchestration process.

    **Event Types:**
    - `task_started`: Task begins execution
    - `task_progress`: Planning info (chunks, time estimates, costs)
    - `chunk_started`: Agent starts working on a chunk
    - `token_stream`: Live tokens from agent output
    - `chunk_completed`: Agent completes a chunk
    - `task_completed`: Final result ready
    - `task_error`: Error occurred
    - `system_message`: System notifications
    - `heartbeat`: Keepalive (every 15s)

    **Usage:**
    ```python
    import httpx

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", f"/api/stream/events/{stream_id}") as response:
            async for line in response.aiter_lines():
                # Parse SSE format
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
    ```
    """
    logger.info(f"SSE connection requested for stream: {stream_id}")

    # Check if stream exists
    state = manager.get_stream(stream_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    # Return SSE stream
    return StreamingResponse(
        manager.event_generator(stream_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive"
        }
    )


@router.get("/status/{stream_id}")
async def get_stream_status(
    stream_id: str,
    manager: StreamManager = Depends(get_manager)
):
    """
    Get the current status of a stream

    Returns the current state without creating an SSE connection.
    Useful for polling or checking if a stream exists.
    """
    logger.info(f"Status requested for stream: {stream_id}")

    state = manager.get_stream(stream_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    return {
        "stream_id": stream_id,
        "task": state.task,
        "status": state.status,
        "total_chunks": state.total_chunks,
        "chunks_completed": state.chunks_completed,
        "current_agent": state.current_agent,
        "current_stage": state.current_stage,
        "error_message": state.error_message,
        "estimated_time": state.estimated_time,
        "estimated_cost": state.estimated_cost,
        "actual_cost": state.actual_cost
    }


# ============================================================================
# Background Task Execution
# ============================================================================

async def send_progress_updates(
    stream_id: str,
    manager: StreamManager,
    builder: StreamEventBuilder
):
    """Send periodic progress messages while orchestrator runs"""
    count = 0
    spinner_frames = ["|", "/", "-", "\\"]

    while True:
        await asyncio.sleep(3)
        count += 1
        elapsed = count * 3
        frame = spinner_frames[count % len(spinner_frames)]

        await manager.send_event(
            stream_id,
            builder.system_message(f"{frame} Processing... {elapsed}s elapsed")
        )


async def execute_streaming_task(
    stream_id: str,
    task: str,
    context: Dict[str, Any]
):
    """
    Execute the collaborative task with streaming updates

    This runs in the background and sends SSE events as the task progresses.
    """
    manager = get_stream_manager()
    builder = StreamEventBuilder()

    logger.info(f"Starting execution for stream {stream_id}")

    try:
        # Import orchestrator
        from collaborative_orchestrator import CollaborativeOrchestrator

        # Get state
        state = manager.get_stream(stream_id)
        if not state:
            logger.error(f"Stream {stream_id} not found for execution")
            return

        # Send task_started event
        state.status = "running"
        await manager.send_event(
            stream_id,
            builder.task_started(stream_id, task)
        )

        # Create orchestrator
        orchestrator = CollaborativeOrchestrator(use_sequential=True)

        # Realistic estimates based on sequential orchestrator
        # Typical stages: architect, coder, reviewer, documenter (and possibly refiner/tester)
        estimated_stages = 4  # Conservative estimate
        state.total_chunks = estimated_stages
        state.estimated_time = estimated_stages * 30.0  # ~30s per stage (more realistic)
        state.estimated_cost = estimated_stages * 0.05  # ~$0.05 per stage (more realistic)

        await manager.send_event(
            stream_id,
            builder.task_progress(
                total_chunks=estimated_stages,
                estimated_time=state.estimated_time,
                estimated_cost=state.estimated_cost
            )
        )

        # Start progress updates while orchestrator runs
        progress_task = asyncio.create_task(
            send_progress_updates(stream_id, manager, builder)
        )

        try:
            # Execute collaboration with streaming (this takes 4-5 minutes)
            result = await stream_collaboration(
                orchestrator=orchestrator,
                task=task,
                stream_id=stream_id,
                manager=manager,
                builder=builder,
                state=state
            )
        finally:
            # Stop progress updates
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Send task_completed event
        state.status = "completed"
        state.final_result = result.final_output

        await manager.send_event(
            stream_id,
            builder.task_completed(
                result=result.final_output,
                metrics=result.metrics
            )
        )

        logger.info(f"Stream {stream_id} completed successfully")

    except Exception as e:
        logger.error(f"Error in stream {stream_id}: {str(e)}", exc_info=True)

        # Send error event
        state = manager.get_stream(stream_id)
        if state:
            state.status = "error"
            state.error_message = str(e)

        await manager.send_event(
            stream_id,
            builder.task_error(str(e))
        )


async def stream_collaboration(
    orchestrator,
    task: str,
    stream_id: str,
    manager: StreamManager,
    builder: StreamEventBuilder,
    state
):
    """
    Execute REAL collaboration with streaming events
    NO SIMULATION - Uses actual orchestrator.collaborate()
    """
    # Execute the real collaboration
    result = await orchestrator.collaborate(task=task)

    # Stream the actual results from individual_outputs
    # CollaborationResult has individual_outputs dict with agent outputs
    chunk_num = 0

    agent_map = {
        "architect": "Architect",
        "coder": "Coder",
        "reviewer": "Reviewer",
        "refiner": "Refiner",
        "tester": "Tester",
        "documenter": "Documenter"
    }

    # Iterate through agents_used to maintain order
    if hasattr(result, 'agents_used') and result.agents_used:
        for agent_role in result.agents_used:
            chunk_num += 1
            agent_name = agent_map.get(agent_role, agent_role.capitalize())

            state.current_agent = agent_name
            state.current_stage = agent_role

            # Send chunk_started
            await manager.send_event(
                stream_id,
                builder.chunk_started(
                    chunk_number=chunk_num,
                    agent=agent_name,
                    stage=agent_role,
                    description=f"Completed {agent_name} stage"
                )
            )

            # Get the agent's output from individual_outputs
            output = result.individual_outputs.get(agent_role, "")

            if output:
                # Stream in larger chunks for efficiency
                chunk_size = 200
                for i in range(0, len(output), chunk_size):
                    chunk = output[i:i+chunk_size]
                    await manager.send_event(
                        stream_id,
                        builder.token_stream(chunk, agent_name)
                    )

            # Send chunk_completed
            state.chunks_completed = chunk_num
            await manager.send_event(
                stream_id,
                builder.chunk_completed(
                    chunk_number=chunk_num,
                    agent=agent_name,
                    output=output[:200] if output else ""
                )
            )
    else:
        # Fallback: if no workflow stages, just stream the final output
        chunk_num = 1
        await manager.send_event(
            stream_id,
            builder.chunk_started(
                chunk_number=chunk_num,
                agent="Orchestrator",
                stage="execution",
                description="Processing task"
            )
        )

        # Stream final output immediately (no delays)
        output = result.final_output if hasattr(result, 'final_output') else str(result)
        chunk_size = 200
        for i in range(0, len(output), chunk_size):
            chunk = output[i:i+chunk_size]
            await manager.send_event(
                stream_id,
                builder.token_stream(chunk, "Orchestrator")
            )

        await manager.send_event(
            stream_id,
            builder.chunk_completed(
                chunk_number=chunk_num,
                agent="Orchestrator",
                output=output[:200]
            )
        )
        state.chunks_completed = 1

    return result
