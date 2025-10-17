"""
Streaming API Router for Facilitair
Provides SSE endpoints for real-time collaborative orchestration
"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.streaming.sse_handler import (
    get_stream_manager,
    StreamEventBuilder,
    StreamManager
)
from backend.services.task_executor import execute_streaming_task

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

        # Create stream state (async now)
        state = await manager.create_stream(
            stream_id=stream_id,
            task=request.task,
            context=request.context
        )

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
