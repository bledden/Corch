"""
Task Execution Service for Facilitair Streaming

Handles background task execution and streaming event generation.
Isolated from router layer to prevent circular imports.
"""

import asyncio
import logging
from typing import Dict, Any

from backend.streaming.sse_handler import (
    get_stream_manager,
    StreamEventBuilder,
    StreamManager
)

# Configure logging
logger = logging.getLogger(__name__)


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
        from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator

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
