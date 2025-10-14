"""
SSE (Server-Sent Events) Handler for Facilitair Streaming
Manages event formatting, stream state, and SSE protocol compliance.
"""

import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StreamEventType(str, Enum):
    """SSE event types"""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    CHUNK_STARTED = "chunk_started"
    TOKEN_STREAM = "token_stream"
    CHUNK_COMPLETED = "chunk_completed"
    TASK_COMPLETED = "task_completed"
    TASK_ERROR = "task_error"
    SYSTEM_MESSAGE = "system_message"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamState:
    """Maintains state for an active stream"""
    stream_id: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "initializing"  # initializing, running, completed, error
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    total_chunks: Optional[int] = None
    chunks_completed: int = 0
    current_agent: Optional[str] = None
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    final_result: Optional[str] = None

    # Metrics
    estimated_time: Optional[float] = None
    estimated_cost: Optional[float] = None
    actual_cost: float = 0.0


class SSEFormatter:
    """Formats events according to SSE protocol"""

    @staticmethod
    def format_event(event_type: str, data: Dict[str, Any]) -> str:
        """
        Format an SSE event according to spec:
        event: <event_type>
        data: <json_data>

        (blank line)
        """
        lines = []
        lines.append(f"event: {event_type}")

        # JSON encode data
        json_data = json.dumps(data, ensure_ascii=False)
        lines.append(f"data: {json_data}")

        # Blank line to signal end of event
        lines.append("")

        return "\n".join(lines) + "\n"

    @staticmethod
    def format_comment(comment: str) -> str:
        """Format SSE comment (for keepalive)"""
        return f": {comment}\n\n"


class StreamEventBuilder:
    """Builds SSE events with proper formatting"""

    def __init__(self, formatter: SSEFormatter = None):
        self.formatter = formatter or SSEFormatter()

    def task_started(self, stream_id: str, task: str) -> str:
        """Build task_started event"""
        return self.formatter.format_event(
            StreamEventType.TASK_STARTED,
            {
                "stream_id": stream_id,
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
        )

    def task_progress(
        self,
        total_chunks: int,
        estimated_time: Optional[float] = None,
        estimated_cost: Optional[float] = None
    ) -> str:
        """Build task_progress event with planning info"""
        data = {
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat()
        }

        if estimated_time is not None:
            data["estimated_time"] = estimated_time

        if estimated_cost is not None:
            data["estimated_cost"] = estimated_cost

        return self.formatter.format_event(StreamEventType.TASK_PROGRESS, data)

    def chunk_started(
        self,
        chunk_number: int,
        agent: str,
        stage: str,
        description: Optional[str] = None
    ) -> str:
        """Build chunk_started event"""
        data = {
            "chunk_number": chunk_number,
            "agent": agent,
            "stage": stage,
            "timestamp": datetime.now().isoformat()
        }

        if description:
            data["description"] = description

        return self.formatter.format_event(StreamEventType.CHUNK_STARTED, data)

    def token_stream(self, tokens: str, agent: Optional[str] = None) -> str:
        """Build token_stream event"""
        data = {"tokens": tokens}

        if agent:
            data["agent"] = agent

        return self.formatter.format_event(StreamEventType.TOKEN_STREAM, data)

    def chunk_completed(
        self,
        chunk_number: int,
        agent: str,
        output: Optional[str] = None
    ) -> str:
        """Build chunk_completed event"""
        data = {
            "chunk_number": chunk_number,
            "agent": agent,
            "timestamp": datetime.now().isoformat()
        }

        if output:
            data["output"] = output

        return self.formatter.format_event(StreamEventType.CHUNK_COMPLETED, data)

    def task_completed(
        self,
        result: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build task_completed event"""
        data = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        if metrics:
            data["metrics"] = metrics

        return self.formatter.format_event(StreamEventType.TASK_COMPLETED, data)

    def task_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Build task_error event"""
        data = {
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        if details:
            data["details"] = details

        return self.formatter.format_event(StreamEventType.TASK_ERROR, data)

    def system_message(self, message: str) -> str:
        """Build system_message event"""
        return self.formatter.format_event(
            StreamEventType.SYSTEM_MESSAGE,
            {
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        )

    def heartbeat(self) -> str:
        """Build heartbeat event"""
        return self.formatter.format_event(
            StreamEventType.HEARTBEAT,
            {"timestamp": datetime.now().isoformat()}
        )


class StreamManager:
    """Manages multiple active streams"""

    def __init__(self):
        self.streams: Dict[str, StreamState] = {}
        self.event_queues: Dict[str, asyncio.Queue] = {}
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}

    def create_stream(
        self,
        stream_id: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamState:
        """Create a new stream"""
        state = StreamState(
            stream_id=stream_id,
            task=task,
            context=context or {},
            status="initializing"
        )

        self.streams[stream_id] = state
        self.event_queues[stream_id] = asyncio.Queue()

        # Schedule cleanup after 1 hour
        self._schedule_cleanup(stream_id, timeout=3600)

        return state

    def get_stream(self, stream_id: str) -> Optional[StreamState]:
        """Get stream state by ID"""
        return self.streams.get(stream_id)

    async def send_event(self, stream_id: str, event: str):
        """Send an event to a stream"""
        if stream_id in self.event_queues:
            await self.event_queues[stream_id].put(event)

    async def event_generator(
        self,
        stream_id: str,
        heartbeat_interval: int = 15
    ) -> AsyncIterator[str]:
        """
        Generate SSE events for a stream
        Includes heartbeat to keep connection alive
        """
        if stream_id not in self.event_queues:
            yield SSEFormatter().format_comment(f"Stream {stream_id} not found")
            return

        queue = self.event_queues[stream_id]
        builder = StreamEventBuilder()

        last_heartbeat = datetime.now().timestamp()

        while True:
            try:
                # Wait for event with timeout for heartbeat
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=heartbeat_interval
                )

                yield event

                # Check if stream is complete
                state = self.get_stream(stream_id)
                if state and state.status in ["completed", "error"]:
                    break

            except asyncio.TimeoutError:
                # Send heartbeat
                now = datetime.now().timestamp()
                if now - last_heartbeat >= heartbeat_interval:
                    yield builder.heartbeat()
                    last_heartbeat = now
            except asyncio.CancelledError:
                break
            except Exception as e:
                yield builder.task_error(f"Stream error: {str(e)}")
                break

        # Final heartbeat before closing
        yield SSEFormatter().format_comment("Stream complete")

    def _schedule_cleanup(self, stream_id: str, timeout: int):
        """Schedule stream cleanup after timeout"""
        async def cleanup():
            await asyncio.sleep(timeout)
            await self.cleanup_stream(stream_id)

        task = asyncio.create_task(cleanup())
        self._cleanup_tasks[stream_id] = task

    async def cleanup_stream(self, stream_id: str):
        """Clean up a stream and its resources"""
        if stream_id in self.streams:
            del self.streams[stream_id]

        if stream_id in self.event_queues:
            # Drain queue
            queue = self.event_queues[stream_id]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            del self.event_queues[stream_id]

        if stream_id in self._cleanup_tasks:
            task = self._cleanup_tasks[stream_id]
            if not task.done():
                task.cancel()
            del self._cleanup_tasks[stream_id]

    async def cleanup_all(self):
        """Clean up all streams"""
        stream_ids = list(self.streams.keys())
        for stream_id in stream_ids:
            await self.cleanup_stream(stream_id)


# Global stream manager instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get or create global stream manager"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
