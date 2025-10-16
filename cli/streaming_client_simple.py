#!/usr/bin/env python3
"""
Simple CLI Streaming Client for Facilitair Collaborative Orchestration
Stream-based output (no complex Live UI) - more reliable for all terminal types
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Configuration
API_BASE = os.getenv("FACILITAIR_API_BASE", "http://localhost:8000")


class SimpleStreamingClient:
    """Simple CLI client with stream-based output"""

    def __init__(self):
        self.console = Console()
        self.stream_id: Optional[str] = None
        self.task_description: str = ""
        self.current_agent: Optional[str] = None
        self.chunk_count: int = 0
        self.total_chunks: Optional[int] = None

    async def create_stream(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new stream via POST /api/stream/task"""
        headers = {"Content-Type": "application/json"}
        url = f"{API_BASE}/api/stream/task"

        payload = {
            "task": task,
            "context": context or {},
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                self.stream_id = data.get("stream_id")
                self.task_description = task

                return self.stream_id
        except Exception as e:
            self.console.print(f"[red]Error creating stream: {e}[/red]")
            raise

    async def consume_sse(self, stream_id: str):
        """Consume SSE events from GET /api/stream/events/{stream_id}"""
        url = f"{API_BASE}/api/stream/events/{stream_id}"

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    event_type = None
                    data_buffer = []

                    async for line in response.aiter_lines():
                        # Skip comments only
                        if line.startswith(":"):
                            continue

                        # Blank line signals end of event
                        if not line or not line.strip():
                            if event_type and data_buffer:
                                data_str = "\n".join(data_buffer)
                                try:
                                    data = json.loads(data_str)
                                    yield event_type, data
                                except json.JSONDecodeError:
                                    pass

                            # Reset for next event
                            event_type = None
                            data_buffer = []
                            continue

                        # Parse SSE fields
                        if line.startswith("event:"):
                            event_type = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            data_buffer.append(line.split(":", 1)[1].strip())

        except httpx.HTTPStatusError as e:
            self.console.print(f"[red]HTTP {e.response.status_code}: {e.response.text}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

    def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Process SSE events and print output"""

        if event_type == "task_started":
            self.console.print("\n[bold green]>> Task Started[/bold green]")
            self.console.print(f"[dim]Stream ID: {data.get('stream_id')}[/dim]\n")

        elif event_type == "task_progress":
            # Extract planning information
            self.total_chunks = data.get("total_chunks")
            estimated_time = data.get("estimated_time")
            estimated_cost = data.get("estimated_cost")

            plan_parts = [f"[cyan]Plan: {self.total_chunks} stages[/cyan]"]
            if estimated_time:
                plan_parts.append(f"~{estimated_time:.1f}s")
            if estimated_cost:
                plan_parts.append(f"~${estimated_cost:.4f}")

            self.console.print(" | ".join(plan_parts) + "\n")

        elif event_type == "chunk_started":
            self.chunk_count += 1
            agent_name = data.get("agent", "Agent")
            stage = data.get("stage", "working")
            description = data.get("description", "")

            # Map to agent metaphor (NO EMOJIS - ASCII only)
            if "architect" in agent_name.lower() or "architecture" in stage.lower():
                display_name = "Architect"
            elif "coder" in agent_name.lower() or "implementation" in stage.lower():
                display_name = "Coder"
            elif "review" in agent_name.lower() or "review" in stage.lower():
                display_name = "Reviewer"
            elif "refine" in agent_name.lower() or "refinement" in stage.lower():
                display_name = "Refiner"
            elif "doc" in agent_name.lower() or "documentation" in stage.lower():
                display_name = "Documenter"
            else:
                display_name = agent_name

            self.current_agent = display_name

            # Print agent header
            progress = f"({self.chunk_count}/{self.total_chunks})" if self.total_chunks else f"({self.chunk_count})"
            self.console.print(f"\n[bold blue]>> {display_name}[/bold blue] {progress}")

            if description:
                self.console.print(f"[dim italic]{description}[/dim italic]")

            # ASCII progress indicator
            self.console.print("[dim]" + "." * 60 + "[/dim]")

        elif event_type == "token_stream":
            # Print tokens inline (no newline)
            tokens = data.get("tokens", "")
            if tokens:
                self.console.print(tokens, end="", highlight=False)

        elif event_type == "chunk_completed":
            chunk_num = data.get("chunk_number", self.chunk_count)
            self.console.print(f"\n[green][OK] Stage {chunk_num} complete[/green]\n")

        elif event_type == "task_completed":
            result = data.get("result", "")
            metrics = data.get("metrics", {})

            self.console.print("\n\n" + "=" * 70)
            self.console.print("[bold green][OK] Task Completed Successfully![/bold green]")
            self.console.print("=" * 70 + "\n")

            if result:
                panel = Panel(
                    result,
                    title="[bold]Final Result[/bold]",
                    border_style="green",
                    padding=(1, 2)
                )
                self.console.print(panel)

            if metrics:
                self.console.print(f"\n[dim]Metrics: {json.dumps(metrics, indent=2)}[/dim]")

        elif event_type == "task_error":
            error_msg = data.get("error", "Unknown error")
            self.console.print(f"\n[bold red][FAIL] Error: {error_msg}[/bold red]\n")

        elif event_type == "system_message":
            msg = data.get("message", "")
            self.console.print(f"[dim italic][INFO]  {msg}[/dim italic]")

        elif event_type == "heartbeat":
            # Silent heartbeat
            pass

    async def stream_task(self, task: str, context: Optional[Dict[str, Any]] = None):
        """Main method: Create stream and consume events"""

        # Print header
        self.console.print("\n" + "=" * 70)
        self.console.print("[bold cyan][GOAL] Facilitair Streaming Collaboration[/bold cyan]")
        self.console.print("=" * 70)
        self.console.print(f"\n[bold]Task:[/bold] {task}\n")

        # Create stream
        self.console.print("[dim]Creating stream...[/dim]")
        stream_id = await self.create_stream(task, context)
        self.console.print(f"[dim]Connected to stream {stream_id[:8]}...[/dim]")

        # Consume events
        try:
            async for event_type, data in self.consume_sse(stream_id):
                self.handle_event(event_type, data)

                # Exit on completion or error
                if event_type in ["task_completed", "task_error"]:
                    break
        except KeyboardInterrupt:
            self.console.print("\n\n[yellow][WARNING]  Stream interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n\n[red][FAIL] Stream error: {e}[/red]")

        self.console.print("\n" + "=" * 70 + "\n")


async def main():
    """Main entry point"""

    # Parse command line arguments
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python cli/streaming_client_simple.py \"Your task here\"[/yellow]")
        console.print("[dim]Example: python cli/streaming_client_simple.py \"Build a REST API with JWT auth\"[/dim]")
        sys.exit(1)

    task = sys.argv[1]

    # Optional: Parse context from JSON
    context = {}
    if len(sys.argv) > 2:
        try:
            context = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            console.print("[yellow]Warning: Could not parse context JSON, using empty context[/yellow]")

    # Create client and run
    client = SimpleStreamingClient()

    try:
        await client.stream_task(task, context)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
