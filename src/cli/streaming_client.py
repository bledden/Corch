#!/usr/bin/env python3
"""
CLI Streaming Client for Facilitair Collaborative Orchestration
Connects to existing SSE endpoints and visualizes streaming debate in terminal.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

# Configuration
API_BASE = os.getenv("FACILITAIR_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("FAC_API_KEY", "")


class StreamingDebateClient:
    """CLI client for streaming collaborative orchestration."""

    def __init__(self):
        self.console = Console()
        self.stream_id: Optional[str] = None
        self.task_description: str = ""
        self.agent_outputs: Dict[str, str] = {}
        self.current_agent: Optional[str] = None
        self.final_result: Optional[str] = None
        self.chunk_count: int = 0
        self.total_chunks: Optional[int] = None
        self.estimated_time: Optional[float] = None
        self.estimated_cost: Optional[float] = None
        self.status: str = "initializing"
        self.error_message: Optional[str] = None

    async def create_stream(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new stream via POST /api/stream/task"""
        headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
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
        headers = {"X-API-Key": API_KEY}
        url = f"{API_BASE}/api/stream/events/{stream_id}"

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()

                    event_type = None
                    data_buffer = []

                    async for line in response.aiter_lines():
                        # Skip empty lines and comments
                        if not line or line.startswith(":"):
                            continue

                        # Blank line signals end of event
                        if not line.strip():
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
            self.error_message = f"HTTP {e.response.status_code}: {e.response.text}"
            self.status = "error"
        except Exception as e:
            self.error_message = str(e)
            self.status = "error"

    def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Process SSE events and update internal state"""

        if event_type == "task_started":
            self.status = "running"
            self.console.print("[bold green]Task Started[/bold green]")

        elif event_type == "task_progress":
            # Extract planning information
            self.total_chunks = data.get("total_chunks")
            self.estimated_time = data.get("estimated_time")
            self.estimated_cost = data.get("estimated_cost")

            plan_text = f"Plan: {self.total_chunks} chunks"
            if self.estimated_time:
                plan_text += f", ~{self.estimated_time:.1f}s"
            if self.estimated_cost:
                plan_text += f", ~${self.estimated_cost:.4f}"

            self.console.print(f"[cyan]{plan_text}[/cyan]")

        elif event_type == "chunk_started":
            self.chunk_count += 1
            agent_name = data.get("agent", "Agent")
            stage = data.get("stage", "working")

            # Map to debate metaphor
            if "architect" in agent_name.lower() or "architecture" in stage.lower():
                display_agent = "Architect"
                action = "Analyzing requirements..."
            elif "coder" in agent_name.lower() or "implementation" in stage.lower():
                display_agent = "Coder"
                action = "Writing implementation..."
            elif "review" in agent_name.lower() or "review" in stage.lower():
                display_agent = "Reviewer"
                action = "Analyzing code quality..."
            elif "refine" in agent_name.lower() or "refinement" in stage.lower():
                display_agent = "Refiner"
                action = "Improving solution..."
            elif "doc" in agent_name.lower() or "documentation" in stage.lower():
                display_agent = "Documenter"
                action = "Creating documentation..."
            else:
                display_agent = f"{agent_name}"
                action = data.get("description", "Processing...")

            self.current_agent = display_agent
            self.agent_outputs[display_agent] = ""

            chunk_info = f"[bold blue]{display_agent}[/bold blue]: {action}"
            if self.total_chunks:
                chunk_info += f" [dim]({self.chunk_count}/{self.total_chunks})[/dim]"

            self.console.print(chunk_info)

        elif event_type == "token_stream":
            # Append tokens to current agent's output
            tokens = data.get("tokens", "")
            if self.current_agent and tokens:
                self.agent_outputs[self.current_agent] = \
                    self.agent_outputs.get(self.current_agent, "") + tokens

        elif event_type == "chunk_completed":
            chunk_num = data.get("chunk_number", self.chunk_count)
            self.console.print(f"[green]Chunk {chunk_num} complete[/green]")

        elif event_type == "task_completed":
            self.status = "completed"
            self.final_result = data.get("result", "")
            self.console.print("[bold green]Task Completed![/bold green]")

        elif event_type == "task_error":
            self.status = "error"
            self.error_message = data.get("error", "Unknown error")
            self.console.print(f"[bold red]Error: {self.error_message}[/bold red]")

        elif event_type == "system_message":
            msg = data.get("message", "")
            self.console.print(f"[dim italic]{msg}[/dim italic]")

        elif event_type == "heartbeat":
            # Silent heartbeat, just keep connection alive
            pass

    def create_dashboard(self) -> Panel:
        """Create live dashboard showing current state"""

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="agents", size=None),
            Layout(name="footer", size=3)
        )

        # Header: Task info
        header_text = Text()
        header_text.append("Task: ", style="bold cyan")
        header_text.append(self.task_description[:80], style="white")
        if len(self.task_description) > 80:
            header_text.append("...", style="dim")

        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Agents: Current outputs
        agents_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            border_style="blue"
        )
        agents_table.add_column("Agent", style="cyan", width=15)
        agents_table.add_column("Output", style="white")

        for agent, output in self.agent_outputs.items():
            # Truncate output for display
            display_output = output[:200]
            if len(output) > 200:
                display_output += "..."

            agents_table.add_row(agent, display_output)

        layout["agents"].update(Panel(agents_table, title="Agent Outputs", border_style="blue"))

        # Footer: Status
        footer_text = Text()
        if self.status == "running":
            footer_text.append("Status: ", style="bold yellow")
            footer_text.append("STREAMING", style="bold yellow")
        elif self.status == "completed":
            footer_text.append("Status: ", style="bold green")
            footer_text.append("COMPLETED", style="bold green")
        elif self.status == "error":
            footer_text.append("Status: ", style="bold red")
            footer_text.append(f"ERROR - {self.error_message}", style="bold red")
        else:
            footer_text.append("Status: ", style="bold blue")
            footer_text.append(self.status.upper(), style="bold blue")

        if self.total_chunks:
            footer_text.append(f"  |  Progress: {self.chunk_count}/{self.total_chunks}", style="dim")

        layout["footer"].update(Panel(footer_text, border_style="cyan"))

        return Panel(layout, title="Facilitair Streaming Debate", border_style="bright_blue")

    async def stream_task(self, task: str, context: Optional[Dict[str, Any]] = None, live_ui: bool = True):
        """Main method: Create stream and consume events"""

        # Create stream
        self.console.print(f"[bold cyan]Creating stream for task...[/bold cyan]")
        stream_id = await self.create_stream(task, context)
        self.console.print(f"[green]Stream created: {stream_id}[/green]\n")

        if live_ui:
            # Live dashboard mode
            with Live(self.create_dashboard(), refresh_per_second=4, console=self.console) as live:
                async for event_type, data in self.consume_sse(stream_id):
                    self.handle_event(event_type, data)
                    live.update(self.create_dashboard())

                    # Exit on completion or error
                    if self.status in ["completed", "error"]:
                        break
        else:
            # Simple streaming mode (just handle events)
            async for event_type, data in self.consume_sse(stream_id):
                self.handle_event(event_type, data)

                if self.status in ["completed", "error"]:
                    break

        # Print final result
        if self.final_result:
            self.console.print("\n[bold green]=== Final Result ===[/bold green]")
            self.console.print(self.final_result)

        return self.final_result


async def main():
    """Main entry point"""

    # Parse command line arguments
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python cli/streaming_client.py \"Your task here\"[/yellow]")
        console.print("[dim]Example: python cli/streaming_client.py \"Build a REST API with JWT auth\"[/dim]")
        sys.exit(1)

    task = sys.argv[1]

    # Optional: Parse context from JSON file or args
    context = {}
    if len(sys.argv) > 2:
        try:
            context = json.loads(sys.argv[2])
        except json.JSONDecodeError:
            console.print("[yellow]Warning: Could not parse context JSON, using empty context[/yellow]")

    # Check for API key
    if not API_KEY:
        console.print("[red]Error: FAC_API_KEY environment variable not set[/red]")
        console.print("[dim]Set it with: export FAC_API_KEY='your-api-key'[/dim]")
        sys.exit(1)

    # Create client and run
    client = StreamingDebateClient()

    try:
        await client.stream_task(task, context, live_ui=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stream interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
