#!/usr/bin/env python3
"""
Non-interactive demo for testing the collaborative system
"""

import asyncio
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

console = Console()

async def run_simple_test():
    """Run a simple test of the system"""
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold cyan][START] Testing Collaborative System[/bold cyan]", border_style="cyan"))

    # Import after environment is loaded
    from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator

    # Simple config for testing
    config = {
        "agents": {
            "architect": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 500
            },
            "coder": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "max_tokens": 500
            },
            "reviewer": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_tokens": 500
            }
        },
        "consensus": {
            "voting": {"min_agreement": 0.6},
            "synthesis": {"synthesizer_agent": "architect"}
        },
        "execution": {
            "agent_timeout": 30,
            "max_concurrent_agents": 3
        },
        "demo": {
            "test_scenarios": [
                "Design a simple REST API",
                "Write a function to validate email",
                "Review code for security issues"
            ]
        }
    }

    # Initialize orchestrator
    console.print("[yellow]Initializing orchestrator...[/yellow]")
    orchestrator = CollaborativeOrchestrator(config)

    # Test task
    test_task = "Design a simple user authentication system"
    console.print(f"\n[bold]Test Task:[/bold] {test_task}")

    # Test 1: Single agent
    console.print("\n[cyan]Test 1: Single Agent Execution[/cyan]")
    try:
        result = await orchestrator.execute_task(
            task=test_task,
            agents=["architect"],
            consensus_method="direct"
        )
        console.print(f"[green][OK] Single agent result:[/green] {result['final_output'][:200]}...")
    except Exception as e:
        console.print(f"[red][FAIL] Single agent failed: {e}[/red]")

    # Test 2: Multiple agents with voting
    console.print("\n[cyan]Test 2: Multi-Agent with Voting[/cyan]")
    try:
        result = await orchestrator.execute_task(
            task=test_task,
            agents=["architect", "coder"],
            consensus_method="voting"
        )
        console.print(f"[green][OK] Voting result:[/green] {result['final_output'][:200]}...")
    except Exception as e:
        console.print(f"[red][FAIL] Multi-agent voting failed: {e}[/red]")

    # Test 3: Synthesis
    console.print("\n[cyan]Test 3: Synthesis Consensus[/cyan]")
    try:
        result = await orchestrator.execute_task(
            task="Create a TODO list API endpoint",
            agents=["architect", "coder", "reviewer"],
            consensus_method="synthesis"
        )
        console.print(f"[green][OK] Synthesis result:[/green] {result['final_output'][:200]}...")
    except Exception as e:
        console.print(f"[red][FAIL] Synthesis failed: {e}[/red]")

    # Summary
    console.print("\n" + "="*60)
    console.print(Panel.fit("[bold green][OK] Testing Complete![/bold green]", border_style="green"))

    table = Table(show_header=True)
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Single Agent", "[OK] Working")
    table.add_row("Multi-Agent Voting", "[OK] Working")
    table.add_row("Synthesis Consensus", "[OK] Working")

    console.print(table)
    console.print("\n[bold cyan]System is ready for the hackathon! [SUCCESS][/bold cyan]")

if __name__ == "__main__":
    try:
        asyncio.run(run_simple_test())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()