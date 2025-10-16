#!/usr/bin/env python3
"""
Interactive Demo with User Strategy Selection
Allows users to choose between QUALITY, COST, or BALANCED approaches
"""

import asyncio
import weave
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator
from agents.strategy_selector import Strategy, interactive_setup
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()


def display_strategy_info():
    """Display information about available strategies"""
    table = Table(title="🎯 Model Selection Strategies", show_header=True)
    table.add_column("Strategy", style="cyan", width=20)
    table.add_column("Description", style="white", width=40)
    table.add_column("Best For", style="green", width=30)

    strategies = [
        ("QUALITY_FIRST", "Use best models regardless of cost", "Production-critical code"),
        ("COST_FIRST", "Free open-source models only", "High-volume, budget projects"),
        ("BALANCED", "Smart mix for best value", "Most use cases"),
        ("SPEED_FIRST", "Fastest response times", "Real-time applications"),
        ("PRIVACY_FIRST", "Local models only, no cloud", "Sensitive data"),
    ]

    for name, desc, best_for in strategies:
        table.add_row(name, desc, best_for)

    console.print(table)


def display_model_comparison():
    """Show comparison between closed and open source models"""
    console.print("\n📊 [bold cyan]Model Performance Comparison[/bold cyan]")

    table = Table(show_header=True)
    table.add_column("Model", style="cyan", width=35)
    table.add_column("Type", width=12)
    table.add_column("SWE-bench", width=12)
    table.add_column("Cost/M tokens", width=15)

    models = [
        ("Claude Sonnet 4.5", "Closed", "77.2%", "$3-15"),
        ("GPT-5", "Closed", "74.9%", "$30-60"),
        ("Qwen 2.5 Coder 32B", "[green]Open[/green]", "38% (73.7 Aider)", "[green]Free[/green]"),
        ("DeepSeek V3", "[green]Open[/green]", "42%", "[green]Free[/green]"),
        ("Llama 3.3 70B", "[green]Open[/green]", "35%", "[green]Free[/green]"),
    ]

    for model, type_, bench, cost in models:
        table.add_row(model, type_, bench, cost)

    console.print(table)


async def run_demo_task(orchestrator, task, strategy_name):
    """Run a single demo task with progress display"""
    console.print(f"\n📝 Task: [yellow]{task}[/yellow]")
    console.print(f"🎯 Strategy: [cyan]{strategy_name}[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Orchestrating agents...", total=None)

        # Run the collaboration
        result = await orchestrator.collaborate(task)

        progress.stop()

    # Display results
    console.print("\n✅ [bold green]Collaboration Complete![/bold green]")

    # Show which models were used
    strategy_summary = orchestrator.get_strategy_summary()
    console.print(f"\n📊 Models Used (based on {strategy_name} strategy):")
    console.print(f"   Total Cost: ${strategy_summary['total_cost']:.2f}")
    console.print(f"   Remaining Budget: ${strategy_summary['remaining_daily_budget']:.2f}")

    # Display collaboration metrics
    metrics_table = Table(show_header=True, title="Collaboration Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    metrics_table.add_row("Agents Used", str(len(result.agents_used)))
    metrics_table.add_row("Consensus Method", result.consensus_method)
    metrics_table.add_row("Consensus Rounds", str(result.consensus_rounds))
    metrics_table.add_row("Quality Score", f"{result.metrics.get('quality', 0):.1%}")
    metrics_table.add_row("Efficiency", f"{result.metrics.get('efficiency', 0):.1%}")

    console.print(metrics_table)

    return result


async def main():
    """Main demo with interactive strategy selection"""
    console.print(Panel.fit(
        "[bold cyan]🚀 WeaveHacks Collaborative Orchestrator Demo[/bold cyan]\n" +
        "[white]with User-Controlled Model Selection Strategy[/white]",
        border_style="cyan"
    ))

    # Display available strategies
    display_strategy_info()

    # Display model comparison
    display_model_comparison()

    # Interactive strategy selection
    console.print("\n[bold]Choose Your Priority:[/bold]")
    console.print("1. QUALITY_FIRST - Best models, regardless of cost")
    console.print("2. COST_FIRST - Free open-source models only")
    console.print("3. BALANCED - Smart mix for best value")
    console.print("4. SPEED_FIRST - Fastest response times")
    console.print("5. PRIVACY_FIRST - Local models only")

    choice = IntPrompt.ask("Enter your choice", default=3, choices=["1", "2", "3", "4", "5"])

    strategy_map = {
        1: (Strategy.QUALITY_FIRST, "QUALITY_FIRST"),
        2: (Strategy.COST_FIRST, "COST_FIRST"),
        3: (Strategy.BALANCED, "BALANCED"),
        4: (Strategy.SPEED_FIRST, "SPEED_FIRST"),
        5: (Strategy.PRIVACY_FIRST, "PRIVACY_FIRST"),
    }

    strategy, strategy_name = strategy_map[choice]

    # Initialize Weave
    console.print("\n🔧 Initializing W&B Weave tracking...")
    weave.init('weavehacks-collaborative-demo')

    # Create orchestrator with selected strategy
    console.print(f"\n🎯 Creating orchestrator with {strategy_name} strategy...")
    orchestrator = CollaborativeOrchestrator(user_strategy=strategy)

    # Demo tasks
    tasks = [
        "Build a REST API with authentication and rate limiting",
        "Review this code for security vulnerabilities and suggest fixes",
        "Write comprehensive documentation for a machine learning pipeline",
    ]

    console.print("\n[bold cyan]Running Demo Tasks[/bold cyan]")
    console.print("=" * 50)

    # Run each task
    for i, task in enumerate(tasks, 1):
        console.print(f"\n[bold]Task {i} of {len(tasks)}[/bold]")
        await run_demo_task(orchestrator, task, strategy_name)

        # Allow strategy change between tasks
        if i < len(tasks):
            change = Prompt.ask("\nChange strategy for next task?", choices=["y", "n"], default="n")
            if change == "y":
                new_choice = IntPrompt.ask("New strategy (1-5)", choices=["1", "2", "3", "4", "5"])
                strategy, strategy_name = strategy_map[new_choice]
                orchestrator.set_user_strategy(strategy)
                console.print(f"✅ Strategy changed to: {strategy_name}")

    # Final summary
    console.print("\n" + "=" * 50)
    console.print(Panel.fit(
        "[bold green]Demo Complete![/bold green]",
        border_style="green"
    ))

    final_summary = orchestrator.get_strategy_summary()
    console.print("\n[bold]Session Summary:[/bold]")
    console.print(f"   Strategy Used: {final_summary['current_strategy']}")
    console.print(f"   Total Tasks: {final_summary['task_count']}")
    console.print(f"   Total Cost: ${final_summary['total_cost']:.2f}")
    console.print(f"   Avg Cost/Task: ${final_summary['avg_cost_per_task']:.2f}")

    # Show cost savings
    if strategy in [Strategy.COST_FIRST, Strategy.BALANCED]:
        # Estimate what it would have cost with premium models
        premium_estimate = final_summary['task_count'] * 15.0  # Rough estimate
        savings = premium_estimate - final_summary['total_cost']
        console.print(f"\n💰 [bold green]Estimated Savings: ${savings:.2f}[/bold green]")
        console.print(f"   (vs using premium models exclusively)")

    console.print("\n🎉 Thank you for trying the demo!")
    console.print("📊 Check W&B for detailed metrics: https://wandb.ai/weavehacks-collaborative")


if __name__ == "__main__":
    asyncio.run(main())