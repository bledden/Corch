#!/usr/bin/env python3
"""
Execution script for trained collaborative orchestrator
Shows the difference between untrained and trained collaboration
"""

import asyncio
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.syntax import Syntax
import weave
from collaborative_orchestrator import SelfImprovingCollaborativeOrchestrator

console = Console()


async def compare_generations(task: str, show_outputs: bool = False):
    """Compare untrained vs trained orchestrator on the same task"""

    console.print(Panel.fit(
        f"[bold cyan]Task:[/bold cyan] {task}",
        border_style="cyan"
    ))

    # Create two orchestrators
    untrained = SelfImprovingCollaborativeOrchestrator()
    trained = SelfImprovingCollaborativeOrchestrator()

    # Simulate training for the trained one
    console.print("\n[yellow]Training orchestrator (simulated fast training)...[/yellow]")

    # Quick training loop
    training_tasks = [
        "Design a REST API",
        "Implement authentication",
        "Review security",
        "Write documentation",
        "Research best practices"
    ]

    for _ in range(5):  # 5 quick generations
        for training_task in training_tasks:
            await trained.collaborate(training_task)
        trained.advance_generation()

    console.print("[green]✓ Training complete![/green]\n")

    # Now execute the actual task with both
    console.print("[bold]Executing with UNTRAINED orchestrator...[/bold]")
    untrained_result = await untrained.collaborate(task)

    console.print("[bold]Executing with TRAINED orchestrator...[/bold]")
    trained_result = await trained.collaborate(task)

    # Display comparison
    console.print("\n[bold cyan]═══ Results Comparison ═══[/bold cyan]\n")

    # Create comparison table
    comparison = Table(show_header=True, header_style="bold magenta")
    comparison.add_column("Metric", style="cyan", width=20)
    comparison.add_column("Untrained", style="red", width=25)
    comparison.add_column("Trained", style="green", width=25)
    comparison.add_column("Improvement", style="yellow", width=15)

    # Add agent selection row
    comparison.add_row(
        "Agents Selected",
        ", ".join(untrained_result.agents_used),
        ", ".join(trained_result.agents_used),
        "✓" if trained_result.agents_used != untrained_result.agents_used else "-"
    )

    # Add consensus method row
    comparison.add_row(
        "Consensus Method",
        untrained_result.consensus_method,
        trained_result.consensus_method,
        "✓" if trained_result.consensus_method != untrained_result.consensus_method else "-"
    )

    # Add metrics rows
    metrics_to_compare = ["quality", "efficiency", "harmony", "cost"]
    for metric in metrics_to_compare:
        untrained_val = untrained_result.metrics[metric]
        trained_val = trained_result.metrics[metric]

        if metric == "cost":
            improvement = f"-{((untrained_val - trained_val) / untrained_val * 100):.1f}%"
            comparison.add_row(
                metric.capitalize(),
                f"${untrained_val:.2f}",
                f"${trained_val:.2f}",
                improvement if trained_val < untrained_val else "-"
            )
        else:
            improvement = f"+{((trained_val - untrained_val) / untrained_val * 100):.1f}%"
            comparison.add_row(
                metric.capitalize(),
                f"{untrained_val:.3f}",
                f"{trained_val:.3f}",
                improvement if trained_val > untrained_val else "-"
            )

    # Add consensus rounds
    comparison.add_row(
        "Consensus Rounds",
        str(untrained_result.consensus_rounds),
        str(trained_result.consensus_rounds),
        f"-{untrained_result.consensus_rounds - trained_result.consensus_rounds}"
        if trained_result.consensus_rounds < untrained_result.consensus_rounds else "-"
    )

    # Add conflicts resolved
    comparison.add_row(
        "Conflicts",
        str(untrained_result.conflicts_resolved),
        str(trained_result.conflicts_resolved),
        f"-{untrained_result.conflicts_resolved - trained_result.conflicts_resolved}"
        if trained_result.conflicts_resolved < untrained_result.conflicts_resolved else "-"
    )

    console.print(comparison)

    # Show individual outputs if requested
    if show_outputs:
        console.print("\n[bold cyan]═══ Individual Agent Outputs ═══[/bold cyan]\n")

        # Untrained outputs
        untrained_panel = Panel(
            "\n".join([f"[yellow]{agent}:[/yellow]\n{output[:200]}..."
                      for agent, output in untrained_result.individual_outputs.items()]),
            title="[red]Untrained Collaboration[/red]",
            border_style="red"
        )

        # Trained outputs
        trained_panel = Panel(
            "\n".join([f"[yellow]{agent}:[/yellow]\n{output[:200]}..."
                      for agent, output in trained_result.individual_outputs.items()]),
            title="[green]Trained Collaboration[/green]",
            border_style="green"
        )

        console.print(Columns([untrained_panel, trained_panel]))

    # Show learning insights
    if hasattr(trained, 'task_type_patterns'):
        task_type = trained._classify_task(task)
        pattern = trained.task_type_patterns.get(task_type, {})

        if pattern and pattern.get('best_agents'):
            console.print("\n[bold cyan]═══ Learning Insights ═══[/bold cyan]\n")

            insights = f"""
The trained orchestrator learned that for [yellow]{task_type}[/yellow] tasks:
• Best team: [green]{', '.join(pattern['best_agents'])}[/green]
• Optimal consensus: [cyan]{pattern['best_consensus']}[/cyan]
• Team size: [magenta]{pattern['optimal_team_size']}[/magenta] agents
• Success rate: [green]{pattern.get('success_rate', 0):.1%}[/green]
            """

            console.print(Panel(insights, title="🧠 What the System Learned", border_style="cyan"))


@click.command()
@click.argument('task', required=False)
@click.option('--show-outputs', is_flag=True, help='Show individual agent outputs')
@click.option('--demo', is_flag=True, help='Run demo with multiple tasks')
def execute(task, show_outputs, demo):
    """Execute a task with trained collaborative orchestrator"""

    console.print(Panel.fit(
        "[bold cyan]Self-Improving Collaborative Orchestrator[/bold cyan]\n"
        "Comparing untrained vs trained collaboration",
        border_style="cyan"
    ))

    if demo:
        # Run demo with multiple example tasks
        demo_tasks = [
            "Build a REST API with authentication and deploy it to production",
            "Review this codebase for security vulnerabilities and create a report",
            "Design a microservices architecture for a video streaming platform",
            "Write comprehensive documentation for our API including tutorials",
            "Research and recommend a database scaling strategy"
        ]

        async def run_demo():
            for demo_task in demo_tasks:
                console.print(f"\n[bold blue]{'═' * 60}[/bold blue]\n")
                await compare_generations(demo_task, show_outputs)
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()

        asyncio.run(run_demo())

    elif task:
        # Execute specific task
        asyncio.run(compare_generations(task, show_outputs))

    else:
        # Interactive mode
        console.print("\n[yellow]Enter a task to execute (or 'quit' to exit):[/yellow]")

        async def interactive_loop():
            while True:
                task_input = console.input("\n[cyan]Task>[/cyan] ")

                if task_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if task_input.strip():
                    await compare_generations(task_input, show_outputs)

        asyncio.run(interactive_loop())


if __name__ == "__main__":
    execute()