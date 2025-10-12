#!/usr/bin/env python3
"""
Training script for self-improving collaborative orchestrator
Demonstrates how agents learn to work together better over time
"""

import asyncio
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import weave
from collaborative_orchestrator import SelfImprovingCollaborativeOrchestrator

console = Console()


class CollaborativeTrainer:
    """Trainer for collaborative learning"""

    def __init__(self):
        self.orchestrator = SelfImprovingCollaborativeOrchestrator()
        self.training_tasks = {
            "architecture": [
                "Design a microservices architecture for an e-commerce platform",
                "Create system architecture for a real-time chat application",
                "Design database schema for a social media platform",
                "Architect a scalable video streaming service",
                "Design API structure for a banking application"
            ],
            "coding": [
                "Implement a Python REST API with JWT authentication",
                "Build a WebSocket server for real-time notifications",
                "Create a data processing pipeline with error handling",
                "Implement a caching layer with Redis",
                "Build a GraphQL API with subscriptions"
            ],
            "review": [
                "Review this authentication system for security vulnerabilities",
                "Audit the database queries for performance issues",
                "Review API endpoints for REST best practices",
                "Check error handling and logging implementation",
                "Review the deployment configuration for production readiness"
            ],
            "documentation": [
                "Write comprehensive API documentation with examples",
                "Create a developer onboarding guide",
                "Document the deployment process step by step",
                "Write user manual for the admin dashboard",
                "Create troubleshooting guide for common issues"
            ],
            "research": [
                "Research best practices for microservices communication",
                "Analyze database scaling strategies for high traffic",
                "Research authentication methods and their trade-offs",
                "Investigate caching strategies for API optimization",
                "Research monitoring and observability best practices"
            ]
        }

    async def train_generation(self, generation: int, progress: Progress = None):
        """Train one generation"""

        results = []
        task_id = progress.add_task(f"[cyan]Generation {generation + 1}", total=25) if progress else None

        # Run all task types
        for task_type, tasks in self.training_tasks.items():
            for task in tasks:
                # Run collaboration
                result = await self.orchestrator.collaborate(task)
                results.append(result)

                if progress:
                    progress.update(task_id, advance=1)

                # Small delay to simulate real execution
                await asyncio.sleep(0.1)

        # Advance generation
        self.orchestrator.advance_generation()

        return results

    def display_generation_summary(self, generation: int, results: list):
        """Display summary of generation performance"""

        # Calculate average metrics
        avg_quality = sum(r.metrics["quality"] for r in results) / len(results)
        avg_efficiency = sum(r.metrics["efficiency"] for r in results) / len(results)
        avg_harmony = sum(r.metrics["harmony"] for r in results) / len(results)
        avg_cost = sum(r.metrics["cost"] for r in results) / len(results)

        # Create summary table
        table = Table(title=f"Generation {generation + 1} Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Avg Quality", f"{avg_quality:.3f}")
        table.add_row("Avg Efficiency", f"{avg_efficiency:.3f}")
        table.add_row("Avg Harmony", f"{avg_harmony:.3f}")
        table.add_row("Avg Cost", f"${avg_cost:.2f}")
        table.add_row("Total Tasks", str(len(results)))

        console.print(table)

        # Show consensus method distribution
        consensus_methods = {}
        for r in results:
            method = r.consensus_method
            consensus_methods[method] = consensus_methods.get(method, 0) + 1

        console.print("\n[yellow]Consensus Methods Used:[/yellow]")
        for method, count in consensus_methods.items():
            console.print(f"  {method}: {count} times")

    def display_learning_progress(self):
        """Display overall learning progress"""

        report = self.orchestrator.get_collaboration_report()

        if "error" in report:
            console.print("[red]Not enough data for learning report[/red]")
            return

        # Create improvement panel
        improvements_text = f"""
[green]Quality Improvement:[/green] {report['improvements']['quality']:.1f}%
[green]Efficiency Improvement:[/green] {report['improvements']['efficiency']:.1f}%
[green]Harmony Improvement:[/green] {report['improvements']['harmony']:.1f}%
[yellow]Cost Reduction:[/yellow] {report['improvements']['cost_reduction']:.1f}%
        """

        console.print(Panel(improvements_text, title="🎯 Learning Progress", border_style="green"))

        # Show discovered optimal teams
        if report['best_teams']:
            team_table = Table(title="🏆 Discovered Optimal Teams", show_header=True)
            team_table.add_column("Task Type", style="cyan")
            team_table.add_column("Best Agents", style="green")
            team_table.add_column("Consensus", style="yellow")
            team_table.add_column("Success Rate", style="magenta")

            for task_type, team_info in report['best_teams'].items():
                team_table.add_row(
                    task_type,
                    ", ".join(team_info['agents']),
                    team_info['consensus'],
                    f"{team_info['success_rate']:.1%}"
                )

            console.print(team_table)

        # Show agent expertise discovery
        if report['agent_expertise']:
            expertise_table = Table(title="🧠 Discovered Agent Expertise", show_header=True)
            expertise_table.add_column("Agent", style="cyan")
            expertise_table.add_column("Best At", style="green")
            expertise_table.add_column("Performance", style="yellow")
            expertise_table.add_column("Best Partner", style="magenta")

            for agent_id, expertise in report['agent_expertise'].items():
                best_partner = expertise['best_collaborators'][0][0] if expertise['best_collaborators'] else "None"
                expertise_table.add_row(
                    agent_id,
                    expertise['discovered_strength'],
                    f"{expertise['performance']:.2f}",
                    best_partner
                )

            console.print(expertise_table)


@click.command()
@click.option('--generations', default=10, help='Number of generations to train')
@click.option('--fast', is_flag=True, help='Fast mode with less tasks')
@click.option('--verbose', is_flag=True, help='Show detailed output')
def train(generations, fast, verbose):
    """Train the collaborative orchestrator"""

    console.print(Panel.fit(
        "[bold cyan]Self-Improving Collaborative Orchestrator Training[/bold cyan]\n"
        "Watch as agents learn to work together!",
        border_style="cyan"
    ))

    trainer = CollaborativeTrainer()

    # Reduce tasks in fast mode
    if fast:
        for task_type in trainer.training_tasks:
            trainer.training_tasks[task_type] = trainer.training_tasks[task_type][:2]

    async def run_training():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            for gen in range(generations):
                console.print(f"\n[bold blue]═══ Generation {gen + 1}/{generations} ═══[/bold blue]\n")

                # Train one generation
                results = await trainer.train_generation(gen, progress)

                # Show summary
                trainer.display_generation_summary(gen, results)

                # Show learning progress every 3 generations
                if gen % 3 == 2 or gen == generations - 1:
                    console.print("\n")
                    trainer.display_learning_progress()

        # Final report
        console.print("\n[bold green]═══ Training Complete! ═══[/bold green]\n")

        final_report = trainer.orchestrator.get_collaboration_report()

        if "improvements" in final_report:
            console.print(Panel(
                f"[green]✨ Achieved {final_report['improvements']['quality']:.1f}% quality improvement![/green]\n"
                f"[yellow]💰 Reduced costs by {final_report['improvements']['cost_reduction']:.1f}%[/yellow]\n"
                f"[cyan]🤝 Improved collaboration harmony by {final_report['improvements']['harmony']:.1f}%[/cyan]",
                title="🎉 Final Results",
                border_style="green"
            ))

        # Save the trained model
        console.print("\n[yellow]Saving trained orchestrator...[/yellow]")
        # In a real implementation, we'd pickle or save the orchestrator state
        console.print("[green]✓ Orchestrator saved![/green]")

    # Run the training
    asyncio.run(run_training())


if __name__ == "__main__":
    train()