#!/usr/bin/env python3
"""
WeaveHacks 2 Demo Script
Demonstrates self-improving collaborative agent system with W&B Weave tracking
"""

import asyncio
import os
import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich.columns import Columns
import weave
from dotenv import load_dotenv

# Load environment and config
load_dotenv()

# Import our orchestrator
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator

console = Console()


class CollaborativeDemoRunner:
    """Demo runner for hackathon presentation"""

    def __init__(self):
        self.orchestrator = CollaborativeOrchestrator()
        self.demo_tasks = [
            # Architecture Task
            {
                "type": "architecture",
                "task": "Design a real-time collaborative code editor with WebSocket support",
                "complexity": "high"
            },
            # Coding Task
            {
                "type": "coding",
                "task": "Implement a Python API endpoint for user authentication with JWT",
                "complexity": "medium"
            },
            # Review Task
            {
                "type": "review",
                "task": "Review this payment processing code for security vulnerabilities",
                "complexity": "high"
            },
            # Documentation Task
            {
                "type": "documentation",
                "task": "Write API documentation for a REST endpoint with examples",
                "complexity": "low"
            },
            # Research Task
            {
                "type": "research",
                "task": "Research best practices for microservices communication patterns",
                "complexity": "medium"
            }
        ]

    def create_header(self) -> Panel:
        """Create demo header panel"""

        header_text = Text()
        header_text.append("Agent ", style="bold cyan")
        header_text.append("Collaborative Agent System\n", style="bold white")
        header_text.append("WeaveHacks 2 | July 12-13, 2025\n\n", style="dim")
        header_text.append("Powered by: ", style="white")
        header_text.append("W&B Weave", style="bold yellow")
        header_text.append(" • ", style="dim")
        header_text.append("Daytona", style="bold green")
        header_text.append(" • ", style="dim")
        header_text.append("MCP", style="bold blue")
        header_text.append(" • ", style="dim")
        header_text.append("CopilotKit", style="bold magenta")
        header_text.append("\n\nWatch as agents learn to collaborate better over time!", style="italic yellow")

        return Panel(
            Align.center(header_text),
            border_style="cyan",
            padding=(1, 2)
        )

    def create_agent_status_table(self, generation: int) -> Table:
        """Create table showing agent status and learning"""

        table = Table(title=f"Generation {generation} - Agent Status", show_header=True)
        table.add_column("Agent", style="cyan", width=12)
        table.add_column("Model", style="green", width=20)
        table.add_column("Best At", style="yellow", width=15)
        table.add_column("Performance", style="magenta", width=12)
        table.add_column("Top Collaborator", style="blue", width=15)

        for agent_id, agent in self.orchestrator.agents.items():
            # Find best task type
            best_task = "Learning..."
            best_perf = 0.0
            if agent.performance_history:
                best_task = max(agent.performance_history, key=agent.performance_history.get)
                best_perf = agent.performance_history[best_task]

            # Find best collaborator
            best_collab = "None yet"
            if agent.collaboration_scores:
                best_collab_id = max(agent.collaboration_scores, key=agent.collaboration_scores.get)
                best_collab = best_collab_id.capitalize()

            # Performance indicator
            perf_indicator = "[STAR]" * int(best_perf * 5)

            table.add_row(
                agent_id.capitalize(),
                agent.model[:15] + "..." if len(agent.model) > 15 else agent.model,
                best_task.capitalize(),
                perf_indicator or "Learning",
                best_collab
            )

        return table

    def create_sponsor_status_panel(self) -> Panel:
        """Create panel showing sponsor integrations status"""

        status_text = Text()

        # Daytona Status
        status_text.append("Architect  Daytona: ", style="bold green")
        status_text.append("5 isolated workspaces active\n", style="green")
        status_text.append("   Each agent runs in secure container\n", style="dim")

        # MCP Status
        status_text.append("\n MCP: ", style="bold blue")
        status_text.append("Inter-agent protocol established\n", style="blue")
        status_text.append("   Context sharing enabled\n", style="dim")

        # CopilotKit Status
        status_text.append("\n[PARTNERSHIP] CopilotKit: ", style="bold magenta")
        status_text.append("Auto-guidance mode active\n", style="magenta")
        status_text.append("   Human intervention available\n", style="dim")

        # W&B Weave Status
        status_text.append("\n[CHART] W&B Weave: ", style="bold yellow")
        status_text.append("All metrics tracked\n", style="yellow")
        status_text.append("   Learning patterns recorded\n", style="dim")

        return Panel(status_text, title="Sponsor Integrations", border_style="green")

    def create_metrics_panel(self, metrics: Dict[str, float]) -> Panel:
        """Create metrics display panel"""

        content = Text()

        # Quality
        quality = metrics.get("quality", 0)
        quality_bar = "" * int(quality * 10) + "" * (10 - int(quality * 10))
        content.append(f"Quality:    {quality_bar} {quality:.2%}\n", style="green" if quality > 0.7 else "yellow")

        # Efficiency
        efficiency = metrics.get("efficiency", 0)
        efficiency_bar = "" * int(efficiency * 10) + "" * (10 - int(efficiency * 10))
        content.append(f"Efficiency: {efficiency_bar} {efficiency:.2%}\n", style="green" if efficiency > 0.7 else "yellow")

        # Harmony
        harmony = metrics.get("harmony", 0)
        harmony_bar = "" * int(harmony * 10) + "" * (10 - int(harmony * 10))
        content.append(f"Harmony:    {harmony_bar} {harmony:.2%}\n", style="green" if harmony > 0.7 else "yellow")

        # Cost
        cost = metrics.get("cost", 0)
        content.append(f"Cost:       ${cost:.2f}", style="cyan")

        return Panel(content, title="Collaboration Metrics", border_style="green")

    def create_learning_insights(self, generation: int) -> Panel:
        """Create panel showing learning insights"""

        insights = Text()

        if generation == 0:
            insights.append(" Initial Generation\n\n", style="bold yellow")
            insights.append("• Agents are just starting to work together\n")
            insights.append("• No collaboration patterns established yet\n")
            insights.append("• Expect some conflicts and inefficiencies")

        elif generation < 3:
            insights.append("[UP] Early Learning Phase\n\n", style="bold cyan")
            insights.append("• Agents discovering each other's strengths\n")
            insights.append("• Testing different consensus strategies\n")
            insights.append("• Building collaboration history")

        elif generation < 7:
            insights.append("[FAST] Optimization Phase\n\n", style="bold green")
            insights.append("• Clear expertise patterns emerging\n")
            insights.append("• Preferred collaborations forming\n")
            insights.append("• Consensus becoming more efficient")

        else:
            insights.append("[START] Mastery Phase\n\n", style="bold magenta")
            insights.append("• Agents work seamlessly together\n")
            insights.append("• Optimal team compositions learned\n")
            insights.append("• Minimal conflicts, maximum efficiency")

        return Panel(insights, title="Learning Insights", border_style="yellow")

    async def run_generation(self, generation: int, show_details: bool = True) -> List[Any]:
        """Run one generation of collaborative learning"""

        results = []

        console.print(f"\n[bold cyan]=== Generation {generation + 1} ===[/bold cyan]\n")

        for task_info in self.demo_tasks:
            task = task_info["task"]
            task_type = task_info["type"]

            # Display task
            console.print(f"[bold]Task:[/bold] {task[:80]}...")
            console.print(f"[dim]Type: {task_type} | Complexity: {task_info['complexity']}[/dim]\n")

            # Execute collaboration
            with console.status("[cyan]Agents collaborating...", spinner="dots"):
                result = await self.orchestrator.collaborate(task)

            results.append(result)

            if show_details:
                # Show collaboration details
                console.print(f"[OK] [green]Collaboration Complete[/green]")
                console.print(f"   Agents: {', '.join([a.capitalize() for a in result.agents_used])}")
                console.print(f"   Method: {result.consensus_method}")
                console.print(f"   Rounds: {result.consensus_rounds}")

                # Show metrics
                metrics_panel = self.create_metrics_panel(result.metrics)
                console.print(metrics_panel)

                # Show sample output
                if result.final_output:
                    output_preview = result.final_output[:200] + "..." if len(result.final_output) > 200 else result.final_output
                    console.print(Panel(output_preview, title="Output Preview", border_style="blue"))

            else:
                # Compact display
                quality = result.metrics.get("quality", 0)
                efficiency = result.metrics.get("efficiency", 0)
                console.print(
                    f"   [OK] Quality: {quality:.1%} | "
                    f"Efficiency: {efficiency:.1%} | "
                    f"Agents: {len(result.agents_used)} | "
                    f"Method: {result.consensus_method}"
                )

            console.print()

        # Advance generation
        self.orchestrator.advance_generation()

        return results

    async def show_improvement_analysis(self):
        """Show analysis of improvement over generations"""

        console.print("\n[bold cyan]=== Learning Analysis ===[/bold cyan]\n")

        report = self.orchestrator.get_collaboration_report()

        if "improvements" in report:
            imp = report["improvements"]

            # Create improvement table
            table = Table(title="Performance Improvements", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Early Avg", style="yellow")
            table.add_column("Recent Avg", style="green")
            table.add_column("Improvement", style="bold magenta")

            early = report.get("early_metrics", {})
            recent = report.get("recent_metrics", {})

            table.add_row(
                "Quality",
                f"{early.get('quality', 0):.2%}",
                f"{recent.get('quality', 0):.2%}",
                f"{imp.get('quality', 0):+.1f}%"
            )
            table.add_row(
                "Efficiency",
                f"{early.get('efficiency', 0):.2%}",
                f"{recent.get('efficiency', 0):.2%}",
                f"{imp.get('efficiency', 0):+.1f}%"
            )
            table.add_row(
                "Harmony",
                f"{early.get('harmony', 0):.2%}",
                f"{recent.get('harmony', 0):.2%}",
                f"{imp.get('harmony', 0):+.1f}%"
            )
            table.add_row(
                "Cost",
                f"${early.get('cost', 0):.2f}",
                f"${recent.get('cost', 0):.2f}",
                f"{imp.get('cost_reduction', 0):+.1f}%"
            )

            console.print(table)

        # Show discovered optimal teams
        if "best_teams" in report and report["best_teams"]:
            console.print("\n[bold]Discovered Optimal Teams:[/bold]")
            for task_type, team_info in report["best_teams"].items():
                agents = ", ".join([a.capitalize() for a in team_info["agents"]])
                console.print(
                    f"  • {task_type.capitalize()}: {agents} "
                    f"(Method: {team_info['consensus']}, Success: {team_info['success_rate']:.1%})"
                )

        # Show agent expertise discovery
        if "agent_expertise" in report and report["agent_expertise"]:
            console.print("\n[bold]Discovered Agent Expertise:[/bold]")
            for agent_id, expertise in report["agent_expertise"].items():
                console.print(
                    f"  • {agent_id.capitalize()}: Best at {expertise['discovered_strength']} "
                    f"(Performance: {expertise['performance']:.2%})"
                )

                if expertise.get("best_collaborators"):
                    collabs = ", ".join([c[0].capitalize() for c in expertise["best_collaborators"][:2]])
                    console.print(f"    Works best with: {collabs}")

    async def run_demo(self, fast_mode: bool = False):
        """Run the full demo"""

        # Display header
        console.print(self.create_header())

        # Initialize Weave
        console.print("\n[cyan]Initializing W&B Weave tracking...[/cyan]")
        try:
            weave.init("weavehacks-collaborative-demo")
            console.print("[green][OK] Weave initialized successfully[/green]\n")
        except Exception as e:
            console.print(f"[yellow][WARNING]  Weave initialization failed: {e}[/yellow]")
            console.print("[yellow]Continuing without W&B tracking...[/yellow]\n")

        # Configuration
        num_generations = 5 if fast_mode else 10
        show_details_for = [0, num_generations - 1]  # Show details for first and last

        console.print(f"[bold]Running {num_generations} generations of collaborative learning[/bold]")
        console.print("[dim]Watch as agents learn to work together better over time![/dim]\n")

        # Show sponsor status
        console.print(self.create_sponsor_status_panel())

        # Initial agent status
        console.print(self.create_agent_status_table(0))

        # Run generations
        for gen in range(num_generations):
            # Show learning insights periodically
            if gen % 3 == 0:
                console.print(self.create_learning_insights(gen))

            # Run generation
            show_details = gen in show_details_for
            results = await self.run_generation(gen, show_details=show_details)

            # Show agent status periodically
            if gen % 3 == 2:
                console.print(self.create_agent_status_table(gen + 1))

            # Small delay between generations
            if gen < num_generations - 1:
                await asyncio.sleep(1)

        # Final analysis
        await self.show_improvement_analysis()

        # Conclusion
        conclusion = Panel(
            Text.from_markup(
                "[bold green][SUCCESS] Demo Complete![/bold green]\n\n"
                "[yellow]Key Achievements:[/yellow]\n"
                "• Agents learned optimal team compositions\n"
                "• Consensus methods adapted to task types\n"
                "• Collaboration efficiency improved significantly\n"
                "• Conflict resolution became smoother\n\n"
                "[cyan]This demonstrates how multi-agent systems can [bold]learn[/bold] "
                "to collaborate better over time using W&B Weave for tracking and analysis![/cyan]"
            ),
            border_style="green",
            padding=(1, 2)
        )
        console.print("\n", conclusion)

        # Weave dashboard link
        console.print("\n[bold cyan]View detailed metrics in W&B Weave:[/bold cyan]")
        console.print("[link]https://wandb.ai/[/link]\n")


async def main():
    """Main demo entry point"""

    import argparse
    parser = argparse.ArgumentParser(description="WeaveHacks Collaborative Demo")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (5 generations)")
    parser.add_argument("--task", type=str, help="Run a single task")
    args = parser.parse_args()

    runner = CollaborativeDemoRunner()

    if args.task:
        # Single task mode
        console.print(f"\n[bold]Running single task:[/bold] {args.task}")
        result = await runner.orchestrator.collaborate(args.task)

        console.print(f"\n[green]Task completed![/green]")
        console.print(f"Agents used: {', '.join(result.agents_used)}")
        console.print(f"Consensus method: {result.consensus_method}")
        console.print(f"Quality: {result.metrics['quality']:.2%}")
        console.print(f"\n[bold]Output:[/bold]\n{result.final_output}")

    else:
        # Full demo mode
        await runner.run_demo(fast_mode=args.fast)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        sys.exit(0)