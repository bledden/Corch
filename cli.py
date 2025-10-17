#!/usr/bin/env python3
"""
Facilitair CLI - Command-line interface for collaborative AI orchestration
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.json import JSON
import weave
from dotenv import load_dotenv
import logging

# Configure logging
log_dir = Path('test_results/logs')
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'facilitair_cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('facilitair_cli')

# Load environment
load_dotenv()

from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator
from utils.api_key_validator import APIKeyValidator

console = Console()


def validate_api_keys() -> dict:
    """Validate API keys and return results"""
    validator = APIKeyValidator()
    all_valid, results = validator.validate_all()

    return {
        'all_valid': all_valid,
        'results': {
            r.key_name: {
                'valid': r.is_valid,
                'message': r.error_message if not r.is_valid else 'Valid',
                'required': r.is_required
            }
            for r in results
        }
    }


class FacilitairCLI:
    """Main CLI application class"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize CLI with optional config file"""
        self.config = self._load_config(config_path)
        self.orchestrator = None
        logger.info("FacilitairCLI initialized")

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
                return config

        # Default configuration
        return {
            "use_sequential": True,
            "max_iterations": 3,
            "temperature": 0.2,
            "output_format": "text",  # text, json, markdown
            "verbose": False
        }

    async def initialize_orchestrator(self):
        """Initialize the orchestrator with telemetry"""
        if self.orchestrator is None:
            logger.info("Initializing orchestrator...")

            # Initialize W&B Weave
            weave.init("facilitair/cli")

            self.orchestrator = CollaborativeOrchestrator(
                use_sequential=self.config["use_sequential"]
            )
            logger.info("Orchestrator initialized successfully")

    async def collaborate(
        self,
        task: str,
        output_format: Optional[str] = None,
        save_to: Optional[str] = None
    ) -> dict:
        """Execute a collaborative task"""
        logger.info(f"Starting collaboration task: {task[:50]}...")

        await self.initialize_orchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task_progress = progress.add_task(
                "[cyan]Collaborating with agents...",
                total=None
            )

            try:
                result = await self.orchestrator.collaborate(
                    task=task,
                    force_agents=None
                )

                progress.update(task_progress, completed=True)
                logger.info(f"Collaboration completed successfully. Agents: {result.agents_used}")

                result_dict = {
                    "task": task,
                    "success": result.success,
                    "agents_used": result.agents_used,
                    "output": result.final_output,
                    "metrics": result.metrics,
                    "consensus_method": result.consensus_method,
                    "duration": f"{result.metrics.get('duration', 0):.2f}s"
                }

                # Save to file if requested
                if save_to:
                    output_path = Path(save_to)
                    output_path.write_text(json.dumps(result_dict, indent=2))
                    logger.info(f"Results saved to {save_to}")

                return result_dict

            except Exception as e:
                logger.error(f"Collaboration failed: {str(e)}", exc_info=True)
                progress.update(task_progress, description="[red]Failed")
                raise

    def display_result(self, result: dict, format: str = "text"):
        """Display result in specified format"""
        if format == "json":
            console.print(JSON(json.dumps(result, indent=2)))
        elif format == "markdown":
            console.print(f"# Task: {result['task']}\n")
            console.print(f"**Status**: {'Success' if result['success'] else 'Failed'}\n")
            console.print(f"**Agents**: {', '.join(result['agents_used'])}\n")
            console.print(f"**Duration**: {result['duration']}\n")
            console.print(f"\n## Output\n\n{result['output']}")
        else:  # text
            # Display header
            console.print(Panel.fit(
                f"[bold cyan]Task:[/bold cyan] {result['task']}\n"
                f"[bold green]Status:[/bold green] {'Success' if result['success'] else 'Failed'}\n"
                f"[bold yellow]Duration:[/bold yellow] {result['duration']}",
                border_style="cyan",
                title="[bold]Facilitair Result[/bold]"
            ))

            # Display agents
            agents_table = Table(title="Agents Used", show_header=True)
            agents_table.add_column("Agent", style="cyan")
            for agent in result['agents_used']:
                agents_table.add_row(agent)
            console.print(agents_table)

            # Display metrics
            if result.get('metrics'):
                metrics_table = Table(title="Metrics", show_header=True)
                metrics_table.add_column("Metric", style="yellow")
                metrics_table.add_column("Value", style="green")
                for key, value in result['metrics'].items():
                    if isinstance(value, float):
                        metrics_table.add_row(key.capitalize(), f"{value:.3f}")
                    else:
                        metrics_table.add_row(key.capitalize(), str(value))
                console.print(metrics_table)

            # Display output
            console.print(Panel(
                result['output'],
                title="[bold]Output[/bold]",
                border_style="green"
            ))


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Facilitair - Collaborative AI Orchestration CLI"""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = FacilitairCLI(config_path=config)

    if verbose:
        ctx.obj['cli'].config['verbose'] = True
        logger.setLevel(logging.DEBUG)


@cli.command()
@click.argument('task')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'markdown']), default='text', help='Output format')
@click.option('--save', '-s', help='Save result to file')
@click.option('--sequential/--consensus', default=True, help='Use sequential or consensus mode')
@click.option('--stream/--no-stream', default=False, help='Enable streaming debate interface')
@click.pass_context
def collaborate(ctx, task, format, save, sequential, stream):
    """Execute a collaborative task with AI agents"""
    logger.info(f"CLI collaborate command: task='{task[:50]}...'")

    # Validate API keys
    console.print("[yellow]Validating API keys...[/yellow]")
    validation = validate_api_keys()

    if not validation.get('all_valid'):
        console.print("[red]API key validation failed![/red]")
        for key, status in validation.get('results', {}).items():
            if not status.get('valid'):
                console.print(f"  [red]X[/red] {key}: {status.get('message')}")
        sys.exit(1)

    console.print("[green]API keys validated[/green]\n")

    cli_obj = ctx.obj['cli']
    cli_obj.config['use_sequential'] = sequential

    if stream:
        # Use streaming debate interface
        async def run_streaming():
            from src.cli.cli_streaming_debate import CLIDebateInterface
            interface = CLIDebateInterface()
            await cli_obj.initialize_orchestrator()
            await interface.stream_debate(task, cli_obj.orchestrator)

        asyncio.run(run_streaming())
    else:
        # Use standard interface
        async def run_task():
            result = await cli_obj.collaborate(task, output_format=format, save_to=save)
            cli_obj.display_result(result, format=format)

        asyncio.run(run_task())


@cli.command()
@click.option('--tasks', '-n', default=10, help='Number of tasks to evaluate')
@click.option('--compare-baseline/--no-baseline', default=True, help='Compare against single-model baseline')
@click.pass_context
def evaluate(ctx, tasks, compare_baseline):
    """Run evaluation comparing sequential vs baseline"""
    logger.info(f"CLI evaluate command: tasks={tasks}, compare_baseline={compare_baseline}")

    console.print(Panel.fit(
        "[bold cyan]Running Evaluation[/bold cyan]\n"
        f"Tasks: {tasks}\n"
        f"Compare baseline: {compare_baseline}",
        border_style="cyan"
    ))

    if compare_baseline:
        console.print("[yellow]Running sequential vs baseline evaluation...[/yellow]")
        import subprocess
        result = subprocess.run(
            ["python3", "run_sequential_vs_baseline_eval.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            console.print("[green]Evaluation completed successfully![/green]")
            console.print(result.stdout)
        else:
            console.print("[red]Evaluation failed![/red]")
            console.print(result.stderr)
            sys.exit(1)
    else:
        console.print("[yellow]Running comprehensive evaluation...[/yellow]")
        import subprocess
        result = subprocess.run(
            ["python3", "run_comprehensive_eval.py"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            console.print("[green]Evaluation completed successfully![/green]")
            console.print(result.stdout)
        else:
            console.print("[red]Evaluation failed![/red]")
            console.print(result.stderr)
            sys.exit(1)


@cli.command()
@click.option('--port', '-p', default=8000, help='Port to run server on')
@click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
@click.pass_context
def serve(ctx, port, host):
    """Start the REST API server"""
    logger.info(f"CLI serve command: host={host}, port={port}")

    console.print(Panel.fit(
        "[bold cyan]Starting Facilitair API Server[/bold cyan]\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Docs: http://{host}:{port}/docs",
        border_style="cyan"
    ))

    import subprocess
    subprocess.run([
        "uvicorn",
        "api:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ])


@cli.command()
def health():
    """Check system health and API keys"""
    logger.info("CLI health command")

    console.print("[bold cyan]Facilitair Health Check[/bold cyan]\n")

    # Check API keys
    console.print("[yellow]Checking API keys...[/yellow]")
    validation = validate_api_keys()

    health_table = Table(title="API Key Status", show_header=True)
    health_table.add_column("Key", style="cyan")
    health_table.add_column("Status", style="yellow")
    health_table.add_column("Message", style="white")

    for key, status in validation.get('results', {}).items():
        status_icon = "OK" if status.get('valid') else "FAIL"
        health_table.add_row(
            key,
            status_icon,
            status.get('message', '')
        )

    console.print(health_table)

    if validation.get('all_valid'):
        console.print("\n[bold green]All systems operational[/bold green]")
    else:
        console.print("\n[bold red]Some systems not operational[/bold red]")
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', default='facilitair_config.json', help='Output config file path')
def init(output):
    """Initialize a new configuration file"""
    logger.info(f"CLI init command: output={output}")

    config = {
        "use_sequential": True,
        "max_iterations": 3,
        "temperature": 0.2,
        "output_format": "text",
        "verbose": False,
        "agents": {
            "architect": {"enabled": True},
            "coder": {"enabled": True},
            "reviewer": {"enabled": True},
            "documenter": {"enabled": True}
        }
    }

    with open(output, 'w') as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]Configuration file created: {output}[/green]")
    console.print("\nEdit this file to customize your Facilitair settings.")


if __name__ == "__main__":
    cli()
