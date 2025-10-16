#!/usr/bin/env python3
"""
Service Setup and Connection Test Script
Ensures all services are properly connected before running the demo
"""

import os
import sys
import asyncio
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import weave

console = Console()

def check_environment():
    """Check and setup environment variables"""
    console.print(Panel.fit("[bold cyan] Checking Environment Setup[/bold cyan]", border_style="cyan"))

    # Load .env if exists
    env_loaded = load_dotenv()

    if not env_loaded:
        console.print("[yellow][WARNING]  No .env file found. Creating from template...[/yellow]")
        # Create .env file with instructions
        with open('.env', 'w') as f:
            f.write("""# W&B Configuration (REQUIRED for tracking)
WANDB_API_KEY=your_wandb_api_key_here  # Get from: https://wandb.ai/settings
WANDB_PROJECT=weavehacks-collaborative
WANDB_ENTITY=your_wandb_entity  # Your W&B username or team

# LLM API Keys (At least one required for real execution)
# For OpenAI: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic: https://console.anthropic.com/settings/keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Google: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# OpenRouter API (for open-source models) - Optional
# Get from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_key_here

# Model Selection Strategy (QUALITY_FIRST, COST_FIRST, BALANCED)
DEFAULT_STRATEGY=BALANCED

# Demo Settings
DEMO_GENERATIONS=5
DEMO_TASKS_PER_TYPE=3
""")
        console.print("[green][OK] Created .env file. Please add your API keys![/green]")
        console.print("\n[bold red]Action Required:[/bold red]")
        console.print("1. Edit .env file and add your API keys")
        console.print("2. At minimum, add WANDB_API_KEY for W&B Weave tracking")
        console.print("3. Add at least one LLM API key (OpenAI, Anthropic, or Google)")
        return False

    return True


def check_api_keys():
    """Check which API keys are available"""
    console.print("\n[bold]API Key Status:[/bold]")

    table = Table(show_header=True)
    table.add_column("Service", style="cyan", width=20)
    table.add_column("Status", width=15)
    table.add_column("Notes", width=40)

    services = {
        "W&B Weave": ("WANDB_API_KEY", "Required for tracking"),
        "OpenAI": ("OPENAI_API_KEY", "For GPT models"),
        "Anthropic": ("ANTHROPIC_API_KEY", "For Claude models"),
        "Google": ("GOOGLE_API_KEY", "For Gemini models"),
        "OpenRouter": ("OPENROUTER_API_KEY", "For open-source models"),
    }

    available_llms = []

    for service, (env_var, notes) in services.items():
        key = os.getenv(env_var)
        if key and key not in [f"your_{env_var.lower()}_here", "demo_mode_no_key_required",
                               "demo_mode_anthropic", "demo_mode_google", "demo_mode_openrouter",
                               "will_use_from_environment"]:
            status = "[green][OK] Connected[/green]"
            if service != "W&B Weave":
                available_llms.append(service)
        elif env_var == "OPENAI_API_KEY" and os.getenv(env_var):
            # Special check for OpenAI from environment
            status = "[green][OK] Connected[/green]"
            available_llms.append(service)
        elif key and "demo_mode" in key:
            status = "[yellow][WARNING] Demo mode[/yellow]"
        else:
            status = "[red][FAIL] Missing[/red]"

        table.add_row(service, status, notes)

    console.print(table)

    # Check if we have minimum requirements
    wandb_key = os.getenv("WANDB_API_KEY")
    has_wandb = wandb_key and wandb_key not in ["your_wandb_api_key_here", "demo_mode_no_key_required"]
    demo_mode = wandb_key == "demo_mode_no_key_required"

    if demo_mode:
        console.print("\n[bold yellow][WARNING]  Running in demo mode without W&B tracking[/bold yellow]")
        console.print("The demo will use simulated metrics.")
    elif not has_wandb:
        console.print("\n[bold red][WARNING]  Warning: W&B API key not configured![/bold red]")
        console.print("The demo will run but without tracking metrics.")

    if not available_llms:
        console.print("\n[bold yellow][WARNING]  No LLM API keys configured![/bold yellow]")
        console.print("The demo will run in simulation mode.")
        return False, has_wandb

    return True, has_wandb


def test_wandb_connection():
    """Test W&B Weave connection"""
    console.print("\n[bold]Testing W&B Weave Connection:[/bold]")

    try:
        # Initialize Weave
        weave.init('weavehacks-test')

        # Create a simple op to test
        @weave.op()
        def test_op(x: int) -> int:
            return x * 2

        # Run test
        result = test_op(21)

        if result == 42:
            console.print("[green][OK] W&B Weave connection successful![/green]")
            return True
        else:
            console.print("[red][FAIL] W&B Weave test failed[/red]")
            return False

    except Exception as e:
        console.print(f"[red][FAIL] W&B Weave error: {e}[/red]")
        return False


async def test_llm_connection():
    """Test LLM API connections"""
    console.print("\n[bold]Testing LLM Connections:[/bold]")

    try:
        from agents.llm_client import LLMClient

        # Create a minimal config for testing
        test_config = {
            "agents": {
                "test": {
                    "model": "gpt-4o-mini",  # Use a cheap model for testing
                    "temperature": 0.7,
                    "max_tokens": 100
                }
            }
        }

        client = LLMClient(test_config)

        # Test with a simple prompt
        response = await client.execute_llm(
            agent_id="test",
            task="Say 'Hello, WeaveHacks!' in exactly 3 words.",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100
        )

        if response.content and not response.error:
            console.print(f"[green][OK] LLM responded: {response.content[:50]}...[/green]")
            return True
        else:
            console.print(f"[yellow][WARNING]  LLM simulation mode active[/yellow]")
            return False

    except ImportError as e:
        console.print(f"[yellow][WARNING]  LLM client not found: {e}[/yellow]")
        console.print("Creating basic LLM client for testing...")

        # Test OpenAI API directly if available
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Say 'Hello, WeaveHacks!' in exactly 3 words."}],
                    max_tokens=20
                )
                content = response.choices[0].message.content
                console.print(f"[green][OK] OpenAI API works: {content}[/green]")
                return True
            except Exception as e2:
                console.print(f"[yellow][WARNING]  OpenAI test failed: {e2}[/yellow]")

        console.print("Will use simulation mode for demo")
        return False

    except Exception as e:
        console.print(f"[yellow][WARNING]  LLM test failed: {e}[/yellow]")
        console.print("Will use simulation mode for demo")
        return False


def check_dependencies():
    """Check if all Python dependencies are installed"""
    console.print("\n[bold]Checking Python Dependencies:[/bold]")

    required = [
        "weave",
        "dotenv",
        "yaml",
        "numpy",
        "openai",
        "anthropic",
        "google.generativeai",
        "rich",
        "click",
        "aiohttp"
    ]

    missing = []
    for module in required:
        try:
            if '.' in module:
                parts = module.split('.')
                __import__(parts[0])
            else:
                __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        console.print(f"[red][FAIL] Missing dependencies: {', '.join(missing)}[/red]")
        console.print("\n[bold]Run:[/bold] pip install -r requirements.txt")
        return False

    console.print("[green][OK] All dependencies installed[/green]")
    return True


async def main():
    """Main setup and test flow"""
    console.print(Panel.fit(
        "[bold cyan][START] WeaveHacks Collaborative Orchestrator Setup[/bold cyan]",
        border_style="cyan"
    ))

    # Check dependencies
    if not check_dependencies():
        console.print("\n[bold red]Please install missing dependencies first![/bold red]")
        sys.exit(1)

    # Check environment
    if not check_environment():
        console.print("\n[bold yellow]Please configure your .env file and run this script again![/bold yellow]")
        sys.exit(1)

    # Check API keys
    has_llm, has_wandb = check_api_keys()

    # Test connections
    if has_wandb:
        wandb_ok = test_wandb_connection()
    else:
        wandb_ok = False
        console.print("\n[yellow]Skipping W&B test (no API key)[/yellow]")

    if has_llm:
        llm_ok = await test_llm_connection()
    else:
        llm_ok = False
        console.print("\n[yellow]Running in simulation mode (no LLM keys)[/yellow]")

    # Summary
    console.print("\n" + "="*50)
    console.print(Panel.fit("[bold]Setup Summary[/bold]", border_style="green"))

    status_table = Table(show_header=False)
    status_table.add_column("Component", style="cyan", width=30)
    status_table.add_column("Status", width=20)

    status_table.add_row("Dependencies", "[green][OK] Ready[/green]")
    status_table.add_row("Environment", "[green][OK] Configured[/green]")
    status_table.add_row(
        "W&B Weave",
        "[green][OK] Connected[/green]" if wandb_ok else "[yellow][WARNING]  Not configured[/yellow]"
    )
    status_table.add_row(
        "LLM APIs",
        "[green][OK] Connected[/green]" if llm_ok else "[yellow][WARNING]  Simulation mode[/yellow]"
    )

    console.print(status_table)

    # Recommendations
    console.print("\n[bold]Next Steps:[/bold]")

    if wandb_ok and llm_ok:
        console.print("[green]Refiner Everything is ready! You can now run:[/green]")
        console.print("   python demo.py                    # Basic demo")
        console.print("   python demo_with_strategy.py      # Interactive strategy demo")
    elif not wandb_ok:
        console.print("[yellow]1. Add your W&B API key to .env for tracking[/yellow]")
        console.print("[yellow]2. You can still run demos in simulation mode[/yellow]")
    elif not llm_ok:
        console.print("[yellow]1. Add LLM API keys to .env for real execution[/yellow]")
        console.print("[yellow]2. Demo will run in simulation mode[/yellow]")

    console.print("\n[bold cyan]Ready to showcase at WeaveHacks 2! [SUCCESS][/bold cyan]")


if __name__ == "__main__":
    asyncio.run(main())