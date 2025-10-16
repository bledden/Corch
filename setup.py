#!/usr/bin/env python3
"""
Setup script for WeaveHacks Collaborative Orchestrator
Initializes W&B Weave and verifies API keys
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import weave
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()

def check_environment():
    """Check and setup environment variables"""

    # Load existing .env file if it exists
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        console.print("[yellow]No .env file found. Creating from .env.example...[/yellow]")

        if env_example_path.exists():
            # Copy .env.example to .env
            import shutil
            shutil.copy(env_example_path, env_path)
            console.print("[green]Created .env file from .env.example[/green]")
            console.print("[yellow]Please edit .env and add your API keys[/yellow]")
        else:
            console.print("[red]No .env.example file found![/red]")
            sys.exit(1)

    load_dotenv()

    # Check for required API keys
    required_keys = {
        "WANDB_API_KEY": "W&B API key for Weave tracking",
        "OPENAI_API_KEY": "OpenAI API key (optional)",
        "ANTHROPIC_API_KEY": "Anthropic API key (optional)",
        "GOOGLE_API_KEY": "Google AI API key (optional)"
    }

    missing_keys = []
    optional_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]

    table = Table(title="Environment Check", show_header=True)
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")

    for key, description in required_keys.items():
        value = os.getenv(key)
        if value and value != f"your_{key.lower()}_here":
            status = "[OK] Set"
            style = "green"
        elif key in optional_keys:
            status = "[WARNING]  Optional"
            style = "yellow"
        else:
            status = "[FAIL] Missing"
            style = "red"
            missing_keys.append(key)

        table.add_row(key, f"[{style}]{status}[/{style}]", description)

    console.print(table)

    if "WANDB_API_KEY" in missing_keys:
        console.print("\n[red]WANDB_API_KEY is required for Weave tracking![/red]")
        console.print("Get your API key from: https://wandb.ai/authorize")

        if Confirm.ask("Would you like to enter your W&B API key now?"):
            api_key = Prompt.ask("Enter your W&B API key", password=True)

            # Update .env file
            with open(env_path, 'r') as f:
                lines = f.readlines()

            with open(env_path, 'w') as f:
                updated = False
                for line in lines:
                    if line.startswith("WANDB_API_KEY="):
                        f.write(f"WANDB_API_KEY={api_key}\n")
                        updated = True
                    else:
                        f.write(line)

                if not updated:
                    f.write(f"\nWANDB_API_KEY={api_key}\n")

            os.environ["WANDB_API_KEY"] = api_key
            console.print("[green]W&B API key saved to .env[/green]")

    # Check for at least one LLM provider
    llm_keys = [k for k in optional_keys if os.getenv(k) and os.getenv(k) != f"your_{k.lower()}_here"]

    if not llm_keys:
        console.print("\n[yellow]Warning: No LLM API keys found. The system will use simulated responses.[/yellow]")
        console.print("For real LLM integration, add at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")

    return len(missing_keys) == 0

def initialize_weave():
    """Initialize W&B Weave"""

    console.print("\n[cyan]Initializing W&B Weave...[/cyan]")

    project_name = os.getenv("WANDB_PROJECT", "weavehacks-collaborative")
    entity = os.getenv("WANDB_ENTITY")

    try:
        if entity:
            weave.init(f"{entity}/{project_name}")
        else:
            weave.init(project_name)

        console.print(f"[green][OK] Weave initialized with project: {project_name}[/green]")

        # Test logging
        weave.log({"setup": "initialized", "test": True})
        console.print("[green][OK] Test log successful[/green]")

        return True

    except Exception as e:
        console.print(f"[red]Failed to initialize Weave: {str(e)}[/red]")
        console.print("[yellow]Please check your W&B API key and network connection[/yellow]")
        return False

def verify_dependencies():
    """Verify that all required dependencies are installed"""

    console.print("\n[cyan]Verifying dependencies...[/cyan]")

    required_packages = [
        ("weave", "W&B Weave"),
        ("rich", "Rich terminal UI"),
        ("yaml", "YAML configuration"),
        ("dotenv", "Environment variables"),
        ("numpy", "Numerical computations")
    ]

    optional_packages = [
        ("openai", "OpenAI API"),
        ("anthropic", "Anthropic API"),
        ("google.generativeai", "Google AI API")
    ]

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, name in required_packages:
        try:
            if "." in package:
                parts = package.split(".")
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(package)
            console.print(f"[OK] {name} installed")
        except ImportError:
            missing_required.append((package, name))
            console.print(f"[FAIL] {name} missing")

    # Check optional packages
    for package, name in optional_packages:
        try:
            if "." in package:
                parts = package.split(".")
                module = __import__(parts[0])
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(package)
            console.print(f"[OK] {name} installed (optional)")
        except ImportError:
            missing_optional.append((package, name))
            console.print(f"[WARNING]  {name} not installed (optional)")

    if missing_required:
        console.print("\n[red]Missing required packages![/red]")
        console.print("Run: pip install -r requirements.txt")
        return False

    if missing_optional:
        console.print("\n[yellow]Some optional packages are missing.[/yellow]")
        console.print("For full functionality, run: pip install -r requirements.txt")

    return True

def main():
    """Main setup function"""

    console.print(Panel.fit(
        "[bold cyan]WeaveHacks Collaborative Orchestrator Setup[/bold cyan]\n"
        "Self-Improving Multi-Agent Collaboration System",
        border_style="cyan"
    ))

    # Step 1: Check environment
    console.print("\n[bold]Step 1: Environment Configuration[/bold]")
    env_ok = check_environment()

    # Step 2: Verify dependencies
    console.print("\n[bold]Step 2: Dependency Verification[/bold]")
    deps_ok = verify_dependencies()

    if not deps_ok:
        console.print("\n[red]Please install missing dependencies before continuing[/red]")
        sys.exit(1)

    # Step 3: Initialize Weave
    if env_ok and os.getenv("WANDB_API_KEY"):
        console.print("\n[bold]Step 3: W&B Weave Initialization[/bold]")
        weave_ok = initialize_weave()

        if weave_ok:
            console.print("\n[bold green][OK] Setup Complete![/bold green]")
            console.print("\nYou can now run:")
            console.print("  [cyan]python train.py[/cyan] - Train the collaborative system")
            console.print("  [cyan]python execute.py[/cyan] - Run tasks with the trained system")
            console.print("  [cyan]python demo.py[/cyan] - Run the hackathon demo")
        else:
            console.print("\n[yellow]Setup partially complete. Weave initialization failed.[/yellow]")
            console.print("The system will still work but without W&B tracking.")
    else:
        console.print("\n[yellow]Setup partially complete. Please configure your API keys.[/yellow]")
        console.print("Edit the .env file and run setup.py again.")

    console.print("\n[dim]For more information, see README.md[/dim]")

if __name__ == "__main__":
    main()