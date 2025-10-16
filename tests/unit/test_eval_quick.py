"""Quick test of evaluation system with 5 tasks"""

import asyncio
from run_comprehensive_eval import run_single_evaluation, generate_statistics, print_results_dashboard
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator, Strategy
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from datetime import datetime
import weave
import json

console = Console()

QUICK_TASKS = [
    {
        "id": 1,
        "category": "coding_easy",
        "description": "Write a Python function to check if a number is prime",
        "complexity": 0.2,
        "expected_agents": ["coder"],
        "sensitive_data": False,
    },
    {
        "id": 2,
        "category": "coding_medium",
        "description": "Implement a binary search algorithm in Python",
        "complexity": 0.5,
        "expected_agents": ["coder", "reviewer"],
        "sensitive_data": False,
    },
    {
        "id": 3,
        "category": "architecture",
        "description": "Design a scalable URL shortener system architecture",
        "complexity": 0.8,
        "expected_agents": ["architect", "reviewer"],
        "sensitive_data": False,
    },
    {
        "id": 4,
        "category": "debugging",
        "description": "Debug: def add(a, b): return a + b + 1  # Should just add",
        "complexity": 0.6,
        "expected_agents": ["coder", "reviewer"],
        "sensitive_data": False,
    },
    {
        "id": 5,
        "category": "documentation",
        "description": "Write API documentation for a REST service",
        "complexity": 0.4,
        "expected_agents": ["documenter", "researcher"],
        "sensitive_data": False,
    },
]

async def main():
    console.print("\n[bold cyan] Quick Evaluation Test (5 tasks)[/bold cyan]\n")

    # Init weave
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    weave.init(f"facilitair/weavehacks-quick-test-{run_id}")

    # Init orchestrator
    console.print(" Initializing orchestrator...")
    orchestrator = CollaborativeOrchestrator(
        use_sponsors=True,
        user_strategy=Strategy.BALANCED
    )

    # Run tasks
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task_progress = progress.add_task("[cyan]Running tests...", total=len(QUICK_TASKS))

        for task in QUICK_TASKS:
            result = await run_single_evaluation(orchestrator, task, progress, task_progress)
            results.append(result)
            console.print(f"[OK] Task {task['id']}: {'[OK] Success' if result['success'] else '[FAIL] Failed'}")

    # Save results
    with open(f'quick_test_results_{run_id}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate stats
    stats = generate_statistics(results)

    # Print dashboard
    print_results_dashboard(results, stats, run_id)

    console.print(f"\n[bold green][OK] Test complete! View traces at:[/bold green]")
    console.print(f"https://wandb.ai/facilitair/weavehacks-quick-test-{run_id}/weave\n")

if __name__ == "__main__":
    asyncio.run(main())
