#!/usr/bin/env python3
"""
10-Task Real Evaluation
5 Open Source Models + 5 Closed Source Models
Covers all categories and APIs with REAL LLM calls
"""

import asyncio
import weave
import yaml
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.orchestrators.sequential_orchestrator import SequentialCollaborativeOrchestrator

console = Console()

# 10 diverse tasks covering all categories
EVAL_TASKS = [
    # CODING TASKS (Easy, Medium, Hard)
    {
        "id": 1,
        "category": "coding_easy",
        "description": "Write a Python function to check if a number is prime",
        "model_type": "open_source",
        "target_model": "qwen/qwen3-coder-plus"
    },
    {
        "id": 2,
        "category": "coding_medium",
        "description": "Implement a binary search tree with insert, search, and delete operations",
        "model_type": "closed_source",
        "target_model": "openai/gpt-4"
    },
    {
        "id": 3,
        "category": "coding_hard",
        "description": "Create a thread-safe LRU cache with O(1) get and put operations",
        "model_type": "open_source",
        "target_model": "deepseek/deepseek-v3"
    },

    # ARCHITECTURE
    {
        "id": 4,
        "category": "architecture",
        "description": "Design a microservices architecture for an e-commerce platform with event-driven communication",
        "model_type": "closed_source",
        "target_model": "anthropic/claude-sonnet-4.5"
    },

    # DATA PROCESSING
    {
        "id": 5,
        "category": "data_processing",
        "description": "Write a Python script to parse CSV files, clean data (handle nulls, duplicates), and export to JSON",
        "model_type": "open_source",
        "target_model": "meta-llama/llama-3.3-70b-instruct"
    },

    # DEBUGGING
    {
        "id": 6,
        "category": "debugging",
        "description": "Debug a recursive function causing stack overflow and refactor to use iteration",
        "model_type": "closed_source",
        "target_model": "google/gemini-2.0-flash-exp"
    },

    # TESTING
    {
        "id": 7,
        "category": "testing",
        "description": "Write comprehensive unit tests for a REST API endpoint with edge cases and mocking",
        "model_type": "open_source",
        "target_model": "mistralai/mistral-large-2411"
    },

    # DOCUMENTATION
    {
        "id": 8,
        "category": "documentation",
        "description": "Write API documentation for a user authentication service with code examples",
        "model_type": "closed_source",
        "target_model": "openai/gpt-4-turbo"
    },

    # OPTIMIZATION
    {
        "id": 9,
        "category": "optimization",
        "description": "Optimize a slow database query with proper indexing and query rewriting",
        "model_type": "open_source",
        "target_model": "qwen/qwen3-coder-plus"
    },

    # WEB DEVELOPMENT
    {
        "id": 10,
        "category": "web_development",
        "description": "Build a React component with hooks for real-time data updates using WebSockets",
        "model_type": "closed_source",
        "target_model": "anthropic/claude-sonnet-4.5"
    }
]

async def run_task(orchestrator, task, progress, task_progress):
    """Run a single task with real LLM calls"""
    progress.update(task_progress, description=f"[cyan]Task {task['id']}: {task['category']}...")

    start_time = time.time()

    try:
        result = await orchestrator.execute_workflow(
            task=task['description'],
            max_iterations=2,
            temperature=0.7
        )

        duration = time.time() - start_time

        return {
            "task_id": task['id'],
            "category": task['category'],
            "model_type": task['model_type'],
            "target_model": task['target_model'],
            "success": result.success,
            "duration_seconds": duration,
            "stages_completed": len(result.stages),
            "iterations": result.iterations,
            "output_length": len(result.final_output),
            "output_preview": result.final_output[:200] + "..." if len(result.final_output) > 200 else result.final_output
        }

    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[red]Task {task['id']} failed: {e}")
        return {
            "task_id": task['id'],
            "category": task['category'],
            "model_type": task['model_type'],
            "target_model": task['target_model'],
            "success": False,
            "duration_seconds": duration,
            "error": str(e)
        }

async def main():
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan] 10-Task Real Evaluation - Open Source vs Closed Source[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Weave
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    weave.init(f"facilitair/10-task-real-eval-{run_id}")

    # Initialize orchestrator
    console.print("[cyan] Initializing sequential orchestrator...")
    orchestrator = SequentialCollaborativeOrchestrator(config)

    console.print("[green][OK] Sequential collaboration enabled (Facilitair_v2 architecture)\n")

    # Show task breakdown
    open_source_count = sum(1 for t in EVAL_TASKS if t['model_type'] == 'open_source')
    closed_source_count = sum(1 for t in EVAL_TASKS if t['model_type'] == 'closed_source')

    table = Table(title="Task Distribution")
    table.add_column("Model Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Categories", style="green")

    open_categories = set(t['category'] for t in EVAL_TASKS if t['model_type'] == 'open_source')
    closed_categories = set(t['category'] for t in EVAL_TASKS if t['model_type'] == 'closed_source')

    table.add_row("Open Source", str(open_source_count), ", ".join(open_categories))
    table.add_row("Closed Source", str(closed_source_count), ", ".join(closed_categories))

    console.print(table)
    console.print()

    # Run tasks
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for task in EVAL_TASKS:
            task_progress = progress.add_task(description="", total=None)
            result = await run_task(orchestrator, task, progress, task_progress)
            results.append(result)
            progress.remove_task(task_progress)

            # Show immediate result
            status = "[OK]" if result.get('success') else "[FAIL]"
            console.print(f"{status} Task {task['id']} ({task['category']}): {result.get('duration_seconds', 0):.2f}s")

    console.print("\n[bold cyan]" + "="*80)
    console.print("[CHART] RESULTS")
    console.print("="*80 + "[/bold cyan]\n")

    # Analyze results
    open_source_results = [r for r in results if r['model_type'] == 'open_source']
    closed_source_results = [r for r in results if r['model_type'] == 'closed_source']

    def calc_stats(results_list):
        if not results_list:
            return {
                "success_rate": 0,
                "avg_duration": 0,
                "total_tasks": 0
            }
        return {
            "success_rate": sum(1 for r in results_list if r.get('success', False)) / len(results_list) * 100,
            "avg_duration": sum(r.get('duration_seconds', 0) for r in results_list) / len(results_list),
            "total_tasks": len(results_list),
            "successful": sum(1 for r in results_list if r.get('success', False))
        }

    open_stats = calc_stats(open_source_results)
    closed_stats = calc_stats(closed_source_results)
    overall_stats = calc_stats(results)

    # Results table
    results_table = Table(title="Performance Comparison")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Open Source", style="green")
    results_table.add_column("Closed Source", style="magenta")
    results_table.add_column("Overall", style="yellow")

    results_table.add_row(
        "Success Rate",
        f"{open_stats['successful']}/{open_stats['total_tasks']} ({open_stats['success_rate']:.1f}%)",
        f"{closed_stats['successful']}/{closed_stats['total_tasks']} ({closed_stats['success_rate']:.1f}%)",
        f"{overall_stats['successful']}/{overall_stats['total_tasks']} ({overall_stats['success_rate']:.1f}%)"
    )
    results_table.add_row(
        "Avg Duration",
        f"{open_stats['avg_duration']:.2f}s",
        f"{closed_stats['avg_duration']:.2f}s",
        f"{overall_stats['avg_duration']:.2f}s"
    )

    console.print(results_table)
    console.print()

    # Detailed results
    detail_table = Table(title="Detailed Results")
    detail_table.add_column("ID", style="cyan")
    detail_table.add_column("Category", style="green")
    detail_table.add_column("Model", style="magenta")
    detail_table.add_column("Status", style="yellow")
    detail_table.add_column("Duration", style="blue")

    for r in results:
        status = "[OK] Success" if r.get('success') else f"[FAIL] Failed"
        if 'error' in r:
            status += f" ({r['error'][:30]}...)"

        detail_table.add_row(
            str(r['task_id']),
            r['category'],
            r['target_model'].split('/')[-1][:20],
            status,
            f"{r.get('duration_seconds', 0):.2f}s"
        )

    console.print(detail_table)

    console.print(f"\n[bold green][OK] Evaluation Complete![/bold green]")
    console.print(f"Results logged to Weave: https://wandb.ai/facilitair/10-task-real-eval-{run_id}/weave\n")

    return results

if __name__ == "__main__":
    asyncio.run(main())
