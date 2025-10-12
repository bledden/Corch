"""
Comprehensive Evaluation Suite for WeaveHacks Demo
Runs 100 diverse tasks across complexity levels and categories
"""

import asyncio
import weave
from datetime import datetime
from typing import List, Dict, Any
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
import numpy as np

from collaborative_orchestrator import CollaborativeOrchestrator, Strategy

console = Console()

# ============================================================================
# EVALUATION TASK DATASET - 100 Diverse Tasks
# ============================================================================

EVALUATION_TASKS = {
    # CODING TASKS (25 tasks) - Complexity: Low to High
    "coding_easy": [
        "Write a Python function to check if a number is prime",
        "Create a function that reverses a string",
        "Write a function to find the largest number in a list",
        "Implement a function to check if a string is a palindrome",
        "Create a function to calculate factorial",
    ],
    "coding_medium": [
        "Implement a binary search algorithm in Python",
        "Create a function to merge two sorted lists",
        "Write a class for a basic stack data structure",
        "Implement a function to find the longest common substring",
        "Create a decorator that caches function results",
        "Write a function to validate email addresses with regex",
        "Implement a breadth-first search for a graph",
        "Create a function to detect cycles in a linked list",
        "Write a priority queue implementation",
        "Implement a trie data structure",
    ],
    "coding_hard": [
        "Implement a red-black tree with insertion and deletion",
        "Create a lock-free concurrent queue",
        "Write an async web crawler with rate limiting",
        "Implement a diff algorithm (similar to git diff)",
        "Create a compiler lexer and parser for a simple language",
        "Write a distributed consensus algorithm (Raft basics)",
        "Implement a B-tree database index",
        "Create a neural network from scratch with backpropagation",
        "Write a garbage collector algorithm",
        "Implement a JIT compiler for a simple bytecode",
    ],

    # DEBUGGING TASKS (15 tasks)
    "debugging": [
        "Debug: def add(a, b): return a + b + 1  # Should just add",
        "Debug: for i in range(10): print(i) print('done')  # Indentation error",
        "Debug: list = [1,2,3]; list.append(list[10])  # Index error",
        "Debug: def divide(a, b): return a / b  # Handle division by zero",
        "Debug: import josn; data = josn.loads('{\"a\": 1}')  # Typo in import",
        "Debug: x = '5'; y = 3; print(x + y)  # Type error",
        "Debug: while True: pass  # Add break condition",
        "Debug: def recursive(n): return recursive(n-1)  # Missing base case",
        "Debug: dict = {'a': 1}; print(dict['b'])  # KeyError",
        "Debug: list1 = [1,2,3]; list2 = list1; list2.append(4)  # Shallow copy issue",
        "Debug: async def f(): return 1; print(f())  # Needs await",
        "Debug: class A: pass; a = A(); print(a.x)  # AttributeError",
        "Debug: file = open('test.txt'); content = file.read()  # Resource leak",
        "Debug: import threading; x = 0; [threading.Thread(target=lambda: x+1).start() for _ in range(100)]  # Race condition",
        "Debug: def memoize(f): cache = {}; return lambda *args: cache.setdefault(args, f(*args))  # Unhashable dict values",
    ],

    # ARCHITECTURE TASKS (15 tasks)
    "architecture": [
        "Design a scalable URL shortener system architecture",
        "Design a real-time chat application with presence",
        "Design a distributed caching system (Redis-like)",
        "Design a recommendation engine architecture",
        "Design a video streaming platform (Netflix-like)",
        "Design a payment processing system",
        "Design an e-commerce order fulfillment system",
        "Design a social media feed ranking system",
        "Design a search engine indexing pipeline",
        "Design a ride-sharing matching system (Uber-like)",
        "Design a distributed logging and monitoring system",
        "Design a content delivery network (CDN)",
        "Design a fraud detection system for transactions",
        "Design a multi-tenant SaaS database architecture",
        "Design a real-time analytics dashboard system",
    ],

    # DATA PROCESSING TASKS (15 tasks)
    "data_processing": [
        "Parse and analyze 1000 CSV records for trends",
        "Extract structured data from unstructured text",
        "Clean and normalize a messy dataset",
        "Aggregate and summarize time-series data",
        "Join and merge multiple data sources",
        "Detect anomalies in sensor data",
        "Build a ETL pipeline for data warehousing",
        "Implement data deduplication algorithm",
        "Create a data validation and quality framework",
        "Design a feature engineering pipeline for ML",
        "Implement stream processing for real-time data",
        "Create a data versioning system",
        "Build a data catalog and metadata manager",
        "Implement incremental data processing",
        "Design a data lake architecture",
    ],

    # OPTIMIZATION TASKS (10 tasks)
    "optimization": [
        "Optimize a slow database query with proper indexes",
        "Reduce memory usage in a data processing pipeline",
        "Optimize an API endpoint from 2s to <100ms",
        "Improve algorithm time complexity from O(n²) to O(n log n)",
        "Optimize Docker image size from 1GB to <100MB",
        "Reduce frontend bundle size and improve load time",
        "Optimize batch processing throughput",
        "Improve cache hit rate from 60% to 95%",
        "Optimize ML model inference latency",
        "Reduce cloud infrastructure costs by 50%",
    ],

    # TESTING TASKS (10 tasks)
    "testing": [
        "Write comprehensive unit tests for a REST API",
        "Create integration tests for a microservice",
        "Write property-based tests for a sorting algorithm",
        "Create load tests for a web service",
        "Write security tests for authentication",
        "Create chaos engineering tests",
        "Write mutation tests to validate test quality",
        "Create smoke tests for deployment verification",
        "Write performance regression tests",
        "Create end-to-end user journey tests",
    ],

    # DOCUMENTATION TASKS (10 tasks)
    "documentation": [
        "Write API documentation for a REST service",
        "Create user onboarding guide with examples",
        "Write technical design document for a feature",
        "Create runbook for production incidents",
        "Write contributing guide for open source project",
        "Create architecture decision record (ADR)",
        "Write migration guide for major version upgrade",
        "Create troubleshooting guide with common issues",
        "Write performance tuning guide",
        "Create security best practices document",
    ],
}

def generate_all_tasks() -> List[Dict[str, Any]]:
    """Generate complete list of 100 evaluation tasks with metadata"""

    tasks = []
    task_id = 1

    # Complexity mapping
    complexity_map = {
        "coding_easy": 0.2,
        "coding_medium": 0.5,
        "coding_hard": 0.9,
        "debugging": 0.6,
        "architecture": 0.8,
        "data_processing": 0.7,
        "optimization": 0.8,
        "testing": 0.5,
        "documentation": 0.4,
    }

    for category, task_list in EVALUATION_TASKS.items():
        for task_description in task_list:
            tasks.append({
                "id": task_id,
                "category": category,
                "description": task_description,
                "complexity": complexity_map[category],
                "expected_agents": _suggest_agents(category),
                "sensitive_data": False,
            })
            task_id += 1

    return tasks

def _suggest_agents(category: str) -> List[str]:
    """Suggest optimal agents for each category"""
    agent_map = {
        "coding_easy": ["coder"],
        "coding_medium": ["coder", "reviewer"],
        "coding_hard": ["architect", "coder", "reviewer"],
        "debugging": ["coder", "reviewer"],
        "architecture": ["architect", "researcher"],
        "data_processing": ["researcher", "coder"],
        "optimization": ["architect", "coder", "reviewer"],
        "testing": ["reviewer", "coder"],
        "documentation": ["documenter", "researcher"],
    }
    return agent_map.get(category, ["coder", "reviewer"])

# ============================================================================
# EVALUATION RUNNER
# ============================================================================

@weave.op()
async def run_single_evaluation(
    orchestrator: CollaborativeOrchestrator,
    task: Dict[str, Any],
    progress: Progress,
    task_progress_id
) -> Dict[str, Any]:
    """Run a single evaluation task and track results"""

    start_time = datetime.now()

    try:
        # Execute with collaborative orchestrator
        result = await orchestrator.collaborate(task["description"])

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate evaluation metrics
        eval_result = {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "complexity": task["complexity"],
            "agents_used": result.agents_used,
            "consensus_method": result.consensus_method,
            "duration_seconds": duration,
            "quality_score": result.metrics["quality"],
            "efficiency_score": result.metrics["efficiency"],
            "harmony_score": result.metrics["harmony"],
            "overall_score": result.metrics["overall"],
            "conflicts_resolved": result.conflicts_resolved,
            "consensus_rounds": result.consensus_rounds,
            "success": result.metrics["overall"] > 0.6,
            "timestamp": start_time.isoformat(),
        }

        progress.update(task_progress_id, advance=1)

        return eval_result

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        progress.update(task_progress_id, advance=1)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "complexity": task["complexity"],
            "agents_used": [],
            "consensus_method": "none",
            "duration_seconds": duration,
            "quality_score": 0.0,
            "efficiency_score": 0.0,
            "harmony_score": 0.0,
            "overall_score": 0.0,
            "conflicts_resolved": 0,
            "consensus_rounds": 0,
            "success": False,
            "error": str(e),
            "timestamp": start_time.isoformat(),
        }

@weave.op()
async def run_comprehensive_evaluation(batch_size: int = 5):
    """Run comprehensive evaluation with 100 tasks"""

    console.print("\n" + "="*70)
    console.print("[bold cyan]🚀 WeaveHacks Comprehensive Evaluation Suite[/bold cyan]")
    console.print("="*70 + "\n")

    # Initialize Weave tracking
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    weave.init(f"facilitair/weavehacks-eval-{run_id}")

    # Generate all tasks
    all_tasks = generate_all_tasks()
    console.print(f"📋 Generated {len(all_tasks)} evaluation tasks")

    # Show category breakdown
    category_counts = {}
    for task in all_tasks:
        category_counts[task["category"]] = category_counts.get(task["category"], 0) + 1

    table = Table(title="Task Distribution")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Complexity", style="yellow")

    for category, count in sorted(category_counts.items()):
        complexity = all_tasks[0]["complexity"] if all_tasks else 0
        for task in all_tasks:
            if task["category"] == category:
                complexity = task["complexity"]
                break
        table.add_row(category, str(count), f"{complexity:.1f}")

    console.print(table)
    console.print()

    # Initialize orchestrator
    console.print("🔧 Initializing collaborative orchestrator...")
    orchestrator = CollaborativeOrchestrator(
        use_sponsors=True,
        user_strategy=Strategy.BALANCED
    )

    # Run evaluations with progress tracking
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task_progress = progress.add_task(
            "[cyan]Running evaluations...",
            total=len(all_tasks)
        )

        # Run in batches to avoid overwhelming the system
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]

            # Run batch concurrently
            batch_results = await asyncio.gather(*[
                run_single_evaluation(orchestrator, task, progress, task_progress)
                for task in batch
            ])

            results.extend(batch_results)

            # Small delay between batches
            await asyncio.sleep(1)

    # Save results
    results_file = f"evaluation_results_{run_id}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n✅ Saved detailed results to: [cyan]{results_file}[/cyan]")

    return results, run_id

# ============================================================================
# STATISTICS AND VISUALIZATION
# ============================================================================

def generate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive statistics from evaluation results"""

    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r["success"])
    failed_tasks = total_tasks - successful_tasks

    # Overall metrics
    overall_scores = [r["overall_score"] for r in results]
    quality_scores = [r["quality_score"] for r in results]
    efficiency_scores = [r["efficiency_score"] for r in results]
    harmony_scores = [r["harmony_score"] for r in results]
    durations = [r["duration_seconds"] for r in results]

    # Category breakdown
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0,
                "success": 0,
                "avg_quality": [],
                "avg_duration": [],
            }
        category_stats[cat]["total"] += 1
        if result["success"]:
            category_stats[cat]["success"] += 1
        category_stats[cat]["avg_quality"].append(result["quality_score"])
        category_stats[cat]["avg_duration"].append(result["duration_seconds"])

    # Calculate category averages
    for cat in category_stats:
        category_stats[cat]["success_rate"] = (
            category_stats[cat]["success"] / category_stats[cat]["total"] * 100
        )
        category_stats[cat]["avg_quality_score"] = np.mean(category_stats[cat]["avg_quality"])
        category_stats[cat]["avg_duration_seconds"] = np.mean(category_stats[cat]["avg_duration"])

    # Agent usage stats
    agent_usage = {}
    consensus_usage = {}

    for result in results:
        for agent in result["agents_used"]:
            agent_usage[agent] = agent_usage.get(agent, 0) + 1

        method = result["consensus_method"]
        consensus_usage[method] = consensus_usage.get(method, 0) + 1

    return {
        "summary": {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "avg_overall_score": np.mean(overall_scores),
            "avg_quality_score": np.mean(quality_scores),
            "avg_efficiency_score": np.mean(efficiency_scores),
            "avg_harmony_score": np.mean(harmony_scores),
            "avg_duration_seconds": np.mean(durations),
            "total_duration_seconds": sum(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
        },
        "category_breakdown": category_stats,
        "agent_usage": agent_usage,
        "consensus_usage": consensus_usage,
    }

def print_results_dashboard(results: List[Dict[str, Any]], stats: Dict[str, Any], run_id: str):
    """Print beautiful results dashboard"""

    console.print("\n" + "="*70)
    console.print("[bold green]📊 EVALUATION RESULTS DASHBOARD[/bold green]")
    console.print("="*70 + "\n")

    # Summary panel
    summary = stats["summary"]
    summary_text = f"""
[bold cyan]Overall Performance[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Tasks:         {summary['total_tasks']}
Successful:          {summary['successful_tasks']} ([green]{summary['success_rate']:.1f}%[/green])
Failed:              {summary['failed_tasks']}

[bold cyan]Quality Metrics[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Overall Score:       {summary['avg_overall_score']:.2f} / 1.00
Quality Score:       {summary['avg_quality_score']:.2f} / 1.00
Efficiency Score:    {summary['avg_efficiency_score']:.2f} / 1.00
Harmony Score:       {summary['avg_harmony_score']:.2f} / 1.00

[bold cyan]Performance[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Avg Duration:        {summary['avg_duration_seconds']:.2f}s
Total Time:          {summary['total_duration_seconds'] / 60:.1f} minutes
Fastest Task:        {summary['min_duration']:.2f}s
Slowest Task:        {summary['max_duration']:.2f}s
"""
    console.print(Panel(summary_text, title="Summary", border_style="green"))

    # Category breakdown table
    console.print("\n[bold cyan]📂 Performance by Category[/bold cyan]\n")
    cat_table = Table()
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Tasks", justify="right")
    cat_table.add_column("Success Rate", justify="right")
    cat_table.add_column("Avg Quality", justify="right")
    cat_table.add_column("Avg Duration", justify="right")

    for cat, data in sorted(stats["category_breakdown"].items()):
        success_color = "green" if data["success_rate"] > 80 else "yellow" if data["success_rate"] > 60 else "red"
        cat_table.add_row(
            cat,
            str(data["total"]),
            f"[{success_color}]{data['success_rate']:.1f}%[/{success_color}]",
            f"{data['avg_quality_score']:.2f}",
            f"{data['avg_duration_seconds']:.2f}s",
        )

    console.print(cat_table)

    # Agent usage table
    console.print("\n[bold cyan]🤖 Agent Usage Statistics[/bold cyan]\n")
    agent_table = Table()
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Times Used", justify="right", style="magenta")
    agent_table.add_column("Percentage", justify="right")

    total_agent_uses = sum(stats["agent_usage"].values())
    for agent, count in sorted(stats["agent_usage"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_agent_uses * 100) if total_agent_uses > 0 else 0
        agent_table.add_row(agent, str(count), f"{percentage:.1f}%")

    console.print(agent_table)

    # Consensus methods table
    console.print("\n[bold cyan]🤝 Consensus Method Usage[/bold cyan]\n")
    consensus_table = Table()
    consensus_table.add_column("Method", style="cyan")
    consensus_table.add_column("Times Used", justify="right", style="magenta")
    consensus_table.add_column("Percentage", justify="right")

    total_consensus_uses = sum(stats["consensus_usage"].values())
    for method, count in sorted(stats["consensus_usage"].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_consensus_uses * 100) if total_consensus_uses > 0 else 0
        consensus_table.add_row(method, str(count), f"{percentage:.1f}%")

    console.print(consensus_table)

    # W&B Weave link
    console.print("\n" + "="*70)
    console.print(f"[bold cyan]📈 View detailed traces in W&B Weave:[/bold cyan]")
    console.print(f"[link]https://wandb.ai/facilitair/weavehacks-eval-{run_id}/weave[/link]")
    console.print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""

    try:
        # Run comprehensive evaluation
        results, run_id = await run_comprehensive_evaluation(batch_size=5)

        # Generate statistics
        console.print("\n📊 Generating statistics...")
        stats = generate_statistics(results)

        # Save statistics
        stats_file = f"evaluation_stats_{run_id}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        console.print(f"✅ Saved statistics to: [cyan]{stats_file}[/cyan]")

        # Print results dashboard
        print_results_dashboard(results, stats, run_id)

        console.print("\n[bold green]✨ Evaluation complete! Results are ready for presentation.[/bold green]\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Evaluation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n\n[red]❌ Error during evaluation: {e}[/red]")
        raise

if __name__ == "__main__":
    asyncio.run(main())
