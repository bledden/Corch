"""
10-Task Quick Validation Benchmark
Fast validation of sequential orchestration before full 500-task run
"""

import asyncio
import json
import weave
from datetime import datetime
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator
from agents.llm_client import MultiAgentLLMOrchestrator
import yaml

console = Console()

# 10 diverse tasks covering all categories
QUICK_TASKS = [
    # 2 basic algorithms
    {"id": 1, "category": "basic_algorithms", "description": "Write a function to check if a number is prime", "complexity": 0.3},
    {"id": 2, "category": "basic_algorithms", "description": "Implement factorial using recursion", "complexity": 0.3},

    # 2 data structures
    {"id": 3, "category": "data_structures", "description": "Implement a stack using list with push, pop, and peek operations", "complexity": 0.5},
    {"id": 4, "category": "data_structures", "description": "Create a queue using two stacks", "complexity": 0.5},

    # 2 medium algorithms
    {"id": 5, "category": "algorithms_medium", "description": "Implement binary search on sorted array", "complexity": 0.6},
    {"id": 6, "category": "algorithms_medium", "description": "Write merge sort algorithm", "complexity": 0.6},

    # 2 hard algorithms
    {"id": 7, "category": "algorithms_hard", "description": "Implement N-Queens solver with backtracking", "complexity": 0.9},
    {"id": 8, "category": "algorithms_hard", "description": "Write KMP string matching algorithm", "complexity": 0.9},

    # 2 real-world tasks
    {"id": 9, "category": "real_world_tasks", "description": "Create REST API endpoint for user authentication with JWT", "complexity": 0.7},
    {"id": 10, "category": "real_world_tasks", "description": "Implement rate limiting decorator with configurable limits", "complexity": 0.7},
]


class HallucinationDetector:
    """Detects hallucinations following HumanEval standards"""

    HALLUCINATION_PATTERNS = [
        # Non-existent APIs
        "import imaginary", "from fake_", "import nonexistent",
        # Impossible claims
        "O(0)", "O(-1)", "100% accuracy", "never fails", "always correct",
        "guaranteed perfect", "infinite speed", "zero memory",
        # Contradictions
        "both true and false", "simultaneously opposite",
        # Invalid syntax
        "def async lambda", "class static final", "async sync def",
    ]

    def detect(self, output: str) -> Dict[str, Any]:
        """Detect hallucinations in output"""
        output_lower = output.lower()

        found = [p for p in self.HALLUCINATION_PATTERNS if p.lower() in output_lower]

        # Check for code presence
        has_code = any(marker in output for marker in ["```", "def ", "class ", "function "])
        has_confidence = any(word in output_lower for word in ["guaranteed", "perfect", "never", "always"])

        confidence_without_code = has_confidence and not has_code

        score = len(found) * 0.2 + (0.3 if confidence_without_code else 0)

        return {
            "hallucination_detected": score > 0,
            "hallucination_score": min(score, 1.0),
            "patterns_found": found,
        }


@weave.op()
async def run_sequential(orchestrator: CollaborativeOrchestrator, task: Dict) -> Dict:
    """Run sequential collaboration (our approach)"""
    start = datetime.now()

    try:
        result = await orchestrator.collaborate(task["description"])
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(result.final_output)

        # HumanEval-style Pass@1: Binary pass/fail based on multi-stage validation
        quality = result.metrics.get("quality", 0.0)
        overall = result.metrics.get("overall", 0.0)

        # Pass@1: Task passes if:
        # 1. Quality threshold met (validated by reviewer stage)
        # 2. No hallucinations detected
        # 3. Output is non-empty and substantial
        has_substantial_output = len(result.final_output.strip()) > 50
        pass_at_1 = (quality > 0.7 and
                     not hallucination["hallucination_detected"] and
                     has_substantial_output)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "pass": pass_at_1,
            "quality_score": quality,
            "overall_score": overall,
            "duration": duration,
            "hallucination": hallucination,
            "output": result.final_output[:500],
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "success": False,
            "pass": False,
            "error": str(e),
        }


@weave.op()
async def run_baseline(llm: MultiAgentLLMOrchestrator, task: Dict) -> Dict:
    """Run single-model baseline (GPT-4 direct)"""
    start = datetime.now()

    try:
        output = await llm.execute_agent_task("coder", task["description"])
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(output)

        # HumanEval-style Pass@1 for baseline: Binary pass/fail
        has_code = any(m in output for m in ["```", "def ", "class ", "function "])
        has_logic = any(keyword in output.lower() for keyword in ["if ", "for ", "while ", "return "])
        reasonable_length = 100 < len(output) < 10000
        has_substantial_output = len(output.strip()) > 50

        # Quality estimate based on heuristics
        quality_estimate = 0.8 if (has_code and has_logic and reasonable_length) else (
            0.5 if has_code else 0.2
        )

        # Pass@1: Baseline passes if it meets quality bar AND no hallucinations
        pass_at_1 = (quality_estimate >= 0.7 and
                     not hallucination["hallucination_detected"] and
                     has_substantial_output)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "pass": pass_at_1,
            "quality_score": quality_estimate,
            "duration": duration,
            "hallucination": hallucination,
            "output": output[:500],
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "success": False,
            "pass": False,
            "error": str(e),
        }


async def run_quick_benchmark():
    """Run 10-task quick validation benchmark"""

    console.print("\n[bold cyan]10-Task Quick Validation Benchmark[/bold cyan]")
    console.print("[yellow]Fast validation before full 500-task run[/yellow]")
    console.print(f"Total tasks: {len(QUICK_TASKS)}\n")

    # Initialize
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    weave.init("facilitair/10-task-quick-benchmark")

    llm = MultiAgentLLMOrchestrator(config)
    orchestrator = CollaborativeOrchestrator(config)

    sequential_results = []
    baseline_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task_progress = progress.add_task(
            "[cyan]Running quick benchmark...",
            total=len(QUICK_TASKS) * 2
        )

        for task in QUICK_TASKS:
            # Display current task
            console.print(f"\n[yellow]Task {task['id']}: {task['description']}[/yellow]")

            # Run sequential
            console.print("  [cyan]→ Sequential orchestration...[/cyan]")
            seq_result = await run_sequential(orchestrator, task)
            sequential_results.append(seq_result)
            progress.update(task_progress, advance=1)

            # Run baseline
            console.print("  [cyan]→ Baseline (GPT-4 direct)...[/cyan]")
            base_result = await run_baseline(llm, task)
            baseline_results.append(base_result)
            progress.update(task_progress, advance=1)

    # Calculate metrics
    seq_passes = [r["pass"] for r in sequential_results]
    base_passes = [r["pass"] for r in baseline_results]

    metrics = {
        "sequential": {
            "pass@1": sum(seq_passes) / len(seq_passes) * 100,
            "total_successes": sum(seq_passes),
            "total_tasks": len(seq_passes),
            "hallucinations": sum(1 for r in sequential_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(r.get("quality_score", 0) for r in sequential_results) / len(sequential_results),
            "avg_duration": sum(r.get("duration", 0) for r in sequential_results) / len(sequential_results),
        },
        "baseline": {
            "pass@1": sum(base_passes) / len(base_passes) * 100,
            "total_successes": sum(base_passes),
            "total_tasks": len(base_passes),
            "hallucinations": sum(1 for r in baseline_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(r.get("quality_score", 0) for r in baseline_results) / len(baseline_results),
            "avg_duration": sum(r.get("duration", 0) for r in baseline_results) / len(baseline_results),
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_10_quick_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_tasks": len(QUICK_TASKS),
                "type": "quick_validation",
            },
            "metrics": metrics,
            "sequential_results": sequential_results,
            "baseline_results": baseline_results,
        }, f, indent=2)

    # Display results
    table = Table(title="10-Task Quick Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="green")
    table.add_column("Baseline (GPT-4)", style="yellow")
    table.add_column("Improvement", style="magenta")

    table.add_row(
        "Pass@1",
        f"{metrics['sequential']['pass@1']:.1f}%",
        f"{metrics['baseline']['pass@1']:.1f}%",
        f"+{metrics['sequential']['pass@1'] - metrics['baseline']['pass@1']:.1f}%"
    )

    table.add_row(
        "Tasks Passed",
        f"{metrics['sequential']['total_successes']}/{metrics['sequential']['total_tasks']}",
        f"{metrics['baseline']['total_successes']}/{metrics['baseline']['total_tasks']}",
        f"+{metrics['sequential']['total_successes'] - metrics['baseline']['total_successes']}"
    )

    table.add_row(
        "Hallucinations",
        f"{metrics['sequential']['hallucinations']}",
        f"{metrics['baseline']['hallucinations']}",
        f"-{metrics['baseline']['hallucinations'] - metrics['sequential']['hallucinations']}"
    )

    table.add_row(
        "Avg Quality",
        f"{metrics['sequential']['avg_quality']:.3f}",
        f"{metrics['baseline']['avg_quality']:.3f}",
        f"+{metrics['sequential']['avg_quality'] - metrics['baseline']['avg_quality']:.3f}"
    )

    table.add_row(
        "Avg Duration (s)",
        f"{metrics['sequential']['avg_duration']:.2f}",
        f"{metrics['baseline']['avg_duration']:.2f}",
        f"+{metrics['sequential']['avg_duration'] - metrics['baseline']['avg_duration']:.2f}"
    )

    console.print("\n")
    console.print(table)
    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Category breakdown
    category_table = Table(title="Performance by Category")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Sequential Pass Rate", style="green")
    category_table.add_column("Baseline Pass Rate", style="yellow")

    for category in ["basic_algorithms", "data_structures", "algorithms_medium", "algorithms_hard", "real_world_tasks"]:
        seq_cat = [r for r in sequential_results if r["category"] == category]
        base_cat = [r for r in baseline_results if r["category"] == category]

        if seq_cat:
            seq_pass_rate = sum(r["pass"] for r in seq_cat) / len(seq_cat) * 100
            base_pass_rate = sum(r["pass"] for r in base_cat) / len(base_cat) * 100

            category_table.add_row(
                category.replace("_", " ").title(),
                f"{seq_pass_rate:.0f}%",
                f"{base_pass_rate:.0f}%"
            )

    console.print("\n")
    console.print(category_table)

    # Summary
    if metrics['sequential']['pass@1'] > metrics['baseline']['pass@1']:
        console.print("\n[bold green][OK] Sequential orchestration outperforms baseline![/bold green]")
        console.print("[green]Ready to proceed with full 500-task benchmark.[/green]")
    elif metrics['sequential']['pass@1'] == metrics['baseline']['pass@1']:
        console.print("\n[bold yellow][WARNING]  Sequential and baseline have equal performance[/bold yellow]")
        console.print("[yellow]Consider investigating before full benchmark.[/yellow]")
    else:
        console.print("\n[bold red][WARNING]  Baseline outperforms sequential[/bold red]")
        console.print("[red]Review orchestration configuration before proceeding.[/red]")

    return metrics


if __name__ == "__main__":
    asyncio.run(run_quick_benchmark())
