"""
10-Task Benchmark with Real Quality Evaluation
Uses CodeQualityEvaluator for objective quality metrics
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
from src.evaluation.quality_evaluator import CodeQualityEvaluator, detect_language
import yaml

console = Console()

# 10 diverse tasks
QUICK_TASKS = [
    {"id": 1, "category": "basic_algorithms", "description": "Write a function to check if a number is prime", "complexity": 0.3},
    {"id": 2, "category": "basic_algorithms", "description": "Implement factorial using recursion", "complexity": 0.3},
    {"id": 3, "category": "data_structures", "description": "Implement a stack using list with push, pop, and peek operations", "complexity": 0.5},
    {"id": 4, "category": "data_structures", "description": "Create a queue using two stacks", "complexity": 0.5},
    {"id": 5, "category": "algorithms_medium", "description": "Implement binary search on sorted array", "complexity": 0.6},
    {"id": 6, "category": "algorithms_medium", "description": "Write merge sort algorithm", "complexity": 0.6},
    {"id": 7, "category": "algorithms_hard", "description": "Implement N-Queens solver with backtracking", "complexity": 0.9},
    {"id": 8, "category": "algorithms_hard", "description": "Write KMP string matching algorithm", "complexity": 0.9},
    {"id": 9, "category": "real_world_tasks", "description": "Create REST API endpoint for user authentication with JWT", "complexity": 0.7},
    {"id": 10, "category": "real_world_tasks", "description": "Implement rate limiting decorator with configurable limits", "complexity": 0.7},
]


class HallucinationDetector:
    """Detects hallucinations in code"""

    HALLUCINATION_PATTERNS = [
        "import imaginary", "from fake_", "import nonexistent",
        "O(0)", "O(-1)", "100% accuracy", "never fails", "always correct",
        "guaranteed perfect", "infinite speed", "zero memory",
        "both true and false", "simultaneously opposite",
        "def async lambda", "class static final", "async sync def",
    ]

    def detect(self, output: str) -> Dict[str, Any]:
        """Detect hallucinations"""
        output_lower = output.lower()
        found = [p for p in self.HALLUCINATION_PATTERNS if p.lower() in output_lower]

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
async def run_sequential(orchestrator: CollaborativeOrchestrator, task: Dict, evaluator: CodeQualityEvaluator) -> Dict:
    """Run sequential collaboration with REAL quality evaluation"""
    start = datetime.now()

    try:
        result = await orchestrator.collaborate(task["description"])
        duration = (datetime.now() - start).total_seconds()

        # Hallucination detection
        detector = HallucinationDetector()
        hallucination = detector.detect(result.final_output)

        # REAL quality evaluation
        language = detect_language(result.final_output)
        quality_result = evaluator.evaluate(result.final_output, task["description"], language)

        # Pass@1: Uses REAL quality score
        has_substantial_output = len(result.final_output.strip()) > 50
        pass_at_1 = (quality_result.overall > 0.7 and
                     not hallucination["hallucination_detected"] and
                     has_substantial_output)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "pass": pass_at_1,
            "quality_score": quality_result.overall,  # REAL score!
            "quality_dimensions": quality_result.dimensions,  # Detailed breakdown
            "language": language,
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
            "quality_score": 0.0,
            "error": str(e),
        }


@weave.op()
async def run_baseline(llm: MultiAgentLLMOrchestrator, task: Dict, evaluator: CodeQualityEvaluator) -> Dict:
    """Run baseline with REAL quality evaluation"""
    start = datetime.now()

    try:
        output = await llm.execute_agent_task("coder", task["description"])
        duration = (datetime.now() - start).total_seconds()

        # Hallucination detection
        detector = HallucinationDetector()
        hallucination = detector.detect(output)

        # REAL quality evaluation
        language = detect_language(output)
        quality_result = evaluator.evaluate(output, task["description"], language)

        # Pass@1: Uses REAL quality score
        has_substantial_output = len(output.strip()) > 50
        pass_at_1 = (quality_result.overall > 0.7 and
                     not hallucination["hallucination_detected"] and
                     has_substantial_output)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "pass": pass_at_1,
            "quality_score": quality_result.overall,  # REAL score!
            "quality_dimensions": quality_result.dimensions,  # Detailed breakdown
            "language": language,
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
            "quality_score": 0.0,
            "error": str(e),
        }


async def run_benchmark():
    """Run 10-task benchmark with REAL quality evaluation"""

    console.print("\n[bold cyan]10-Task Benchmark with Real Quality Evaluation[/bold cyan]")
    console.print("[yellow]Objective metrics: syntax, completeness, quality, docs, error handling, tests[/yellow]")
    console.print(f"Total tasks: {len(QUICK_TASKS)}\n")

    # Initialize
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    weave.init("facilitair/10-task-benchmark-v2")

    llm = MultiAgentLLMOrchestrator(config)
    orchestrator = CollaborativeOrchestrator(config)
    evaluator = CodeQualityEvaluator(pass_threshold=0.7)

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
            "[cyan]Running benchmark...",
            total=len(QUICK_TASKS) * 2
        )

        for task in QUICK_TASKS:
            console.print(f"\n[yellow]Task {task['id']}: {task['description']}[/yellow]")

            # Sequential
            console.print("  [cyan]-> Sequential orchestration...[/cyan]")
            seq_result = await run_sequential(orchestrator, task, evaluator)
            sequential_results.append(seq_result)
            console.print(f"     Quality: {seq_result.get('quality_score', 0):.2f}")
            progress.update(task_progress, advance=1)

            # Baseline
            console.print("  [cyan]-> Baseline (GPT-4 direct)...[/cyan]")
            base_result = await run_baseline(llm, task, evaluator)
            baseline_results.append(base_result)
            console.print(f"     Quality: {base_result.get('quality_score', 0):.2f}")
            progress.update(task_progress, advance=1)

    # Calculate metrics
    seq_passes = [r["pass"] for r in sequential_results if "pass" in r]
    base_passes = [r["pass"] for r in baseline_results if "pass" in r]

    seq_qualities = [r["quality_score"] for r in sequential_results if "quality_score" in r]
    base_qualities = [r["quality_score"] for r in baseline_results if "quality_score" in r]

    metrics = {
        "sequential": {
            "pass@1": sum(seq_passes) / len(seq_passes) * 100 if seq_passes else 0,
            "total_successes": sum(seq_passes),
            "total_tasks": len(seq_passes),
            "hallucinations": sum(1 for r in sequential_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(seq_qualities) / len(seq_qualities) if seq_qualities else 0,
            "min_quality": min(seq_qualities) if seq_qualities else 0,
            "max_quality": max(seq_qualities) if seq_qualities else 0,
            "avg_duration": sum(r.get("duration", 0) for r in sequential_results) / len(sequential_results),
        },
        "baseline": {
            "pass@1": sum(base_passes) / len(base_passes) * 100 if base_passes else 0,
            "total_successes": sum(base_passes),
            "total_tasks": len(base_passes),
            "hallucinations": sum(1 for r in baseline_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(base_qualities) / len(base_qualities) if base_qualities else 0,
            "min_quality": min(base_qualities) if base_qualities else 0,
            "max_quality": max(base_qualities) if base_qualities else 0,
            "avg_duration": sum(r.get("duration", 0) for r in baseline_results) / len(baseline_results),
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_10_v2_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_tasks": len(QUICK_TASKS),
                "type": "real_quality_evaluation",
                "evaluator": "CodeQualityEvaluator",
            },
            "metrics": metrics,
            "sequential_results": sequential_results,
            "baseline_results": baseline_results,
        }, f, indent=2)

    # Display results
    table = Table(title="10-Task Benchmark Results (Real Quality Scores)")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="green")
    table.add_column("Baseline (GPT-4)", style="yellow")
    table.add_column("Difference", style="magenta")

    table.add_row(
        "Pass@1",
        f"{metrics['sequential']['pass@1']:.1f}%",
        f"{metrics['baseline']['pass@1']:.1f}%",
        f"{metrics['sequential']['pass@1'] - metrics['baseline']['pass@1']:+.1f}%"
    )

    table.add_row(
        "Avg Quality (Real)",
        f"{metrics['sequential']['avg_quality']:.3f}",
        f"{metrics['baseline']['avg_quality']:.3f}",
        f"{metrics['sequential']['avg_quality'] - metrics['baseline']['avg_quality']:+.3f}"
    )

    table.add_row(
        "Min Quality",
        f"{metrics['sequential']['min_quality']:.3f}",
        f"{metrics['baseline']['min_quality']:.3f}",
        f"{metrics['sequential']['min_quality'] - metrics['baseline']['min_quality']:+.3f}"
    )

    table.add_row(
        "Max Quality",
        f"{metrics['sequential']['max_quality']:.3f}",
        f"{metrics['baseline']['max_quality']:.3f}",
        f"{metrics['sequential']['max_quality'] - metrics['baseline']['max_quality']:+.3f}"
    )

    table.add_row(
        "Hallucinations",
        f"{metrics['sequential']['hallucinations']}",
        f"{metrics['baseline']['hallucinations']}",
        f"{metrics['baseline']['hallucinations'] - metrics['sequential']['hallucinations']:+d}"
    )

    table.add_row(
        "Avg Duration (s)",
        f"{metrics['sequential']['avg_duration']:.1f}",
        f"{metrics['baseline']['avg_duration']:.1f}",
        f"{metrics['sequential']['avg_duration'] - metrics['baseline']['avg_duration']:+.1f}"
    )

    console.print("\n")
    console.print(table)

    # Category breakdown
    category_table = Table(title="Performance by Category (Real Quality)")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Sequential Avg Quality", style="green")
    category_table.add_column("Baseline Avg Quality", style="yellow")

    for category in ["basic_algorithms", "data_structures", "algorithms_medium", "algorithms_hard", "real_world_tasks"]:
        seq_cat = [r for r in sequential_results if r.get("category") == category]
        base_cat = [r for r in baseline_results if r.get("category") == category]

        if seq_cat:
            seq_avg = sum(r.get("quality_score", 0) for r in seq_cat) / len(seq_cat)
            base_avg = sum(r.get("quality_score", 0) for r in base_cat) / len(base_cat)

            category_table.add_row(
                category.replace("_", " ").title(),
                f"{seq_avg:.3f}",
                f"{base_avg:.3f}"
            )

    console.print("\n")
    console.print(category_table)
    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Interpretation
    if metrics['sequential']['avg_quality'] > metrics['baseline']['avg_quality']:
        console.print("\n[bold green]Sequential orchestration produces higher quality code![/bold green]")
    elif metrics['sequential']['avg_quality'] < metrics['baseline']['avg_quality']:
        console.print("\n[bold yellow]Baseline produces higher quality code[/bold yellow]")
    else:
        console.print("\n[bold white]Quality is equivalent between approaches[/bold white]")

    return metrics


if __name__ == "__main__":
    asyncio.run(run_benchmark())
