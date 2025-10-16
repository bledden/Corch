"""
Sequential Collaboration vs Single-Model Baseline Evaluation
Compares the new sequential collaboration against single model requests.
Includes hallucination detection and quality metrics.
"""

import asyncio
import json
import weave
from datetime import datetime
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator, Strategy
from agents.llm_client import MultiAgentLLMOrchestrator

console = Console()

# Evaluation tasks (same as comprehensive eval)
EVALUATION_TASKS = {
    "coding_easy": [
        {"id": 1, "category": "coding_easy", "description": "Write a Python function to calculate factorial", "complexity": 0.2},
        {"id": 2, "category": "coding_easy", "description": "Create a function that reverses a string", "complexity": 0.2},
        {"id": 3, "category": "coding_easy", "description": "Write a function to find the maximum in a list", "complexity": 0.2},
    ],
    "coding_medium": [
        {"id": 4, "category": "coding_medium", "description": "Implement a binary search algorithm", "complexity": 0.5},
        {"id": 5, "category": "coding_medium", "description": "Create a function to merge two sorted lists", "complexity": 0.5},
        {"id": 6, "category": "coding_medium", "description": "Write a function to validate email addresses using regex", "complexity": 0.5},
    ],
    "coding_hard": [
        {"id": 7, "category": "coding_hard", "description": "Implement a LRU cache with O(1) operations", "complexity": 0.9},
        {"id": 8, "category": "coding_hard", "description": "Create a function to solve N-Queens problem", "complexity": 0.9},
    ],
    "debugging": [
        {"id": 9, "category": "debugging", "description": "Debug a recursive function that's causing stack overflow", "complexity": 0.6},
        {"id": 10, "category": "debugging", "description": "Fix a memory leak in Python code", "complexity": 0.6},
    ],
}

# Flatten tasks
ALL_TASKS = []
for category_tasks in EVALUATION_TASKS.values():
    ALL_TASKS.extend(category_tasks)


class HallucinationDetector:
    """Detects hallucinations in LLM outputs"""

    HALLUCINATION_INDICATORS = [
        # Claims about non-existent features/APIs
        "import imaginary_module",
        "from fake_library import",
        ".nonexistent_method()",

        # Impossible claims
        "O(0) time complexity",
        "100% accuracy guaranteed",
        "never fails",
        "always works",

        # Contradictions
        "both true and false",
        "simultaneously",

        # Made-up syntax
        "def async lambda",
        "class static final",
    ]

    def detect_hallucinations(self, output: str) -> Dict[str, Any]:
        """Detect hallucinations in output"""
        output_lower = output.lower()

        found_indicators = []
        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator.lower() in output_lower:
                found_indicators.append(indicator)

        # Check for excessive confidence without code
        has_code = "```" in output or "def " in output or "class " in output
        has_excessive_confidence = any(word in output_lower for word in ["guaranteed", "perfect", "never fails", "always works"])

        confidence_without_substance = has_excessive_confidence and not has_code

        hallucination_score = len(found_indicators) * 0.2
        if confidence_without_substance:
            hallucination_score += 0.3

        return {
            "has_hallucinations": hallucination_score > 0,
            "hallucination_score": min(hallucination_score, 1.0),
            "indicators_found": found_indicators,
            "confidence_without_substance": confidence_without_substance
        }


@weave.op()
async def run_sequential_collaborative(
    orchestrator: CollaborativeOrchestrator,
    task: Dict[str, Any]
) -> Dict[str, Any]:
    """Run sequential collaboration on a task"""

    start_time = datetime.now()

    try:
        result = await orchestrator.collaborate(task["description"])

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Detect hallucinations
        detector = HallucinationDetector()
        hallucination_analysis = detector.detect_hallucinations(result.final_output)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "complexity": task["complexity"],
            "method": "sequential_collaboration",
            "agents_used": result.agents_used,
            "stages": result.consensus_rounds,
            "iterations": result.conflicts_resolved,
            "duration_seconds": duration,
            "output": result.final_output,
            "quality_score": result.metrics["quality"],
            "success": result.metrics["overall"] > 0.6,
            "hallucination_detected": hallucination_analysis["has_hallucinations"],
            "hallucination_score": hallucination_analysis["hallucination_score"],
            "hallucination_details": hallucination_analysis,
            "timestamp": start_time.isoformat(),
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "method": "sequential_collaboration",
            "success": False,
            "error": str(e),
            "hallucination_detected": False,
            "hallucination_score": 0.0,
        }


@weave.op()
async def run_single_model_baseline(
    llm_client: MultiAgentLLMOrchestrator,
    task: Dict[str, Any],
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """Run single model baseline on a task"""

    start_time = datetime.now()

    try:
        # Single model execution (no collaboration)
        output = await llm_client.execute_agent_task(
            agent_id="coder",  # Use coder agent config
            task=task["description"]
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Detect hallucinations
        detector = HallucinationDetector()
        hallucination_analysis = detector.detect_hallucinations(output)

        # Simple quality heuristic (has code, reasonable length)
        has_code = "```" in output or "def " in output
        reasonable_length = 50 < len(output) < 5000
        quality_score = 0.7 if (has_code and reasonable_length) else 0.4

        return {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "complexity": task["complexity"],
            "method": "single_model_baseline",
            "model": model,
            "duration_seconds": duration,
            "output": output,
            "quality_score": quality_score,
            "success": quality_score > 0.6,
            "hallucination_detected": hallucination_analysis["has_hallucinations"],
            "hallucination_score": hallucination_analysis["hallucination_score"],
            "hallucination_details": hallucination_analysis,
            "timestamp": start_time.isoformat(),
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "description": task["description"],
            "method": "single_model_baseline",
            "success": False,
            "error": str(e),
            "hallucination_detected": False,
            "hallucination_score": 0.0,
        }


async def run_comparison_evaluation():
    """Run full comparison evaluation"""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print("\n" + "=" * 80)
    console.print(" Sequential Collaboration vs Single-Model Baseline Evaluation")
    console.print("=" * 80)
    console.print()

    # Initialize Weave
    weave.init(f"facilitair/sequential-vs-baseline-{run_id}")

    # Initialize orchestrator (sequential only)
    console.print(" Initializing sequential orchestrator...")
    orchestrator = CollaborativeOrchestrator(
        use_sequential=True,
        use_sponsors=False,
        user_strategy=Strategy.BALANCED
    )

    llm_client = orchestrator.llm_orchestrator

    console.print(f"[LIST] Running {len(ALL_TASKS)} tasks with both methods...")
    console.print()

    sequential_results = []
    baseline_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        task_progress = progress.add_task(
            f"Running evaluations...",
            total=len(ALL_TASKS) * 2
        )

        for task in ALL_TASKS:
            # Run sequential collaboration
            console.print(f"   Sequential: {task['description'][:50]}...")
            seq_result = await run_sequential_collaborative(orchestrator, task)
            sequential_results.append(seq_result)
            progress.update(task_progress, advance=1)

            # Run single model baseline
            console.print(f"   Baseline:   {task['description'][:50]}...")
            baseline_result = await run_single_model_baseline(llm_client, task)
            baseline_results.append(baseline_result)
            progress.update(task_progress, advance=1)

            console.print()

    # Calculate statistics
    console.print("\n" + "=" * 80)
    console.print("[CHART] RESULTS COMPARISON")
    console.print("=" * 80)
    console.print()

    # Sequential stats
    seq_success = sum(1 for r in sequential_results if r.get("success", False))
    seq_hallucinations = sum(1 for r in sequential_results if r.get("hallucination_detected", False))
    seq_avg_quality = sum(r.get("quality_score", 0) for r in sequential_results) / len(sequential_results)
    seq_avg_duration = sum(r.get("duration_seconds", 0) for r in sequential_results) / len(sequential_results)
    seq_avg_hallucination_score = sum(r.get("hallucination_score", 0) for r in sequential_results) / len(sequential_results)

    # Baseline stats
    base_success = sum(1 for r in baseline_results if r.get("success", False))
    base_hallucinations = sum(1 for r in baseline_results if r.get("hallucination_detected", False))
    base_avg_quality = sum(r.get("quality_score", 0) for r in baseline_results) / len(baseline_results)
    base_avg_duration = sum(r.get("duration_seconds", 0) for r in baseline_results) / len(baseline_results)
    base_avg_hallucination_score = sum(r.get("hallucination_score", 0) for r in baseline_results) / len(baseline_results)

    # Create comparison table
    table = Table(title="Sequential vs Baseline Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential Collaboration", style="green")
    table.add_column("Single-Model Baseline", style="yellow")
    table.add_column("Winner", style="magenta")

    table.add_row(
        "Success Rate",
        f"{seq_success}/{len(sequential_results)} ({seq_success/len(sequential_results)*100:.1f}%)",
        f"{base_success}/{len(baseline_results)} ({base_success/len(baseline_results)*100:.1f}%)",
        "[ACHIEVEMENT] Sequential" if seq_success > base_success else "[ACHIEVEMENT] Baseline"
    )

    table.add_row(
        "Avg Quality Score",
        f"{seq_avg_quality:.3f}",
        f"{base_avg_quality:.3f}",
        "[ACHIEVEMENT] Sequential" if seq_avg_quality > base_avg_quality else "[ACHIEVEMENT] Baseline"
    )

    table.add_row(
        "Hallucinations Detected",
        f"{seq_hallucinations} ({seq_hallucinations/len(sequential_results)*100:.1f}%)",
        f"{base_hallucinations} ({base_hallucinations/len(baseline_results)*100:.1f}%)",
        "[ACHIEVEMENT] Sequential" if seq_hallucinations < base_hallucinations else "[ACHIEVEMENT] Baseline"
    )

    table.add_row(
        "Avg Hallucination Score",
        f"{seq_avg_hallucination_score:.3f}",
        f"{base_avg_hallucination_score:.3f}",
        "[ACHIEVEMENT] Sequential" if seq_avg_hallucination_score < base_avg_hallucination_score else "[ACHIEVEMENT] Baseline"
    )

    table.add_row(
        "Avg Duration (sec)",
        f"{seq_avg_duration:.2f}",
        f"{base_avg_duration:.2f}",
        "[ACHIEVEMENT] Baseline" if base_avg_duration < seq_avg_duration else "[ACHIEVEMENT] Sequential"
    )

    console.print(table)
    console.print()

    # Save results
    results_file = f"sequential_vs_baseline_results_{run_id}.json"
    comparison_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "sequential": {
                "success_count": seq_success,
                "success_rate": seq_success / len(sequential_results),
                "avg_quality": seq_avg_quality,
                "hallucinations_detected": seq_hallucinations,
                "hallucination_rate": seq_hallucinations / len(sequential_results),
                "avg_hallucination_score": seq_avg_hallucination_score,
                "avg_duration": seq_avg_duration,
            },
            "baseline": {
                "success_count": base_success,
                "success_rate": base_success / len(baseline_results),
                "avg_quality": base_avg_quality,
                "hallucinations_detected": base_hallucinations,
                "hallucination_rate": base_hallucinations / len(baseline_results),
                "avg_hallucination_score": base_avg_hallucination_score,
                "avg_duration": base_avg_duration,
            }
        },
        "sequential_results": sequential_results,
        "baseline_results": baseline_results,
    }

    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)

    console.print(f"[MEMORY] Results saved to: {results_file}")
    console.print()
    console.print("=" * 80)
    console.print("[OK] Evaluation Complete!")
    console.print("=" * 80)

    return comparison_data


if __name__ == "__main__":
    asyncio.run(run_comparison_evaluation())
