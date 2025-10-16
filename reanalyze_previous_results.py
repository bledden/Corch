"""
Re-analyze previous benchmark results with real quality evaluation
"""

import json
import asyncio
from quality_evaluator import CodeQualityEvaluator, detect_language
from rich.console import Console
from rich.table import Table

console = Console()


def extract_code_from_output(output: str) -> str:
    """Extract actual code from output (may contain markdown blocks)"""
    # Try to extract from markdown code blocks
    import re
    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', output, re.DOTALL)

    if code_blocks:
        # Return the first substantial code block
        for block in code_blocks:
            if len(block.strip()) > 50:
                return block

    # No code blocks found, return the whole output
    return output


def reanalyze_results():
    """Re-analyze previous benchmark results with real quality evaluation"""

    # Load previous results
    with open("benchmark_10_quick_results_20251014_161425.json") as f:
        old_results = json.load(f)

    evaluator = CodeQualityEvaluator(pass_threshold=0.7)

    # Re-analyze sequential results
    console.print("\n[bold cyan]Re-analyzing Sequential Results[/bold cyan]\n")
    sequential_reanalyzed = []

    for result in old_results["sequential_results"]:
        task_id = result["task_id"]
        output = result["output"]

        # Extract code
        code = extract_code_from_output(output)

        # Detect language and evaluate
        language = detect_language(code)
        quality = evaluator.evaluate(code, f"Task {task_id}", language)

        console.print(f"Task {task_id}: Quality={quality.overall:.3f} (was hardcoded 0.8)")
        console.print(f"  Dimensions: Correctness={quality.dimensions['correctness']:.2f}, "
                     f"Completeness={quality.dimensions['completeness']:.2f}, "
                     f"Quality={quality.dimensions['code_quality']:.2f}")

        sequential_reanalyzed.append({
            "task_id": task_id,
            "old_quality": result.get("quality_score", 0.8),
            "new_quality": quality.overall,
            "dimensions": quality.dimensions,
            "language": language,
            "passed_old": result.get("pass", False),
            "passed_new": quality.passed,
        })

    # Re-analyze baseline results
    console.print("\n[bold cyan]Re-analyzing Baseline Results[/bold cyan]\n")
    baseline_reanalyzed = []

    for result in old_results["baseline_results"]:
        task_id = result["task_id"]
        output = result["output"]

        # Extract code
        code = extract_code_from_output(output)

        # Detect language and evaluate
        language = detect_language(code)
        quality = evaluator.evaluate(code, f"Task {task_id}", language)

        console.print(f"Task {task_id}: Quality={quality.overall:.3f} (was hardcoded 0.8)")
        console.print(f"  Dimensions: Correctness={quality.dimensions['correctness']:.2f}, "
                     f"Completeness={quality.dimensions['completeness']:.2f}, "
                     f"Quality={quality.dimensions['code_quality']:.2f}")

        baseline_reanalyzed.append({
            "task_id": task_id,
            "old_quality": result.get("quality_score", 0.8),
            "new_quality": quality.overall,
            "dimensions": quality.dimensions,
            "language": language,
            "passed_old": result.get("pass", False),
            "passed_new": quality.passed,
        })

    # Calculate new metrics
    seq_new_qualities = [r["new_quality"] for r in sequential_reanalyzed]
    base_new_qualities = [r["new_quality"] for r in baseline_reanalyzed]

    seq_new_passes = [r["passed_new"] for r in sequential_reanalyzed]
    base_new_passes = [r["passed_new"] for r in baseline_reanalyzed]

    # Display comparison table
    console.print("\n")
    table = Table(title="Reanalysis Results - Old vs New Quality Scores")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential (Old)", style="yellow")
    table.add_column("Sequential (New)", style="green")
    table.add_column("Baseline (Old)", style="yellow")
    table.add_column("Baseline (New)", style="green")

    table.add_row(
        "Avg Quality",
        f"{old_results['metrics']['sequential']['avg_quality']:.3f}",
        f"{sum(seq_new_qualities) / len(seq_new_qualities):.3f}",
        f"{old_results['metrics']['baseline']['avg_quality']:.3f}",
        f"{sum(base_new_qualities) / len(base_new_qualities):.3f}"
    )

    table.add_row(
        "Min Quality",
        "N/A",
        f"{min(seq_new_qualities):.3f}",
        "N/A",
        f"{min(base_new_qualities):.3f}"
    )

    table.add_row(
        "Max Quality",
        "N/A",
        f"{max(seq_new_qualities):.3f}",
        "N/A",
        f"{max(base_new_qualities):.3f}"
    )

    table.add_row(
        "Pass@1",
        f"{old_results['metrics']['sequential']['pass@1']:.1f}%",
        f"{sum(seq_new_passes) / len(seq_new_passes) * 100:.1f}%",
        f"{old_results['metrics']['baseline']['pass@1']:.1f}%",
        f"{sum(base_new_passes) / len(base_new_passes) * 100:.1f}%"
    )

    console.print(table)

    # Detailed task-by-task comparison
    console.print("\n")
    detail_table = Table(title="Task-by-Task Quality Comparison")
    detail_table.add_column("Task", style="cyan")
    detail_table.add_column("Sequential Old", style="yellow")
    detail_table.add_column("Sequential New", style="green")
    detail_table.add_column("Baseline Old", style="yellow")
    detail_table.add_column("Baseline New", style="green")
    detail_table.add_column("Winner", style="magenta")

    for i in range(len(sequential_reanalyzed)):
        seq = sequential_reanalyzed[i]
        base = baseline_reanalyzed[i]

        winner = "Sequential" if seq["new_quality"] > base["new_quality"] else "Baseline" if base["new_quality"] > seq["new_quality"] else "Tie"

        detail_table.add_row(
            f"Task {seq['task_id']}",
            f"{seq['old_quality']:.3f}",
            f"{seq['new_quality']:.3f}",
            f"{base['old_quality']:.3f}",
            f"{base['new_quality']:.3f}",
            winner
        )

    console.print(detail_table)

    # Summary
    console.print("\n[bold]Key Findings:[/bold]")
    console.print(f"1. Old scores were all identical (0.8), masking real quality differences")
    console.print(f"2. New scores show real variation (Sequential: {min(seq_new_qualities):.3f}-{max(seq_new_qualities):.3f}, "
                 f"Baseline: {min(base_new_qualities):.3f}-{max(base_new_qualities):.3f})")

    seq_avg_new = sum(seq_new_qualities) / len(seq_new_qualities)
    base_avg_new = sum(base_new_qualities) / len(base_new_qualities)

    if seq_avg_new > base_avg_new:
        console.print(f"3. [green]Sequential produces higher quality code on average ({seq_avg_new:.3f} vs {base_avg_new:.3f})[/green]")
    elif base_avg_new > seq_avg_new:
        console.print(f"3. [yellow]Baseline produces higher quality code on average ({base_avg_new:.3f} vs {seq_avg_new:.3f})[/yellow]")
    else:
        console.print(f"3. [white]Both approaches produce equivalent quality ({seq_avg_new:.3f})[/white]")

    # Count wins
    seq_wins = sum(1 for i in range(len(sequential_reanalyzed))
                   if sequential_reanalyzed[i]["new_quality"] > baseline_reanalyzed[i]["new_quality"])
    base_wins = sum(1 for i in range(len(baseline_reanalyzed))
                    if baseline_reanalyzed[i]["new_quality"] > sequential_reanalyzed[i]["new_quality"])
    ties = len(sequential_reanalyzed) - seq_wins - base_wins

    console.print(f"4. Task wins: Sequential={seq_wins}, Baseline={base_wins}, Ties={ties}")

    # Save reanalyzed results
    with open("benchmark_10_reanalyzed_results.json", "w") as f:
        json.dump({
            "metadata": {
                "source": "benchmark_10_quick_results_20251014_161425.json",
                "reanalyzed_with": "CodeQualityEvaluator",
            },
            "sequential": sequential_reanalyzed,
            "baseline": baseline_reanalyzed,
            "summary": {
                "sequential": {
                    "old_avg": old_results['metrics']['sequential']['avg_quality'],
                    "new_avg": seq_avg_new,
                    "new_min": min(seq_new_qualities),
                    "new_max": max(seq_new_qualities),
                    "task_wins": seq_wins,
                },
                "baseline": {
                    "old_avg": old_results['metrics']['baseline']['avg_quality'],
                    "new_avg": base_avg_new,
                    "new_min": min(base_new_qualities),
                    "new_max": max(base_new_qualities),
                    "task_wins": base_wins,
                }
            }
        }, f, indent=2)

    console.print("\n[green]Results saved to: benchmark_10_reanalyzed_results.json[/green]")


if __name__ == "__main__":
    reanalyze_results()
