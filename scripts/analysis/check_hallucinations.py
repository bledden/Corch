"""
Check Existing Benchmark Results for Hallucinations

Analyzes completed benchmark tasks to detect if any outputs
are hallucinations (code that doesn't match task requirements).
"""

import json
import sys
from src.evaluation.semantic_relevance_checker import SemanticRelevanceChecker
from src.evaluation.quality_evaluator import detect_language
from rich.console import Console
from rich.table import Table

console = Console()


def check_benchmark_for_hallucinations(results_file: str):
    """
    Analyze benchmark results file for hallucinations.
    """

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {results_file}[/red]")
        return

    checker = SemanticRelevanceChecker()

    # Support both old and new result file formats
    if "sequential" in data and "baseline" in data:
        # New format (reanalyzed)
        sequential_results = data.get("sequential", [])
        baseline_results = data.get("baseline", [])
    elif "sequential_results" in data and "baseline_results" in data:
        # Old format (original benchmark)
        sequential_results = data.get("sequential_results", [])
        baseline_results = data.get("baseline_results", [])

    console.print(f"\n[cyan]Checking {len(sequential_results)} Sequential results for hallucinations...[/cyan]")
    seq_hallucinations, seq_avg_rel, seq_avg_req = analyze_results(checker, sequential_results, "Sequential")

    console.print(f"\n[cyan]Checking {len(baseline_results)} Baseline results for hallucinations...[/cyan]")
    base_hallucinations, base_avg_rel, base_avg_req = analyze_results(checker, baseline_results, "Baseline")

    # Summary
    console.print("\n" + "="*70)
    console.print("[bold]HALLUCINATION DETECTION SUMMARY[/bold]")
    console.print("="*70)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Approach", style="cyan", width=15)
    table.add_column("Total Tasks", justify="right")
    table.add_column("Hallucinations", justify="right")
    table.add_column("Hallucination Rate", justify="right")
    table.add_column("Avg Relevance", justify="right")

    seq_rate = (len(seq_hallucinations) / len(sequential_results) * 100) if sequential_results else 0
    base_rate = (len(base_hallucinations) / len(baseline_results) * 100) if baseline_results else 0

    table.add_row(
        "Sequential",
        str(len(sequential_results)),
        str(len(seq_hallucinations)),
        f"{seq_rate:.1f}%",
        f"{seq_avg_rel:.3f}"
    )
    table.add_row(
        "Baseline",
        str(len(baseline_results)),
        str(len(base_hallucinations)),
        f"{base_rate:.1f}%",
        f"{base_avg_rel:.3f}"
    )

    console.print(table)

    # Show specific hallucinations
    if seq_hallucinations:
        console.print("\n[yellow]Sequential Hallucinations:[/yellow]")
        for h in seq_hallucinations:
            console.print(f"  Task {h['task_id']}: {h['task_description'][:60]}...")
            console.print(f"    Relevance: {h['relevance_score']:.2f} | {h['reason']}")

    if base_hallucinations:
        console.print("\n[yellow]Baseline Hallucinations:[/yellow]")
        for h in base_hallucinations:
            console.print(f"  Task {h['task_id']}: {h['task_description'][:60]}...")
            console.print(f"    Relevance: {h['relevance_score']:.2f} | {h['reason']}")


def analyze_results(checker, results, approach_name):
    """
    Analyze results for hallucinations.
    Returns tuple of (hallucinations_list, avg_relevance, avg_requirement_score).
    """
    hallucinations = []
    relevance_scores = []
    requirement_scores = []

    for i, result in enumerate(results, 1):
        # Support both old and new formats
        task_desc = result.get("task_description") or result.get("task", "")
        code = result.get("final_output") or result.get("output", "")

        if not code or not task_desc:
            continue

        # Detect language
        language = detect_language(code)

        # Check relevance
        relevance_score, details = checker.check_relevance(code, task_desc, language)

        # Also check specific requirements
        req_score, req_details = checker.check_task_specific_requirements(code, task_desc)

        # Store scores
        relevance_scores.append(relevance_score)
        requirement_scores.append(req_score)

        # Store scores in result
        result["relevance_score"] = relevance_score
        result["requirement_score"] = req_score

        # Consider it a hallucination if:
        # 1. Relevance score < 0.5 (likely wrong problem)
        # 2. Requirement score < 0.5 AND relevance < 0.7 (missing key requirements)
        is_hallucination = (
            relevance_score < 0.5 or
            (req_score < 0.5 and relevance_score < 0.7)
        )

        if is_hallucination:
            hallucinations.append({
                "task_id": result.get("task_id", i),
                "task_description": task_desc,
                "relevance_score": relevance_score,
                "requirement_score": req_score,
                "reason": details.get("reasoning", "Low keyword match") if "reasoning" in details else f"Missing: {req_details.get('requirements_missing', [])}",
                "code_preview": code[:200]
            })

        # Progress indicator
        if i % 10 == 0:
            console.print(f"  Analyzed {i}/{len(results)} tasks...")

    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    avg_requirement = sum(requirement_scores) / len(requirement_scores) if requirement_scores else 0

    return hallucinations, avg_relevance, avg_requirement


def main():
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python check_hallucinations.py <results_file.json>[/yellow]")
        console.print("\nExample:")
        console.print("  python check_hallucinations.py benchmark_10_reanalyzed_results.json")
        return

    results_file = sys.argv[1]
    check_benchmark_for_hallucinations(results_file)


if __name__ == "__main__":
    main()
