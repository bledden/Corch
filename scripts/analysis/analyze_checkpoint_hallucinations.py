"""
Analyze checkpoint file for hallucinations using task descriptions from benchmark
"""

import json
from src.evaluation.semantic_relevance_checker import SemanticRelevanceChecker
from src.evaluation.quality_evaluator import detect_language
from rich.console import Console
from rich.table import Table

console = Console()

# Task descriptions from run_100_task_benchmark.py
TASK_DESCRIPTIONS = {
    1: "Implement password hashing with bcrypt, salt, and pepper",
    2: "Create SQL query builder with parameterized queries to prevent injection",
    3: "Build JWT token validator with signature verification and expiry checks",
    4: "Implement file upload handler with type validation and size limits",
    5: "Create user input sanitizer for XSS prevention",
    6: "Build CSRF token generator and validator",
    7: "Implement rate limiter with Redis to prevent DoS",
    8: "Create secure session manager with HTTP-only cookies",
    9: "Build API key generator with cryptographically secure randomness",
    10: "Implement secure password reset flow with token expiry",
    11: "Create content security policy header builder",
    12: "Build encrypted data storage with AES-256",
    13: "Implement OAuth2 authorization code flow",
    14: "Create permissions middleware with role-based access control",
    15: "Build audit logger for security events",
    16: "Implement secure random token generator for 2FA",
    17: "Create API request signer with HMAC",
    18: "Build secure file downloader with path traversal prevention",
    19: "Implement secure WebSocket connection with authentication",
    20: "Create database backup encryption tool",
}


def analyze_checkpoint(checkpoint_file: str):
    """Analyze checkpoint for hallucinations"""

    with open(checkpoint_file, 'r') as f:
        data = json.load(f)

    sequential = data.get('sequential', [])
    baseline = data.get('baseline', [])

    console.print(f"\n[bold cyan]Hallucination Analysis: First 20 Tasks[/bold cyan]\n")
    console.print(f"Checkpoint: {checkpoint_file}\n")

    checker = SemanticRelevanceChecker(use_llm_judge=False)

    # Analyze both approaches
    console.print("[cyan]Analyzing Sequential results...[/cyan]")
    seq_results = analyze_results(checker, sequential, "Sequential")

    console.print("\n[cyan]Analyzing Baseline results...[/cyan]")
    base_results = analyze_results(checker, baseline, "Baseline")

    # Summary table
    console.print("\n" + "="*80)
    console.print("[bold]HALLUCINATION DETECTION RESULTS[/bold]")
    console.print("="*80)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Sequential", justify="right", width=15)
    table.add_column("Baseline", justify="right", width=15)
    table.add_column("Winner", justify="center", width=15)

    # Hallucination count
    winner = "Sequential" if seq_results['hallucination_count'] < base_results['hallucination_count'] else "Baseline"
    if seq_results['hallucination_count'] == base_results['hallucination_count']:
        winner = "Tie"
    table.add_row(
        "Hallucinations Detected",
        str(seq_results['hallucination_count']),
        str(base_results['hallucination_count']),
        f"[green]{winner}[/green]"
    )

    # Hallucination rate
    seq_rate = seq_results['hallucination_count'] / 20 * 100
    base_rate = base_results['hallucination_count'] / 20 * 100
    winner = "Sequential" if seq_rate < base_rate else "Baseline"
    if seq_rate == base_rate:
        winner = "Tie"
    table.add_row(
        "Hallucination Rate",
        f"{seq_rate:.1f}%",
        f"{base_rate:.1f}%",
        f"[green]{winner}[/green]"
    )

    # Avg relevance score
    winner = "Sequential" if seq_results['avg_relevance'] > base_results['avg_relevance'] else "Baseline"
    table.add_row(
        "Avg Relevance Score",
        f"{seq_results['avg_relevance']:.3f}",
        f"{base_results['avg_relevance']:.3f}",
        f"[green]{winner}[/green]"
    )

    # Avg requirement score
    winner = "Sequential" if seq_results['avg_requirement'] > base_results['avg_requirement'] else "Baseline"
    table.add_row(
        "Avg Requirement Score",
        f"{seq_results['avg_requirement']:.3f}",
        f"{base_results['avg_requirement']:.3f}",
        f"[green]{winner}[/green]"
    )

    # Quality score (from benchmark)
    seq_quality = sum(r.get('quality_score', 0) for r in sequential) / len(sequential)
    base_quality = sum(r.get('quality_score', 0) for r in baseline) / len(baseline)
    winner = "Sequential" if seq_quality > base_quality else "Baseline"
    table.add_row(
        "Avg Quality Score",
        f"{seq_quality:.3f}",
        f"{base_quality:.3f}",
        f"[green]{winner}[/green]"
    )

    console.print(table)

    # Show specific hallucinations
    if seq_results['hallucinations']:
        console.print("\n[yellow]Sequential Hallucinations:[/yellow]")
        for h in seq_results['hallucinations']:
            console.print(f"  [red]Task {h['task_id']}:[/red] {h['task_desc']}")
            console.print(f"    Relevance: {h['relevance']:.2f} | Requirement: {h['requirement']:.2f}")
            console.print(f"    Reason: {h['reason']}")

    if base_results['hallucinations']:
        console.print("\n[yellow]Baseline Hallucinations:[/yellow]")
        for h in base_results['hallucinations']:
            console.print(f"  [red]Task {h['task_id']}:[/red] {h['task_desc']}")
            console.print(f"    Relevance: {h['relevance']:.2f} | Requirement: {h['requirement']:.2f}")
            console.print(f"    Reason: {h['reason']}")

    # Key insights
    console.print("\n" + "="*80)
    console.print("[bold]KEY INSIGHTS[/bold]")
    console.print("="*80)

    if seq_results['hallucination_count'] < base_results['hallucination_count']:
        console.print(f"[OK] Sequential has [green]{base_results['hallucination_count'] - seq_results['hallucination_count']} fewer hallucinations[/green] than Baseline")
        console.print("   This validates the hypothesis that multi-stage review catches off-topic outputs!")
    elif seq_results['hallucination_count'] > base_results['hallucination_count']:
        console.print(f"[WARNING]  Baseline has [yellow]{seq_results['hallucination_count'] - base_results['hallucination_count']} fewer hallucinations[/yellow] than Sequential")
        console.print("   Unexpected - may need to investigate why Sequential is generating more hallucinations")
    else:
        console.print("= Both approaches have the same hallucination count")

    console.print(f"\n[OK] Sequential relevance: {seq_results['avg_relevance']:.3f} vs Baseline: {base_results['avg_relevance']:.3f}")
    console.print(f"[OK] Sequential requirement: {seq_results['avg_requirement']:.3f} vs Baseline: {base_results['avg_requirement']:.3f}")

    if seq_results['avg_relevance'] > base_results['avg_relevance']:
        console.print("   → Sequential code is more semantically relevant to tasks")

    if seq_results['avg_requirement'] > base_results['avg_requirement']:
        console.print("   → Sequential code better meets task-specific requirements")


def analyze_results(checker, results, approach_name):
    """Analyze results for hallucinations"""

    hallucinations = []
    relevance_scores = []
    requirement_scores = []

    for result in results:
        task_id = result.get('task_id', 0)
        task_desc = TASK_DESCRIPTIONS.get(task_id, "")
        code = result.get('output', '')

        if not code or not task_desc:
            continue

        # Detect language
        language = detect_language(code)

        # Check relevance
        relevance_score, rel_details = checker.check_relevance(code, task_desc, language)

        # Check requirements
        req_score, req_details = checker.check_task_specific_requirements(code, task_desc)

        relevance_scores.append(relevance_score)
        requirement_scores.append(req_score)

        # Flag hallucination if:
        # 1. Relevance < 0.5 (clearly wrong problem)
        # 2. Requirement < 0.5 AND relevance < 0.7 (missing key requirements)
        is_hallucination = (
            relevance_score < 0.5 or
            (req_score < 0.5 and relevance_score < 0.7)
        )

        if is_hallucination:
            missing_reqs = req_details.get('requirements_missing', [])
            reason = f"Missing requirements: {missing_reqs}" if missing_reqs else "Low keyword match"
            if 'missing_keywords' in rel_details:
                reason += f" | Missing keywords: {rel_details['missing_keywords']}"

            hallucinations.append({
                'task_id': task_id,
                'task_desc': task_desc,
                'relevance': relevance_score,
                'requirement': req_score,
                'reason': reason
            })

    return {
        'hallucinations': hallucinations,
        'hallucination_count': len(hallucinations),
        'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
        'avg_requirement': sum(requirement_scores) / len(requirement_scores) if requirement_scores else 0
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        checkpoint_file = sys.argv[1]
    else:
        # Find most recent checkpoint
        import glob
        checkpoints = glob.glob("/Users/bledden/Documents/weavehacks-collaborative/benchmark_100_checkpoint*.json")
        if not checkpoints:
            console.print("[red]No checkpoint files found![/red]")
            sys.exit(1)
        checkpoint_file = sorted(checkpoints)[-1]
        console.print(f"Using most recent checkpoint: {checkpoint_file}")

    analyze_checkpoint(checkpoint_file)
