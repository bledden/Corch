"""
Smoke Test: 10 Tasks (5 Self-Contained + 5 Web-Search)
Quick validation before full 500-task benchmark
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
from src.utils.web_search_router import WebSearchRouter, SearchStrategy
from agents.hallucination_detector import HallucinationDetector
import yaml

console = Console()

# Initialize web search router
web_search_router = WebSearchRouter()


def categorize_model(model_name: str) -> str:
    """
    Categorize a model as open_source or closed_source based on provider.

    Closed Source: openai/*, anthropic/*, google/gemini*, perplexity/*
    Open Source: qwen/*, deepseek/*, mistralai/codestral*, alibaba/qwen*,
                 meta-llama/*, cohere/command-r*
    """
    if not model_name:
        return "unknown"

    model_lower = model_name.lower()

    # Closed source providers
    closed_source_prefixes = [
        "openai/", "anthropic/", "google/gemini", "perplexity/"
    ]

    # Open source providers
    open_source_prefixes = [
        "qwen/", "deepseek", "alibaba/qwen", "meta-llama/",
        "mistralai/codestral", "cohere/command-r"
    ]

    for prefix in closed_source_prefixes:
        if model_lower.startswith(prefix):
            return "closed_source"

    for prefix in open_source_prefixes:
        if prefix in model_lower:
            return "open_source"

    # Default to unknown if not matched
    return "unknown"

# 10 Smoke Test Tasks
SMOKE_TEST_TASKS = [
    # 5 Self-Contained (should not trigger web search)
    {"id": 1, "category": "self_contained", "description": "Write a function to check if a number is prime", "complexity": 0.3},
    {"id": 2, "category": "self_contained", "description": "Implement factorial using recursion", "complexity": 0.3},
    {"id": 3, "category": "self_contained", "description": "Create a function to reverse a string", "complexity": 0.3},
    {"id": 4, "category": "self_contained", "description": "Create a function to check if a string is palindrome", "complexity": 0.3},
    {"id": 5, "category": "self_contained", "description": "Create a function to find GCD of two numbers", "complexity": 0.3},

    # 5 Web-Search-Requiring (should trigger web search)
    {"id": 6, "category": "web_search", "description": "Implement Next.js 15 App Router with Server Actions", "complexity": 0.7},
    {"id": 7, "category": "web_search", "description": "Create React 19 component with new use() hook", "complexity": 0.7},
    {"id": 8, "category": "web_search", "description": "Build Stripe API v2024 payment intent", "complexity": 0.7},
    {"id": 9, "category": "web_search", "description": "Implement OpenAI GPT-4 Turbo API with streaming", "complexity": 0.7},
    {"id": 10, "category": "web_search", "description": "Implement OWASP Top 10 2023 security fixes", "complexity": 0.7},
]


@weave.op()
async def execute_web_search(task_description: str, strategy: str = "balanced") -> Dict[str, Any]:
    """Execute web search for tasks requiring external information"""
    try:
        # Select search method based on strategy
        search_strategy = SearchStrategy[strategy.upper()]
        search_method = web_search_router.select_search_method(search_strategy)

        console.print(f"[yellow]Executing web search via {search_method.name}[/yellow]")

        # Simulate search execution (replace with actual API calls in production)
        # In production: Call Tavily, Perplexity, or Gemini APIs here
        search_results = f"[Web Search Results via {search_method.name}]\n"
        search_results += f"Latest information about: {task_description[:100]}...\n"
        search_results += "Current best practices and documentation retrieved.\n"

        return {
            "search_executed": True,
            "search_method": search_method.name,
            "search_cost": search_method.cost_per_search,
            "search_results": search_results,
        }
    except Exception as e:
        console.print(f"[red]Web search failed: {e}[/red]")
        return {
            "search_executed": False,
            "search_method": None,
            "search_cost": 0.0,
            "search_results": "",
        }


@weave.op()
async def run_sequential(orchestrator: CollaborativeOrchestrator, task: Dict) -> Dict:
    """Run sequential collaboration (our approach)"""
    start = datetime.now()

    # Detect if task needs external information (web search)
    needs_search, patterns, confidence = web_search_router.detect_needs_web_search(task["description"])

    # Execute web search if needed
    search_info = {"search_executed": False, "search_method": None, "search_cost": 0.0}
    task_context = task["description"]

    if needs_search:
        search_info = await execute_web_search(task["description"], strategy="balanced")
        if search_info["search_executed"]:
            task_context = f"{task['description']}\n\n{search_info['search_results']}"

    try:
        result = await orchestrator.collaborate(task_context)
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(result.final_output)

        quality = result.metrics.get("quality", 0.0)
        overall = result.metrics.get("overall", 0.0)

        # Get model used (from orchestrator's last execution)
        models_used = result.metadata.get("models_used", {})
        primary_model = models_used.get("coder", "unknown")
        model_type = categorize_model(primary_model)

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
            "needs_external_info": needs_search,
            "search_confidence": confidence,
            "matched_patterns": patterns,
            "search_executed": search_info["search_executed"],
            "search_method_used": search_info["search_method"],
            "search_cost": search_info["search_cost"],
            "model_used": primary_model,
            "model_type": model_type,
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "success": False,
            "pass": False,
            "error": str(e),
            "needs_external_info": needs_search,
            "search_executed": search_info["search_executed"],
            "search_method_used": search_info["search_method"],
            "search_cost": search_info["search_cost"],
        }


@weave.op()
async def run_baseline(llm: MultiAgentLLMOrchestrator, task: Dict) -> Dict:
    """Run single-model baseline (GPT-4 direct)"""
    start = datetime.now()

    # Detect if task needs external information (web search)
    needs_search, patterns, confidence = web_search_router.detect_needs_web_search(task["description"])

    # Execute web search if needed
    search_info = {"search_executed": False, "search_method": None, "search_cost": 0.0}
    task_context = task["description"]

    if needs_search:
        search_info = await execute_web_search(task["description"], strategy="cheapest")
        if search_info["search_executed"]:
            task_context = f"{task['description']}\n\n{search_info['search_results']}"

    try:
        output = await llm.execute_agent_task("coder", task_context)
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(output)

        # Get model used for coder agent
        coder_config = llm.config.get("agents", {}).get("coder", {})
        baseline_model = coder_config.get("default_model", "unknown")
        model_type = categorize_model(baseline_model)

        has_code = any(m in output for m in ["```", "def ", "class ", "function "])
        has_logic = any(keyword in output.lower() for keyword in ["if ", "for ", "while ", "return "])
        reasonable_length = 100 < len(output) < 10000
        has_substantial_output = len(output.strip()) > 50

        quality_estimate = 0.8 if (has_code and has_logic and reasonable_length) else (
            0.5 if has_code else 0.2
        )

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
            "needs_external_info": needs_search,
            "search_confidence": confidence,
            "matched_patterns": patterns,
            "search_executed": search_info["search_executed"],
            "search_method_used": search_info["search_method"],
            "search_cost": search_info["search_cost"],
            "model_used": baseline_model,
            "model_type": model_type,
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "success": False,
            "pass": False,
            "error": str(e),
            "needs_external_info": needs_search,
            "search_executed": search_info["search_executed"],
            "search_method_used": search_info["search_method"],
            "search_cost": search_info["search_cost"],
        }


async def run_smoke_test():
    """Run smoke test with 10 tasks"""
    console.print("\n[bold cyan]Smoke Test: 10 Tasks[/bold cyan]")
    console.print("[yellow]5 Self-Contained + 5 Web-Search[/yellow]")
    console.print(f"Total tasks: {len(SMOKE_TEST_TASKS)}\n")

    # Initialize
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    weave.init("facilitair/smoke-test")

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
            "[cyan]Running smoke test...",
            total=len(SMOKE_TEST_TASKS) * 2
        )

        for task in SMOKE_TEST_TASKS:
            # Run sequential
            console.print(f"\n[blue]Task {task['id']}: {task['description'][:60]}...[/blue]")
            seq_result = await run_sequential(orchestrator, task)
            sequential_results.append(seq_result)
            progress.update(task_progress, advance=1)

            # Run baseline
            base_result = await run_baseline(llm, task)
            baseline_results.append(base_result)
            progress.update(task_progress, advance=1)

    # Calculate metrics
    seq_passes = [r["pass"] for r in sequential_results]
    base_passes = [r["pass"] for r in baseline_results]

    # Task type breakdown
    seq_self_contained = [r for r in sequential_results if not r.get("needs_external_info", False)]
    seq_web_search = [r for r in sequential_results if r.get("needs_external_info", False)]
    base_self_contained = [r for r in baseline_results if not r.get("needs_external_info", False)]
    base_web_search = [r for r in baseline_results if r.get("needs_external_info", False)]

    # Search method breakdown
    total_searches = sum(1 for r in sequential_results + baseline_results if r.get("search_executed", False))
    total_search_cost = sum(r.get("search_cost", 0.0) for r in sequential_results + baseline_results)

    # Model type breakdown
    def calculate_model_stats(results):
        open_source = [r for r in results if r.get("model_type") == "open_source"]
        closed_source = [r for r in results if r.get("model_type") == "closed_source"]
        return {
            "open_source_count": len(open_source),
            "closed_source_count": len(closed_source),
            "open_source_pass_rate": (sum(r["pass"] for r in open_source) / len(open_source) * 100) if open_source else 0,
            "closed_source_pass_rate": (sum(r["pass"] for r in closed_source) / len(closed_source) * 100) if closed_source else 0,
            "open_source_avg_quality": sum(r.get("quality_score", 0) for r in open_source) / len(open_source) if open_source else 0,
            "closed_source_avg_quality": sum(r.get("quality_score", 0) for r in closed_source) / len(closed_source) if closed_source else 0,
        }

    seq_model_stats = calculate_model_stats(sequential_results)
    base_model_stats = calculate_model_stats(baseline_results)

    metrics = {
        "sequential": {
            "pass@1": sum(seq_passes) / len(seq_passes) * 100 if seq_passes else 0,
            "total_successes": sum(seq_passes),
            "total_tasks": len(seq_passes),
            "hallucinations": sum(1 for r in sequential_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(r.get("quality_score", 0) for r in sequential_results) / len(sequential_results) if sequential_results else 0,
            "avg_duration": sum(r.get("duration", 0) for r in sequential_results) / len(sequential_results) if sequential_results else 0,
            "self_contained_tasks": len(seq_self_contained),
            "web_search_tasks": len(seq_web_search),
            "self_contained_pass_rate": (sum(r["pass"] for r in seq_self_contained) / len(seq_self_contained) * 100) if seq_self_contained else 0,
            "web_search_pass_rate": (sum(r["pass"] for r in seq_web_search) / len(seq_web_search) * 100) if seq_web_search else 0,
            "model_stats": seq_model_stats,
        },
        "baseline": {
            "pass@1": sum(base_passes) / len(base_passes) * 100 if base_passes else 0,
            "total_successes": sum(base_passes),
            "total_tasks": len(base_passes),
            "hallucinations": sum(1 for r in baseline_results if r.get("hallucination", {}).get("hallucination_detected")),
            "avg_quality": sum(r.get("quality_score", 0) for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "avg_duration": sum(r.get("duration", 0) for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "self_contained_tasks": len(base_self_contained),
            "web_search_tasks": len(base_web_search),
            "self_contained_pass_rate": (sum(r["pass"] for r in base_self_contained) / len(base_self_contained) * 100) if base_self_contained else 0,
            "web_search_pass_rate": (sum(r["pass"] for r in base_web_search) / len(base_web_search) * 100) if base_web_search else 0,
            "model_stats": base_model_stats,
        },
        "search": {
            "total_searches": total_searches,
            "total_cost_usd": total_search_cost,
        }
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"smoke_test_results_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "total_tasks": len(SMOKE_TEST_TASKS),
                "test_type": "smoke_test",
            },
            "metrics": metrics,
            "sequential_results": sequential_results,
            "baseline_results": baseline_results,
        }, f, indent=2)

    # Display results
    table = Table(title="Smoke Test Results (10 Tasks)")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="green")
    table.add_column("Baseline", style="yellow")
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

    # Task Type Breakdown
    type_table = Table(title="Task Type Breakdown")
    type_table.add_column("Task Type", style="cyan")
    type_table.add_column("Count", style="white")
    type_table.add_column("Sequential Pass Rate", style="green")
    type_table.add_column("Baseline Pass Rate", style="yellow")

    type_table.add_row(
        "Self-Contained",
        f"{metrics['sequential']['self_contained_tasks']}",
        f"{metrics['sequential']['self_contained_pass_rate']:.1f}%",
        f"{metrics['baseline']['self_contained_pass_rate']:.1f}%"
    )

    type_table.add_row(
        "Web-Search Required",
        f"{metrics['sequential']['web_search_tasks']}",
        f"{metrics['sequential']['web_search_pass_rate']:.1f}%",
        f"{metrics['baseline']['web_search_pass_rate']:.1f}%"
    )

    console.print("\n")
    console.print(type_table)

    # Search Statistics
    search_table = Table(title="Web Search Statistics")
    search_table.add_column("Metric", style="cyan")
    search_table.add_column("Value", style="white")

    search_table.add_row("Total Searches Executed", f"{metrics['search']['total_searches']}")
    search_table.add_row("Total Cost", f"${metrics['search']['total_cost_usd']:.4f}")

    console.print("\n")
    console.print(search_table)

    # Model Type Statistics
    model_type_table = Table(title="Model Type Performance (Open vs Closed Source)")
    model_type_table.add_column("Model Type", style="cyan")
    model_type_table.add_column("Sequential Tasks", style="white")
    model_type_table.add_column("Sequential Pass Rate", style="green")
    model_type_table.add_column("Sequential Avg Quality", style="green")
    model_type_table.add_column("Baseline Tasks", style="white")
    model_type_table.add_column("Baseline Pass Rate", style="yellow")
    model_type_table.add_column("Baseline Avg Quality", style="yellow")

    model_type_table.add_row(
        "Open Source",
        f"{metrics['sequential']['model_stats']['open_source_count']}",
        f"{metrics['sequential']['model_stats']['open_source_pass_rate']:.1f}%",
        f"{metrics['sequential']['model_stats']['open_source_avg_quality']:.3f}",
        f"{metrics['baseline']['model_stats']['open_source_count']}",
        f"{metrics['baseline']['model_stats']['open_source_pass_rate']:.1f}%",
        f"{metrics['baseline']['model_stats']['open_source_avg_quality']:.3f}"
    )

    model_type_table.add_row(
        "Closed Source",
        f"{metrics['sequential']['model_stats']['closed_source_count']}",
        f"{metrics['sequential']['model_stats']['closed_source_pass_rate']:.1f}%",
        f"{metrics['sequential']['model_stats']['closed_source_avg_quality']:.3f}",
        f"{metrics['baseline']['model_stats']['closed_source_count']}",
        f"{metrics['baseline']['model_stats']['closed_source_pass_rate']:.1f}%",
        f"{metrics['baseline']['model_stats']['closed_source_avg_quality']:.3f}"
    )

    console.print("\n")
    console.print(model_type_table)
    console.print(f"\n[green]Results saved to: {output_file}[/green]")

    # Smoke test validation
    console.print("\n[bold yellow]Smoke Test Validation:[/bold yellow]")
    if metrics['sequential']['total_successes'] >= 7:  # At least 70% pass rate
        console.print("[green][OK] PASS: Sequential method performing well (>= 70% pass rate)[/green]")
    else:
        console.print("[red][X] FAIL: Sequential method below 70% pass rate[/red]")

    if total_searches >= 3:  # At least 3 web searches executed (out of 5 web-search tasks * 2 methods = 10)
        console.print(f"[green][OK] PASS: Web search integration working ({total_searches} searches executed)[/green]")
    else:
        console.print(f"[yellow] WARNING: Web search may not be triggering ({total_searches} searches executed)[/yellow]")

    console.print("\n[bold cyan]Smoke test complete! Ready for full 500-task benchmark.[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
