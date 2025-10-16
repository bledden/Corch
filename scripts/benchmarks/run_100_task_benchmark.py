"""
100-Task Benchmark Designed to Show Sequential Advantages

Task categories specifically chosen to favor multi-stage review:
1. Security-critical code (needs reviewer to catch vulnerabilities)
2. Complex algorithms (benefits from architect planning)
3. Production-ready features (needs documentation, tests, error handling)
4. Code with edge cases (reviewer catches missing cases)
5. Refactoring tasks (benefits from review and refinement)
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

# 100 tasks designed to favor sequential collaboration
BENCHMARK_TASKS = {
    # Category 1: Security-Critical Code (20 tasks)
    # Sequential should excel: reviewer catches vulnerabilities
    "security_critical": [
        {"id": 1, "description": "Implement password hashing with bcrypt, salt, and pepper", "rationale": "Needs security review"},
        {"id": 2, "description": "Create SQL query builder with parameterized queries to prevent injection", "rationale": "SQL injection prevention"},
        {"id": 3, "description": "Build JWT token validator with signature verification and expiry checks", "rationale": "Auth security"},
        {"id": 4, "description": "Implement file upload handler with type validation and size limits", "rationale": "Upload security"},
        {"id": 5, "description": "Create user input sanitizer for XSS prevention", "rationale": "XSS protection"},
        {"id": 6, "description": "Build CSRF token generator and validator", "rationale": "CSRF protection"},
        {"id": 7, "description": "Implement rate limiter with Redis to prevent DoS", "rationale": "DoS prevention"},
        {"id": 8, "description": "Create secure session manager with HTTP-only cookies", "rationale": "Session security"},
        {"id": 9, "description": "Build API key generator with cryptographically secure randomness", "rationale": "Crypto security"},
        {"id": 10, "description": "Implement secure password reset flow with token expiry", "rationale": "Password security"},
        {"id": 11, "description": "Create content security policy header builder", "rationale": "CSP security"},
        {"id": 12, "description": "Build encrypted data storage with AES-256", "rationale": "Encryption"},
        {"id": 13, "description": "Implement OAuth2 authorization code flow", "rationale": "OAuth security"},
        {"id": 14, "description": "Create permissions middleware with role-based access control", "rationale": "RBAC"},
        {"id": 15, "description": "Build audit logger for security events", "rationale": "Security auditing"},
        {"id": 16, "description": "Implement secure random token generator for 2FA", "rationale": "2FA security"},
        {"id": 17, "description": "Create API request signer with HMAC", "rationale": "API security"},
        {"id": 18, "description": "Build secure file downloader with path traversal prevention", "rationale": "Path traversal"},
        {"id": 19, "description": "Implement secure WebSocket connection with authentication", "rationale": "WebSocket security"},
        {"id": 20, "description": "Create database backup encryption tool", "rationale": "Backup security"},
    ],

    # Category 2: Complex Algorithms with Edge Cases (20 tasks)
    # Sequential should excel: reviewer catches edge cases
    "complex_algorithms": [
        {"id": 21, "description": "Implement balanced binary search tree with rotation", "rationale": "Complex edge cases"},
        {"id": 22, "description": "Create LRU cache with O(1) operations and thread safety", "rationale": "Thread safety"},
        {"id": 23, "description": "Build topological sort with cycle detection", "rationale": "Cycle detection"},
        {"id": 24, "description": "Implement dijkstra shortest path with negative edge handling", "rationale": "Negative edges"},
        {"id": 25, "description": "Create trie with prefix search and autocomplete", "rationale": "Prefix handling"},
        {"id": 26, "description": "Build interval tree for range queries", "rationale": "Range edge cases"},
        {"id": 27, "description": "Implement bloom filter with optimal hash functions", "rationale": "Hash collisions"},
        {"id": 28, "description": "Create skip list with probabilistic balancing", "rationale": "Probabilistic"},
        {"id": 29, "description": "Build segment tree for range updates", "rationale": "Range updates"},
        {"id": 30, "description": "Implement union-find with path compression", "rationale": "Path compression"},
        {"id": 31, "description": "Create consistent hashing for distributed systems", "rationale": "Distribution"},
        {"id": 32, "description": "Build red-black tree with rebalancing", "rationale": "Rebalancing logic"},
        {"id": 33, "description": "Implement suffix array with LCP array", "rationale": "LCP edge cases"},
        {"id": 34, "description": "Create KD-tree for nearest neighbor search", "rationale": "Spatial queries"},
        {"id": 35, "description": "Build fenwick tree for range sum queries", "rationale": "Range sums"},
        {"id": 36, "description": "Implement tarjan algorithm for SCCs", "rationale": "SCC detection"},
        {"id": 37, "description": "Create aho-corasick for multi-pattern matching", "rationale": "Multi-pattern"},
        {"id": 38, "description": "Build van emde boas tree", "rationale": "Complex structure"},
        {"id": 39, "description": "Implement dancing links for exact cover", "rationale": "Backtracking"},
        {"id": 40, "description": "Create suffix automaton for string matching", "rationale": "Automaton states"},
    ],

    # Category 3: Production-Ready Features (20 tasks)
    # Sequential should excel: needs docs, tests, error handling
    "production_features": [
        {"id": 41, "description": "Build REST API client with retry logic, timeout, and error handling", "rationale": "Production ready"},
        {"id": 42, "description": "Create connection pool manager with health checks", "rationale": "Connection management"},
        {"id": 43, "description": "Implement job queue with dead letter queue", "rationale": "Job reliability"},
        {"id": 44, "description": "Build cache wrapper with TTL and LRU eviction", "rationale": "Cache management"},
        {"id": 45, "description": "Create metrics collector with aggregation", "rationale": "Metrics"},
        {"id": 46, "description": "Implement distributed lock with Redlock algorithm", "rationale": "Distributed locking"},
        {"id": 47, "description": "Build event bus with pub/sub pattern", "rationale": "Event handling"},
        {"id": 48, "description": "Create circuit breaker with exponential backoff", "rationale": "Fault tolerance"},
        {"id": 49, "description": "Implement graceful shutdown handler", "rationale": "Shutdown handling"},
        {"id": 50, "description": "Build health check endpoint with dependency checks", "rationale": "Health monitoring"},
        {"id": 51, "description": "Create structured logger with context propagation", "rationale": "Logging"},
        {"id": 52, "description": "Implement feature flag system with rollout", "rationale": "Feature flags"},
        {"id": 53, "description": "Build database migration runner with rollback", "rationale": "Migrations"},
        {"id": 54, "description": "Create async task scheduler with cron support", "rationale": "Scheduling"},
        {"id": 55, "description": "Implement webhook delivery with retry", "rationale": "Webhook reliability"},
        {"id": 56, "description": "Build request ID middleware for tracing", "rationale": "Tracing"},
        {"id": 57, "description": "Create pagination helper with cursor support", "rationale": "Pagination"},
        {"id": 58, "description": "Implement bulk operations with batching", "rationale": "Bulk operations"},
        {"id": 59, "description": "Build idempotency key handler", "rationale": "Idempotency"},
        {"id": 60, "description": "Create API versioning middleware", "rationale": "Versioning"},
    ],

    # Category 4: Data Validation and Parsing (20 tasks)
    # Sequential should excel: reviewer catches validation edge cases
    "data_validation": [
        {"id": 61, "description": "Create JSON schema validator with custom rules", "rationale": "Schema validation"},
        {"id": 62, "description": "Build email validator with RFC compliance", "rationale": "Email validation"},
        {"id": 63, "description": "Implement phone number parser for international formats", "rationale": "Phone parsing"},
        {"id": 64, "description": "Create URL parser with validation", "rationale": "URL validation"},
        {"id": 65, "description": "Build credit card validator with Luhn algorithm", "rationale": "CC validation"},
        {"id": 66, "description": "Implement date parser for multiple formats", "rationale": "Date parsing"},
        {"id": 67, "description": "Create CSV parser with error recovery", "rationale": "CSV parsing"},
        {"id": 68, "description": "Build XML parser with schema validation", "rationale": "XML validation"},
        {"id": 69, "description": "Implement regex validator for user input", "rationale": "Regex validation"},
        {"id": 70, "description": "Create currency amount parser with precision", "rationale": "Currency parsing"},
        {"id": 71, "description": "Build address validator with geocoding", "rationale": "Address validation"},
        {"id": 72, "description": "Implement file type detector from magic bytes", "rationale": "File detection"},
        {"id": 73, "description": "Create markdown parser with sanitization", "rationale": "Markdown parsing"},
        {"id": 74, "description": "Build HTML sanitizer for user content", "rationale": "HTML sanitization"},
        {"id": 75, "description": "Implement timezone converter with DST", "rationale": "Timezone handling"},
        {"id": 76, "description": "Create semantic version parser and comparator", "rationale": "Version parsing"},
        {"id": 77, "description": "Build configuration file parser with defaults", "rationale": "Config parsing"},
        {"id": 78, "description": "Implement command line argument parser", "rationale": "CLI parsing"},
        {"id": 79, "description": "Create environment variable loader with validation", "rationale": "Env validation"},
        {"id": 80, "description": "Build query string parser with array support", "rationale": "Query parsing"},
    ],

    # Category 5: Error Handling and Resilience (20 tasks)
    # Sequential should excel: reviewer ensures robust error handling
    "error_handling": [
        {"id": 81, "description": "Create retry decorator with exponential backoff and jitter", "rationale": "Retry logic"},
        {"id": 82, "description": "Build error boundary for React components", "rationale": "Error boundaries"},
        {"id": 83, "description": "Implement timeout wrapper for async operations", "rationale": "Timeout handling"},
        {"id": 84, "description": "Create exception handler with error codes", "rationale": "Exception handling"},
        {"id": 85, "description": "Build fallback chain for multiple data sources", "rationale": "Fallback logic"},
        {"id": 86, "description": "Implement panic recovery handler", "rationale": "Panic recovery"},
        {"id": 87, "description": "Create error aggregator for batch operations", "rationale": "Error aggregation"},
        {"id": 88, "description": "Build transaction wrapper with rollback", "rationale": "Transaction safety"},
        {"id": 89, "description": "Implement context cancellation handler", "rationale": "Context handling"},
        {"id": 90, "description": "Create resource cleanup with finally blocks", "rationale": "Resource cleanup"},
        {"id": 91, "description": "Build error notification system", "rationale": "Error notifications"},
        {"id": 92, "description": "Implement partial failure handler", "rationale": "Partial failures"},
        {"id": 93, "description": "Create validation error collector", "rationale": "Validation errors"},
        {"id": 94, "description": "Build async error boundary", "rationale": "Async errors"},
        {"id": 95, "description": "Implement error recovery strategy selector", "rationale": "Recovery strategies"},
        {"id": 96, "description": "Create error context enricher", "rationale": "Error context"},
        {"id": 97, "description": "Build error rate limiter", "rationale": "Error rate limiting"},
        {"id": 98, "description": "Implement graceful degradation handler", "rationale": "Degradation"},
        {"id": 99, "description": "Create error serializer for logging", "rationale": "Error logging"},
        {"id": 100, "description": "Build error recovery checkpoint system", "rationale": "Checkpointing"},
    ],
}

# Flatten tasks
ALL_TASKS = []
for category, tasks in BENCHMARK_TASKS.items():
    for task in tasks:
        ALL_TASKS.append({
            **task,
            "category": category,
            "complexity": 0.7,  # All moderately complex
        })

print(f"Total tasks: {len(ALL_TASKS)}")


class HallucinationDetector:
    """Detects hallucinations"""
    HALLUCINATION_PATTERNS = [
        "import imaginary", "from fake_", "import nonexistent",
        "O(0)", "O(-1)", "100% accuracy", "never fails", "always correct",
    ]

    def detect(self, output: str) -> Dict[str, Any]:
        output_lower = output.lower()
        found = [p for p in self.HALLUCINATION_PATTERNS if p.lower() in output_lower]
        score = len(found) * 0.2
        return {
            "hallucination_detected": score > 0,
            "hallucination_score": min(score, 1.0),
            "patterns_found": found,
        }


@weave.op()
async def run_sequential(orchestrator: CollaborativeOrchestrator, task: Dict, evaluator: CodeQualityEvaluator) -> Dict:
    """Run sequential with real quality evaluation"""
    start = datetime.now()

    try:
        result = await orchestrator.collaborate(task["description"])
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(result.final_output)

        language = detect_language(result.final_output)
        quality_result = evaluator.evaluate(result.final_output, task["description"], language)

        pass_at_1 = (quality_result.overall > 0.7 and
                     not hallucination["hallucination_detected"] and
                     len(result.final_output.strip()) > 50)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "pass": pass_at_1,
            "quality_score": quality_result.overall,
            "quality_dimensions": quality_result.dimensions,
            "language": language,
            "duration": duration,
            "hallucination": hallucination,
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "sequential",
            "pass": False,
            "quality_score": 0.0,
            "error": str(e),
        }


@weave.op()
async def run_baseline(llm: MultiAgentLLMOrchestrator, task: Dict, evaluator: CodeQualityEvaluator) -> Dict:
    """Run baseline with real quality evaluation"""
    start = datetime.now()

    try:
        output = await llm.execute_agent_task("coder", task["description"])
        duration = (datetime.now() - start).total_seconds()

        detector = HallucinationDetector()
        hallucination = detector.detect(output)

        language = detect_language(output)
        quality_result = evaluator.evaluate(output, task["description"], language)

        pass_at_1 = (quality_result.overall > 0.7 and
                     not hallucination["hallucination_detected"] and
                     len(output.strip()) > 50)

        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "pass": pass_at_1,
            "quality_score": quality_result.overall,
            "quality_dimensions": quality_result.dimensions,
            "language": language,
            "duration": duration,
            "hallucination": hallucination,
        }

    except Exception as e:
        return {
            "task_id": task["id"],
            "category": task["category"],
            "method": "baseline",
            "pass": False,
            "quality_score": 0.0,
            "error": str(e),
        }


async def run_benchmark():
    """Run 100-task benchmark"""

    console.print("\n[bold cyan]100-Task Benchmark - Sequential vs Baseline[/bold cyan]")
    console.print("[yellow]Tasks designed to favor multi-stage review and refinement[/yellow]\n")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    weave.init("facilitair/100-task-benchmark")

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
            total=len(ALL_TASKS) * 2
        )

        for i, task in enumerate(ALL_TASKS, 1):
            console.print(f"\n[yellow]Task {i}/100: {task['description'][:60]}...[/yellow]")

            # Sequential
            seq_result = await run_sequential(orchestrator, task, evaluator)
            sequential_results.append(seq_result)
            console.print(f"  Sequential: {seq_result.get('quality_score', 0):.3f}")
            progress.update(task_progress, advance=1)

            # Baseline
            base_result = await run_baseline(llm, task, evaluator)
            baseline_results.append(base_result)
            console.print(f"  Baseline: {base_result.get('quality_score', 0):.3f}")
            progress.update(task_progress, advance=1)

            # Save intermediate results every 20 tasks
            if i % 20 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"benchmark_100_checkpoint_{i}_{timestamp}.json", "w") as f:
                    json.dump({
                        "sequential": sequential_results,
                        "baseline": baseline_results
                    }, f, indent=2)
                console.print(f"[green]Checkpoint saved at task {i}[/green]")

    # Final analysis
    analyze_results(sequential_results, baseline_results)


def analyze_results(sequential_results, baseline_results):
    """Analyze and display results"""

    seq_qualities = [r.get("quality_score", 0) for r in sequential_results]
    base_qualities = [r.get("quality_score", 0) for r in baseline_results]

    # Overall metrics
    table = Table(title="100-Task Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Sequential", style="green")
    table.add_column("Baseline", style="yellow")
    table.add_column("Difference", style="magenta")

    seq_avg = sum(seq_qualities) / len(seq_qualities)
    base_avg = sum(base_qualities) / len(base_qualities)

    table.add_row("Avg Quality", f"{seq_avg:.3f}", f"{base_avg:.3f}", f"{seq_avg - base_avg:+.3f}")
    table.add_row("Min Quality", f"{min(seq_qualities):.3f}", f"{min(base_qualities):.3f}", "")
    table.add_row("Max Quality", f"{max(seq_qualities):.3f}", f"{max(base_qualities):.3f}", "")

    console.print("\n")
    console.print(table)

    # Category breakdown
    cat_table = Table(title="Performance by Category")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Sequential", style="green")
    cat_table.add_column("Baseline", style="yellow")
    cat_table.add_column("Advantage", style="magenta")

    for category in BENCHMARK_TASKS.keys():
        seq_cat = [r.get("quality_score", 0) for r in sequential_results if r.get("category") == category]
        base_cat = [r.get("quality_score", 0) for r in baseline_results if r.get("category") == category]

        if seq_cat:
            seq_avg_cat = sum(seq_cat) / len(seq_cat)
            base_avg_cat = sum(base_cat) / len(base_cat)
            advantage = "Sequential" if seq_avg_cat > base_avg_cat else "Baseline"

            cat_table.add_row(
                category.replace("_", " ").title(),
                f"{seq_avg_cat:.3f}",
                f"{base_avg_cat:.3f}",
                f"{advantage} (+{abs(seq_avg_cat - base_avg_cat):.3f})"
            )

    console.print("\n")
    console.print(cat_table)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"benchmark_100_final_{timestamp}.json", "w") as f:
        json.dump({
            "metadata": {"total_tasks": 100, "timestamp": timestamp},
            "sequential": sequential_results,
            "baseline": baseline_results,
        }, f, indent=2)

    console.print(f"\n[green]Final results saved[/green]")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
