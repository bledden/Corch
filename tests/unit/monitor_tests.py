#!/usr/bin/env python3
"""
Real-time CLI dashboard for monitoring running test processes.
Shows live progress, task completion, and results for all background tests.
"""

import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()


@dataclass
class TestStatus:
    """Status of a running test"""
    bash_id: str
    strategy: str
    current_task: int
    total_tasks: int
    status: str  # running, completed, killed, error
    sequential_pass: Optional[int] = None
    sequential_total: Optional[int] = None
    baseline_pass: Optional[int] = None
    baseline_total: Optional[int] = None
    last_update: datetime = None
    error_message: Optional[str] = None


class TestMonitor:
    """Monitor and display status of background test processes"""

    def __init__(self):
        self.tests: Dict[str, TestStatus] = {}
        self.console = Console()

    def get_running_tests(self) -> List[str]:
        """Get list of running test bash IDs"""
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Look for python test processes
            bash_ids = []
            for line in result.stdout.split('\n'):
                if 'run_smoke_test.py' in line or 'run_10_task' in line or 'run_comprehensive' in line:
                    # Extract bash ID from process info if available
                    # For now, we'll use predefined IDs
                    pass

            # Return known test IDs (from system reminders)
            return ['fe6e5e', '8c42af', 'd9ac2f']
        except Exception as e:
            return []

    def parse_test_output(self, bash_id: str, output: str) -> TestStatus:
        """Parse test output to extract current status"""

        # Initialize default status
        status = TestStatus(
            bash_id=bash_id,
            strategy="UNKNOWN",
            current_task=0,
            total_tasks=10,
            status="running",
            last_update=datetime.now()
        )

        # Extract strategy
        strategy_match = re.search(r'Model selection strategy: (\w+)', output)
        if strategy_match:
            status.strategy = strategy_match.group(1)

        # Extract current task number
        task_matches = re.findall(r'Task (\d+):', output)
        if task_matches:
            status.current_task = int(task_matches[-1])

        # Check for completion
        if 'SEQUENTIAL RESULTS' in output:
            # Extract sequential results
            seq_match = re.search(r'Pass rate: (\d+)/(\d+)', output)
            if seq_match:
                status.sequential_pass = int(seq_match.group(1))
                status.sequential_total = int(seq_match.group(2))

        if 'BASELINE RESULTS' in output:
            # Extract baseline results
            base_match = re.search(r'Pass rate: (\d+)/(\d+)', output.split('BASELINE RESULTS')[1])
            if base_match:
                status.baseline_pass = int(base_match.group(1))
                status.baseline_total = int(base_match.group(2))

            # If both results present, test is complete
            if status.sequential_pass is not None:
                status.status = "completed"

        # Check for errors
        if 'ERROR' in output or 'FAILED' in output:
            error_match = re.search(r'\[ERROR\].*?:(.+?)(?:\n|$)', output)
            if error_match:
                status.error_message = error_match.group(1).strip()[:100]

        return status

    def fetch_bash_output(self, bash_id: str) -> Optional[str]:
        """Fetch output from a background bash process"""
        try:
            # Use the BashOutput tool via subprocess
            # This is a simplified version - in practice, you'd need to interface with Claude Code's tools
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=2
            )

            # For now, return placeholder
            # In practice, this would call the actual BashOutput tool
            return None
        except Exception:
            return None

    def create_dashboard(self) -> Table:
        """Create the dashboard table"""
        table = Table(
            title="[START] Facilitair Test Monitor Dashboard",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            title_style="bold cyan"
        )

        table.add_column("Test ID", style="cyan", width=8)
        table.add_column("Strategy", style="yellow", width=10)
        table.add_column("Progress", style="green", width=12)
        table.add_column("Status", style="blue", width=10)
        table.add_column("Sequential", style="magenta", width=12)
        table.add_column("Baseline", style="magenta", width=12)
        table.add_column("Last Update", style="dim", width=10)

        for bash_id, test in sorted(self.tests.items()):
            # Progress bar
            progress_pct = (test.current_task / test.total_tasks * 100) if test.total_tasks > 0 else 0
            progress_bar = self._create_progress_bar(progress_pct, test.current_task, test.total_tasks)

            # Status indicator
            status_emoji = {
                "running": "[RUNNING]",
                "completed": "[OK]",
                "killed": "",
                "error": "[FAIL]"
            }.get(test.status, "")

            status_text = f"{status_emoji} {test.status.upper()}"

            # Sequential results
            if test.sequential_pass is not None and test.sequential_total is not None:
                seq_pct = (test.sequential_pass / test.sequential_total * 100) if test.sequential_total > 0 else 0
                seq_text = f"{test.sequential_pass}/{test.sequential_total} ({seq_pct:.0f}%)"
            else:
                seq_text = "Pending..."

            # Baseline results
            if test.baseline_pass is not None and test.baseline_total is not None:
                base_pct = (test.baseline_pass / test.baseline_total * 100) if test.baseline_total > 0 else 0
                base_text = f"{test.baseline_pass}/{test.baseline_total} ({base_pct:.0f}%)"
            else:
                base_text = "Pending..."

            # Last update time
            time_ago = "N/A"
            if test.last_update:
                delta = (datetime.now() - test.last_update).total_seconds()
                if delta < 60:
                    time_ago = f"{int(delta)}s ago"
                else:
                    time_ago = f"{int(delta/60)}m ago"

            table.add_row(
                bash_id[:6],
                test.strategy,
                progress_bar,
                status_text,
                seq_text,
                base_text,
                time_ago
            )

        return table

    def _create_progress_bar(self, percentage: float, current: int, total: int) -> str:
        """Create a text progress bar"""
        width = 10
        filled = int(width * percentage / 100)
        bar = "" * filled + "" * (width - filled)
        return f"{bar} {current}/{total}"

    def update_test_status(self, bash_id: str, output: str):
        """Update status for a specific test"""
        status = self.parse_test_output(bash_id, output)
        self.tests[bash_id] = status

    def run(self, test_ids: List[str], refresh_interval: float = 2.0):
        """Run the live dashboard"""

        # Initialize tests
        for test_id in test_ids:
            self.tests[test_id] = TestStatus(
                bash_id=test_id,
                strategy="Loading...",
                current_task=0,
                total_tasks=10,
                status="running",
                last_update=datetime.now()
            )

        with Live(self.create_dashboard(), refresh_per_second=1, console=console) as live:
            try:
                while True:
                    # Update dashboard
                    dashboard = self.create_dashboard()

                    # Add legend
                    legend = Text()
                    legend.append("\n\nLegend: ", style="bold")
                    legend.append("[RUNNING] Running  ", style="green")
                    legend.append("[OK] Completed  ", style="blue")
                    legend.append("[FAIL] Error  ", style="red")
                    legend.append(" Killed  ", style="yellow")
                    legend.append("\n\nPress Ctrl+C to exit", style="dim italic")

                    # Combine table and legend
                    full_display = Panel(
                        Text.assemble(dashboard, legend),
                        border_style="cyan",
                        padding=(1, 2)
                    )

                    live.update(dashboard)

                    # Check if all tests completed
                    all_done = all(
                        test.status in ["completed", "killed", "error"]
                        for test in self.tests.values()
                    )

                    if all_done:
                        console.print("\n[bold green][OK] All tests completed![/bold green]")
                        break

                    time.sleep(refresh_interval)

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitor stopped by user[/yellow]")


def main():
    """Main entry point"""
    console.print("[bold cyan][START] Facilitair Test Monitor[/bold cyan]")
    console.print("[dim]Starting dashboard...[/dim]\n")

    # Get test IDs to monitor (from command line or default to recent tests)
    import sys
    if len(sys.argv) > 1:
        test_ids = sys.argv[1:]
    else:
        # Default to the three most recent tests
        test_ids = ['fe6e5e', '8c42af', 'd9ac2f']

    console.print(f"[blue]Monitoring tests:[/blue] {', '.join(test_ids)}\n")

    monitor = TestMonitor()

    try:
        monitor.run(test_ids)
    except Exception as e:
        console.print(f"[red]Error running monitor: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
