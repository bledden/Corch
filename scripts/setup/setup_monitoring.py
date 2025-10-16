#!/usr/bin/env python3
"""
Setup monitoring for currently running tests.
This script reads bash outputs and writes them to log files for the monitor to display.

Usage:
    python3 setup_monitoring.py fe6e5e 8c42af d9ac2f

This will continuously fetch outputs from the specified bash IDs and write them to
/tmp/facilitair_logs/ so that monitor.py can display them in real-time.

In another terminal, run:
    python3 monitor.py fe6e5e 8c42af d9ac2f
"""

import os
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("/tmp/facilitair_logs")


def get_bash_output(bash_id: str) -> str:
    """
    Simulate getting bash output.
    In the actual environment, this would use Claude Code's BashOutput tool.
    """
    # This is a placeholder - in the real implementation,
    # we'd need to integrate with Claude Code's tool system
    return f"Test {bash_id} is running..."


def write_log(bash_id: str, content: str):
    """Write content to the log file for a test"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"test_{bash_id}.log"

    with open(log_file, 'w') as f:
        f.write(content)


def setup_monitoring_for_tests(bash_ids: list[str]):
    """Continuously fetch outputs and write to logs"""
    print(f"Setting up monitoring for: {', '.join(bash_ids)}")
    print(f"Log directory: {LOG_DIR}")
    print("\nRun this in another terminal to see the dashboard:")
    print(f"  python3 monitor.py {' '.join(bash_ids)}\n")
    print("Press Ctrl+C to stop\n")

    iteration = 0
    try:
        while True:
            for bash_id in bash_ids:
                # Fetch latest output
                output = get_bash_output(bash_id)

                # Write to log file
                write_log(bash_id, output)

            iteration += 1
            if iteration % 10 == 0:
                print(f"Updated logs (iteration {iteration})")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nMonitoring setup stopped")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 setup_monitoring.py <bash_id1> [bash_id2] [bash_id3]")
        print("Example: python3 setup_monitoring.py fe6e5e 8c42af d9ac2f")
        sys.exit(1)

    bash_ids = sys.argv[1:]
    setup_monitoring_for_tests(bash_ids)


if __name__ == "__main__":
    main()
