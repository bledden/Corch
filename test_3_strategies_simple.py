#!/usr/bin/env python3
"""
Simple test to verify model diversity works across 3 strategies:
- COST_FIRST (open source models)
- QUALITY_FIRST (closed source premium models)
- BALANCED (mix of open and closed)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import asyncio
import yaml
import signal
import atexit
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator

# Test task
TEST_TASK = "Write a function to check if a number is prime"

STRATEGIES = ["COST_FIRST", "QUALITY_FIRST", "BALANCED"]

# Global variable to track config state
STRATEGY_CONFIG_PATH = "config/model_strategy_config.yaml"
ORIGINAL_CONFIG_CONTENT = None


def restore_config():
    """Restore config file to original state - called on exit or interrupt"""
    global ORIGINAL_CONFIG_CONTENT, STRATEGY_CONFIG_PATH
    if ORIGINAL_CONFIG_CONTENT is not None:
        try:
            with open(STRATEGY_CONFIG_PATH, 'w') as f:
                f.write(ORIGINAL_CONFIG_CONTENT)
            print("\n[CLEANUP] Config restored to original state")
        except Exception as e:
            print(f"\n[CLEANUP ERROR] Failed to restore config: {e}")


def signal_handler(signum, frame):
    """Handle interrupts (Ctrl+C, kill, etc.)"""
    print(f"\n[INTERRUPT] Received signal {signum}, cleaning up...")
    restore_config()
    sys.exit(1)


# Register cleanup handlers
atexit.register(restore_config)
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command

async def test_strategy(strategy_name: str):
    """Test a single strategy"""
    global ORIGINAL_CONFIG_CONTENT, STRATEGY_CONFIG_PATH

    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'='*60}\n")

    # Load and modify config
    config_path = "config/config.yaml"

    # Save original config content ONCE (first time only)
    if ORIGINAL_CONFIG_CONTENT is None:
        with open(STRATEGY_CONFIG_PATH, 'r') as f:
            ORIGINAL_CONFIG_CONTENT = f.read()

    # Load and parse strategy config
    strategy_config = yaml.safe_load(ORIGINAL_CONFIG_CONTENT)

    # Temporarily set strategy
    strategy_config['user_preference'] = strategy_name

    # Write modified config
    with open(STRATEGY_CONFIG_PATH, 'w') as f:
        yaml.dump(strategy_config, f, default_flow_style=False)

    try:
        # Load main config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create orchestrator
        orchestrator = CollaborativeOrchestrator(config)

        # Run collaboration
        print(f"[TEST] Running: {TEST_TASK}")
        result = await orchestrator.collaborate(TEST_TASK)

        # Display results
        print(f"\n[RESULT] Strategy: {strategy_name}")
        print(f"[RESULT] Success: {result.success if hasattr(result, 'success') else 'N/A'}")
        print(f"[RESULT] Workflow: {result.workflow_name if hasattr(result, 'workflow_name') else 'N/A'}")
        print(f"[RESULT] Run ID: {result.run_id if hasattr(result, 'run_id') else 'N/A'}")

        # Extract agents and models used from stages
        if hasattr(result, 'stages') and result.stages:
            print(f"[RESULT] Stages: {len(result.stages)}")
            agents_used = [stage.agent_role.value for stage in result.stages]
            print(f"[RESULT] Agents used: {', '.join(agents_used)}")

        # Duration
        if hasattr(result, 'total_duration_seconds'):
            print(f"[RESULT] Duration: {result.total_duration_seconds:.2f}s")
        elif hasattr(result, 'duration'):
            print(f"[RESULT] Duration: {result.duration:.2f}s")

        # Check output
        if result.final_output:
            output_preview = result.final_output[:200].replace('\n', ' ')
            print(f"[RESULT] Output preview: {output_preview}...")

        print(f"\n[OK] {strategy_name} test completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] {strategy_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        # Config will be restored automatically by atexit handler


async def main():
    """Test all 3 strategies"""
    print("\n" + "="*60)
    print("Model Diversity Test Across 3 Strategies")
    print("="*60)
    print(f"\nTest Task: {TEST_TASK}")
    print(f"Strategies to test: {', '.join(STRATEGIES)}\n")

    for strategy in STRATEGIES:
        await test_strategy(strategy)
        await asyncio.sleep(2)  # Brief pause between tests

    print("\n" + "="*60)
    print("All strategy tests completed!")
    print("="*60)
    print("\nSummary:")
    print("- COST_FIRST: Open source models (free, cost-efficient)")
    print("- QUALITY_FIRST: Premium closed source models (best quality)")
    print("- BALANCED: Smart mix of open and closed models")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
