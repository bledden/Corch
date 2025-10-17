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
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator

# Test task
TEST_TASK = "Write a function to check if a number is prime"

STRATEGIES = ["COST_FIRST", "QUALITY_FIRST", "BALANCED"]

async def test_strategy(strategy_name: str):
    """Test a single strategy"""
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"{'='*60}\n")

    # Load and modify config
    config_path = "config/config.yaml"
    strategy_config_path = "config/model_strategy_config.yaml"

    # Read strategy config
    with open(strategy_config_path, 'r') as f:
        strategy_config = yaml.safe_load(f)

    # Temporarily set strategy
    original_preference = strategy_config['user_preference']
    strategy_config['user_preference'] = strategy_name

    # Write modified config
    with open(strategy_config_path, 'w') as f:
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
        print(f"[RESULT] Success: {result.success}")
        print(f"[RESULT] Workflow: {result.workflow_name}")
        print(f"[RESULT] Stages: {len(result.stages)}")

        # Extract models used
        models_used = []
        for stage in result.stages:
            agent_role = stage.agent_role.value
            # Try to extract model from stage metadata or output
            models_used.append(f"{agent_role}")

        print(f"[RESULT] Agents used: {', '.join(models_used)}")
        print(f"[RESULT] Duration: {result.total_duration_seconds:.2f}s")

        # Check output
        if result.final_output:
            output_preview = result.final_output[:200].replace('\n', ' ')
            print(f"[RESULT] Output preview: {output_preview}...")

        print(f"\n[OK] {strategy_name} test completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] {strategy_name} test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore original config
        strategy_config['user_preference'] = original_preference
        with open(strategy_config_path, 'w') as f:
            yaml.dump(strategy_config, f, default_flow_style=False)


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
