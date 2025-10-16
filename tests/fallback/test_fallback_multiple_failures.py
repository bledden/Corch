"""
Test 2: Multiple Consecutive Failures
Forces multiple model failures to test cascading fallback
"""

import asyncio
import yaml
import os
from dotenv import load_dotenv
import weave

load_dotenv()

# Import with manual mode enabled
from agents.llm_client import MultiAgentLLMOrchestrator

# Initialize Weave
weave.init("facilitair/fallback-test-multiple-failures")

async def test_multiple_failures():
    """Test fallback through multiple consecutive failures"""

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # FORCE MULTIPLE INVALID MODELS
    config["agents"]["coder"]["default_model"] = "invalid/first-bad-model"

    # Create chain of failures -> finally a working model
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/first-bad-model",      # Fail #1
        "invalid/second-bad-model",     # Fail #2 (if user retries)
        "invalid/third-bad-model",      # Fail #3 (if user retries again)
        "qwen/qwen3-coder-plus",        # Should work!
        "deepseek/deepseek-chat",       # Backup
    ]

    print("=" * 80)
    print("TEST 2: MULTIPLE CONSECUTIVE FAILURES")
    print("=" * 80)
    print(f"Primary model: {config['agents']['coder']['default_model']}")
    print(f"Expected to fail 3 times before success")
    print(f"Failure chain: {config['agents']['coder']['candidate_models'][:3]}")
    print(f"Working fallback: {config['agents']['coder']['candidate_models'][3]}")
    print("=" * 80)
    print()

    # Create orchestrator with MANUAL MODE enabled
    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    # Try to execute a simple task
    task = "Create a function to add two numbers"

    print(f"\n[START] Executing task: {task}")
    print("[WARNING]  This will require multiple fallback attempts\n")

    try:
        result = await orchestrator.execute_agent_task("coder", task)
        print(f"\n[OK] SUCCESS after multiple retries!")
        print(f"Result: {result[:200]}...\n")
    except Exception as e:
        print(f"\n[FAIL] FAILED after all attempts: {e}\n")

    print("=" * 80)
    print("TEST 2 COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_multiple_failures())
