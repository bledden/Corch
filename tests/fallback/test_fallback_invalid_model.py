"""
Test 1: Invalid Model ID Fallback
Forces failure with an invalid model ID to test fallback system
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
weave.init("facilitair/fallback-test-invalid-model")

async def test_invalid_model_fallback():
    """Test fallback when model ID is invalid"""

    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # FORCE INVALID MODEL in coder config
    config["agents"]["coder"]["default_model"] = "invalid/nonexistent-model-xyz"

    # Keep valid fallback models
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/nonexistent-model-xyz",  # Will fail
        "qwen/qwen3-coder-plus",  # Should work as fallback
        "deepseek/deepseek-chat",  # Backup
    ]

    print("=" * 80)
    print("TEST 1: INVALID MODEL ID FALLBACK")
    print("=" * 80)
    print(f"Primary model: {config['agents']['coder']['default_model']}")
    print(f"Expected to fail with: 'invalid model ID' error")
    print(f"Fallback models available: {config['agents']['coder']['candidate_models'][1:]}")
    print("=" * 80)
    print()

    # Create orchestrator with MANUAL MODE enabled
    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    # Try to execute a simple task
    task = "Write a simple hello world function in Python"

    print(f"\n[START] Executing task: {task}")
    print("[WARNING]  This should trigger fallback when invalid model fails\n")

    try:
        result = await orchestrator.execute_agent_task("coder", task)
        print(f"\n[OK] SUCCESS! Task completed with result:")
        print(f"{result[:200]}...\n")
    except Exception as e:
        print(f"\n[FAIL] FAILED: {e}\n")

    print("=" * 80)
    print("TEST 1 COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_invalid_model_fallback())
