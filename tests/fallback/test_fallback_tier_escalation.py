"""
Test 3: Tier Escalation Test
Forces failure with Tier 3 model to test escalation to higher tiers
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
weave.init("facilitair/fallback-test-tier-escalation")

async def test_tier_escalation():
    """Test fallback escalation from Tier 3 -> Tier 2 -> Tier 1"""

    # Load config
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Start with invalid Tier 3 model (budget)
    config["agents"]["coder"]["default_model"] = "invalid/budget-model"

    # Set up tier escalation: Tier 3 (fail) -> Tier 2 (fail) -> Tier 1 (work)
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/budget-model",                      # Tier 3 - Will fail
        "invalid/mid-tier-model",                    # Tier 2 - Will fail
        "openai/gpt-5",                              # Tier 1 - Should work!
        "anthropic/claude-sonnet-4.5",               # Tier 1 backup
    ]

    print("=" * 80)
    print("TEST 3: TIER ESCALATION (Budget → Premium)")
    print("=" * 80)
    print(f"Starting with: Tier 3 (budget) - {config['agents']['coder']['default_model']}")
    print(f"Expected failure, escalate to Tier 2")
    print(f"Tier 2 also fails, escalate to Tier 1 (premium)")
    print(f"Tier 1 success: {config['agents']['coder']['candidate_models'][2]}")
    print("=" * 80)
    print()

    # Set fallback config to demonstrate tier escalation
    if "fallback" not in config:
        config["fallback"] = {}

    config["fallback"]["mode"] = "interactive"  # User can see tier escalation

    print("[IDEA] This test shows tier escalation: Budget → Balanced → Premium")
    print("[IDEA] User will be prompted to approve moving to higher-cost tier\n")

    # Create orchestrator with MANUAL MODE enabled
    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    # Try to execute a task
    task = "Implement a binary search algorithm"

    print(f"\n[START] Executing task: {task}")
    print("[WARNING]  This should escalate from Tier 3 → Tier 2 → Tier 1\n")

    try:
        result = await orchestrator.execute_agent_task("coder", task)
        print(f"\n[OK] SUCCESS with Tier 1 (premium) model!")
        print(f"Result: {result[:200]}...\n")
    except Exception as e:
        print(f"\n[FAIL] FAILED even after tier escalation: {e}\n")

    print("=" * 80)
    print("TEST 3 COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_tier_escalation())
