"""
Automated Fallback Tests - No Interactive Prompts
Tests fallback system in AUTO mode (can run in background)
"""

import asyncio
import yaml
import os
from dotenv import load_dotenv
import weave
from rich.console import Console

load_dotenv()
console = Console()

# Import with manual mode enabled
from agents.llm_client import MultiAgentLLMOrchestrator

# Initialize Weave
weave.init("facilitair/fallback-auto-test")

async def test_auto_fallback():
    """Test automatic fallback without user prompts"""

    console.print("\n[bold cyan]=" * 40)
    console.print("[bold cyan]AUTOMATED FALLBACK TEST - AUTO MODE[/bold cyan]")
    console.print("[bold cyan]=" * 40 + "\n")

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Configure AUTO MODE in fallback settings
    if "fallback" not in config:
        config["fallback"] = {}

    config["fallback"]["mode"] = "auto"  # No prompts, auto-escalate
    config["fallback"]["auto_rules"] = {
        "try_same_tier_first": True,
        "max_tier_jump": 2,
        "on_rate_limit": "try_next_in_tier",
        "on_api_error": "escalate_one_tier",
        "on_timeout": "try_faster_model",
        "on_invalid_model": "escalate_one_tier"
    }

    # Test 1: Invalid model -> should auto-fallback to valid model
    console.print("[bold yellow]TEST 1: Invalid Model Auto-Fallback[/bold yellow]")
    config["agents"]["coder"]["default_model"] = "invalid/nonexistent-model"
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/nonexistent-model",
        "qwen/qwen3-coder-plus",  # Should automatically use this
        "deepseek/deepseek-chat",
    ]

    console.print(f"[dim]Primary (will fail): {config['agents']['coder']['default_model']}[/dim]")
    console.print(f"[dim]Auto-fallback to: {config['agents']['coder']['candidate_models'][1]}[/dim]\n")

    # Create orchestrator with MANUAL MODE enabled + AUTO fallback
    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    task = "Write a hello world function in Python"
    console.print(f"[cyan][START] Task: {task}[/cyan]")

    try:
        result = await orchestrator.execute_agent_task("coder", task)
        console.print(f"[green][OK] SUCCESS! Auto-fallback worked![/green]")
        console.print(f"[dim]Result preview: {result[:150]}...[/dim]\n")
        return True
    except Exception as e:
        console.print(f"[red][FAIL] FAILED: {e}[/red]\n")
        return False

async def test_tier_escalation_auto():
    """Test automatic tier escalation"""

    console.print("[bold yellow]TEST 2: Tier Escalation (Auto)[/bold yellow]")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Configure AUTO MODE
    config["fallback"] = {
        "mode": "auto",
        "auto_rules": {
            "try_same_tier_first": True,
            "max_tier_jump": 2,
            "on_invalid_model": "escalate_one_tier"
        }
    }

    # Start with Tier 3 (budget), escalate to Tier 1 (premium)
    config["agents"]["coder"]["default_model"] = "invalid/tier3-budget-model"
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/tier3-budget-model",      # Tier 3 - fails
        "invalid/tier2-balanced-model",    # Tier 2 - fails
        "openai/gpt-5",                    # Tier 1 - works!
    ]

    console.print(f"[dim]Starting: Tier 3 (budget)[/dim]")
    console.print(f"[dim]Auto-escalate through: Tier 2 → Tier 1[/dim]")
    console.print(f"[dim]Expected success: {config['agents']['coder']['candidate_models'][2]}[/dim]\n")

    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    task = "Implement binary search"
    console.print(f"[cyan][START] Task: {task}[/cyan]")

    try:
        result = await orchestrator.execute_agent_task("coder", task)
        console.print(f"[green][OK] SUCCESS! Escalated to Tier 1 automatically![/green]")
        console.print(f"[dim]Result preview: {result[:150]}...[/dim]\n")
        return True
    except Exception as e:
        console.print(f"[red][FAIL] FAILED: {e}[/red]\n")
        return False

async def test_multiple_agents_fallback():
    """Test fallback across multiple agents"""

    console.print("[bold yellow]TEST 3: Multiple Agents with Fallback[/bold yellow]")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Configure AUTO MODE
    config["fallback"] = {
        "mode": "auto",
        "auto_rules": {
            "on_invalid_model": "escalate_one_tier"
        }
    }

    # Break multiple agents
    config["agents"]["architect"]["default_model"] = "invalid/architect-model"
    config["agents"]["architect"]["candidate_models"] = [
        "invalid/architect-model",
        "openai/gpt-5-pro",  # Works
    ]

    config["agents"]["coder"]["default_model"] = "invalid/coder-model"
    config["agents"]["coder"]["candidate_models"] = [
        "invalid/coder-model",
        "qwen/qwen3-coder-plus",  # Works
    ]

    console.print(f"[dim]Architect fallback: invalid → {config['agents']['architect']['candidate_models'][1]}[/dim]")
    console.print(f"[dim]Coder fallback: invalid → {config['agents']['coder']['candidate_models'][1]}[/dim]\n")

    orchestrator = MultiAgentLLMOrchestrator(config, manual_mode=True)

    # Test architect
    console.print(f"[cyan][START] Testing Architect agent...[/cyan]")
    try:
        result1 = await orchestrator.execute_agent_task("architect", "Design a REST API")
        console.print(f"[green][OK] Architect fallback worked![/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Architect failed: {e}[/red]")
        return False

    # Test coder
    console.print(f"[cyan][START] Testing Coder agent...[/cyan]")
    try:
        result2 = await orchestrator.execute_agent_task("coder", "Implement the API")
        console.print(f"[green][OK] Coder fallback worked![/green]\n")
        return True
    except Exception as e:
        console.print(f"[red][FAIL] Coder failed: {e}[/red]\n")
        return False

async def main():
    """Run all fallback tests"""

    console.print("\n[bold magenta]=" * 40)
    console.print("[bold magenta]STARTING AUTOMATED FALLBACK TESTS[/bold magenta]")
    console.print("[bold magenta]=" * 40 + "\n")

    results = []

    # Test 1: Basic invalid model fallback
    result1 = await test_auto_fallback()
    results.append(("Invalid Model Fallback", result1))

    # Test 2: Tier escalation
    result2 = await test_tier_escalation_auto()
    results.append(("Tier Escalation", result2))

    # Test 3: Multiple agents
    result3 = await test_multiple_agents_fallback()
    results.append(("Multiple Agents", result3))

    # Summary
    console.print("\n[bold magenta]=" * 40)
    console.print("[bold magenta]TEST RESULTS SUMMARY[/bold magenta]")
    console.print("[bold magenta]=" * 40 + "\n")

    for test_name, passed in results:
        status = "[green][OK] PASSED[/green]" if passed else "[red][FAIL] FAILED[/red]"
        console.print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    console.print(f"\n[bold]Total: {total_passed}/{len(results)} tests passed[/bold]\n")

if __name__ == "__main__":
    asyncio.run(main())
