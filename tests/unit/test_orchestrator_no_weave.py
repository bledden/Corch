#!/usr/bin/env python3
"""
Test 4: Collaborative orchestrator WITHOUT Weave
"""
import asyncio
from dotenv import load_dotenv
import yaml

load_dotenv()

async def test_orchestrator_no_weave():
    """Test orchestrator without Weave initialization"""
    print("Test 4: Collaborative orchestrator WITHOUT Weave")
    print("=" * 60)

    try:
        print("\n1. Loading config...")
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        print("\n2. Testing LLM client directly (no orchestrator, no weave)...")
        from agents.llm_client import LLMClient

        llm_client = LLMClient(config=config)

        print("\n3. Calling LLM via LLMClient...")
        # This will still have @weave.op() but no weave.init()
        # So it should work fine
        response = await llm_client.execute_llm(
            agent_id="coder",
            task="Write a simple function to add two numbers",
            model="qwen/qwen3-coder-plus",
            temperature=0.7,
            max_tokens=200
        )

        print(f"[OK] LLMClient call successful!")
        print(f"Response preview: {response.content[:100]}...")

        print("\n" + "=" * 60)
        print("[OK] Test 4 PASSED: LLMClient works without weave.init!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_no_weave())
    exit(0 if success else 1)
