#!/usr/bin/env python3
"""
Test 3: Single LLM call WITH Weave decorator
"""
import weave
import asyncio
import litellm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_llm_with_weave():
    """Test LLM call with Weave tracking"""
    print("Test 3: Single LLM call WITH Weave decorator")
    print("=" * 60)

    try:
        # Initialize Weave
        print("\n1. Initializing Weave...")
        weave.init("facilitair/llm-weave-test")
        print("✅ Weave initialized")

        # Configure LiteLLM
        litellm.api_key = os.getenv("OPENROUTER_API_KEY")
        litellm.api_base = "https://openrouter.ai/api/v1"

        # Test LLM call WITH @weave.op() decorator
        @weave.op()
        async def call_llm_with_weave(prompt: str) -> str:
            """LLM call wrapped in Weave tracking"""
            response = await litellm.acompletion(
                model="openrouter/qwen/qwen3-coder-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content

        print("\n2. Calling LLM with @weave.op() decorator...")
        result = await call_llm_with_weave("Write a function to add two numbers")

        print(f"✅ LLM call successful!")
        print(f"Response preview: {result[:150]}...")

        print("\n" + "=" * 60)
        print("✅ Test 3 PASSED: LLM with Weave works!")
        return True

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_with_weave())
    exit(0 if success else 1)
