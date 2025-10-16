#!/usr/bin/env python3
"""
Direct LLM test WITHOUT Weave to isolate the hanging issue
"""
import asyncio
import os
import litellm
from dotenv import load_dotenv

load_dotenv()

async def test_direct_llm_call():
    """Test LiteLLM → OpenRouter directly without Weave"""
    print("Testing direct LLM call without Weave...")
    print(f"OpenRouter API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}...")

    # Configure LiteLLM
    litellm.api_key = os.getenv("OPENROUTER_API_KEY")
    litellm.api_base = "https://openrouter.ai/api/v1"

    try:
        print("\nCalling OpenRouter with model: openrouter/qwen/qwen3-coder-plus")
        response = await litellm.acompletion(
            model="openrouter/qwen/qwen3-coder-plus",
            messages=[{"role": "user", "content": "Write a function to add two numbers"}],
            temperature=0.7,
            max_tokens=500,
        )

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        print(f"\n[OK] SUCCESS!")
        print(f"Tokens used: {tokens}")
        print(f"Response preview: {content[:200]}...")
        return True

    except Exception as e:
        print(f"\n[FAIL] FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_llm_call())
    exit(0 if success else 1)
