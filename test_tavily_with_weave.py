#!/usr/bin/env python3
"""
Test 6: Tavily web search WITH Weave tracking
"""
import weave
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_tavily_with_weave():
    """Test Tavily web search with Weave tracking"""
    print("Test 6: Tavily web search WITH Weave tracking")
    print("=" * 60)

    try:
        # Initialize Weave
        print("\n1. Initializing Weave...")
        weave.init("facilitair/tavily-weave-test")
        print("✅ Weave initialized")

        # Check for Tavily API key
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key or tavily_key.startswith("tvly-"):
            print(f"\n2. Checking Tavily API key...")
            print(f"✅ Tavily API key found: {tavily_key[:15]}...")
        else:
            print("⚠️  Tavily API key not found or invalid")
            return False

        # Test Tavily search WITH @weave.op() decorator
        from web_search_router import WebSearchRouter

        print("\n3. Creating WebSearchRouter...")
        router = WebSearchRouter()

        @weave.op()
        async def search_with_weave(query: str) -> dict:
            """Web search wrapped in Weave tracking"""
            result = await router.search(
                query=query,
                strategy="balanced",
                max_results=3
            )
            return result

        print("\n4. Executing web search with @weave.op() decorator...")
        result = await search_with_weave("What is the latest Python version released in 2024?")

        print(f"✅ Web search successful!")
        print(f"Search executed: {result.get('search_executed', False)}")
        print(f"Search method: {result.get('search_method', 'N/A')}")
        print(f"Results count: {len(result.get('results', []))}")
        if result.get('results'):
            print(f"First result preview: {result['results'][0][:100]}...")

        print("\n" + "=" * 60)
        print("✅ Test 6 PASSED: Tavily with Weave works!")
        return True

    except Exception as e:
        print(f"\n❌ Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tavily_with_weave())
    exit(0 if success else 1)
