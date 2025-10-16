#!/usr/bin/env python3
"""Quick Tavily integration test"""

import os
from tavily import TavilyClient

def test_tavily():
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        print("[FAIL] TAVILY_API_KEY not set!")
        print("Set with: export TAVILY_API_KEY='tvly-your-key'")
        return False

    print(f"[OK] TAVILY_API_KEY found: {api_key[:10]}...")

    try:
        client = TavilyClient(api_key=api_key)

        # Quick test search
        print("\nReviewer Testing Tavily search...")
        result = client.search("WeaveHacks 2 AI collaboration", max_results=2)

        print(f"[OK] Tavily search successful! Found {len(result.get('results', []))} results")

        if result.get('results'):
            print(f"\n First result:")
            print(f"   Title: {result['results'][0].get('title', 'N/A')}")
            print(f"   URL: {result['results'][0].get('url', 'N/A')}")
            print(f"   Content: {result['results'][0].get('content', 'N/A')[:100]}...")

        print("\n[OK] TAVILY INTEGRATION WORKING!")
        return True

    except Exception as e:
        print(f"[FAIL] Tavily error: {e}")
        return False

if __name__ == "__main__":
    success = test_tavily()
    exit(0 if success else 1)
