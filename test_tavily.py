#!/usr/bin/env python3
"""Quick Tavily integration test"""

import os
import asyncio
from tavily import TavilyClient

async def test_tavily():
    api_key = os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        print("❌ TAVILY_API_KEY not set!")
        print("Get your key from: https://tavily.com")
        return False
    
    print(f"✅ TAVILY_API_KEY found: {api_key[:10]}...")
    
    try:
        client = TavilyClient(api_key=api_key)
        
        # Quick test search
        print("\n🔍 Testing Tavily search...")
        result = client.search("Python sequential collaboration best practices", max_results=2)
        
        print(f"✅ Search successful! Found {len(result.get('results', []))} results")
        
        if result.get('results'):
            print(f"\nFirst result: {result['results'][0].get('title', 'N/A')}")
            print(f"URL: {result['results'][0].get('url', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tavily error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tavily())
    exit(0 if success else 1)
