#!/usr/bin/env python3
"""
Test 2: Weave connection and basic tracking
"""
import weave
import asyncio
from dotenv import load_dotenv

# Load environment variables (for WANDB_API_KEY)
load_dotenv()

async def test_weave_basic():
    """Test basic Weave initialization and tracking"""
    print("Test 2: Weave connection and basic tracking")
    print("=" * 60)

    try:
        # Initialize Weave
        print("\n1. Initializing Weave...")
        weave.init("facilitair/weave-basic-test")
        print("[OK] Weave initialized successfully")

        # Test basic weave.op decorator
        @weave.op()
        def simple_sync_function(x: int) -> int:
            return x * 2

        print("\n2. Testing sync function with @weave.op()...")
        result = simple_sync_function(5)
        print(f"[OK] Sync function result: {result}")

        # Test async weave.op decorator
        @weave.op()
        async def simple_async_function(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 3

        print("\n3. Testing async function with @weave.op()...")
        result = await simple_async_function(5)
        print(f"[OK] Async function result: {result}")

        print("\n" + "=" * 60)
        print("[OK] Test 2 PASSED: Weave connection works!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_weave_basic())
    exit(0 if success else 1)
