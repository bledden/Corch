#!/usr/bin/env python3
"""Direct SSE test - minimal client to debug event delivery"""

import httpx
import asyncio

async def test_sse():
    stream_id = "5cd920c5-d455-4573-b509-d8e5be303432"  # From your recent test
    url = f"http://localhost:8000/api/stream/events/{stream_id}"

    print(f"Connecting to: {url}")
    print("Waiting for events... (Ctrl+C to stop)\n")

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", url, headers={"X-API-Key": "test-key"}) as response:
            print(f"Response status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}\n")

            count = 0
            async for line in response.aiter_lines():
                count += 1
                print(f"[{count}] {repr(line)}")

                if count > 50:  # Stop after 50 lines
                    print("\n... stopping after 50 lines")
                    break

if __name__ == "__main__":
    asyncio.run(test_sse())
