#!/usr/bin/env python3
"""Test streaming with a fresh stream"""

import httpx
import asyncio
import json

async def main():
    base_url = "http://localhost:8000"

    # Step 1: Create a new stream
    print("Creating new stream...")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/api/stream/task",
            headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
            json={"task": "Write a hello world function", "stream": True}
        )
        data = response.json()
        stream_id = data["stream_id"]
        print(f"Stream created: {stream_id}\n")

    # Step 2: Connect to SSE immediately
    print("Connecting to SSE...")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", f"{base_url}/api/stream/events/{stream_id}",
                                 headers={"X-API-Key": "test-key"}) as response:
            print(f"Status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('content-type')}")
            print("=" * 70)
            print("\nReceiving events:\n")

            event_count = 0
            async for line in response.aiter_lines():
                event_count += 1
                print(f"[{event_count:3d}] {line}")

                # Stop after seeing task_completed or 100 events
                if "task_completed" in line or event_count > 100:
                    print(f"\n... stopping (saw {event_count} lines)")
                    break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
