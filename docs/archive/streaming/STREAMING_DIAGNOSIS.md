# Streaming Diagnosis - The Real Problem

## What's Happening

1. **Client connects** → Creates stream → Gets stream_id
2. **Background task starts** → Calls `orchestrator.collaborate(task)`
3. **Orchestrator runs** → Takes 4-5 minutes to complete all stages
4. **Client waits** → Sees nothing, times out, disconnects
5. **Orchestrator finishes** → Tries to stream results but client is gone

## The Core Issue

**Post-generation streaming doesn't work when generation takes 4-5 minutes.**

The orchestrator blocks for the entire workflow:
- Stage 1 (Architecture): 25 seconds
- Stage 2 (Implementation): 31 seconds
- Stage 3 (Review): 7 seconds
- Stage 4 (Refinement iter 1): ~50 seconds
- Stage 4 (Refinement iter 2): ~50 seconds
- Stage 4 (Refinement iter 3): ~50 seconds
- Stage 5 (Documentation): ?? seconds

**Total: 4-5+ minutes** with NO output to client

## Why Client Disconnects

SSE connections with no data will:
1. Browser/client timeout (typically 30-60 seconds)
2. User gives up waiting
3. Intermediate proxies drop connection

We send heartbeats every 15s, but that's not enough - the client needs to see PROGRESS.

## Solutions

### Option 1: Send Progress Events DURING Orchestration (Quick)

Modify `execute_streaming_task` to send progress updates while orchestrator runs:

```python
async def execute_streaming_task(...):
    # Start progress task
    progress_task = asyncio.create_task(send_progress_updates(stream_id))

    # Execute orchestrator (blocks for 4-5 min)
    result = await stream_collaboration(...)

    # Stop progress
    progress_task.cancel()

    # Stream results
    ...

async def send_progress_updates(stream_id):
    """Send periodic 'working...' messages"""
    count = 0
    while True:
        await asyncio.sleep(5)
        count += 1
        await manager.send_event(
            stream_id,
            builder.system_message(f"Processing... ({count * 5}s elapsed)")
        )
```

### Option 2: Hook Into Sequential Orchestrator (Proper)

Modify `sequential_orchestrator.py` to accept a callback for real-time stage updates:

```python
# In sequential_orchestrator.py
async def execute_workflow(
    self,
    task: str,
    on_stage_start: Optional[Callable] = None,
    on_stage_complete: Optional[Callable] = None
):
    for stage in stages:
        if on_stage_start:
            await on_stage_start(stage)

        # Execute stage
        result = await self._execute_stage(stage)

        if on_stage_complete:
            await on_stage_complete(stage, result)
```

Then in streaming.py:
```python
async def on_stage_start(stage):
    await manager.send_event(stream_id, builder.chunk_started(...))

async def on_stage_complete(stage, result):
    # Stream the result immediately
    await stream_output(result.output)
    await manager.send_event(stream_id, builder.chunk_completed(...))

result = await orchestrator.execute_workflow_with_callbacks(
    task=task,
    on_stage_start=on_stage_start,
    on_stage_complete=on_stage_complete
)
```

### Option 3: True Token Streaming (Complex)

Hook into LLM client's streaming API to get tokens as they're generated.

## Recommended Path

**Immediate (5 minutes)**: Option 1 - Add progress messages
**Short-term (30 minutes)**: Option 2 - Hook into orchestrator stages
**Long-term (2+ hours)**: Option 3 - True LLM token streaming

## Current State

- [OK] Real orchestrator working
- [OK] Real LLM calls happening
- [OK] SSE infrastructure complete
- [FAIL] No events sent during 4-5 minute wait
- [FAIL] Client disconnects before completion

The streaming code is correct - we just need to send events DURING execution, not just after.
