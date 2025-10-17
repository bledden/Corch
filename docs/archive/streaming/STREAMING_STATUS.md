# Streaming Implementation Status

## Current State: POST-GENERATION STREAMING [OK]

### What We Have Now

[OK] **Real Orchestrator Integration**
- Calls actual `CollaborativeOrchestrator.collaborate(task)`
- Uses real LLM APIs (OpenRouter, Anthropic, etc.)
- Real agents: Architect, Coder, Reviewer, Documenter
- Real outputs from actual LLMs

[OK] **Post-Generation Streaming**
- After each stage completes, we stream its output
- Chunks output into ~50 character pieces
- Artificial delays (`sleep(0.05)`) to make streaming visible
- SSE events sent as output is chunked

[OK] **Real Results**
- No simulation or fake data
- Actual code/documentation from LLMs
- Real stage transitions
- Accurate workflow results

### What We DON'T Have Yet

[FAIL] **True Token Streaming**
- LLM generates tokens one at a time
- We wait for COMPLETE stage result
- Then chunk and stream it artificially

[FAIL] **Real-time Progress**
- No "Agent is thinking..." while waiting
- No loading animation during LLM calls
- Silent period between stages

### The Flow

```
User submits task
    ↓
POST /api/stream/task (returns stream_id instantly)
    ↓
GET /api/stream/events/{stream_id} (SSE connection)
    ↓
task_started event
    ↓
task_progress event (estimates)
    ↓
[SILENT PERIOD - Orchestrator calls LLM for Architect stage - 15-30s]
    ↓
chunk_started event (Architect)
    ↓
token_stream events (Architect output streamed in chunks with 0.05s delays)
    ↓
chunk_completed event (Architect)
    ↓
[SILENT PERIOD - Orchestrator calls LLM for Coder stage - 15-30s]
    ↓
chunk_started event (Coder)
    ↓
token_stream events (Coder output streamed)
    ↓
... (repeat for each stage)
    ↓
task_completed event
```

### The Problem

**Silent Periods:**
During `[SILENT PERIOD]`, the client sees nothing. The LLM is working but we're not streaming the tokens as they're generated.

## Next Steps to TRUE Real-Time Streaming

### Option 1: Add "Thinking" Indicators (Quick Fix)

Add periodic heartbeat/progress events during LLM calls:

```python
# In streaming.py
async def execute_stage_with_progress(stage_name):
    # Start stage
    await send_event(builder.chunk_started(...))

    # Background task to send "thinking" messages
    async def send_progress():
        count = 0
        while not done:
            await asyncio.sleep(2)
            await send_event(builder.system_message(f"{stage_name} is working... ({count * 2}s)"))
            count += 1

    progress_task = asyncio.create_task(send_progress())

    # Execute stage (blocks until LLM finishes)
    result = await orchestrator.execute_stage(...)

    # Stop progress updates
    done = True
    progress_task.cancel()

    # Stream the result
    ...
```

### Option 2: Integrate True LLM Token Streaming (Proper Fix)

Modify `agents/llm_client.py` to support streaming callbacks:

```python
# In llm_client.py
async def generate_with_streaming(
    self,
    prompt: str,
    on_token: Callable[[str], Awaitable[None]]  # Callback for each token
) -> str:
    full_output = ""

    # For OpenAI/Anthropic/etc with streaming support
    async with client.stream(prompt) as stream:
        async for token in stream:
            full_output += token
            await on_token(token)  # Send to SSE immediately

    return full_output
```

Then in streaming.py:
```python
async def stream_collaboration(...):
    # Define token callback
    async def on_token(token: str):
        await manager.send_event(
            stream_id,
            builder.token_stream(token, current_agent)
        )

    # Execute with streaming
    result = await orchestrator.collaborate_with_streaming(
        task=task,
        on_token=on_token
    )
```

## Recommendation

**For immediate improvement:** Option 1 (add "thinking" indicators)
- Quick to implement
- Eliminates silent periods
- Shows progress during LLM calls

**For true real-time:** Option 2 (true token streaming)
- Requires modifying orchestrator + LLM client
- More complex but provides best UX
- Tokens appear as LLM generates them

## Current Delays

- `sleep(0.05)` in lines 382-389 of streaming.py
- Needed because we're chunking complete output
- Remove when we have true token streaming

## Stage List

From `sequential_orchestrator.py`:

1. **ARCHITECT** - High-level design (Markdown output)
2. **CODER** - Code generation (Code output)
3. **REVIEWER** - Code review (JSON output)
4. **REFINER** - Fix issues (Code output) [Optional, during iteration]
5. **TESTER** - Test generation (Code output) [Optional]
6. **DOCUMENTER** - Documentation (Markdown output)

Typical sequence: Architect → Coder → Reviewer → Documenter (4 stages)
With refinement: Architect → Coder → Reviewer → Refiner → Reviewer (loop) → Documenter (5-6 stages)

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Real orchestrator | [OK] | Fully integrated |
| Real LLM calls | [OK] | Actually calling APIs |
| Real outputs | [OK] | No simulation |
| Post-gen streaming | [OK] | Chunks complete output |
| True token streaming | [FAIL] | Need to hook into LLM stream API |
| Progress indicators | [FAIL] | Silent during LLM calls |
| Artificial delays | [WARNING] | Still needed (0.05s) |

**Bottom line:** We're 80% there! Real collaboration works, streaming works, just need to connect LLM token generation to SSE events for true real-time streaming.
