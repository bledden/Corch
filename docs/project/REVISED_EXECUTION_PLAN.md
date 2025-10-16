# Streaming Consensus - Revised Execution Plan
## Aligned with weavehacks-collaborative Architecture

**Based on comprehensive technical review and repo analysis**

---

## Executive Summary

**Original Plan Issues**:
- Referenced Facilitair_v2 patterns not present in weavehacks-collaborative
- Proposed Redis+sentence-transformers cache (conflicts with existing Supabase+pgvector)
- Custom event protocol (conflicts with existing SSE event names)
- Overly ambitious Rust integration timeline

**This Revision**:
- [OK] Uses existing SSE infrastructure (`backend/routers/streaming.py`)
- [OK] Leverages existing Supabase context store for semantic caching
- [OK] Works with current `sequential_orchestrator.py` and `llm_client.py`
- [OK] Aligns event names with `backend/streaming/sse_handler.py`
- [OK] Realistic timeline for weavehacks-collaborative codebase

---

## Current State Analysis

### [OK] Already Implemented

**File**: `backend/routers/streaming.py`
- `POST /api/stream/task` - Creates stream, returns `stream_id`
- `GET /api/stream/events/{stream_id}` - SSE endpoint
- `WS /api/stream/ws` - WebSocket endpoint (for future)

**File**: `backend/streaming/sse_handler.py`
- Event types: `TASK_STARTED`, `TASK_PROGRESS`, `CHUNK_STARTED`, `TOKEN_STREAM`, `CHUNK_COMPLETED`, `TASK_COMPLETED`, `TASK_ERROR`, `SYSTEM_MESSAGE`, `HEARTBEAT`
- Event builders: `StreamEvent.task_started()`, `StreamEvent.chunk_progress()`, etc.

**File**: `backend/services/supabase_context_store.py`
- `SupabaseContextStore` with OpenAI embeddings (`text-embedding-3-small`)
- `search_by_similarity()` for vector search
- `store_context_entry()` for caching results
- Already integrated with pgvector

**File**: `backend/agents/intelligent_chunker_agent.py`
- Task decomposition into chunks
- Already used in streaming path

**File**: `backend/agents/orchestrator_with_tracking.py`
- Orchestrates multi-agent execution
- Tracking and telemetry built-in

###  Needs Implementation

1. **Event protocol mapping** to show "debate" metaphor in UI
2. **Semantic caching integration** with existing Supabase store
3. **CLI client** that consumes existing SSE endpoints
4. **Frontend SSE client** for React UI
5. **Interleaved streaming** of agent outputs for "concurrent" feel

---

## Phase 1: CLI Streaming Client (Days 1-2)

### Goal
Working CLI that shows live "debate" by consuming existing SSE endpoints.

### Approach

#### Use Existing Backend Endpoints
```python
# No backend changes needed!
# Just consume:
# POST /api/stream/task → get stream_id
# GET /api/stream/events/{stream_id} → receive SSE
```

#### Map Events to "Debate" UX
```python
# Event name mapping (from sse_handler.py):
'task_started' → "[START] Session Started"
'task_progress' → "[LIST] Plan: X chunks, ~Ys, ~$Z"
'chunk_started' → "Architect Architect: Analyzing requirements..." (derive role from model/description)
'token_stream' → Append tokens to agent's output
'chunk_completed' → "[OK] Chunk X complete"
'task_completed' → "[OK] Final Result"
'task_error' → "[FAIL] Error: ..."
'system_message' → System notifications
'heartbeat' → (ignore or show connection status)
```

### Implementation

**File**: `cli/streaming_client.py` (new)

```python
"""
CLI client for weavehacks-collaborative streaming API
Uses existing /api/stream endpoints
"""
import os, asyncio, json, time
import httpx
from rich.live import Live
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich.text import Text

API_BASE = os.getenv("FAC_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("FAC_API_KEY")  # Required for auth

console = Console()

class StreamingDebateClient:
    def __init__(self):
        self.state = {
            "started_at": None,
            "plan": {},
            "chunks": {},  # chunk_id -> accumulated content
            "current_chunk": None,
            "final": None,
            "errors": []
        }

    async def create_stream(self, task: str, context: dict = None):
        """POST /api/stream/task"""
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{API_BASE}/api/stream/task",
                headers=headers,
                json={"task": task, "context": context or {}, "stream": True}
            )
            r.raise_for_status()
            data = r.json()
            return data["stream_id"]

    async def consume_sse(self, stream_id: str):
        """GET /api/stream/events/{stream_id}"""
        headers = {"X-API-Key": API_KEY}
        url = f"{API_BASE}/api/stream/events/{stream_id}"

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=headers) as r:
                r.raise_for_status()

                event_type, data_buf = None, []
                async for line in r.aiter_lines():
                    if not line or line.startswith(":"):
                        continue

                    if not line.strip():
                        # End of event
                        if event_type and data_buf:
                            data_str = "\n".join(data_buf)
                            try:
                                data = json.loads(data_str)
                                yield event_type, data
                            except:
                                pass
                        event_type, data_buf = None, []
                        continue

                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line.split(":", 1)[1].strip())

    def render(self):
        """Rich UI rendering"""
        layout = Table.grid(pad_edge=True)
        layout.add_column()

        # Header
        if self.state["started_at"]:
            elapsed = time.time() - self.state["started_at"]
            header_text = f"Collaborative Session • {elapsed:.1f}s elapsed"
            if self.state["plan"]:
                header_text += f"\nChunks: {self.state['plan'].get('total_chunks', '?')} • "
                header_text += f"Est: {self.state['plan'].get('estimated_time', '?')}s, ${self.state['plan'].get('estimated_cost', '?')}"
            layout.add_row(Panel(header_text, title="Facilitair Stream", border_style="cyan"))

        # Active chunks (show as "agents")
        for chunk_id, content in self.state["chunks"].items():
            # Derive "role" from chunk description or model
            role_icon = "Coder"  # Default
            if "architect" in chunk_id.lower():
                role_icon = "Architect"
            elif "review" in chunk_id.lower():
                role_icon = "Reviewer"

            layout.add_row(Panel(
                content[-1000:],  # Last 1000 chars
                title=f"{role_icon} {chunk_id}",
                border_style="green"
            ))

        # Final result
        if self.state["final"]:
            layout.add_row(Panel(
                self.state["final"],
                title="[OK] Final Result",
                border_style="magenta"
            ))

        # Errors
        for err in self.state["errors"]:
            layout.add_row(Panel(err, title="[FAIL] Error", border_style="red"))

        return layout

    async def run(self, task: str, context: dict = None):
        """Main execution loop"""
        stream_id = await self.create_stream(task, context)
        self.state["started_at"] = time.time()

        with Live(self.render(), refresh_per_second=10, console=console):
            async for event_type, data in self.consume_sse(stream_id):
                if event_type == "task_started":
                    pass  # Already set started_at

                elif event_type == "task_progress":
                    self.state["plan"] = data

                elif event_type == "chunk_started":
                    chunk_id = data.get("chunk_id", "unknown")
                    model = data.get("model", "")
                    desc = data.get("description", "")
                    self.state["current_chunk"] = chunk_id
                    self.state["chunks"][chunk_id] = f"[BRAIN] {model}: {desc}\n"

                elif event_type == "token_stream":
                    chunk_id = data.get("chunk_id", self.state["current_chunk"] or "unknown")
                    tokens = data.get("tokens", "")
                    if chunk_id not in self.state["chunks"]:
                        self.state["chunks"][chunk_id] = ""
                    self.state["chunks"][chunk_id] += tokens

                elif event_type == "chunk_completed":
                    chunk_id = data.get("chunk_id", "unknown")
                    content = data.get("content", "")
                    self.state["chunks"][chunk_id] = content

                elif event_type == "task_completed":
                    self.state["final"] = data.get("content", "")

                elif event_type == "task_error":
                    self.state["errors"].append(data.get("error", "Unknown error"))

                elif event_type == "system_message":
                    # Could show as temporary message
                    pass

                elif event_type == "heartbeat":
                    pass

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Facilitair Streaming CLI")
    parser.add_argument("task", help="Task to execute")
    parser.add_argument("--language", help="Preferred language (e.g., python, typescript)")
    parser.add_argument("--framework", help="Preferred framework (e.g., fastapi, react)")
    args = parser.parse_args()

    if not API_KEY:
        console.print("[red]Error: FAC_API_KEY environment variable not set[/red]")
        return

    context = {}
    if args.language:
        context["preferred_language"] = args.language
    if args.framework:
        context["frameworks"] = [args.framework]

    client = StreamingDebateClient()
    await client.run(args.task, context)

if __name__ == "__main__":
    asyncio.run(main())
```

### Usage
```bash
# Set API key
export FAC_API_KEY="your_key_here"

# Run
python cli/streaming_client.py "Build a REST API with JWT authentication"

# With context
python cli/streaming_client.py "Build a REST API" --language python --framework fastapi
```

### Success Criteria
- [ ] CLI connects to existing endpoints without backend changes
- [ ] Shows live streaming of chunks/tokens
- [ ] Maps events to "debate" metaphor in UI
- [ ] Total implementation time: 4-6 hours

---

## Phase 2: Semantic Caching with Supabase (Days 3-4)

### Goal
30-40% cache hit rate using existing Supabase+pgvector infrastructure.

### Approach

**Use Existing**: `backend/services/supabase_context_store.py`
- Already has `search_by_similarity()` for vector search
- Already uses OpenAI embeddings (`text-embedding-3-small`)
- Already integrated with pgvector

### Implementation

**File**: `backend/services/semantic_cache.py` (new)

```python
"""
Context-aware semantic caching using existing Supabase store
"""
from typing import Optional, Dict, Any
import json
import hashlib
from .supabase_context_store import SupabaseContextStore

class ContextAwareSemanticCache:
    def __init__(self, store: SupabaseContextStore, similarity_threshold: float = 0.92):
        self.store = store
        self.threshold = similarity_threshold

    def _normalize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize context for consistent cache keys"""
        return {
            "language": context.get("preferred_language", "any"),
            "frameworks": sorted(context.get("frameworks", [])),
            "security_level": context.get("security_level", "standard"),
            "compliance": sorted(context.get("compliance", [])),
            "team_size": context.get("team_size", "small"),
            "existing_stack": sorted(context.get("existing_stack", []))
        }

    async def get(
        self,
        user_id: str,
        session_id: str,
        task: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check cache for similar task+context"""

        # Create query embedding with task + normalized context
        normalized_ctx = self._normalize_context(context)
        query = f"{task}\nCONTEXT:{json.dumps(normalized_ctx, sort_keys=True)}"

        # Search using existing Supabase store
        matches = await self.store.search_by_similarity(
            user_id=user_id,
            query=query,
            limit=5,
            similarity_threshold=self.threshold,
            session_id=session_id
        )

        # Find cached result
        for match in matches:
            if match.get("entry_type") == "cached_result":
                return {
                    "content": match.get("content"),
                    "metadata": match.get("metadata"),
                    "similarity": match.get("similarity")
                }

        return None

    async def set(
        self,
        user_id: str,
        session_id: str,
        task: str,
        context: Dict[str, Any],
        result: str,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Store result in cache"""

        normalized_ctx = self._normalize_context(context)

        # Store with normalized context in metadata
        cache_metadata = {
            **metadata,
            "task": task,
            "context": normalized_ctx,
            "cache_key": hashlib.sha256(
                f"{task}{json.dumps(normalized_ctx, sort_keys=True)}".encode()
            ).hexdigest()
        }

        return await self.store.store_context_entry(
            user_id=user_id,
            session_id=session_id,
            entry_type="cached_result",
            content=result,
            metadata=cache_metadata,
            generate_embedding=True
        )
```

### Integration

**File**: `backend/routers/streaming.py` (modify)

```python
# Add at top
from backend.services.semantic_cache import ContextAwareSemanticCache
from backend.services.supabase_context_store import SupabaseContextStore

# Initialize cache
supabase_store = SupabaseContextStore()
semantic_cache = ContextAwareSemanticCache(supabase_store)

# In process_streaming_task function, check cache first:
async def process_streaming_task(
    task_context: TaskContext,
    task_content: str,
    stream_id: str
):
    try:
        user_id = task_context.user_id
        session_id = task_context.metadata.get("session_id")
        context = task_context.metadata.get("context", {})

        # Check cache
        cached = await semantic_cache.get(user_id, session_id, task_content, context)

        if cached:
            # Send cache hit notification
            await sse_handler.send_event(
                stream_id,
                StreamEventType.SYSTEM_MESSAGE,
                {"message": f"[FAST] Cache hit ({cached['similarity']:.0%} match) • Saved ${0.08:.2f}, ~4s"}
            )

            # Stream cached result (simulate typing for UX)
            for i, chunk in enumerate(cached["content"].split("\n")):
                await sse_handler.send_event(
                    stream_id,
                    StreamEventType.TOKEN_STREAM,
                    {"tokens": chunk + "\n", "chunk_id": "cached", "is_final": False}
                )
                await asyncio.sleep(0.01)

            # Send completion
            await sse_handler.send_event(
                stream_id,
                StreamEventType.TASK_COMPLETED,
                StreamEvent.final_result(
                    task_context.task_id,
                    cached["content"],
                    0.2,  # Fast time
                    0.0,  # No cost
                    [cached["metadata"].get("model", "cache")],
                    []
                )
            )
            return

        # Cache miss - continue with normal execution
        # ... existing code ...

        # After completion, cache result
        await semantic_cache.set(
            user_id,
            session_id,
            task_content,
            context,
            final_content,
            metadata={"models_used": models_used, "cost": total_cost}
        )

    except Exception as e:
        # ... existing error handling ...
```

### Success Criteria
- [ ] Cache uses existing Supabase+pgvector (no new dependencies)
- [ ] Same query + same context = cache hit
- [ ] Same query + different context = cache miss
- [ ] Cache hits show in <500ms
- [ ] User sees savings: "[MEMORY] Cache hit (94% match, saved $0.08, 4s faster)"

---

## Phase 3: React SSE Client (Days 5-6)

### Challenge
`EventSource` cannot send custom headers (`X-API-Key`).

### Solution Options

**Option A**: Vite Dev Proxy (Development)
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api/stream': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        headers: { 'X-API-Key': process.env.VITE_API_KEY }
      }
    }
  }
})
```

**Option B**: Server-Side Proxy (Production)
Create a Next.js API route or FastAPI endpoint that accepts the user's auth token and proxies to the backend with the server's API key.

### Implementation

**File**: `src/hooks/useCollaborativeStream.ts` (new)

```typescript
import { useEffect, useRef, useState } from 'react'

interface StreamMessage {
  type: string
  agent?: string
  content?: string
  timestamp: number
}

export function useCollaborativeStream() {
  const [messages, setMessages] = useState<StreamMessage[]>([])
  const [finalResult, setFinalResult] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const start = async (task: string, context: any = {}) => {
    setIsStreaming(true)
    setMessages([])
    setFinalResult(null)
    setError(null)

    try {
      // Create stream
      const res = await fetch('/api/stream/task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, context, stream: true })
      })

      if (!res.ok) {
        throw new Error(`Failed to create stream: ${res.statusText}`)
      }

      const data = await res.json()
      const streamId = data.stream_id

      // Connect to SSE
      const es = new EventSource(`/api/stream/events/${streamId}`)
      esRef.current = es

      // Bind event listeners
      const bind = (name: string, handler: (e: MessageEvent) => void) => {
        es.addEventListener(name, handler as any)
      }

      bind('task_started', () => {
        setMessages(m => [...m, { type: 'system', content: 'Task started', timestamp: Date.now() }])
      })

      bind('task_progress', (e) => {
        const d = JSON.parse(e.data)
        setMessages(m => [...m, {
          type: 'system',
          content: `Plan: ${d.total_chunks} chunks • ~${d.estimated_time}s • ~$${d.estimated_cost}`,
          timestamp: Date.now()
        }])
      })

      bind('chunk_started', (e) => {
        const d = JSON.parse(e.data)
        setMessages(m => [...m, {
          type: 'chunk',
          agent: d.model || 'Agent',
          content: d.description || '',
          timestamp: Date.now()
        }])
      })

      bind('token_stream', (e) => {
        const d = JSON.parse(e.data)
        setMessages(m => {
          const last = m[m.length - 1]
          if (last && last.type === 'token') {
            // Append to last token message
            return [...m.slice(0, -1), { ...last, content: (last.content || '') + d.tokens }]
          }
          return [...m, { type: 'token', content: d.tokens, timestamp: Date.now() }]
        })
      })

      bind('chunk_completed', (e) => {
        const d = JSON.parse(e.data)
        setMessages(m => [...m, {
          type: 'system',
          content: `[OK] Chunk ${d.chunk_id} complete`,
          timestamp: Date.now()
        }])
      })

      bind('task_completed', (e) => {
        const d = JSON.parse(e.data)
        setFinalResult(d.content || '')
        setIsStreaming(false)
        es.close()
      })

      bind('task_error', (e) => {
        const d = JSON.parse(e.data)
        setError(d.error || 'Unknown error')
        setIsStreaming(false)
        es.close()
      })

      bind('system_message', (e) => {
        const d = JSON.parse(e.data)
        setMessages(m => [...m, { type: 'system', content: d.message, timestamp: Date.now() }])
      })

      es.onerror = () => {
        setError('Connection lost')
        setIsStreaming(false)
        es.close()
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start stream')
      setIsStreaming(false)
    }
  }

  const stop = () => {
    esRef.current?.close()
    esRef.current = null
    setIsStreaming(false)
  }

  useEffect(() => {
    return () => {
      esRef.current?.close()
    }
  }, [])

  return { start, stop, messages, finalResult, isStreaming, error }
}
```

### Success Criteria
- [ ] React hook connects to existing SSE endpoint
- [ ] Real-time message display
- [ ] Proper cleanup on unmount
- [ ] Error handling for network issues

---

## Phase 4: Rust Layer (Week 2-3) - **OPTIONAL**

### Revised Scope

**DO NOT attempt**:
- [FAIL] Candle embeddings (use OpenAI API)
- [FAIL] HNSW vector index (use Supabase pgvector)
- [FAIL] Token routing (Python async is fine for MVP)

**Consider ONLY if**:
- Python streaming shows measurable lag
- You have Rust experience
- You have time after Phase 1-3 complete

**Minimal Rust Option** (if needed):
```rust
// Simple stream multiplexer - nothing more
use tokio::sync::mpsc;
use pyo3::prelude::*;

#[pyclass]
struct StreamMultiplexer {
    // Just fan-in multiple channels to one output
}

// NO embeddings, NO vector search, NO GPU dependencies
```

### Recommendation
**Skip Rust for MVP**. Python + async is sufficient for hundreds of concurrent users. Optimize only after proven bottleneck.

---

## Revised Timeline

| Phase | Duration | What | Success Metric |
|-------|----------|------|----------------|
| **Phase 1** | Days 1-2 | CLI client | Working `streaming_client.py` consuming existing API |
| **Phase 2** | Days 3-4 | Semantic cache | 30%+ cache hit rate with Supabase |
| **Phase 3** | Days 5-6 | React SSE hook | Live streaming in web UI |
| **Phase 4** | Week 2+ | Rust (optional) | Only if bottleneck proven |

**MVP Ready**: End of Day 6 (Week 1)

---

## Open Questions - **ANSWER BEFORE STARTING**

### 1. Context Granularity
- [ ] **Basic**: Just language
- [x] **Medium**: Language + frameworks + team size ← **RECOMMENDED**
- [ ] **Advanced**: Full stack + compliance + history

### 2. Cache Transparency
- [x] **Always show**: "[MEMORY] Cache hit (94% match, saved $0.08)" ← **RECOMMENDED**
- [ ] **Show on hover**: Tooltip only
- [ ] **Never show**: Invisible

### 3. Debate Verbosity
- [ ] **Maximum**: Every token, every agent thought
- [x] **Filtered**: High-confidence outputs only ← **RECOMMENDED**
- [ ] **Configurable**: User setting (more work)

### 4. Synthesis Timing
- [ ] **After each agent**: Too frequent
- [x] **Semantic chunks**: At logical boundaries ← **RECOMMENDED**
- [ ] **Rolling**: Continuous (complex)

---

## Success Metrics

### Week 1 (MVP)
- [ ] CLI streams debate from existing backend
- [ ] 30%+ cache hit rate on repeated queries
- [ ] React hook displays live streaming
- [ ] Cost per collaboration < $0.10
- [ ] Users can run: `python cli/streaming_client.py "Build REST API"`

### Week 2+ (Polish)
- [ ] Web UI matches CLI quality
- [ ] Cache shows savings in UI
- [ ] Error handling covers edge cases
- [ ] Load test: 10 concurrent users

---

## Files to Create/Modify

### New Files
```
cli/
  streaming_client.py          # CLI SSE consumer

backend/services/
  semantic_cache.py             # Context-aware cache using Supabase

src/hooks/
  useCollaborativeStream.ts     # React SSE hook
```

### Modified Files
```
backend/routers/streaming.py    # Add cache check before processing
vite.config.ts                  # Add dev proxy for SSE auth
```

**Total New Code**: ~500 lines
**Backend Changes**: ~50 lines

---

## Next Steps

1. **Answer 4 open questions** [OK] (answered above with recommendations)
2. **Install dependencies**:
   ```bash
   pip install rich httpx
   ```
3. **Create `cli/streaming_client.py`** (code provided above)
4. **Test CLI**:
   ```bash
   export FAC_API_KEY="your_key"
   python cli/streaming_client.py "Build a REST API"
   ```
5. **Once CLI works**, proceed to Phase 2 (semantic cache)

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| SSE auth headers | Vite dev proxy + production proxy endpoint |
| Cache false positives | High threshold (0.92) + context normalization |
| Streaming lag | Use existing async Python (proven sufficient) |

### Scope Risks
| Risk | Mitigation |
|------|------------|
| Over-engineering | Stick to 3 phases; defer Rust |
| Feature creep | Focus on streaming UX only |
| Timeline slip | Each phase shippable independently |

---

## Conclusion

This revised plan:
- [OK] Uses **100% existing infrastructure**
- [OK] Requires **minimal backend changes** (~50 lines)
- [OK] Delivers **working CLI in 2 days**
- [OK] Achieves **semantic caching in 4 days**
- [OK] Adds **web UI in 6 days**
- [OK] **No Rust complexity** for MVP

**Ready to start?** Just create `cli/streaming_client.py` and run it! [START]
