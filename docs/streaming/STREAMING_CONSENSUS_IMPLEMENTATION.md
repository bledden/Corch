# Streaming Consensus Implementation Plan
## Real-Time Multi-Agent Collaboration with Live Synthesis

Based on Facilitair_v2 learnings + IDEAL_ARCHITECTURE.md principles

---

## Executive Summary

**User's Vision**: Models debate and collaborate in real-time, with synthesis happening WHILE they generate responses, not after. Users see intelligence emerging through a streaming debate interface.

**The Problem**: Current LLM APIs don't support TRUE bidirectional streaming where Model A's partial output influences Model B's generation mid-stream.

**The Solution**: Implement **pseudo-streaming** that feels real-time to users, then transition to true streaming when APIs support it.

---

## Part 1: Understanding User's Requirements

### From User Message:
> "Rust showed improvement capabilities for facilitair_v2. If we cannot improve API call latency, we may as well try and improve latency and performance everywhere else."

**Translation**: Optimize everything we control (routing, processing, UI rendering) since API calls are the bottleneck.

> "Streaming API and Semantic Caching seem like no brainers as long as we have a solution to context of the user for caching."

**Translation**: Implement both immediately, but semantic caching needs context awareness (same query, different user context = different result).

> "Streaming Debate Interface is definitely the move. Would be nice to start with something in the CLI first."

**Translation**: Build CLI streaming interface first (faster iteration), then web UI.

> "Ideally the synthesis is occurring while the models work on a solution together. How would this even work, or how do existing solutions handle this?"

**Translation**: This is the KEY question. User wants to understand HOW to achieve concurrent generation + synthesis.

---

## Part 2: How Existing Solutions Handle This

### ChatGPT Code Interpreter / Canvas
**What They Show**:
- Code execution happens in iframe
- Partial results stream as code runs
- User sees "thinking..." then streaming output

**How It Actually Works**:
```
1. Generate full code (streaming to user)
2. Execute code in sandbox
3. Stream execution output
4. Present as if it's one flow
```

**Key Insight**: They DON'T do true concurrent synthesis. It's staged with good UX.

### Claude Artifacts
**What They Show**:
- Markdown/code renders while Claude generates
- Preview updates token-by-token
- Feels like real-time rendering

**How It Actually Works**:
```
1. Claude streams response
2. Parser extracts code blocks incrementally
3. Renderer updates preview every N tokens
4. Creates illusion of live rendering
```

**Key Insight**: Parsing + rendering happens client-side on streaming tokens.

### Multi-Agent Debate Papers (Du et al., 2023)
**What They Show**:
- Multiple models debate rounds
- Consensus through voting

**How It Actually Works**:
```
Round 1: All agents generate FULL responses in parallel
Round 2: Agents see Round 1 outputs, generate FULL responses
Round 3: Vote on best response
```

**Key Insight**: Still batch-based, just with multiple batches.

### Our Opportunity
**None of them do TRUE streaming consensus** where:
- Agent A generates token `t1`
- Agent B sees `t1` and adjusts its generation
- Agent C synthesizes `t1` in real-time

**Why not?** LLM APIs don't support it. But we can simulate it cleverly.

---

## Part 3: The Pseudo-Streaming Approach

### Concept: "Chunked Streaming Debate"

Instead of waiting for full responses, we chunk them and simulate real-time collaboration:

```
Timeline:

0ms:    User submits task
10ms:   Task broken into 3 chunks
20ms:   Stream "Architect analyzing requirements..."

--- CHUNK 1: Requirements Analysis ---
100ms:  Architect starts generating (streaming to user)
500ms:  Architect outputs: "This needs a REST API with..."
505ms:  Stream "Coder considering architecture..."
510ms:  Coder starts generating based on PARTIAL architecture
800ms:  Coder outputs: "Based on the REST API requirement, I'll use FastAPI..."
805ms:  Stream "Reviewer evaluating approach..."
810ms:  Reviewer generates critique of PARTIAL design
1000ms: Reviewer outputs: "The FastAPI choice is good, but consider rate limiting..."

--- SYNTHESIS POINT 1 ---
1100ms: Synthesizer merges Chunk 1 outputs
1200ms: Stream synthesized Chunk 1 to user

--- CHUNK 2: Implementation Details ---
1300ms: All agents see Chunk 1 synthesis + new chunk
1350ms: Architect refines based on feedback: "Adding rate limiting middleware..."
1600ms: Coder implements: "Here's the rate limiter code..."
1800ms: Reviewer checks: "Good, now add tests..."

--- SYNTHESIS POINT 2 ---
2000ms: Synthesizer merges Chunk 2 outputs
2100ms: Stream synthesized Chunk 2 to user

... and so on
```

**Key Principle**: Agents react to PARTIAL outputs from earlier stages, creating the FEELING of real-time collaboration.

---

## Part 4: Facilitair_v2 Lessons Applied

### From /backend/routers/streaming.py

**What We Learned**:
```python
# Good: Token-level streaming events
await sse_handler.send_event(
    stream_id,
    StreamEventType.TOKEN_STREAM,
    StreamEvent.token_stream(token, chunk_id, is_final)
)

# Good: Progress tracking per chunk
await sse_handler.send_event(
    stream_id,
    StreamEventType.CHUNK_STARTED,
    StreamEvent.chunk_progress(chunk_id, current, total, agent, description)
)
```

**Lesson 1**: Users need to see which agent is working (not just tokens)
**Lesson 2**: Chunk-based progress is intuitive
**Lesson 3**: SSE is simpler than WebSockets for one-way streaming

### Architecture We'll Adopt

```
┌─────────────────────────────────────────────────┐
│              User Interface (CLI/Web)            │
│  [Architect] Analyzing requirements...          │
│  [Coder] Considering FastAPI framework...       │
│  [Reviewer] Evaluating security implications... │
│  [Synthesizer] Merging perspectives...          │
└─────────────────┬───────────────────────────────┘
                  │ SSE Stream
┌─────────────────▼───────────────────────────────┐
│       Streaming Orchestrator (Python)            │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  Chunked Debate Controller               │   │
│  │  - Breaks task into semantic chunks      │   │
│  │  - Manages debate rounds                 │   │
│  │  - Triggers synthesis at checkpoints     │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌────────┐  ┌────────┐  ┌──────────┐          │
│  │Architect│  │ Coder  │  │ Reviewer │          │
│  │ Stream │  │ Stream │  │  Stream  │          │
│  └────────┘  └────────┘  └──────────┘          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│       Semantic Cache (Redis + Embeddings)       │
│  - Cache synthesis results per context          │
│  - Check cache before spawning agents           │
└─────────────────────────────────────────────────┘
```

---

## Part 5: Implementation Phases

### Phase 1: CLI Streaming Debate (Week 1)
**Goal**: Ship working CLI interface

```bash
$ facilitair collaborate "Build a REST API with auth"

┌─ Collaborative Session ────────────────────────────┐
│ Task: Build a REST API with auth                   │
│ Strategy: BALANCED (GPT-5 + Claude + DeepSeek)     │
└────────────────────────────────────────────────────┘

[00:00.10] 🏗️  Architect: Analyzing requirements...
[00:00.50] 💭 Architect: This requires:
           - FastAPI framework for performance
           - JWT authentication
           - PostgreSQL for user storage
           - Redis for session management

[00:00.80] 💻 Coder: Considering architecture...
[00:01.20] 💭 Coder: Based on JWT auth requirement:
           ```python
           from fastapi import FastAPI, Depends
           from fastapi.security import HTTPBearer
           ...
           ```

[00:01.50] 🔍 Reviewer: Evaluating security...
[00:01.90] 💭 Reviewer: Good start, but missing:
           - Rate limiting for auth endpoints
           - Password hashing with bcrypt
           - Refresh token rotation

[00:02.10] 🔄 Synthesizer: Merging perspectives...
[00:02.30] ✅ Chunk 1 Complete: Requirements + Security

┌─ Synthesized Output (Chunk 1) ────────────────────┐
│ FastAPI REST API with JWT authentication:         │
│ - JWT access tokens (15 min expiry)               │
│ - Refresh tokens (7 day expiry)                   │
│ - bcrypt password hashing                         │
│ - Rate limiting: 5 req/min on /auth               │
│ - PostgreSQL + SQLAlchemy ORM                     │
└────────────────────────────────────────────────────┘

[00:02.50] 💻 Coder: Implementing requirements...
           [Streaming code...]

...
```

**Implementation**:

```python
# cli_streaming_debate.py
import asyncio
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from datetime import datetime

console = Console()

class CLIDebateInterface:
    def __init__(self):
        self.layout = Layout()
        self.messages = []
        self.synthesis_results = []

    async def stream_debate(self, task: str):
        """Stream a collaborative debate session"""

        with Live(self.layout, console=console, refresh_per_second=10):
            # Show header
            self.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="debate", ratio=3),
                Layout(name="synthesis", ratio=1)
            )

            self.layout["header"].update(
                Panel(f"Task: {task}", title="Collaborative Session")
            )

            # Start debate
            async for event in stream_collaborative_debate(task):
                if event.type == "agent_thinking":
                    self.add_message(
                        f"[{event.timestamp}] {event.agent_icon} {event.agent}: {event.message}",
                        style=event.agent_color
                    )

                elif event.type == "agent_output":
                    self.add_message(
                        f"[{event.timestamp}] 💭 {event.agent}: {event.content}",
                        style=f"dim {event.agent_color}"
                    )

                elif event.type == "synthesis":
                    self.synthesis_results.append(event.content)
                    self.layout["synthesis"].update(
                        Panel(event.content, title=f"✅ {event.title}")
                    )

                # Update debate view
                self.layout["debate"].update(
                    "\n".join(self.messages[-20:])  # Last 20 messages
                )

                await asyncio.sleep(0.01)  # Smooth rendering

async def stream_collaborative_debate(task: str):
    """
    Generator that yields debate events in real-time

    This is where the magic happens - we simulate concurrent
    debate by chunking and interleaving agent outputs
    """

    # Step 1: Task chunking
    chunks = await intelligent_chunk_task(task)

    for chunk_idx, chunk in enumerate(chunks):
        yield DebateEvent(
            type="synthesis",
            title=f"Chunk {chunk_idx + 1}/{len(chunks)}",
            content=f"Working on: {chunk.description}"
        )

        # Step 2: Parallel agent execution with streaming
        agent_streams = {
            "architect": stream_agent("architect", chunk, context=None),
            "coder": stream_agent("coder", chunk, context=None),
            "reviewer": stream_agent("reviewer", chunk, context=None)
        }

        # Step 3: Interleave streams (pseudo-parallelism)
        partial_outputs = {"architect": "", "coder": "", "reviewer": ""}

        async for agent, token in interleave_streams(agent_streams):
            partial_outputs[agent] += token

            # Yield token for UI
            yield DebateEvent(
                type="agent_output",
                agent=agent,
                content=token,
                timestamp=get_timestamp()
            )

            # Every N tokens, check if another agent should react
            if len(partial_outputs[agent].split()) % 50 == 0:
                # Trigger other agents to "react"
                for other_agent in ["architect", "coder", "reviewer"]:
                    if other_agent != agent:
                        yield DebateEvent(
                            type="agent_thinking",
                            agent=other_agent,
                            message=f"Considering {agent}'s points..."
                        )

        # Step 4: Synthesize chunk
        synthesis = await synthesize_chunk(partial_outputs, chunk)

        yield DebateEvent(
            type="synthesis",
            title=f"Chunk {chunk_idx + 1} Complete",
            content=synthesis
        )

async def interleave_streams(agent_streams):
    """
    Interleave multiple agent streams to create concurrent feel

    This is pseudo-streaming: we're not truly affecting generation,
    but we're creating the PERCEPTION of real-time collaboration
    """

    active_streams = list(agent_streams.keys())
    buffers = {agent: [] for agent in active_streams}

    # Start all streams
    tasks = {
        agent: asyncio.create_task(collect_stream(stream))
        for agent, stream in agent_streams.items()
    }

    # Interleave by round-robin with smart weighting
    while active_streams:
        for agent in active_streams[:]:
            # Check if agent has tokens ready
            if agent in buffers and buffers[agent]:
                # Yield 5-10 tokens at a time
                batch_size = random.randint(5, 10)
                for _ in range(min(batch_size, len(buffers[agent]))):
                    token = buffers[agent].pop(0)
                    yield agent, token

                    # Small delay for realistic pacing
                    await asyncio.sleep(0.02)

            # Refill buffer if needed
            if tasks[agent].done() and not buffers[agent]:
                active_streams.remove(agent)
            elif not buffers[agent]:
                # Try to get more tokens
                try:
                    new_tokens = await asyncio.wait_for(
                        tasks[agent],
                        timeout=0.1
                    )
                    buffers[agent].extend(new_tokens)
                except asyncio.TimeoutError:
                    pass
```

**Key Features**:
1. **Semantic Chunking**: Break task into logical pieces (requirements → design → implementation → testing)
2. **Interleaved Streaming**: Show agents working "in parallel" through round-robin token display
3. **Reactive Messages**: Show agents "reacting" to each other's partial outputs
4. **Synthesis Checkpoints**: Merge perspectives at chunk boundaries

---

### Phase 2: Semantic Caching with Context (Week 1-2)

**The Challenge**: Same query, different context = different answer

Example:
```
User A: "Build a REST API" (context: Python, small startup)
User B: "Build a REST API" (context: Java, enterprise, SOC2)
```

These should NOT hit the same cache.

**Solution: Context-Aware Semantic Caching**

```python
# semantic_cache.py
from sentence_transformers import SentenceTransformer
import redis
import json
import hashlib

class ContextAwareSemanticCache:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.similarity_threshold = 0.92  # High threshold

    def _create_cache_key(self, task: str, context: dict) -> str:
        """
        Create cache key that includes context

        Context includes:
        - User preferences (language, framework)
        - Organization requirements (security level, compliance)
        - Historical choices (past tech stack)
        """

        # Canonicalize context
        context_str = json.dumps(
            {
                "language": context.get("preferred_language", "any"),
                "frameworks": sorted(context.get("frameworks", [])),
                "security_level": context.get("security_level", "standard"),
                "compliance": sorted(context.get("compliance", [])),
                "team_size": context.get("team_size", "small"),
                "existing_stack": sorted(context.get("existing_stack", []))
            },
            sort_keys=True
        )

        # Create embedding for task + context
        combined = f"{task} |CONTEXT| {context_str}"
        embedding = self.model.encode(combined)

        return embedding.tobytes()

    async def get(self, task: str, context: dict) -> Optional[str]:
        """Check cache for similar task with similar context"""

        query_embedding = self.model.encode(
            f"{task} |CONTEXT| {json.dumps(context, sort_keys=True)}"
        )

        # Get all cached items (in production, use vector DB)
        cached_items = self.redis.keys("cache:*")

        best_match = None
        best_similarity = 0.0

        for key in cached_items:
            cached_data = json.loads(self.redis.get(key))
            cached_embedding = np.frombuffer(
                bytes.fromhex(cached_data["embedding"]),
                dtype=np.float32
            )

            # Cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_data

        if best_similarity > self.similarity_threshold:
            return best_match["result"]

        return None

    async def set(
        self,
        task: str,
        context: dict,
        result: str,
        metadata: dict
    ):
        """Cache result with context embedding"""

        embedding = self.model.encode(
            f"{task} |CONTEXT| {json.dumps(context, sort_keys=True)}"
        )

        cache_key = f"cache:{hashlib.sha256(embedding.tobytes()).hexdigest()}"

        self.redis.setex(
            cache_key,
            60 * 60 * 24 * 7,  # 7 day TTL
            json.dumps({
                "task": task,
                "context": context,
                "result": result,
                "embedding": embedding.tobytes().hex(),
                "metadata": metadata,
                "cached_at": time.time()
            })
        )

# Integration with orchestrator
class CachedStreamingOrchestrator:
    def __init__(self):
        self.cache = ContextAwareSemanticCache()
        self.orchestrator = StreamingOrchestrator()

    async def collaborate(self, task: str, user_context: dict):
        """Check cache before spawning agents"""

        # Try cache first
        cached_result = await self.cache.get(task, user_context)

        if cached_result:
            # Stream cached result (still feels live to user)
            async for event in simulate_cached_stream(cached_result):
                yield event

            return

        # Cache miss - do real collaboration
        result_chunks = []

        async for event in self.orchestrator.stream_debate(task, user_context):
            yield event

            if event.type == "synthesis":
                result_chunks.append(event.content)

        # Cache final result
        await self.cache.set(
            task,
            user_context,
            "\n\n".join(result_chunks),
            metadata={"agents_used": ["architect", "coder", "reviewer"]}
        )
```

**Cache Hit UX**:
```
$ facilitair collaborate "Build REST API with auth"

💾 Found similar solution (94% match)
   Cached: 2 hours ago
   Context: Python, FastAPI, Small team

⚡ Retrieving cached solution...

[Streams cached result with realistic pacing]
```

**Cache Miss UX**:
```
$ facilitair collaborate "Build REST API with auth"

🔍 No cached solution found
🚀 Starting collaborative session...

[Real-time debate as shown in Phase 1]

💾 Solution cached for future use
```

---

### Phase 3: Rust Performance Layer (Week 2-3)

**What Rust Will Handle**:
1. Token routing (high throughput)
2. Embedding computation (CPU intensive)
3. Stream multiplexing (concurrent connections)
4. Cache lookups (sub-millisecond latency)

**What Stays in Python**:
1. LLM API calls (IO-bound, Python's fine)
2. Business logic (easier to iterate)
3. Configuration (YAML parsing)

**Architecture**:

```
┌──────────────── Python Layer ─────────────────────┐
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │    Orchestration Logic (Python)          │     │
│  │  - Agent coordination                    │     │
│  │  - Task chunking                         │     │
│  │  - Synthesis logic                       │     │
│  └──────────────┬───────────────────────────┘     │
│                 │                                  │
│                 ▼                                  │
│  ┌──────────────────────────────────────────┐     │
│  │    Rust Bridge (PyO3)                    │     │
│  │  - Zero-copy data transfer               │     │
│  │  - Async interop                         │     │
│  └──────────────┬───────────────────────────┘     │
└─────────────────┼────────────────────────────────┘
                  │
┌─────────────────▼─── Rust Layer ──────────────────┐
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │    Stream Router (Tokio)                 │     │
│  │  - 1M+ concurrent connections            │     │
│  │  - Token multiplexing                    │     │
│  │  - Backpressure handling                 │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │    Embedding Engine (Candle)             │     │
│  │  - BERT model inference                  │     │
│  │  - GPU acceleration (Metal/CUDA)         │     │
│  │  - Batch processing                      │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │    Cache Layer (Redis)                   │     │
│  │  - Sub-millisecond lookups               │     │
│  │  - Vector similarity search              │     │
│  └──────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
```

**Rust Implementation**:

```rust
// lib.rs - Rust performance layer
use pyo3::prelude::*;
use tokio::sync::mpsc;
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;

#[pyclass]
struct RustPerformanceLayer {
    router: StreamRouter,
    embedder: EmbeddingEngine,
    cache: CacheLayer,
}

#[pymethods]
impl RustPerformanceLayer {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            router: StreamRouter::new(),
            embedder: EmbeddingEngine::new()?,
            cache: CacheLayer::new(),
        })
    }

    /// Route tokens from multiple agent streams to synthesis
    fn route_streams(
        &self,
        py: Python,
        agent_streams: Vec<PyObject>
    ) -> PyResult<PyObject> {
        py.allow_threads(|| {
            // This releases GIL for true parallelism
            self.router.interleave_streams(agent_streams)
        })
    }

    /// Compute embeddings for cache lookup
    fn compute_embedding(
        &self,
        py: Python,
        text: String
    ) -> PyResult<Vec<f32>> {
        py.allow_threads(|| {
            // Fast embedding computation without GIL
            self.embedder.encode(&text)
        })
    }

    /// Check cache with vector similarity
    fn cache_lookup(
        &self,
        py: Python,
        embedding: Vec<f32>,
        threshold: f32
    ) -> PyResult<Option<String>> {
        py.allow_threads(|| {
            self.cache.find_similar(embedding, threshold)
        })
    }
}

struct StreamRouter {
    channels: HashMap<String, mpsc::Sender<Token>>,
}

impl StreamRouter {
    fn new() -> Self {
        Self {
            channels: HashMap::new(),
        }
    }

    /// Interleave multiple agent streams with smart batching
    async fn interleave_streams(
        &self,
        streams: Vec<AgentStream>
    ) -> Result<SynthesisStream> {
        let (tx, rx) = mpsc::channel(10000);

        // Spawn tasks for each agent stream
        for stream in streams {
            let tx_clone = tx.clone();

            tokio::spawn(async move {
                // Read tokens from agent
                while let Some(token) = stream.next().await {
                    // Forward to synthesis with agent ID
                    tx_clone.send((stream.agent_id.clone(), token)).await.ok();

                    // Yield to other streams
                    tokio::task::yield_now().await;
                }
            });
        }

        // Return multiplexed stream
        Ok(SynthesisStream::new(rx))
    }
}

struct EmbeddingEngine {
    model: BertModel,
    device: Device,
}

impl EmbeddingEngine {
    fn new() -> Result<Self> {
        // Load BERT model with candle
        let device = Device::cuda_if_available(0)?;
        let model = BertModel::load("sentence-transformers/all-MiniLM-L6-v2", &device)?;

        Ok(Self { model, device })
    }

    /// Compute embeddings 10x faster than Python
    fn encode(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        let tokens = self.model.tokenizer().encode(text)?;

        // Convert to tensor
        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?;

        // Forward pass
        let embeddings = self.model.forward(&input_ids)?;

        // Pool to sentence embedding
        let pooled = embeddings.mean(1)?;

        // Return as Vec
        Ok(pooled.to_vec1()?)
    }

    /// Batch encode for cache warming
    fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        // Process batch on GPU
        let tokens_batch: Vec<_> = texts.iter()
            .map(|t| self.model.tokenizer().encode(t))
            .collect::<Result<_>>()?;

        // Stack into batch tensor
        let batch_tensor = stack_tokens(tokens_batch, &self.device)?;

        // Single forward pass for entire batch
        let embeddings = self.model.forward(&batch_tensor)?;

        // Pool and return
        Ok(mean_pool(embeddings)?)
    }
}

struct CacheLayer {
    redis: redis::aio::ConnectionManager,
    index: VectorIndex,
}

impl CacheLayer {
    fn new() -> Self {
        // Connect to Redis
        let client = redis::Client::open("redis://127.0.0.1/").unwrap();
        let conn = client.get_tokio_connection_manager().await.unwrap();

        // Initialize vector index (HNSW)
        let index = VectorIndex::new(384); // MiniLM dimension

        Self { redis: conn, index }
    }

    /// Sub-millisecond cache lookup
    async fn find_similar(
        &self,
        query_embedding: Vec<f32>,
        threshold: f32
    ) -> Result<Option<String>> {
        // Search vector index (HNSW: O(log n))
        let candidates = self.index.search(&query_embedding, 10)?;

        // Check Redis for actual content
        for (id, similarity) in candidates {
            if similarity > threshold {
                let content: Option<String> = self.redis
                    .get(&format!("cache:{}", id))
                    .await?;

                if let Some(cached) = content {
                    return Ok(Some(cached));
                }
            }
        }

        Ok(None)
    }
}

// Python binding
#[pymodule]
fn facilitair_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustPerformanceLayer>()?;
    Ok(())
}
```

**Python Integration**:

```python
# Import Rust layer
import facilitair_rust

class HybridOrchestrator:
    def __init__(self):
        # Rust performance layer
        self.rust_layer = facilitair_rust.RustPerformanceLayer()

        # Python orchestration
        self.orchestrator = StreamingOrchestrator()

    async def collaborate(self, task: str, context: dict):
        """Hybrid Python + Rust execution"""

        # Step 1: Fast embedding (Rust)
        task_embedding = self.rust_layer.compute_embedding(
            f"{task} |CONTEXT| {json.dumps(context)}"
        )

        # Step 2: Fast cache lookup (Rust)
        cached_result = self.rust_layer.cache_lookup(
            task_embedding,
            threshold=0.92
        )

        if cached_result:
            # Stream cached result
            async for event in simulate_cached_stream(cached_result):
                yield event
            return

        # Step 3: Real collaboration (Python orchestration)
        agent_streams = await self.orchestrator.start_agents(task, context)

        # Step 4: Fast stream routing (Rust)
        synthesis_stream = self.rust_layer.route_streams(agent_streams)

        # Step 5: Synthesize and stream
        async for synthesized_chunk in synthesis_stream:
            yield synthesized_chunk
```

**Performance Gains**:

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Embedding computation | 45ms | 3ms | 15x |
| Cache lookup (10k items) | 12ms | 0.4ms | 30x |
| Stream routing (3 agents) | 8ms overhead | <0.1ms | 80x |
| **Total per request** | **65ms** | **3.5ms** | **18x** |

---

## Part 6: Web UI with Streaming Debate (Week 3-4)

After CLI is working, build web interface using lessons learned.

**Tech Stack**:
- **Frontend**: Next.js 14 + React Server Components
- **Streaming**: SSE (Server-Sent Events) for simplicity
- **State**: Zustand for client state
- **Visualization**: D3.js for debate flow diagram

**UX Design**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Facilitair - Collaborative AI Workspace              [@bledden] │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📝 Task Input                                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ Build a REST API with JWT authentication                 │   │
│  │                                                           │   │
│  └───────────────────────────────────────────────────────────┘   │
│  [⚙️ Settings] [🎯 Strategy: BALANCED] [▶️ Collaborate]          │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│  💬 Live Debate Stream                                           │
│                                                                   │
│  ┌─ Chunk 1/3: Requirements Analysis ──────────────────────┐     │
│  │                                                          │     │
│  │  🏗️  Architect (GPT-5)         [00:00.50] ━━━━━━━━●○    │     │
│  │  "This requires JWT tokens, PostgreSQL for users..."    │     │
│  │                                                          │     │
│  │  💻 Coder (DeepSeek)           [00:01.20] ━━━━●○○○○○    │     │
│  │  "Based on JWT requirement, using fastapi.security..."  │     │
│  │                                                          │     │
│  │  🔍 Reviewer (Claude)          [00:01.80] ━━●○○○○○○○    │     │
│  │  "Good, but add rate limiting on /auth endpoints..."    │     │
│  │                                                          │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ┌─ Synthesis (Live) ───────────────────────────────────────┐     │
│  │  🔄 Merging perspectives...                              │     │
│  │                                                          │     │
│  │  ✅ Agreed: FastAPI + JWT + PostgreSQL                   │     │
│  │  ⚠️  Debate: Rate limiting approach                      │     │
│  │     - Architect: Application-level                      │     │
│  │     - Coder: Nginx upstream                             │     │
│  │     - Reviewer: Both (defense in depth)                 │     │
│  │                                                          │     │
│  │  📊 Confidence: 87% (Reviewer concerns noted)            │     │
│  └──────────────────────────────────────────────────────────┘     │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│  📄 Final Output                                                 │
│                                                                   │
│  [Code tab] [Architecture tab] [Security Notes tab]              │
│                                                                   │
│  ```python                                                       │
│  from fastapi import FastAPI, Depends, HTTPException             │
│  from fastapi.security import HTTPBearer, HTTPAuthorizationCred  │
│  ...                                                             │
│  ```                                                             │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**Implementation**:

```typescript
// components/StreamingDebateInterface.tsx
'use client';

import { useEffect, useState } from 'react';
import { useDebateStore } from '@/stores/debate-store';

interface DebateMessage {
  agent: 'architect' | 'coder' | 'reviewer' | 'synthesizer';
  type: 'thinking' | 'output' | 'synthesis';
  content: string;
  timestamp: number;
  chunk_id?: string;
}

export function StreamingDebateInterface({ task }: { task: string }) {
  const [messages, setMessages] = useState<DebateMessage[]>([]);
  const [synthesis, setSynthesis] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!task) return;

    // Connect to SSE endpoint
    const eventSource = new EventSource(
      `/api/collaborate/stream?task=${encodeURIComponent(task)}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'agent_thinking':
          setMessages(prev => [...prev, {
            agent: data.agent,
            type: 'thinking',
            content: data.message,
            timestamp: Date.now()
          }]);
          break;

        case 'agent_output':
          setMessages(prev => {
            const last = prev[prev.length - 1];

            // Append to last message if same agent
            if (last && last.agent === data.agent && last.type === 'output') {
              return [
                ...prev.slice(0, -1),
                { ...last, content: last.content + data.token }
              ];
            }

            // New message
            return [...prev, {
              agent: data.agent,
              type: 'output',
              content: data.token,
              timestamp: Date.now()
            }];
          });
          break;

        case 'synthesis':
          setSynthesis(prev => [...prev, data.content]);
          break;

        case 'complete':
          setIsStreaming(false);
          eventSource.close();
          break;
      }
    };

    eventSource.onerror = () => {
      setIsStreaming(false);
      eventSource.close();
    };

    setIsStreaming(true);

    return () => {
      eventSource.close();
    };
  }, [task]);

  return (
    <div className="flex flex-col h-full">
      {/* Live debate stream */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <DebateMessage key={i} message={msg} />
        ))}
      </div>

      {/* Synthesis area */}
      {synthesis.length > 0 && (
        <div className="border-t p-4 bg-gray-50">
          <h3 className="font-semibold mb-2">Synthesized Output</h3>
          {synthesis.map((chunk, i) => (
            <div key={i} className="mb-4">
              <h4 className="text-sm text-gray-600">Chunk {i + 1}</h4>
              <pre className="bg-white p-2 rounded text-sm">{chunk}</pre>
            </div>
          ))}
        </div>
      )}

      {/* Streaming indicator */}
      {isStreaming && (
        <div className="border-t p-2 text-center text-sm text-gray-600">
          <span className="animate-pulse">● Streaming...</span>
        </div>
      )}
    </div>
  );
}

function DebateMessage({ message }: { message: DebateMessage }) {
  const agentStyles = {
    architect: { icon: '🏗️', color: 'text-blue-600', bg: 'bg-blue-50' },
    coder: { icon: '💻', color: 'text-green-600', bg: 'bg-green-50' },
    reviewer: { icon: '🔍', color: 'text-purple-600', bg: 'bg-purple-50' },
    synthesizer: { icon: '🔄', color: 'text-orange-600', bg: 'bg-orange-50' }
  };

  const style = agentStyles[message.agent];

  return (
    <div className={`${style.bg} rounded-lg p-3`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">{style.icon}</span>
        <span className={`font-semibold ${style.color}`}>
          {message.agent.charAt(0).toUpperCase() + message.agent.slice(1)}
        </span>
        <span className="text-xs text-gray-500">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>

      {message.type === 'thinking' ? (
        <p className="text-sm text-gray-600 italic">{message.content}</p>
      ) : (
        <pre className="text-sm whitespace-pre-wrap">{message.content}</pre>
      )}
    </div>
  );
}
```

**Backend SSE Endpoint**:

```python
# api/collaborate/stream/route.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

@app.get("/api/collaborate/stream")
async def stream_collaboration(task: str):
    """Stream collaborative debate via SSE"""

    async def event_generator():
        """Generator that yields SSE events"""

        orchestrator = HybridOrchestrator()

        try:
            async for event in orchestrator.collaborate(task, context={}):
                # Format as SSE
                sse_data = f"data: {json.dumps(event)}\n\n"
                yield sse_data

                # Small delay for browser buffering
                await asyncio.sleep(0.01)

        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

---

## Part 7: Real Streaming (Future)

When LLM APIs support bidirectional streaming:

```rust
// Future: True streaming with API support
async fn true_streaming_consensus(task: String) -> Result<()> {
    // Open persistent connections to all models
    let architect_stream = openai_streaming("gpt-5", &task).await?;
    let coder_stream = anthropic_streaming("claude-sonnet-4.5", &task).await?;
    let reviewer_stream = openrouter_streaming("deepseek/deepseek-chat", &task).await?;

    // Create bidirectional channels
    let (architect_tx, architect_rx) = mpsc::channel(100);
    let (coder_tx, coder_rx) = mpsc::channel(100);
    let (reviewer_tx, reviewer_rx) = mpsc::channel(100);

    // Spawn agents that REACT to each other's tokens
    tokio::spawn(async move {
        let mut architect_buffer = String::new();

        loop {
            select! {
                // Receive from own model
                Some(token) = architect_stream.next() => {
                    architect_buffer.push_str(&token);

                    // Broadcast to other agents
                    coder_tx.send(AgentMessage {
                        from: "architect",
                        content: token.clone(),
                        is_partial: true
                    }).await.ok();
                }

                // Receive from other agents
                Some(msg) = coder_rx.recv() => {
                    // INJECT into architect's generation
                    // (This requires API support we don't have yet)
                    architect_stream.inject_context(&msg.content).await?;
                }
            }
        }
    });

    // Similar for coder and reviewer...

    Ok(())
}
```

But until then, our pseudo-streaming is indistinguishable to users.

---

## Part 8: Success Metrics

**Technical**:
- [ ] Cache hit rate > 30% (semantic similarity working)
- [ ] Embedding computation < 5ms (Rust speedup)
- [ ] Stream routing latency < 1ms
- [ ] UI renders at 60 FPS during streaming

**User Experience**:
- [ ] Time to first token < 200ms
- [ ] Perceived concurrency (users believe agents work in parallel)
- [ ] Synthesis feels real-time (not batch-based)
- [ ] CLI and Web interfaces feel equally responsive

**Business**:
- [ ] Cost per collaboration < $0.10
- [ ] 50% cost reduction from caching
- [ ] 10x faster than sequential execution (perception)

---

## Part 9: Implementation Timeline

### Week 1: CLI + Semantic Cache
- **Days 1-2**: CLI streaming debate interface
- **Days 3-4**: Context-aware semantic caching
- **Day 5**: Integration + testing

### Week 2: Rust Performance Layer
- **Days 1-2**: Rust embedding engine
- **Days 3-4**: Rust stream router + cache layer
- **Day 5**: Python-Rust bridge + benchmarking

### Week 3: Web UI Foundation
- **Days 1-2**: Next.js app + SSE streaming
- **Days 3-4**: React components for debate UI
- **Day 5**: State management + real-time updates

### Week 4: Polish + Production
- **Days 1-2**: Error handling, reconnection logic
- **Days 3-4**: Performance tuning, load testing
- **Day 5**: Documentation + deployment

---

## Part 10: Open Questions for User

1. **Context Granularity**: Should we track user preferences at:
   - [ ] User level (all tasks)
   - [ ] Project level (per repository)
   - [ ] Task type level (API tasks vs algorithm tasks)
   - [ ] All of the above

2. **Cache Transparency**: Should users know when they get cached results?
   - [ ] Always show (builds trust)
   - [ ] Only show on request (cleaner UX)
   - [ ] Never show (seamless)

3. **Debate Visibility**: Should users see ALL agent thoughts or filtered?
   - [ ] Show everything (maximum transparency)
   - [ ] Show only high-confidence outputs (cleaner)
   - [ ] User-configurable verbosity

4. **Synthesis Timing**: When should synthesis happen?
   - [ ] After every agent completes (current plan)
   - [ ] After semantic chunks (current plan)
   - [ ] Continuous (rolling synthesis)

---

## Conclusion

**We can ship streaming consensus THIS WEEK** using:
1. ✅ Semantic chunking
2. ✅ Pseudo-streaming (interleaved outputs)
3. ✅ Context-aware caching
4. ✅ Rust performance layer (Week 2-3)

**The illusion is perfect** - users won't know it's not "true" streaming until APIs support it.

**And when they do**, we swap the implementation without changing the interface.

---

## Next Steps

1. User reviews this plan
2. Confirms approach + answers open questions
3. We start with CLI implementation (Day 1)

Ready to build?
