# Streaming and Semantic Caching Guide

Complete guide to the streaming debate interface and context-aware semantic caching features.

---

## Overview

This implementation adds two major features to the Facilitair collaborative orchestration system:

1. **CLI Streaming Debate Interface** - Real-time visualization of agent collaboration
2. **Context-Aware Semantic Caching** - Intelligent result caching based on task similarity and context

Both features follow the design patterns from `STREAMING_CONSENSUS_IMPLEMENTATION.md`.

---

## 1. CLI Streaming Debate Interface

### What It Does

The streaming interface provides real-time visualization of the 5-stage collaboration pipeline:

- **Live agent status** - See which agents are thinking/working/done
- **Real-time debate log** - Watch agent outputs as they stream
- **Synthesis panel** - View latest collaborative results
- **Elapsed time tracking** - Monitor session duration
- **Color-coded agents** - [ARCH] Architect (cyan), [CODE] Coder (green), [REVW] Reviewer (yellow), etc.

### Usage

```bash
# Standard output (non-streaming)
python3 cli.py collaborate "Write a factorial function"

# Streaming debate interface
python3 cli.py collaborate "Write a factorial function" --stream
```

### Example Output

```
+- Collaborative Session ----------------------------+
| Task: Write a factorial function                   |
| Strategy: BALANCED                                  |
| Elapsed: [00:00.50]                                |
+----------------------------------------------------+

+- Agent Status ---------+  +- Live Debate ---------------+
| [ARCH]  Architect: done    |  | [00:00.10] [ARCH]  Architect:  |
| [CODE] Coder: working      |  |   Analyzing requirements... |
| [REVW] Reviewer: idle      |  |                             |
| [REFN] Refiner: idle       |  | [00:00.50] > Architect:    |
| [DOCS] Documenter: idle    |  |   This needs a recursive... |
+------------------------+  |                             |
                            | [00:00.80] [CODE] Coder:       |
                            |   Implementing factorial... |
                            +-----------------------------+

+- Latest Synthesis ---------------------------------+
| Sequential workflow stage 1 completed successfully  |
| Architect designed recursive factorial approach    |
+----------------------------------------------------+
```

### Implementation Details

**File**: `cli_streaming_debate.py`

**Key Components**:
- `CLIDebateInterface` - Main interface class using Rich library
- `stream_collaborative_debate()` - Event generator for streaming
- `DebateEvent` - Event dataclass for agent actions

**Integration**:
- Integrated into `cli.py` with `--stream` flag
- Works with existing `CollaborativeOrchestrator`
- Uses Rich's `Live` display for smooth updates

---

## 2. Context-Aware Semantic Caching

### What It Does

The semantic cache stores collaboration results and retrieves them based on:

1. **Task similarity** - Uses sentence embeddings (cosine similarity)
2. **Context matching** - Same query, different context = different cache entry

This ensures:
- [OK] "Build REST API" for Python startup → Cache entry A
- [OK] "Build REST API" for Java enterprise → Cache entry B (different!)
- [OK] "Create RESTful API" for Python startup → Might hit Cache entry A (similar wording)

### Usage

```python
from cached_orchestrator import CachedStreamingOrchestrator

# Initialize with caching enabled
orchestrator = CachedStreamingOrchestrator(cache_enabled=True)

# Define context
context = {
    "preferred_language": "python",
    "frameworks": ["fastapi"],
    "security_level": "standard",
    "team_size": "small"
}

# Collaborate (checks cache first)
result = await orchestrator.collaborate(
    "Build a REST API with authentication",
    context
)

# Result will be cached for future use
```

### Context Structure

The context dictionary supports:

```python
{
    "preferred_language": str,      # e.g., "python", "java", "rust"
    "frameworks": List[str],         # e.g., ["fastapi", "sqlalchemy"]
    "security_level": str,           # e.g., "standard", "enterprise", "high"
    "compliance": List[str],         # e.g., ["SOC2", "HIPAA", "GDPR"]
    "team_size": str,                # e.g., "small", "medium", "large"
    "existing_stack": List[str]      # e.g., ["postgresql", "redis"]
}
```

### Cache Hit UX

When a cache hit occurs:

```
[CACHE] Cache hit! (similarity > 0.92)
   Cached: 2 hours ago
   Context: python, fastapi, standard

* Retrieving cached solution...

[Streams cached result with realistic pacing]
```

### Cache Miss UX

When no match is found:

```
[REVW] Cache miss - starting collaborative session...

[Real-time debate as shown in streaming interface]

[CACHE] Result cached for future use
```

### Performance

| Metric | Value |
|--------|-------|
| **Similarity Threshold** | 0.92 (high precision) |
| **Cache TTL** | 7 days |
| **Embedding Model** | all-MiniLM-L6-v2 (384 dimensions) |
| **Lookup Time** | ~10-50ms (in-memory search) |

### Implementation Details

**File**: `semantic_cache.py`

**Key Components**:
- `ContextAwareSemanticCache` - Main cache class
- `_create_embedding()` - Generates task + context embeddings
- `_canonicalize_context()` - Normalizes context for consistent matching
- `_calculate_similarity()` - Cosine similarity between embeddings

**Storage**:
- **Backend**: Redis (async)
- **Key Format**: `cache:{sha256(embedding)}`
- **Value Format**: JSON with task, context, result, embedding, metadata

**Integration**:
- `CachedStreamingOrchestrator` wraps `CollaborativeOrchestrator`
- Transparent caching - works with existing code
- Cache stats tracking (hits, misses, hit rate)

---

## 3. Combined Usage: Streaming + Caching

The two features work seamlessly together:

```python
from cached_orchestrator import CachedStreamingOrchestrator

orchestrator = CachedStreamingOrchestrator(cache_enabled=True)

context = {
    "preferred_language": "python",
    "frameworks": ["fastapi"],
    "security_level": "standard"
}

# Stream with caching
async for event in orchestrator.stream_collaborate(
    "Build authentication API",
    context
):
    if event["type"] == "cache_hit":
        print(f"[CACHE] {event['message']}")
    elif event["type"] == "output_chunk":
        print(event["content"], end="", flush=True)
    elif event["type"] == "complete":
        print(f"\n[OK] Done! (cached={event['cached']})")
```

### Event Types

Streaming events include:

- `cache_hit` - Cache match found
- `cache_miss` - No cache match
- `output_chunk` - Partial result streaming
- `collaboration_complete` - Real collaboration finished
- `cached` - Result cached for future
- `complete` - Session complete

---

## 4. Setup and Dependencies

### Install Dependencies

```bash
# Core dependencies
pip install sentence-transformers redis numpy

# For CLI streaming
pip install rich

# For orchestrator
pip install -r requirements.txt
```

### Redis Setup

```bash
# Install Redis (macOS)
brew install redis

# Start Redis server
redis-server

# Verify connection
redis-cli ping  # Should return "PONG"
```

### Configuration

The cache uses sensible defaults but can be configured:

```python
cache = ContextAwareSemanticCache(
    redis_url="redis://localhost:6379/0",
    model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.92  # Adjust for more/fewer hits
)
```

**Similarity Threshold Guide**:
- `0.95+` - Very strict (exact matches only)
- `0.92` - **Default** (high precision, some flexibility)
- `0.85-0.90` - Moderate (more cache hits, less precision)
- `< 0.85` - Loose (many hits, risk of incorrect matches)

---

## 5. Testing

### Test Streaming CLI

```bash
# Simple task
python3 cli.py collaborate "Write factorial function" --stream

# Complex task
python3 cli.py collaborate "Build REST API with JWT auth" --stream
```

### Test Semantic Cache

```bash
# Run cache demo
python3 semantic_cache.py
```

Expected output:
```
=== Semantic Cache Demo ===

Scenario 1: Exact match
  First request: None
  Second request: Here's a FastAPI implementation...

Scenario 2: Different context
  Result: None

Scenario 3: Similar wording
  Result: Here's a FastAPI implementation...

Cache Statistics:
  hits: 2
  misses: 2
  total_requests: 4
  hit_rate: 50.0%
```

### Test Cached Orchestrator

```bash
# Run orchestrator demo
python3 cached_orchestrator.py
```

---

## 6. Architecture

### Streaming Architecture

```
+--------------------------------------+
|       CLI (Rich Library)             |
|  - Live display                      |
|  - Agent status panel                |
|  - Debate log panel                  |
|  - Synthesis panel                   |
+--------------+-----------------------+
               | Events
+--------------v-----------------------+
|  CLIDebateInterface                  |
|  - stream_debate()                   |
|  - Event handling                    |
+--------------+-----------------------+
               | Generator
+--------------v-----------------------+
|  stream_collaborative_debate()       |
|  - Yields DebateEvent objects        |
|  - Integrates with orchestrator      |
+--------------+-----------------------+
               |
+--------------v-----------------------+
|  CollaborativeOrchestrator           |
|  - Sequential 5-stage pipeline       |
|  - Architect → Coder → Reviewer →... |
+--------------------------------------+
```

### Caching Architecture

```
+--------------------------------------+
|   User Request                       |
|   - Task: "Build REST API"           |
|   - Context: {language, frameworks}  |
+--------------+-----------------------+
               |
+--------------v-----------------------+
|  CachedStreamingOrchestrator         |
|  1. Create embedding                 |
|  2. Check cache                      |
+--------------+-----------------------+
               |
        +------+------+
        |             |
    Cache Hit    Cache Miss
        |             |
        v             v
+-------------+  +------------------+
| Return      |  | Run              |
| Cached      |  | Collaboration    |
| Result      |  |                  |
+-------------+  +--------+---------+
                          |
                +---------v----------+
                | Cache Result       |
                +--------------------+
```

### Storage Schema

**Redis Key**: `cache:{sha256(embedding)}`

**Redis Value** (JSON):
```json
{
  "task": "Build a REST API with auth",
  "context": {
    "preferred_language": "python",
    "frameworks": ["fastapi"],
    "security_level": "standard"
  },
  "result": "Here's the implementation...",
  "embedding": "a1b2c3...",  // hex-encoded bytes
  "metadata": {
    "agents_used": ["architect", "coder", "reviewer"],
    "quality": 0.92,
    "duration": 15.3
  },
  "cached_at": 1697654321.123
}
```

---

## 7. Performance Considerations

### Embedding Computation

Current: **~45ms** per embedding (CPU)

Future optimization with Rust:
```rust
// Rust performance layer (from design doc)
fn compute_embedding(&self, text: String) -> Vec<f32> {
    // 15x faster: ~3ms
    self.embedder.encode(&text)
}
```

### Cache Lookup

Current: **O(n)** scan through all cache entries

Future optimization:
- Use vector database (Pinecone, Weaviate, Qdrant)
- HNSW index for O(log n) lookup
- Target: **sub-millisecond** lookups

### Memory Usage

- **Embedding model**: ~90MB in memory
- **Per cache entry**: ~2-10KB (depends on result size)
- **1000 entries**: ~10-20MB

---

## 8. Best Practices

### Context Design

**Good Context** (specific, stable):
```python
{
    "preferred_language": "python",
    "frameworks": ["fastapi", "sqlalchemy"],
    "security_level": "enterprise",
    "compliance": ["HIPAA"]
}
```

**Bad Context** (too specific, unstable):
```python
{
    "user_id": "12345",  # Too specific!
    "timestamp": "2024-01-15",  # Changes every request!
    "random_field": "value"  # Unnecessary!
}
```

### Cache Warming

Pre-populate cache with common tasks:

```python
common_tasks = [
    ("Build REST API", python_context),
    ("Implement authentication", python_context),
    ("Create database schema", python_context)
]

for task, context in common_tasks:
    await orchestrator.collaborate(task, context)
```

### Monitoring

Track cache performance:

```python
stats = await orchestrator.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']}")

# Alert if hit rate drops below 20%
if float(stats['hit_rate'].rstrip('%')) < 20:
    print("[WARN]  Low cache hit rate!")
```

---

## 9. Troubleshooting

### Issue: Redis Connection Error

```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
redis-server

# Or specify different Redis URL
orchestrator = CachedStreamingOrchestrator(
    redis_url="redis://your-redis-server:6379/0"
)
```

### Issue: Low Cache Hit Rate

**Possible causes**:
1. Threshold too high (try 0.85 instead of 0.92)
2. Contexts too varied
3. Not enough cached entries yet

**Solution**:
```python
# Lower threshold
cache = ContextAwareSemanticCache(similarity_threshold=0.85)

# Standardize contexts
# Use consistent framework names, security levels, etc.
```

### Issue: Slow Embedding Computation

**Solution** (future):
- Use Rust performance layer (15x speedup)
- Batch multiple embeddings together
- Cache embeddings separately

---

## 10. Roadmap

### Completed [OK]
- [x] CLI streaming debate interface
- [x] Context-aware semantic caching
- [x] Redis integration
- [x] Cache statistics

### In Progress 
- [ ] Web UI streaming interface
- [ ] Rust performance layer
- [ ] Vector database integration

### Planned 
- [ ] Cache analytics dashboard
- [ ] A/B testing framework
- [ ] Multi-user context profiles
- [ ] Cache warming strategies

---

## 11. Related Documentation

- `STREAMING_CONSENSUS_IMPLEMENTATION.md` - Full design document
- `README.md` - Project overview
- `cli_streaming_debate.py` - Streaming CLI implementation
- `semantic_cache.py` - Cache implementation
- `cached_orchestrator.py` - Integrated orchestrator

---

## Questions?

For issues or questions:
1. Check the troubleshooting section above
2. Review the implementation files
3. Open an issue on GitHub

Happy streaming! 
