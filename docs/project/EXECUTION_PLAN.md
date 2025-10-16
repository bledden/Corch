# Streaming Consensus Implementation - Execution Plan
## Ready to Build

Based on the comprehensive review and validation of test results, here's the prioritized execution plan:

---

## [OK] Current Status

**Baseline System**: **100% pass rate** across all 3 strategy tests
- BALANCED strategy: 10/10 tasks passed (100%)
- OPEN strategy: 10/10 tasks passed (100%)
- CLOSED strategy: 10/10 tasks passed (100%)

**What This Means**: Core multi-agent orchestration is solid. Ready to add streaming + optimization layers.

---

## Phase 1: CLI Streaming Debate (Week 1) - **START HERE**

### Goal
Ship a working CLI that users can use this week, with streaming debate visualization.

### Tasks (Days 1-3)

#### Day 1: Basic CLI Streaming Framework
```bash
# Files to create:
cli_streaming_debate.py      # Main CLI interface
streaming_debate_events.py    # Event types for debate
```

**What to Build**:
1. **Rich TUI Framework**:
   - Use `rich` library (already in monitor.py)
   - Live-updating debate panel
   - Agent-specific color coding
   - Timestamp display

2. **Event Stream Generator**:
```python
@dataclass
class DebateEvent:
    type: str  # "agent_thinking", "agent_output", "synthesis"
    agent: str
    content: str
    timestamp: float
    chunk_id: Optional[str]

async def stream_collaborative_debate(task: str):
    """Generator that yields debate events"""
    # Chunk the task
    chunks = await intelligent_chunk_task(task)

    for chunk in chunks:
        # Start all agents on this chunk
        async for event in execute_chunk_debate(chunk):
            yield event
```

3. **Simple Chunking**:
```python
async def intelligent_chunk_task(task: str) -> List[Chunk]:
    """Break task into 2-3 semantic chunks"""
    # For MVP: Use simple rule-based chunking
    return [
        Chunk(id="requirements", description="Analyze requirements"),
        Chunk(id="implementation", description="Implement solution"),
        Chunk(id="review", description="Review and refine")
    ]
```

**Success Criteria**:
- [ ] User can run `python cli_streaming_debate.py "Build REST API"`
- [ ] See agents "debating" with interleaved outputs
- [ ] Synthesis appears at chunk boundaries
- [ ] Total time feels <30s for simple tasks

#### Day 2: Interleaved Agent Streaming
```bash
# Files to modify:
collaborative_orchestrator.py  # Add streaming method
```

**What to Build**:
1. **Stream Router**:
```python
async def interleave_streams(agent_streams: Dict[str, AsyncGenerator]):
    """Round-robin token display from multiple agents"""
    buffers = {agent: [] for agent in agent_streams}

    while any_agent_active:
        for agent in agents:
            if buffers[agent]:
                # Yield 5-10 tokens at a time
                tokens = buffers[agent][:10]
                yield DebateEvent(
                    type="agent_output",
                    agent=agent,
                    content="".join(tokens)
                )
                buffers[agent] = buffers[agent][10:]

            await asyncio.sleep(0.05)  # Realistic pacing
```

2. **Reactive Messages**:
```python
# After N tokens from one agent, show others "reacting"
if token_count % 50 == 0:
    yield DebateEvent(
        type="agent_thinking",
        agent="reviewer",
        content=f"Considering {architect}'s design..."
    )
```

**Success Criteria**:
- [ ] Agents appear to work concurrently (interleaved output)
- [ ] Realistic pacing (not too fast, not too slow)
- [ ] No long gaps where nothing happens

#### Day 3: Integration + Polish
**What to Build**:
1. Connect CLI to existing `CollaborativeOrchestrator`
2. Add cost/time tracking display
3. Handle errors gracefully (model failures, etc.)
4. Create simple docs: `CLI_USAGE.md`

**Success Criteria**:
- [ ] End-to-end working: task input → streaming debate → final result
- [ ] Costs displayed: "[OK] Complete (3.2s, $0.09)"
- [ ] Error messages don't crash CLI

---

## Phase 2: Context-Aware Semantic Caching (Days 4-5)

### Goal
30-40% cache hit rate, reducing cost and latency for repeated queries.

### Tasks

#### Day 4: Basic Semantic Cache
```bash
# Files to create:
semantic_cache.py  # Cache implementation
```

**What to Build**:
```python
from sentence_transformers import SentenceTransformer

class ContextAwareSemanticCache:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.redis = redis.Redis(host='localhost', port=6379)

    async def get(self, task: str, context: dict) -> Optional[str]:
        """Check if we've solved similar task in similar context"""
        # Embed task + context
        embedding = self.model.encode(
            f"{task} |CONTEXT| {json.dumps(context, sort_keys=True)}"
        )

        # Search for similar (cosine similarity > 0.92)
        cached_result = await self._find_similar(embedding, threshold=0.92)

        return cached_result

    async def set(self, task: str, context: dict, result: str):
        """Store result with embedding for future retrieval"""
        # Implementation here
```

**Context Dimensions to Track**:
```python
context = {
    "preferred_language": "python",  # or "typescript", "java", etc.
    "frameworks": ["fastapi", "sqlalchemy"],
    "security_level": "standard",  # or "enterprise", "high"
    "team_size": "small",  # or "medium", "large"
    "existing_stack": ["postgresql", "redis"]
}
```

**Success Criteria**:
- [ ] Same query + same context = cache hit
- [ ] Same query + different context = cache miss
- [ ] Cache hits return in <500ms

#### Day 5: Integration with CLI
**What to Build**:
1. Check cache before starting debate
2. If hit: stream cached result (with indication)
3. If miss: run debate, then cache result
4. Display cache status: "[MEMORY] Cache hit (94% match, saved $0.08, 4.2s faster)"

**Success Criteria**:
- [ ] Run same query twice → second is instant
- [ ] Run query with different context → no cache hit (correct)
- [ ] User sees cache savings

---

## Phase 3: Rust Performance Layer (Week 2-3) - **OPTIONAL FOR MVP**

### Goal
18x faster orchestration overhead (65ms → 3.5ms per request).

### Decision Point
**After Phase 1+2, evaluate**:
- Is current Python performance acceptable?
- Are we seeing latency issues?
- Is Rust complexity worth the gains?

If YES to Rust:
1. Set up Rust project with PyO3
2. Implement embedding computation in Rust (using Candle)
3. Implement stream routing in Rust (using Tokio)
4. Create Python bindings
5. Benchmark improvements

**Time Estimate**: 5-7 days for someone experienced with Rust
**Alternative**: Defer to future if Python is fast enough

---

## Phase 4: Web UI (Week 3-4) - **AFTER CLI WORKS**

### Goal
Professional web interface with streaming debate visualization.

### Approach
1. Start with CLI lessons learned (what UX works?)
2. Use Next.js 14 + React + SSE
3. Create debate visualization component
4. Add syntax highlighting for code outputs
5. Make it mobile-responsive

**Time Estimate**: 5-7 days

---

## Open Questions - **ANSWER THESE FIRST**

Before starting implementation, please confirm:

### 1. Context Granularity
How detailed should user context be?
- [ ] **Basic**: Just programming language
- [ ] **Medium**: Language + frameworks + team size  ← **Recommended**
- [ ] **Advanced**: Full tech stack + compliance + historical choices

### 2. Cache Transparency
Should users know when they get cached results?
- [ ] **Always show**: "[MEMORY] Cache hit (94% match, saved $0.08)"  ← **Recommended** (builds trust)
- [ ] **Show on hover**: Default hides, tooltip reveals
- [ ] **Never show**: Seamless experience

### 3. Debate Verbosity
How much should users see?
- [ ] **Maximum**: Every agent thought + token stream
- [ ] **Filtered**: Only high-confidence outputs  ← **Recommended** (cleaner)
- [ ] **Configurable**: User sets verbosity level (ideal but more work)

### 4. Synthesis Timing
When should synthesis happen?
- [ ] **After each agent**: Synthesize after every agent completes (too frequent)
- [ ] **Semantic chunks**: Synthesize at logical boundaries  ← **Recommended**
- [ ] **Rolling**: Continuous synthesis as tokens arrive (too complex for MVP)

---

## Success Metrics

### Week 1 (CLI + Cache)
- [ ] Working CLI with streaming debate
- [ ] 30%+ cache hit rate on repeated queries
- [ ] User can complete simple task in <30s
- [ ] Cost per collaboration < $0.10

### Week 2-3 (Rust - Optional)
- [ ] 18x faster orchestration overhead
- [ ] Embedding computation < 5ms
- [ ] No degradation in quality

### Week 4 (Web UI - If time)
- [ ] Professional web interface
- [ ] Real-time streaming visualization
- [ ] Mobile-responsive

---

## Risk Mitigation

### Technical Risks
1. **Streaming feels laggy**:
   - Mitigation: Use "agent thinking" placeholders
   - Tune interleaving delays (currently 50ms)

2. **Cache false positives**:
   - Mitigation: High similarity threshold (0.92)
   - Allow user to override: "Use fresh results"

3. **Chunking produces bad splits**:
   - Mitigation: Start with simple 2-3 chunk splits
   - Refine based on user feedback

### Scope Risks
1. **Feature creep (trying to do too much)**:
   - Mitigation: Stick to Phase 1+2 for MVP
   - Get user feedback before Phase 3+4

2. **Perfectionism (over-engineering)**:
   - Mitigation: "Good enough for user value" > "Perfect but late"
   - Ship CLI Week 1, iterate

---

## Next Immediate Steps

### Step 1: Answer Open Questions (30 minutes)
Review and select options for:
- Context granularity
- Cache transparency
- Debate verbosity
- Synthesis timing

### Step 2: Set Up Environment (30 minutes)
```bash
# Install dependencies
pip install rich sentence-transformers redis

# Start Redis (for caching)
brew install redis  # macOS
redis-server
```

### Step 3: Create First File (Day 1 Start)
```bash
# Create CLI file
touch cli_streaming_debate.py

# Implement basic structure
# (See Day 1 tasks above)
```

---

## Timeline Summary

| Phase | Duration | Outcome |
|-------|----------|---------|
| **Phase 1**: CLI Streaming | Days 1-3 | Working CLI users can try |
| **Phase 2**: Semantic Cache | Days 4-5 | 30-40% cost/latency savings |
| **Phase 3**: Rust (Optional) | Week 2-3 | 18x orchestration speedup |
| **Phase 4**: Web UI | Week 3-4 | Professional interface |

**Minimum Viable Product**: Phase 1 + 2 (5 days)
**Production Ready**: Phase 1 + 2 + 4 (3 weeks)
**Performance Optimized**: All phases (4 weeks)

---

## Decision: What to Build First?

Based on your directive **"I'd like to have something people can use this week"**:

### Recommended: **Phase 1 + 2 Only (Week 1)**

**Rationale**:
1. CLI is fastest to build and test
2. Semantic caching provides immediate value (cost savings)
3. Avoids complexity of Rust integration (can add later)
4. Gets user feedback quickly

**Alternative: If you want polish**:
Build Phase 1+2, then pause for user testing before continuing.

---

## Ready to Start?

Once you answer the 4 open questions and confirm the approach, I'm ready to:

1. Create the first file (`cli_streaming_debate.py`)
2. Implement the event streaming system
3. Integrate with existing `CollaborativeOrchestrator`
4. Add semantic caching
5. Test end-to-end

**Estimated time to working prototype**: 3-5 days of focused work.

Let me know if you want to proceed with Phase 1+2, or if you have any adjustments to the plan! [START]
