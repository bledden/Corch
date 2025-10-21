# CodeSwarm Enhanced Architecture
**With RAG, LangGraph, and Latest Models**

---

## 🧠 ENHANCED CONTEXT SOLUTION: RAG + LangGraph

### The Problem You Identified:
- Browser Use scrapes 3000 chars max (limited)
- Models need deep doc context, not just snippets
- Back-and-forth is inefficient

### The Solution: **Hybrid RAG + Live Scraping**

```
Documentation Strategy (3-Layer Context):

Layer 1: RAG Knowledge Base (Neo4j Graph RAG)
├─ Pre-indexed documentation (FastAPI, OAuth, OWASP, pytest)
├─ Semantic search for relevant sections
├─ Graph relationships (e.g., "OAuth requires JWT" → "JWT uses RSA")
└─ Returns: Top 5 most relevant doc sections (up to 10k tokens)

Layer 2: Live Browser Use Scraping
├─ Scrapes LATEST changes (e.g., FastAPI 0.115.0 released yesterday)
├─ Checks for version-specific differences
└─ Returns: Current API signatures, new features

Layer 3: LangGraph Orchestration
├─ Combines RAG results + live scraping
├─ Builds context graph (dependencies, relationships)
├─ Passes comprehensive context to agents
└─ Adaptive: If RAG has it, skip browsing (faster)
```

---

## 🗄️ RAG IMPLEMENTATION: Neo4j + LangChain

### Why Neo4j Graph RAG?
1. **Relationships matter**: "OAuth flow" connects to "JWT tokens", "PKCE", "refresh tokens"
2. **Context traversal**: Can walk graph to find related concepts
3. **Hackathon friendly**: Neo4j offers free AuraDB tier
4. **Judge appeal**: Neo4j is sponsor-adjacent (graph databases hot topic)

### Architecture:

```python
# rag_knowledge_base.py

from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter

class CodeDocRAG:
    """Neo4j-powered RAG for code documentation"""

    def __init__(self):
        # Neo4j AuraDB connection (free tier)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),  # Free AuraDB instance
            username="neo4j",
            password=os.getenv("NEO4J_PASSWORD")
        )

        # Vector store for semantic search
        self.vector_store = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            graph=self.graph,
            node_label="Documentation",
            text_node_properties=["content", "code_examples"],
            embedding_node_property="embedding"
        )

    async def index_documentation(self, docs: List[Dict]):
        """Pre-index documentation (do this once before demo)"""

        # Index FastAPI docs
        fastapi_docs = await self._fetch_fastapi_docs()
        # Index OAuth specs
        oauth_docs = await self._fetch_oauth_specs()
        # Index OWASP guidelines
        owasp_docs = await self._fetch_owasp_docs()
        # Index pytest docs
        pytest_docs = await self._fetch_pytest_docs()

        all_docs = fastapi_docs + oauth_docs + owasp_docs + pytest_docs

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)

        # Create graph relationships
        for chunk in chunks:
            # Extract entities (e.g., "OAuth2", "FastAPI", "JWT")
            entities = await self._extract_entities(chunk.page_content)

            # Create nodes and relationships
            await self._create_doc_node(chunk, entities)

        print(f"[RAG] Indexed {len(chunks)} documentation chunks")

    async def retrieve_context(
        self,
        query: str,
        agent_type: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Retrieve relevant documentation context"""

        # Semantic search in vector store
        results = await self.vector_store.asimilarity_search_with_score(
            query,
            k=top_k
        )

        # Graph traversal for related concepts
        related_docs = await self._traverse_graph(results[0].metadata["id"])

        # Combine results
        context = {
            "primary_docs": [r.page_content for r in results],
            "related_concepts": related_docs,
            "total_tokens": sum(len(r.page_content.split()) for r in results) * 1.3,
            "sources": [r.metadata.get("source", "Unknown") for r in results]
        }

        return context

    async def _traverse_graph(self, doc_id: str) -> List[str]:
        """Walk graph to find related documentation"""

        query = """
        MATCH (d:Documentation {id: $doc_id})-[r:RELATES_TO|REQUIRES|IMPLEMENTS*1..2]->(related:Documentation)
        RETURN related.content as content
        LIMIT 10
        """

        result = await self.graph.query(query, params={"doc_id": doc_id})
        return [r["content"] for r in result]

    async def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from documentation"""
        # Use Claude to extract entities
        from anthropic import Anthropic
        client = Anthropic()

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Extract key technical entities from this documentation:

{text[:1000]}

Return as JSON array of strings: ["entity1", "entity2", ...]"""
            }]
        )

        import json
        entities = json.loads(response.content[0].text)
        return entities
```

---

## 🔀 LANGGRAPH ORCHESTRATION

### Why LangGraph?
1. **State management**: Tracks context across agent calls
2. **Conditional routing**: RAG first, then browse if needed
3. **Parallel execution**: Multiple agents + RAG queries simultaneously
4. **Hackathon friendly**: Clean visual graph representation

### Implementation:

```python
# langgraph_orchestrator.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

class CodeSwarmState(TypedDict):
    """Shared state across all agents"""
    task: str
    rag_context: Dict[str, Any]
    browsed_docs: Dict[str, str]  # {agent: docs}
    agent_contributions: Dict[str, str]  # {agent: code}
    galileo_scores: Dict[str, float]  # {agent: score}
    final_code: str
    quality_score: float

class CodeSwarmGraph:
    """LangGraph-based orchestration"""

    def __init__(self, rag: CodeDocRAG):
        self.rag = rag
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the orchestration graph"""

        workflow = StateGraph(CodeSwarmState)

        # Node 1: RAG Retrieval (parallel for all agents)
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)

        # Node 2: Conditional Browsing (only if RAG insufficient)
        workflow.add_node("browser_scraping", self._browser_scraping_node)

        # Node 3: Agent Code Generation (parallel)
        workflow.add_node("architecture_agent", self._architecture_agent_node)
        workflow.add_node("implementation_agent", self._implementation_agent_node)
        workflow.add_node("security_agent", self._security_agent_node)
        workflow.add_node("testing_agent", self._testing_agent_node)

        # Node 4: Galileo Evaluation
        workflow.add_node("galileo_eval", self._galileo_eval_node)

        # Node 5: Improvement Loop (if scores < 90)
        workflow.add_node("improve_agents", self._improve_agents_node)

        # Node 6: Final Synthesis
        workflow.add_node("synthesize", self._synthesize_node)

        # Define edges
        workflow.set_entry_point("rag_retrieval")

        workflow.add_conditional_edges(
            "rag_retrieval",
            self._should_browse,
            {
                "browse": "browser_scraping",
                "skip": "architecture_agent"  # RAG has enough context
            }
        )

        workflow.add_edge("browser_scraping", "architecture_agent")

        # Parallel agent execution
        workflow.add_edge("architecture_agent", "implementation_agent")
        workflow.add_edge("architecture_agent", "security_agent")
        workflow.add_edge("architecture_agent", "testing_agent")

        # All agents converge to evaluation
        workflow.add_edge("implementation_agent", "galileo_eval")
        workflow.add_edge("security_agent", "galileo_eval")
        workflow.add_edge("testing_agent", "galileo_eval")

        workflow.add_conditional_edges(
            "galileo_eval",
            self._should_improve,
            {
                "improve": "improve_agents",
                "synthesize": "synthesize"
            }
        )

        workflow.add_edge("improve_agents", "galileo_eval")  # Loop back
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    async def _rag_retrieval_node(self, state: CodeSwarmState) -> CodeSwarmState:
        """Retrieve documentation from RAG"""
        print("[LANGGRAPH] 📚 Retrieving from RAG knowledge base...")

        # Parallel RAG queries for each agent specialty
        queries = {
            "architecture": f"{state['task']} - system architecture patterns",
            "implementation": f"{state['task']} - implementation code examples",
            "security": f"{state['task']} - security best practices",
            "testing": f"{state['task']} - testing strategies"
        }

        rag_results = {}
        for agent, query in queries.items():
            context = await self.rag.retrieve_context(query, agent)
            rag_results[agent] = context

        state["rag_context"] = rag_results
        print(f"[LANGGRAPH] ✅ RAG retrieved {sum(c['total_tokens'] for c in rag_results.values()):.0f} tokens")
        return state

    def _should_browse(self, state: CodeSwarmState) -> str:
        """Decide if we need live browsing"""

        # Check RAG coverage
        total_tokens = sum(
            c["total_tokens"]
            for c in state["rag_context"].values()
        )

        # If RAG has >5000 tokens of context, skip browsing
        if total_tokens > 5000:
            print("[LANGGRAPH] ✅ RAG has sufficient context, skipping browser")
            return "skip"
        else:
            print("[LANGGRAPH] ⚠️ RAG context insufficient, triggering browser")
            return "browse"

    async def _browser_scraping_node(self, state: CodeSwarmState) -> CodeSwarmState:
        """Live documentation scraping with Browser Use"""
        from browser_use import Browser

        print("[LANGGRAPH] 🌐 Live documentation scraping...")

        browser = Browser(headless=False)

        # Each agent browses their specialty
        urls = {
            "architecture": "https://oauth.net/2/",
            "implementation": "https://fastapi.tiangolo.com/",
            "security": "https://owasp.org/www-project-top-ten/",
            "testing": "https://docs.pytest.org/"
        }

        browsed_docs = {}
        for agent, url in urls.items():
            page = await browser.goto(url)
            docs = await page.extract_text(max_length=3000)
            browsed_docs[agent] = docs

        state["browsed_docs"] = browsed_docs
        await browser.close()

        print("[LANGGRAPH] ✅ Browser scraping complete")
        return state

    async def _architecture_agent_node(self, state: CodeSwarmState) -> CodeSwarmState:
        """Architecture agent generates system design"""
        from anthropic import Anthropic

        client = Anthropic()

        # Combine RAG + browsed docs
        context = state["rag_context"]["architecture"]["primary_docs"]
        if "browsed_docs" in state:
            context += [state["browsed_docs"]["architecture"]]

        full_context = "\n\n".join(context)

        response = await client.messages.create(
            model="claude-opus-4.1-20250514",  # LATEST Claude Opus 4.1
            max_tokens=3000,
            system="""You are an Architecture Specialist AI.
Your role: Design high-level system architecture.
Focus: API structure, data flow, component organization.
Use the provided documentation to ensure best practices.""",
            messages=[{
                "role": "user",
                "content": f"""Task: {state['task']}

Documentation context:
{full_context[:8000]}

Generate architecture design with:
1. API endpoint structure
2. Data models
3. Component relationships
4. Integration points"""
            }]
        )

        state["agent_contributions"]["architecture"] = response.content[0].text
        print(f"[ARCHITECTURE AGENT] ✅ Generated design ({len(response.content[0].text)} chars)")
        return state

    # Similar nodes for implementation, security, testing agents...

    async def _galileo_eval_node(self, state: CodeSwarmState) -> CodeSwarmState:
        """Galileo evaluates all contributions"""
        from galileo_observe import GalileoObserve

        galileo = GalileoObserve()

        scores = {}
        for agent, code in state["agent_contributions"].items():
            eval_result = await galileo.evaluate(
                input=state["task"],
                output=code,
                metrics=["correctness", "completeness", "quality", "security"]
            )
            scores[agent] = eval_result.aggregate_score

        state["galileo_scores"] = scores
        print(f"[GALILEO] 📊 Scores: {scores}")
        return state

    def _should_improve(self, state: CodeSwarmState) -> str:
        """Decide if agents need to improve"""

        avg_score = sum(state["galileo_scores"].values()) / len(state["galileo_scores"])

        if avg_score < 90:
            print(f"[LANGGRAPH] ⚠️ Avg score {avg_score:.1f} < 90, improving...")
            return "improve"
        else:
            print(f"[LANGGRAPH] ✅ Avg score {avg_score:.1f} ≥ 90, synthesizing...")
            return "synthesize"
```

---

## 🎯 LATEST MODEL ASSIGNMENTS (From Your Facilitair Config)

Based on your `openrouter_client.py`, here are the **LATEST models**:

```python
AGENT_MODEL_MAP = {
    # Architecture Agent: Best reasoning for system design
    "architecture": {
        "model": "anthropic/claude-opus-4.1",  # Claude Opus 4.1 (Aug 5, 2025)
        "reasoning": "Latest Claude Opus - best for architectural reasoning",
        "context": 200000,  # 200k context window
        "cost_per_1m_tokens": {"prompt": 15, "completion": 75}
    },

    # Implementation Agent: Best code generation
    "implementation": {
        "model": "openai/gpt-5",  # GPT-5 (confirmed working)
        "reasoning": "GPT-5 excels at code implementation",
        "context": 128000,
        "cost_per_1m_tokens": {"prompt": 5, "completion": 15}
    },

    # Security Agent: Deep reasoning for security analysis
    "security": {
        "model": "anthropic/claude-opus-4.1",  # Claude Opus 4.1
        "reasoning": "Claude Opus best for security reasoning",
        "context": 200000,
        "cost_per_1m_tokens": {"prompt": 15, "completion": 75}
    },

    # Testing Agent: Fast, cost-effective for test generation
    "testing": {
        "model": "x-ai/grok-4",  # Grok-4 (98% HumanEval)
        "reasoning": "Grok-4 has 98% HumanEval score - perfect for tests",
        "context": 32000,
        "cost_per_1m_tokens": {"prompt": 5, "completion": 10}
    },

    # RAG Entity Extraction: Fast, cheap for extraction tasks
    "rag_extraction": {
        "model": "anthropic/claude-sonnet-4",  # Claude 4 Sonnet
        "reasoning": "Fast Claude Sonnet for entity extraction",
        "context": 200000,
        "cost_per_1m_tokens": {"prompt": 3, "completion": 15}
    }
}
```

### Why These Specific Models?

1. **Claude Opus 4.1** (Architecture + Security):
   - Released Aug 5, 2025 (BLEEDING EDGE)
   - 200k context window (handles massive RAG results)
   - Best reasoning capability for complex system design
   - Excels at security analysis (OWASP compliance)

2. **GPT-5** (Implementation):
   - Confirmed working in your tests
   - Best code generation quality
   - Fast completion times
   - Balances quality vs cost

3. **Grok-4** (Testing):
   - 98% HumanEval score (better than GPT-4)
   - Specifically optimized for code understanding
   - Great for generating comprehensive test suites
   - Lower cost than Claude/GPT-5

4. **Claude Sonnet 4** (RAG):
   - Fast entity extraction
   - Good at structured output (JSON)
   - Cost-effective for high-volume tasks

---

## 🚀 COMPLETE FLOW WITH RAG + LANGGRAPH

```
User Input: "Build secure FastAPI OAuth endpoint"
    ↓
[LangGraph Entry: rag_retrieval node]
    ↓
RAG Knowledge Base (Neo4j):
├─ Query: "FastAPI OAuth patterns"
├─ Vector search: Top 5 similar docs (8,247 tokens)
├─ Graph traversal: Related concepts (JWT, PKCE, refresh tokens)
└─ Returns: Comprehensive documentation context
    ↓
[LangGraph Decision: should_browse?]
├─ RAG tokens: 8,247 > 5,000 threshold
└─ Decision: SKIP browser scraping (RAG sufficient)
    ↓
[LangGraph: 4 Parallel Agent Nodes]
├─ Architecture Agent (Claude Opus 4.1):
│   - Input: RAG docs (OAuth specs, API patterns)
│   - Output: Endpoint structure, data models
│   - Time: 8 seconds
│
├─ Implementation Agent (GPT-5):
│   - Input: RAG docs (FastAPI examples, OAuth flows)
│   - Output: Python code with OAuth2PasswordBearer
│   - Time: 6 seconds
│
├─ Security Agent (Claude Opus 4.1):
│   - Input: RAG docs (OWASP Top 10, JWT security)
│   - Output: Security middleware, token validation
│   - Time: 9 seconds
│
└─ Testing Agent (Grok-4):
    - Input: RAG docs (pytest patterns, FastAPI testing)
    - Output: Comprehensive test suite
    - Time: 5 seconds
    ↓
[LangGraph: galileo_eval node]
Galileo Evaluate:
- Architecture: 94/100 ✅
- Implementation: 87/100 ⚠️
- Security: 96/100 ✅
- Testing: 92/100 ✅
    ↓
[LangGraph Decision: should_improve?]
├─ Avg score: 92.25 ≥ 90
└─ Decision: Proceed to synthesis (no improvement needed)
    ↓
[LangGraph: synthesize node]
Final Code Quality: 92/100
Files generated:
- app/auth.py (Implementation)
- app/models.py (Architecture)
- app/security.py (Security)
- tests/test_auth.py (Testing)
    ↓
Commit to Daytona workspace ✅
Store pattern in WorkOS team library ✅
```

---

## 📊 DEMO ENHANCEMENTS WITH RAG

### Visual Split-Screen Update:

```
┌─────────────────────────────────────────────────────────────┐
│                    CODESWARM IDE                             │
├─────────────────────────────────────────────────────────────┤
│  Task: "Build secure FastAPI OAuth endpoint"                │
├──────────────────────────────┬──────────────────────────────┤
│   RAG KNOWLEDGE BASE         │   LANGGRAPH FLOW             │
│   [Neo4j Graph View]         │   [State Machine View]       │
│                              │                              │
│   📚 Searching...            │   ⚙️  rag_retrieval          │
│   ✅ Found 127 doc chunks    │   ↓                          │
│   🔗 Traversing graph...     │   ⚙️  should_browse?        │
│   ✅ 23 related concepts     │   → SKIP (RAG sufficient)   │
│                              │   ↓                          │
│   Total Context: 8,247 tokens│   ⚙️  4 agents (parallel)  │
│   Sources:                   │   ├─ architecture_agent     │
│   - FastAPI docs v0.115.0    │   ├─ implementation_agent   │
│   - OAuth 2.1 spec          │   ├─ security_agent         │
│   - OWASP API Security      │   └─ testing_agent          │
│   - pytest docs v8.3        │   ↓                          │
│                              │   ⚙️  galileo_eval          │
│   [Graph visualization       │   ↓                          │
│    showing relationships]    │   ⚙️  synthesize            │
│                              │   ✅ COMPLETE                │
├──────────────────────────────┴──────────────────────────────┤
│   AGENT OUTPUTS (4 panels below)                            │
└─────────────────────────────────────────────────────────────┘
```

### Demo Script Enhancement:

```
[0:45-1:15] THE MAGIC
"First, CodeSwarm queries its RAG knowledge base powered by Neo4j..."
[Show graph view with nodes lighting up]

"127 documentation chunks retrieved. 23 related concepts via graph traversal.
8,247 tokens of context - more than enough!"

[Show LangGraph state machine]
"LangGraph decides: Skip browser scraping, RAG has it covered."

[Split screen: 4 agent panels]
"Watch the agents work in parallel:
- Architecture Agent using Claude Opus 4.1 (latest!)
- Implementation Agent using GPT-5
- Security Agent using Claude Opus 4.1
- Testing Agent using Grok-4 (98% HumanEval)

Each agent gets FULL documentation context from RAG - no back-and-forth needed."
```

---

## ⏱️ REVISED TIMELINE WITH RAG

### Pre-Hackathon Setup (Do This Tonight - 1 hour):
- [ ] Set up Neo4j AuraDB (free tier)
- [ ] Index documentation into RAG (FastAPI, OAuth, OWASP, pytest)
- [ ] Test RAG queries
- **Why**: RAG indexing takes time, do it before the clock starts

### Hour 1: Core Setup
- [ ] Project structure
- [ ] LangGraph orchestration skeleton
- [ ] 4 Claude/GPT agents with prompts
- [ ] Test model API calls

### Hour 2: RAG + Browser Integration
- [ ] Connect to pre-indexed RAG
- [ ] Implement RAG retrieval logic
- [ ] Add Browser Use as fallback
- [ ] Test: RAG → agents → code

### Hour 3: Galileo + Learning
- [ ] Galileo evaluation integration
- [ ] Improvement loop in LangGraph
- [ ] Autonomous learning storage
- [ ] Test: 3 iterations showing improvement

### Hour 4: Daytona + WorkOS + Polish
- [ ] Daytona workspace integration
- [ ] WorkOS team authentication
- [ ] Split-screen UI (RAG graph + LangGraph flow + 4 agents)
- [ ] End-to-end testing

### Final 40min: Video + Submission
- [ ] Record demo
- [ ] Write submission
- [ ] Submit!

---

## 🏆 WHY RAG + LANGGRAPH WINS

### Additional Competitive Advantages:

1. **Neo4j Graph RAG** = Novel approach (judges haven't seen this yet)
2. **LangGraph** = Clean architecture (visual flow diagram impresses judges)
3. **Hybrid Strategy** = RAG-first, browse-fallback (best of both worlds)
4. **Latest Models** = Claude Opus 4.1 (Aug 2025), GPT-5, Grok-4 (cutting edge)
5. **No Back-and-Forth** = Single-shot context delivery (solves your concern)

### Talking Points for Judges:

> "Most coding assistants use small context windows and require multiple round trips. CodeSwarm uses Neo4j Graph RAG to load 8,000+ tokens of documentation context in a single query, then orchestrates 4 specialist agents with LangGraph state management. Each agent gets full context - no back-and-forth needed."

> "We're using the latest models: Claude Opus 4.1 released August 5th, GPT-5, and Grok-4 with 98% HumanEval score. This is bleeding-edge AI applied to real developer workflows."

---

## ✅ READY TO BUILD?

**Enhanced Stack**:
- ✅ Neo4j Graph RAG (pre-indexed docs)
- ✅ LangGraph (orchestration + state management)
- ✅ Browser Use (fallback for latest docs)
- ✅ Claude Opus 4.1 + GPT-5 + Grok-4 (latest models)
- ✅ Galileo (quality evaluation)
- ✅ Daytona (workspace integration)
- ✅ WorkOS (team authentication)

**Say "Let's build" and I'll start with pre-hackathon RAG setup!** 🚀
