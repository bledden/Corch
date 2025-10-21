# CodeSwarm - Production Specification
**Final Architecture for Hackathon (4 hours remaining)**

---

## 🎯 FINAL MODEL SELECTION (Maximum Anthropic + Best-in-Class)

### Core 4 Agents (Running in Parallel):

```python
PRODUCTION_MODELS = {
    # Architecture Agent: Best reasoning for system design
    "architecture": "anthropic/claude-sonnet-4.5",
    # ✅ Latest Claude, strongest reasoning
    # ✅ Anthropic sponsor showcase #1

    # Implementation Agent: Best code generation
    "implementation": "openai/gpt-5-pro",
    # ✅ GPT-5 PRO (not Codex) - better general code
    # ✅ Pro variant = flagship quality
    # NOTE: GPT-5-Codex is specialized but GPT-5-Pro is more reliable

    # Security Agent: Fast security expert
    "security": "anthropic/claude-opus-4.1",
    # ✅ Stronger than Haiku for security
    # ✅ Anthropic sponsor showcase #2
    # ✅ Best for OWASP, vulnerability detection

    # Testing Agent: GROK-4 (beats Nemotron for test generation)
    "testing": "x-ai/grok-4",
    # ✅ 98% HumanEval score
    # ✅ Faster than Nemotron
    # ✅ Better at edge case generation
}
```

### Why Grok-4 > Nemotron for Testing:
**Grok-4 advantages**:
- 98% HumanEval (test generation quality metric)
- Faster inference (better for parallel execution)
- Better edge case discovery (from X.AI's training data)
- More creative test scenarios

**Nemotron advantages**:
- Open source (cost-effective)
- Longer context window

**Decision**: **Grok-4** wins for quality. Testing requires creative edge cases, not just bulk generation.

---

## 🖼️ VISION MODEL - Use Cases & Value

### What Vision Model Does:
```python
VISION_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"  # 235B parameters!
```

### Practical Use Cases (Impressive for Demo):

#### Use Case 1: **Screenshot to Code**
```
User uploads: Figma mockup screenshot
    ↓
Vision Agent (Qwen3-VL 235B) analyzes:
- Layout structure (header, sidebar, main content)
- Component hierarchy (buttons, forms, cards)
- Styling (colors, spacing, typography)
- Interactive elements (dropdowns, modals)
    ↓
Generates: React/Tailwind component code
```

**Demo Impact**: "Watch CodeSwarm convert a design mockup to production code in 10 seconds"

#### Use Case 2: **Error Screenshot Debugging**
```
User uploads: Screenshot of error in browser console
    ↓
Vision Agent reads:
- Error message text
- Stack trace
- Console logs
- Network panel
    ↓
Security Agent: Identifies root cause
Implementation Agent: Generates fix
```

**Demo Impact**: "CodeSwarm can debug from screenshots - no copy-pasting errors"

#### Use Case 3: **Architecture Diagram → Code**
```
User uploads: Hand-drawn architecture diagram
    ↓
Vision Agent interprets:
- Boxes (services/components)
- Arrows (data flow)
- Labels (technology choices)
    ↓
Architecture Agent: Generates formal architecture doc
Implementation Agent: Scaffolds microservices
```

**Demo Impact**: "Whiteboard to working code in one step"

### Should We Include Vision in 4h Timeline?

**RECOMMENDATION: No, too risky for 4 hours**

**Why Skip**:
- Vision adds 1-2 hours complexity
- Requires image upload UI
- Demo requires preparing good screenshots
- Core multi-agent system is already impressive

**Include Post-Hackathon**:
- Mention in "Next Steps" slide
- Show architecture diagram (not implementation)
- Say "Vision coming in v2"

**If you insist on vision**: Only for ONE demo use case (screenshot to code), pre-prepare the screenshot

---

## 1️⃣ RAG ARCHITECTURE (Neo4j + LangGraph + Galileo Quality Gate)

### The "Only 90+ Patterns" Quality Gate Explained:

```python
class GalileoQualityGate:
    """Only store high-quality patterns in RAG"""

    async def evaluate_and_store(self, agent_type: str, task: str, code: str):
        """Galileo evaluates → Only 90+ goes to RAG"""

        # 1. Galileo evaluates the code
        evaluation = await galileo.evaluate(
            project="codeswarm",
            input=task,
            output=code,
            metadata={"agent": agent_type},
            metrics=[
                "correctness",      # Does it work?
                "completeness",     # All requirements met?
                "code_quality",     # Clean, maintainable?
                "security",         # No vulnerabilities?
                "best_practices"    # Follows conventions?
            ]
        )

        score = evaluation.aggregate_score  # 0-100

        # 2. Quality gate: Only 90+ patterns stored
        if score >= 90:
            # Store in Neo4j RAG
            await neo4j_rag.store_pattern(
                task=task,
                code=code,
                quality_score=score,
                agent_type=agent_type,
                galileo_feedback=evaluation.feedback
            )

            print(f"✅ Pattern stored (score: {score}/100)")
            return True
        else:
            print(f"❌ Pattern rejected (score: {score}/100 - below threshold)")

            # Don't pollute RAG with mediocre code
            # Instead, use Galileo feedback to improve
            improved_code = await self.improve_with_feedback(
                code=code,
                feedback=evaluation.feedback,
                agent_type=agent_type
            )

            # Re-evaluate improved version
            return await self.evaluate_and_store(agent_type, task, improved_code)
```

### Why This Matters:

**Without Quality Gate** (Bad):
```
Request 1: Generate code → Score 72/100 → Store in RAG ❌
Request 2: Retrieve 72/100 pattern → Generate similar bad code
Request 3: Retrieve more bad patterns → System gets WORSE over time!
```

**With 90+ Quality Gate** (Good):
```
Request 1: Generate code → Score 72/100 → Reject, improve → 91/100 → Store ✅
Request 2: Retrieve 91/100 pattern → Generate better code → 94/100 → Store ✅
Request 3: Retrieve 91-94 patterns → Generate excellent code → 96/100 → Store ✅

System gets BETTER over time (only learns from successes)
```

### The Complete Flow:

```
Agent generates code
    ↓
Galileo evaluates (0-100 score)
    ↓
    ├─ Score ≥ 90 → Store in Neo4j ✅
    │                (Future agents can learn from this)
    │
    └─ Score < 90 → Don't store ❌
                     ↓
                     Use Galileo feedback to improve
                     ↓
                     Re-evaluate until ≥ 90
                     ↓
                     THEN store
```

### Neo4j Storage Structure:

```cypher
// Pattern node
CREATE (p:Pattern {
    task: "Build FastAPI OAuth endpoint",
    code: "class OAuthHandler...",
    quality_score: 94.5,
    agent_type: "implementation",
    galileo_metrics: {
        correctness: 95,
        completeness: 92,
        code_quality: 96,
        security: 94
    },
    timestamp: "2025-10-18T15:30:00Z"
})

// Relationship to related patterns
MATCH (p1:Pattern {task: "FastAPI OAuth"})
MATCH (p2:Pattern {task: "FastAPI JWT"})
CREATE (p1)-[:RELATES_TO {similarity: 0.85}]->(p2)

// Relationship to user who created it (WorkOS)
MATCH (p:Pattern {task: "FastAPI OAuth"})
MATCH (u:User {email: "blake@facilitair.ai"})
CREATE (u)-[:CREATED]->(p)
```

### RAG Retrieval Example:

```python
async def retrieve_high_quality_patterns(task: str, agent_type: str, k: int = 5):
    """Retrieve only 90+ quality patterns from Neo4j"""

    cypher_query = """
    // Find similar patterns with quality ≥ 90
    MATCH (p:Pattern)
    WHERE p.agent_type = $agent_type
      AND p.quality_score >= 90
      AND p.task CONTAINS $keyword

    // Get related patterns too
    OPTIONAL MATCH (p)-[:RELATES_TO*1..2]->(related:Pattern)
    WHERE related.quality_score >= 90

    // Return sorted by quality
    RETURN p, related
    ORDER BY p.quality_score DESC
    LIMIT $k
    """

    results = neo4j.query(cypher_query, {
        "agent_type": agent_type,
        "keyword": extract_keywords(task),
        "k": k
    })

    # All results are guaranteed 90+ quality
    return results
```

---

## 2️⃣ MODEL EXECUTION: PARALLEL vs SEQUENTIAL

### Answer: **HYBRID - Parallel Agents, Sequential Stages**

```python
class CodeSwarmOrchestrator:
    """LangGraph-based orchestration"""

    async def execute(self, task: str):
        """Hybrid parallel + sequential execution"""

        # STAGE 1: RAG Retrieval (Sequential - must go first)
        rag_context = await self.rag_retrieve(task)

        # STAGE 2: Browser Use (Optional - Parallel if RAG insufficient)
        if self.needs_live_docs(rag_context):
            browsed_docs = await asyncio.gather(
                self.browse_docs("architecture", task),
                self.browse_docs("implementation", task),
                self.browse_docs("security", task),
                self.browse_docs("testing", task)
            )

        # STAGE 3: Agent Code Generation (PARALLEL - all 4 agents at once)
        agent_outputs = await asyncio.gather(
            self.architecture_agent(task, rag_context, browsed_docs),
            self.implementation_agent(task, rag_context, browsed_docs),
            self.security_agent(task, rag_context, browsed_docs),
            self.testing_agent(task, rag_context, browsed_docs)
        )
        # ⏱️ Time: ~5-8 seconds (parallel execution)

        # STAGE 4: Galileo Evaluation (PARALLEL - evaluate each agent)
        galileo_scores = await asyncio.gather(
            galileo.evaluate("architecture", agent_outputs[0]),
            galileo.evaluate("implementation", agent_outputs[1]),
            galileo.evaluate("security", agent_outputs[2]),
            galileo.evaluate("testing", agent_outputs[3])
        )
        # ⏱️ Time: ~1-2 seconds (parallel evaluation)

        # STAGE 5: Improvement Loop (SEQUENTIAL - if needed)
        for i, score in enumerate(galileo_scores):
            if score < 90:
                agent_outputs[i] = await self.improve_agent_output(
                    agent_type=["architecture", "implementation", "security", "testing"][i],
                    code=agent_outputs[i],
                    feedback=score.feedback
                )
                # Re-evaluate
                galileo_scores[i] = await galileo.evaluate(agent_type, agent_outputs[i])

        # STAGE 6: Synthesis (Sequential - combines all outputs)
        final_code = await self.synthesize(agent_outputs, galileo_scores)

        # STAGE 7: Store in RAG (Sequential - only if all ≥ 90)
        await self.store_high_quality_patterns(agent_outputs, galileo_scores)

        return final_code
```

### Timing Breakdown:

```
STAGE 1: RAG Retrieval         ~200ms  (Neo4j query)
STAGE 2: Browser Use            ~3s     (4 parallel browsers)
STAGE 3: Agent Generation       ~6s     (4 parallel LLM calls)
STAGE 4: Galileo Evaluation     ~1.5s   (4 parallel evals)
STAGE 5: Improvement (if <90)   ~4s     (re-generate + re-eval)
STAGE 6: Synthesis              ~500ms  (combine outputs)
STAGE 7: RAG Storage            ~300ms  (Neo4j write)

TOTAL: ~15.5 seconds (with improvement loop)
TOTAL: ~11.2 seconds (if all agents score 90+ first try)
```

### Why Hybrid (Not Pure Parallel)?

**Pure Parallel Issues**:
- RAG must go first (agents need context)
- Can't synthesize before agents finish
- Can't store before Galileo evaluates
- Improvement loop requires sequential retries

**Hybrid Benefits**:
- ✅ Agents run in parallel (4x speedup)
- ✅ Evaluations run in parallel (4x speedup)
- ✅ Sequential stages ensure data dependencies
- ✅ Total time: 11-15 seconds (fast enough for demo)

---

## 3️⃣ BROWSER USE - Documentation Scraping

### When Browser Use Activates:

```python
def needs_live_docs(rag_context: Dict) -> bool:
    """Decide if we need Browser Use"""

    # If RAG has 5+ high-quality examples, skip Browser Use
    if len(rag_context["examples"]) >= 5:
        return False

    # If RAG examples are old (>6 months), use Browser Use
    if rag_context["newest_timestamp"] < "2025-04-18":
        return True

    # If task mentions specific library version, use Browser Use
    if "fastapi 0.115" in task.lower():
        return True

    return False
```

### Browser Use Targets (Agent-Specific):

```python
BROWSER_TARGETS = {
    "architecture": [
        "https://oauth.net/2/",
        "https://www.rfc-editor.org/rfc/rfc6749",
        "https://auth0.com/docs/get-started/authentication-and-authorization-flow"
    ],
    "implementation": [
        "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
        "https://fastapi.tiangolo.com/advanced/security/",
        "https://github.com/tiangolo/fastapi/discussions"
    ],
    "security": [
        "https://owasp.org/www-project-top-ten/",
        "https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html",
        "https://cwe.mitre.org/top25/"
    ],
    "testing": [
        "https://docs.pytest.org/en/stable/how-to/fixtures.html",
        "https://fastapi.tiangolo.com/tutorial/testing/",
        "https://docs.python.org/3/library/unittest.mock.html"
    ]
}
```

### Browser Use Implementation:

```python
from browser_use import Browser, BrowserConfig

async def browse_docs(agent_type: str, task: str):
    """Each agent browses its specialized documentation"""

    browser = Browser(BrowserConfig(
        headless=False,  # Visible for demo!
        timeout=10000
    ))

    urls = BROWSER_TARGETS[agent_type]

    # Extract relevant sections from each URL
    docs = []
    for url in urls:
        page = await browser.goto(url)

        # Semantic search within page
        relevant_sections = await page.search(
            query=task,
            max_results=3
        )

        for section in relevant_sections:
            text = await section.extract_text(max_length=1500)
            docs.append({
                "url": url,
                "content": text,
                "relevance_score": section.score
            })

    await browser.close()

    return docs  # Agent gets fresh, relevant documentation
```

---

## 4️⃣ AUTONOMOUS LEARNING - Source & Mechanism

### Source: **Anomaly Hunter's `autonomous_learner.py`**

Location: `/Users/bledden/Documents/anomaly-hunter/src/learning/autonomous_learner.py`

### What We're Reusing:

```python
# From Anomaly Hunter (proven to work!)
class AutonomousLearner:
    """Learns from Galileo evaluations to improve over time"""

    def __init__(self):
        self.patterns = {}  # Successful code patterns
        self.agent_performance = {
            "architecture": {"total": 0, "avg_score": 0, "success_rate": 0},
            "implementation": {"total": 0, "avg_score": 0, "success_rate": 0},
            "security": {"total": 0, "avg_score": 0, "success_rate": 0},
            "testing": {"total": 0, "avg_score": 0, "success_rate": 0}
        }
        self.learning_history = []

    async def learn_from_evaluation(
        self,
        task: str,
        agent_type: str,
        code: str,
        galileo_score: float,
        galileo_feedback: Dict
    ):
        """Store successful patterns (90+) and track performance"""

        # Update agent performance metrics
        self.agent_performance[agent_type]["total"] += 1
        prev_avg = self.agent_performance[agent_type]["avg_score"]
        total = self.agent_performance[agent_type]["total"]
        new_avg = (prev_avg * (total - 1) + galileo_score) / total
        self.agent_performance[agent_type]["avg_score"] = new_avg

        # Update success rate (90+ = success)
        if galileo_score >= 90:
            success_count = self.agent_performance[agent_type].get("successes", 0) + 1
            self.agent_performance[agent_type]["successes"] = success_count
            self.agent_performance[agent_type]["success_rate"] = success_count / total

            # Store pattern in memory (also goes to Neo4j)
            pattern_key = f"{task}_{agent_type}"
            if pattern_key not in self.patterns:
                self.patterns[pattern_key] = []

            self.patterns[pattern_key].append({
                "code": code,
                "score": galileo_score,
                "feedback": galileo_feedback,
                "timestamp": datetime.now().isoformat()
            })

            # Keep only top 5 patterns per task type
            self.patterns[pattern_key] = sorted(
                self.patterns[pattern_key],
                key=lambda p: p["score"],
                reverse=True
            )[:5]

        # Record learning event
        self.learning_history.append({
            "task": task,
            "agent": agent_type,
            "score": galileo_score,
            "learned": galileo_score >= 90,
            "timestamp": datetime.now().isoformat()
        })

    def get_agent_insights(self, agent_type: str) -> Dict:
        """Get performance insights for an agent"""

        perf = self.agent_performance[agent_type]

        return {
            "total_tasks": perf["total"],
            "average_score": round(perf["avg_score"], 1),
            "success_rate": round(perf["success_rate"] * 100, 1),
            "trend": self._calculate_trend(agent_type),
            "best_patterns": len([p for patterns in self.patterns.values()
                                   for p in patterns if p["score"] >= 95])
        }

    def _calculate_trend(self, agent_type: str) -> str:
        """Calculate if agent is improving, stable, or declining"""

        recent = [e for e in self.learning_history[-10:]
                  if e["agent"] == agent_type]

        if len(recent) < 3:
            return "insufficient_data"

        first_half_avg = sum(e["score"] for e in recent[:len(recent)//2]) / (len(recent)//2)
        second_half_avg = sum(e["score"] for e in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        if second_half_avg > first_half_avg + 2:
            return "improving"
        elif second_half_avg < first_half_avg - 2:
            return "declining"
        else:
            return "stable"
```

### How It Integrates with Neo4j RAG:

```python
# Combined system
class CodeSwarmLearning:
    """Autonomous learning + RAG storage"""

    def __init__(self):
        self.learner = AutonomousLearner()  # From Anomaly Hunter
        self.neo4j_rag = Neo4jRAG()          # New RAG system

    async def process_result(
        self,
        task: str,
        agent_type: str,
        code: str,
        galileo_eval: Dict
    ):
        """Learn + Store in one step"""

        score = galileo_eval["aggregate_score"]

        # 1. Autonomous learning (in-memory + local)
        await self.learner.learn_from_evaluation(
            task=task,
            agent_type=agent_type,
            code=code,
            galileo_score=score,
            galileo_feedback=galileo_eval["feedback"]
        )

        # 2. Neo4j RAG storage (only if 90+)
        if score >= 90:
            await self.neo4j_rag.store_pattern(
                task=task,
                code=code,
                quality_score=score,
                agent_type=agent_type,
                galileo_metrics=galileo_eval["metric_scores"]
            )

        # 3. WorkOS team sharing (if score ≥ 95)
        if score >= 95:
            await self.workos_share(
                team_id=current_user.team_id,
                pattern={
                    "task": task,
                    "code": code,
                    "score": score,
                    "agent": agent_type
                }
            )
```

---

## 5️⃣ DAYTONA HOSTING - How CodeSwarm Runs in Daytona

### Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                DAYTONA WORKSPACE                             │
│  (Dev Environment = Where CodeSwarm Lives)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ CodeSwarm Application                                  │ │
│  │ - FastAPI backend (agents, orchestration)              │ │
│  │ - Neo4j (RAG knowledge graph)                          │ │
│  │ - LangGraph (workflow engine)                          │ │
│  │ - Browser Use (doc scraping)                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Generated Code Output                                  │ │
│  │ /workspace/generated/                                  │ │
│  │   ├─ app/                                              │ │
│  │   │   ├─ main.py      (Implementation Agent)           │ │
│  │   │   ├─ models.py    (Architecture Agent)             │ │
│  │   │   ├─ security.py  (Security Agent)                 │ │
│  │   ├─ tests/                                            │ │
│  │   │   └─ test_main.py (Testing Agent)                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Daytona CLI (built-in)                                 │ │
│  │ $ daytona code run pytest                              │ │
│  │ $ daytona code commit -m "CodeSwarm generated"         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Daytona Integration Points:

```python
# daytona_integration.py

from daytona_sdk import DaytonaWorkspace, DaytonaAPI

class DaytonaIntegration:
    """CodeSwarm runs inside Daytona workspace"""

    def __init__(self):
        self.workspace = DaytonaWorkspace.current()
        self.api = DaytonaAPI(
            api_key=os.getenv("DAYTONA_API_KEY"),
            workspace_id=self.workspace.id
        )

    async def commit_generated_code(
        self,
        files: Dict[str, str],
        agent_outputs: Dict,
        galileo_scores: Dict
    ):
        """Commit agent-generated code to Daytona workspace"""

        # Write files to workspace
        for filepath, content in files.items():
            workspace_path = f"/workspace/generated/{filepath}"
            await self.workspace.write_file(workspace_path, content)

        # Git commit with detailed message
        commit_msg = f"""CodeSwarm Multi-Agent Generation

Agents:
- Architecture: {galileo_scores['architecture']}/100
- Implementation: {galileo_scores['implementation']}/100
- Security: {galileo_scores['security']}/100
- Testing: {galileo_scores['testing']}/100

Overall Quality: {sum(galileo_scores.values())/4:.1f}/100

Generated by CodeSwarm AI
"""

        await self.workspace.git_commit(commit_msg)

    async def run_tests_in_daytona(self):
        """Execute tests in Daytona environment"""

        result = await self.workspace.exec(
            command="cd /workspace/generated && pytest tests/ -v --cov"
        )

        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "coverage": self._parse_coverage(result.stdout)
        }

    async def share_workspace_with_team(self, team_id: str):
        """Enable team collaboration via Daytona"""

        await self.api.share_workspace(
            workspace_id=self.workspace.id,
            organization_id=team_id,
            permissions=["read", "write", "execute"]
        )
```

### Demo Flow in Daytona:

```
[Open Daytona workspace in browser]

User: "Build FastAPI OAuth endpoint"

[CodeSwarm interface shows:]
1. RAG retrieving patterns...
2. 4 agents working in parallel...
3. Galileo evaluating...
4. Files being written to workspace...

[Daytona file tree updates in real-time:]
📁 /workspace/generated
  📁 app
    📄 main.py       ✅ (Implementation: 93/100)
    📄 models.py     ✅ (Architecture: 95/100)
    📄 security.py   ✅ (Security: 98/100)
  📁 tests
    📄 test_main.py  ✅ (Testing: 91/100)

[Git panel shows new commit:]
"CodeSwarm Multi-Agent Generation - Quality: 94.25/100"

[Terminal auto-runs tests:]
$ pytest tests/ -v --cov
==================== 18 passed, 92% coverage ====================

[Success! Code is production-ready in Daytona workspace]
```

---

## 6️⃣ WORKOS - Team Knowledge Sharing

### How Teams Share Learning:

```python
from workos import WorkOS

class TeamKnowledgeSharing:
    """WorkOS-powered team collaboration"""

    def __init__(self):
        self.workos = WorkOS(api_key=os.getenv("WORKOS_API_KEY"))
        self.neo4j = Neo4jRAG()

    async def store_team_pattern(
        self,
        user_email: str,
        pattern: Dict,
        galileo_score: float
    ):
        """Store high-quality patterns for entire team"""

        # Get user's organization via WorkOS
        user = await self.workos.sso.get_profile(user_email)
        team_id = user.organization_id

        # Store in Neo4j with team relationship
        cypher = """
        // Create pattern
        CREATE (p:Pattern {
            task: $task,
            code: $code,
            quality_score: $score,
            agent_type: $agent_type,
            timestamp: datetime()
        })

        // Link to team
        MERGE (t:Team {id: $team_id, name: $team_name})
        CREATE (t)-[:HAS_PATTERN]->(p)

        // Link to user who created it
        MERGE (u:User {email: $user_email})
        CREATE (u)-[:CREATED]->(p)
        """

        await self.neo4j.query(cypher, {
            "task": pattern["task"],
            "code": pattern["code"],
            "score": galileo_score,
            "agent_type": pattern["agent_type"],
            "team_id": team_id,
            "team_name": user.organization_name,
            "user_email": user_email
        })

    async def retrieve_team_patterns(
        self,
        user_email: str,
        task: str,
        agent_type: str
    ):
        """Retrieve patterns from user's team"""

        user = await self.workos.sso.get_profile(user_email)
        team_id = user.organization_id

        # Get team's best patterns
        cypher = """
        MATCH (t:Team {id: $team_id})-[:HAS_PATTERN]->(p:Pattern)
        WHERE p.agent_type = $agent_type
          AND p.quality_score >= 90
          AND p.task CONTAINS $keyword
        RETURN p
        ORDER BY p.quality_score DESC
        LIMIT 10
        """

        patterns = await self.neo4j.query(cypher, {
            "team_id": team_id,
            "agent_type": agent_type,
            "keyword": extract_keywords(task)
        })

        return patterns

    async def get_team_leaderboard(self, team_id: str):
        """Show which developers contribute best patterns"""

        cypher = """
        MATCH (t:Team {id: $team_id})-[:HAS_PATTERN]->(p:Pattern)<-[:CREATED]-(u:User)
        WITH u, COUNT(p) as pattern_count, AVG(p.quality_score) as avg_quality
        WHERE avg_quality >= 90
        RETURN u.email, pattern_count, avg_quality
        ORDER BY avg_quality DESC, pattern_count DESC
        LIMIT 10
        """

        leaderboard = await self.neo4j.query(cypher, {"team_id": team_id})

        return leaderboard
```

### Demo: Team Collaboration

```
Developer 1 (Blake): "Build FastAPI OAuth endpoint"
- CodeSwarm generates code
- Galileo: 94/100 ✅
- Pattern stored for Team "Facilitair"

[Switch to Developer 2 account via WorkOS SSO]

Developer 2 (Teammate): "Build FastAPI OAuth with refresh tokens"
- CodeSwarm loads Team "Facilitair" patterns
- Sees Blake's 94/100 OAuth pattern
- Generates improved code using team knowledge
- Galileo: 97/100 ✅ (better because of team learning!)

[Show Team Dashboard:]
Team "Facilitair" Knowledge Base:
- Total patterns: 15
- Average quality: 93.2/100
- Top contributor: Blake (8 patterns, avg 94.5)
- Most used pattern: "FastAPI OAuth" (used 5 times)
```

---

## ✅ FINAL SUMMARY

**We Will Use**:

1. ✅ **RAG**: Neo4j + LangGraph + Galileo 90+ quality gate
2. ✅ **Models**:
   - Architecture: Claude Sonnet 4.5 (Anthropic #1)
   - Implementation: GPT-5 Pro
   - Security: Claude Opus 4.1 (Anthropic #2)
   - Testing: Grok-4 (98% HumanEval)
   - Execution: **Hybrid Parallel** (agents in parallel, stages sequential)
3. ✅ **Browser Use**: Live doc scraping (only if RAG insufficient)
4. ✅ **Autonomous Learning**: From Anomaly Hunter's `autonomous_learner.py`
5. ✅ **Daytona**: CodeSwarm runs IN Daytona workspace, commits code there
6. ✅ **WorkOS**: Team authentication + knowledge sharing across developers

**Vision Models**: Skip for 4-hour timeline (mention in "Next Steps")

**Timeline**: ~11-15 seconds per request, gets faster with learning

---

**Ready to build? We have ~4 hours. Should I start with Hour 1 setup?** 🚀
