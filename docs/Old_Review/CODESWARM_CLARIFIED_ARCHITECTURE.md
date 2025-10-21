# CodeSwarm - Clarified Architecture
**All questions answered + Vision model included**

---

## 🎨 VISION MODEL - INCLUDED (Qwen3-VL 235B)

### Where Vision Fits:

```python
CODESWARM_MODELS = {
    # Core 4 agents (always active)
    "architecture": "anthropic/claude-sonnet-4.5",
    "implementation": "openai/gpt-5-pro",
    "security": "anthropic/claude-opus-4.1",
    "testing": "x-ai/grok-4",

    # Vision agent (activated when user provides images)
    "vision": "qwen/qwen3-vl-235b-a22b-instruct"  # 235B parameters!
}
```

### Vision Agent Use Cases (For Demo):

#### Use Case 1: **Screenshot to Code** (PRIMARY DEMO)
```python
async def vision_to_code(screenshot_path: str, task: str):
    """Convert UI screenshot to React/Tailwind code"""

    vision_agent = VisionAgent(model="qwen/qwen3-vl-235b-a22b-instruct")

    # Step 1: Vision agent analyzes screenshot
    analysis = await vision_agent.analyze_image(
        image_path=screenshot_path,
        prompt=f"""Analyze this UI screenshot for: {task}

        Extract:
        1. Layout structure (header, sidebar, main, footer)
        2. All UI components (buttons, forms, cards, modals)
        3. Color scheme and typography
        4. Spacing and alignment
        5. Interactive elements and their states

        Provide detailed component breakdown."""
    )

    # Step 2: Architecture agent uses vision analysis
    architecture_prompt = f"""Vision Analysis:
{analysis}

Design React component architecture for this UI."""

    architecture = await architecture_agent.generate(architecture_prompt)

    # Step 3: Implementation agent generates code
    implementation_prompt = f"""Architecture:
{architecture}

Vision Details:
{analysis}

Generate production-ready React + Tailwind code."""

    code = await implementation_agent.generate(implementation_prompt)

    return code
```

**Demo Moment**:
```
[Show Figma/screenshot of auth form]
User: "Convert this to React code"

[Vision Agent analyzes - show split screen:]
Vision Model (Qwen3-VL 235B) analyzing...
- Detected: Login form with email/password fields
- Colors: #1F2937 (dark gray), #3B82F6 (blue)
- Components: 2 inputs, 1 button, 1 checkbox, 2 links
- Layout: Centered card, 400px wide

[Other agents use vision output:]
Architecture Agent: Designing component structure...
Implementation Agent: Generating React code...
Security Agent: Adding validation...
Testing Agent: Creating test suite...

[Result: Production-ready React component in 12 seconds]
```

#### Use Case 2: **Error Screenshot Debugging** (BONUS)
```
User uploads: Screenshot of browser console error
Vision Agent reads: Error message + stack trace
Security Agent: Identifies vulnerability
Implementation Agent: Generates fix
```

### Vision Demo Timeline (Adds 2-3 minutes to demo):
- Hour 3.5: Add vision agent integration (30 min)
- Demo prep: Prepare 1-2 good screenshots (Figma mockup)
- Demo script: Show vision as "wow factor" after core demo

---

## 🔄 CODE GENERATION: COLLABORATIVE (Not Sequential)

### You're Right - It's COLLABORATIVE, Not Sequential

**Sequential** would be:
```
Architecture Agent finishes
    ↓
THEN Implementation Agent starts
    ↓
THEN Security Agent starts
    ↓
THEN Testing Agent starts
```
❌ This takes 4x as long!

**Collaborative** (what we're doing):
```
All 4 agents work IN PARALLEL on the SAME task
Each brings their specialized perspective
    ↓
Architecture Agent: Designs API structure
Implementation Agent: Writes the code
Security Agent: Adds security layer
Testing Agent: Creates tests
    ↓
All finish at roughly the same time (~6 seconds)
    ↓
Results SYNTHESIZED into final code
```

### LangGraph Collaborative Workflow:

```python
from langgraph.graph import StateGraph, END

class CollaborativeCodeGen:
    """All agents work together, not sequentially"""

    def build_workflow(self):
        workflow = StateGraph(CodeSwarmState)

        # All agents are PARALLEL nodes
        workflow.add_node("architecture_agent", self._arch_agent)
        workflow.add_node("implementation_agent", self._impl_agent)
        workflow.add_node("security_agent", self._sec_agent)
        workflow.add_node("testing_agent", self._test_agent)

        # RAG retrieval happens FIRST (before agents)
        workflow.add_node("rag_retrieve", self._rag_retrieve)

        # Synthesis happens AFTER agents (combines outputs)
        workflow.add_node("synthesize", self._synthesize)

        # Evaluation happens AFTER synthesis
        workflow.add_node("galileo_evaluate", self._galileo_eval)

        # Edges
        workflow.set_entry_point("rag_retrieve")

        # RAG → All 4 agents IN PARALLEL
        workflow.add_edge("rag_retrieve", "architecture_agent")
        workflow.add_edge("rag_retrieve", "implementation_agent")
        workflow.add_edge("rag_retrieve", "security_agent")
        workflow.add_edge("rag_retrieve", "testing_agent")

        # All agents → Synthesize (wait for all to complete)
        workflow.add_edge("architecture_agent", "synthesize")
        workflow.add_edge("implementation_agent", "synthesize")
        workflow.add_edge("security_agent", "synthesize")
        workflow.add_edge("testing_agent", "synthesize")

        # Synthesize → Evaluate
        workflow.add_edge("synthesize", "galileo_evaluate")

        # Evaluate → Conditional (improve or done)
        workflow.add_conditional_edges(
            "galileo_evaluate",
            self._should_improve,
            {
                "improve": "improve_agents",  # Score < 90
                "done": END                   # Score ≥ 90
            }
        )

        return workflow.compile()
```

### How Agents Collaborate (Collective Blackboard):

```python
class CodeSwarmState:
    """Shared state all agents can access"""

    task: str                          # User's request
    rag_context: Dict[str, Any]        # Retrieved patterns
    browsed_docs: Dict[str, str]       # Live documentation

    # Agent outputs (written concurrently)
    architecture_output: str = ""
    implementation_output: str = ""
    security_output: str = ""
    testing_output: str = ""

    # Synthesis (combines all outputs)
    final_code: str = ""

    # Evaluation
    galileo_scores: Dict[str, float] = {}
    overall_score: float = 0.0
```

**Example of Collaboration**:
```python
async def _impl_agent(self, state: CodeSwarmState):
    """Implementation agent can SEE what Architecture agent is doing"""

    # Wait briefly for architecture to start (50ms)
    await asyncio.sleep(0.05)

    # Check if architecture has partial output yet
    if state.architecture_output:
        # Use architecture's design as context!
        prompt = f"""Task: {state.task}

Architecture being designed:
{state.architecture_output[:500]}  # First 500 chars

RAG patterns:
{state.rag_context}

Generate implementation that matches this architecture."""
    else:
        # Architecture not ready yet, use RAG only
        prompt = f"""Task: {state.task}

RAG patterns:
{state.rag_context}

Generate implementation."""

    code = await self.call_model("openai/gpt-5-pro", prompt)
    state.implementation_output = code
    return state
```

---

## 📊 GALILEO EVALUATION: ITERATE UNTIL 90+

### Answer: **ITERATE (Not Fail)**

```python
class GalileoQualityLoop:
    """Keep improving until we hit 90+ quality"""

    async def evaluate_and_improve(
        self,
        agent_type: str,
        code: str,
        task: str,
        max_iterations: int = 3
    ):
        """Iterate until score ≥ 90 or max iterations reached"""

        iteration = 0
        current_code = code

        while iteration < max_iterations:
            # Evaluate with Galileo
            evaluation = await galileo.evaluate(
                project="codeswarm",
                input=task,
                output=current_code,
                metadata={"agent": agent_type, "iteration": iteration},
                metrics=[
                    "correctness",
                    "completeness",
                    "code_quality",
                    "security",
                    "best_practices"
                ]
            )

            score = evaluation.aggregate_score

            print(f"[{agent_type}] Iteration {iteration + 1}: Score {score}/100")

            # Check if we hit quality threshold
            if score >= 90:
                print(f"[{agent_type}] ✅ Quality threshold met!")
                return {
                    "code": current_code,
                    "score": score,
                    "iterations": iteration + 1,
                    "success": True
                }

            # Score < 90: Improve using Galileo feedback
            print(f"[{agent_type}] ⚠️  Score below 90, improving...")

            improvement_prompt = f"""Your previous code scored {score}/100.

Galileo Feedback:
{json.dumps(evaluation.feedback, indent=2)}

Specific Issues:
{self._extract_issues(evaluation)}

Original Task: {task}

Previous Code:
{current_code}

Generate IMPROVED code that addresses all feedback."""

            # Agent improves code
            current_code = await self.call_agent_model(
                agent_type=agent_type,
                prompt=improvement_prompt
            )

            iteration += 1

        # Max iterations reached, return best attempt
        print(f"[{agent_type}] ⚠️  Max iterations reached, score: {score}/100")
        return {
            "code": current_code,
            "score": score,
            "iterations": iteration,
            "success": False  # Didn't hit 90+
        }

    def _extract_issues(self, evaluation):
        """Extract specific issues from Galileo feedback"""
        issues = []

        for metric, score in evaluation.metric_scores.items():
            if score < 85:  # Problem area
                feedback = evaluation.feedback.get(metric, "")
                issues.append(f"- {metric}: {feedback}")

        return "\n".join(issues)
```

### Example Iteration Flow:

```
User: "Build FastAPI OAuth endpoint"

[Agents generate code in parallel]

[Galileo evaluates all 4 agents:]
- Architecture: 94/100 ✅ (pass)
- Implementation: 87/100 ⚠️ (needs improvement)
- Security: 96/100 ✅ (pass)
- Testing: 89/100 ⚠️ (needs improvement)

[Improvement loop for Implementation:]
Iteration 1: 87/100
Galileo feedback: "Missing error handling for invalid tokens"
→ Agent adds try-catch blocks
→ Re-evaluate

Iteration 2: 92/100 ✅ (pass!)

[Improvement loop for Testing:]
Iteration 1: 89/100
Galileo feedback: "Missing edge case tests for expired tokens"
→ Agent adds edge case tests
→ Re-evaluate

Iteration 2: 91/100 ✅ (pass!)

[All agents now 90+, proceed to synthesis]
Final code quality: 93.25/100

[Store all 4 agent outputs in Neo4j RAG]
```

### What if Agent Can't Hit 90+ After 3 Iterations?

```python
async def handle_quality_failure(self, agent_type: str, best_score: float):
    """Fallback when agent can't hit 90+ after max iterations"""

    print(f"[{agent_type}] Could not reach 90+ (best: {best_score}/100)")

    # Option 1: Use lower-quality output but flag it
    if best_score >= 80:
        print(f"[{agent_type}] Using 80+ output with warning")
        return {
            "code": current_code,
            "score": best_score,
            "warning": "Below quality threshold, manual review recommended"
        }

    # Option 2: Try different model for this agent
    elif best_score < 80:
        print(f"[{agent_type}] Trying backup model...")
        return await self.try_backup_model(agent_type, task)

    # Option 3: Disable this agent for this task
    else:
        print(f"[{agent_type}] Skipping agent, using other 3 agents only")
        return None
```

---

## 🧠 AUTONOMOUS LEARNER - REUSING FROM ANOMALY HUNTER

### Answer: **REUSING, Not Creating from Scratch**

**What We're Doing**:
1. **Copy** `/anomaly-hunter/src/learning/autonomous_learner.py` to new repo
2. **Adapt** it for code generation (instead of anomaly detection)
3. **Integrate** with Neo4j for persistent storage

**Why Reuse**:
- Already proven to work (14 detections in Anomaly Hunter)
- Saves 2-3 hours of development time
- Known performance characteristics

### What the Learner Does (From Anomaly Hunter):

```python
# From Anomaly Hunter (we're copying this file)
class AutonomousLearner:
    """Learns from every evaluation to improve over time"""

    def __init__(self):
        # Track agent performance
        self.agent_stats = {
            "architecture": {
                "total_tasks": 0,
                "avg_score": 0,
                "success_rate": 0,  # % of tasks with score ≥ 90
                "improvement_trend": "stable"
            },
            "implementation": {...},
            "security": {...},
            "testing": {...}
        }

        # Store successful patterns (in-memory cache)
        self.successful_patterns = {}

        # Track learning history
        self.learning_events = []

    async def learn(self, agent_type: str, task: str, code: str, score: float):
        """Learn from this task"""

        # Update statistics
        self.agent_stats[agent_type]["total_tasks"] += 1

        # Update average score
        prev_avg = self.agent_stats[agent_type]["avg_score"]
        total = self.agent_stats[agent_type]["total_tasks"]
        new_avg = (prev_avg * (total - 1) + score) / total
        self.agent_stats[agent_type]["avg_score"] = new_avg

        # Update success rate
        if score >= 90:
            successes = self.agent_stats[agent_type].get("successes", 0) + 1
            self.agent_stats[agent_type]["successes"] = successes
            self.agent_stats[agent_type]["success_rate"] = successes / total

            # Store pattern (only 90+ scores)
            self._store_pattern(agent_type, task, code, score)

        # Calculate trend (improving/stable/declining)
        self._update_trend(agent_type)

        # Log learning event
        self.learning_events.append({
            "agent": agent_type,
            "task": task,
            "score": score,
            "timestamp": datetime.now().isoformat()
        })

    def _store_pattern(self, agent_type: str, task: str, code: str, score: float):
        """Store successful pattern in memory"""

        key = f"{agent_type}:{self._extract_task_type(task)}"

        if key not in self.successful_patterns:
            self.successful_patterns[key] = []

        self.successful_patterns[key].append({
            "code": code,
            "score": score,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only top 5 patterns per type
        self.successful_patterns[key] = sorted(
            self.successful_patterns[key],
            key=lambda p: p["score"],
            reverse=True
        )[:5]

    def get_learned_patterns(self, agent_type: str, task: str):
        """Retrieve learned patterns for similar tasks"""

        key = f"{agent_type}:{self._extract_task_type(task)}"
        return self.successful_patterns.get(key, [])
```

### How We Adapt It for CodeSwarm:

```python
# New file: /codeswarm/src/learning/code_learner.py

from anomaly_hunter.learning.autonomous_learner import AutonomousLearner
from neo4j_rag import Neo4jRAG

class CodeSwarmLearner(AutonomousLearner):
    """Extended learner with Neo4j persistence"""

    def __init__(self):
        super().__init__()  # Use Anomaly Hunter's logic
        self.neo4j = Neo4jRAG()  # Add persistent storage

    async def learn(self, agent_type: str, task: str, code: str, score: float):
        """Learn + persist to Neo4j"""

        # Use Anomaly Hunter's learning logic
        await super().learn(agent_type, task, code, score)

        # ADDITIONAL: Persist to Neo4j (only if 90+)
        if score >= 90:
            await self.neo4j.store_pattern(
                agent_type=agent_type,
                task=task,
                code=code,
                quality_score=score,
                metadata={
                    "avg_score": self.agent_stats[agent_type]["avg_score"],
                    "success_rate": self.agent_stats[agent_type]["success_rate"]
                }
            )

    async def get_context_for_agent(self, agent_type: str, task: str):
        """Get learning context from both memory + Neo4j"""

        # In-memory patterns (fast)
        memory_patterns = self.get_learned_patterns(agent_type, task)

        # Neo4j patterns (team-wide)
        neo4j_patterns = await self.neo4j.retrieve_patterns(agent_type, task, k=5)

        # Combine
        return {
            "memory_patterns": memory_patterns,     # This session
            "neo4j_patterns": neo4j_patterns,       # All time + team
            "agent_stats": self.agent_stats[agent_type]
        }
```

---

## 📁 NEW REPO STRUCTURE

```bash
/Users/bledden/Documents/codeswarm/
├── README.md
├── pyproject.toml
├── .env
├── docker-compose.yml  # Neo4j + FastAPI
│
├── src/
│   ├── __init__.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── architecture_agent.py    # Claude Sonnet 4.5
│   │   ├── implementation_agent.py  # GPT-5 Pro
│   │   ├── security_agent.py        # Claude Opus 4.1
│   │   ├── testing_agent.py         # Grok-4
│   │   └── vision_agent.py          # Qwen3-VL 235B (NEW!)
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── langgraph_workflow.py   # LangGraph state machine
│   │   └── synthesizer.py          # Combines agent outputs
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── neo4j_client.py         # Neo4j connection
│   │   ├── knowledge_graph.py      # Pattern storage/retrieval
│   │   └── embeddings.py           # OpenAI embeddings
│   │
│   ├── learning/
│   │   ├── __init__.py
│   │   └── code_learner.py         # COPIED from Anomaly Hunter + extended
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── galileo_evaluator.py    # Galileo Observe integration
│   │   └── quality_gate.py         # 90+ threshold logic
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── browser_use.py          # Browser Use doc scraping
│   │   ├── openrouter_client.py    # OpenRouter API (COPIED from Facilitair)
│   │   ├── daytona_client.py       # Daytona workspace integration
│   │   └── workos_client.py        # WorkOS team auth
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI app
│   │
│   └── cli/
│       ├── __init__.py
│       └── demo.py                 # CLI for demo
│
├── tests/
│   └── test_workflow.py
│
├── demo/
│   ├── screenshots/                # For vision model demo
│   │   ├── figma_mockup.png
│   │   └── error_screenshot.png
│   └── test_cases.json
│
└── docs/
    ├── ARCHITECTURE.md
    ├── DEMO_SCRIPT.md
    └── SPONSOR_INTEGRATIONS.md
```

---

## ✅ FINAL CLARIFICATIONS SUMMARY

### 1. **Vision Model** ✅
- **Included**: Qwen3-VL 235B (235 billion parameters!)
- **Use Case**: Screenshot → Code (primary demo feature)
- **Timeline**: Add in Hour 3.5 (30 minutes)

### 2. **Agent Collaboration** ✅
- **Type**: COLLABORATIVE (all 4 agents work in parallel)
- **Not**: Sequential (one after another)
- **Tool**: LangGraph with collective blackboard state

### 3. **Galileo Scoring** ✅
- **Below 90**: ITERATE (not fail)
- **Max Iterations**: 3 attempts per agent
- **Process**: Agent → Galileo feedback → Agent improves → Re-evaluate
- **Result**: Keep improving until 90+ or max iterations

### 4. **Autonomous Learner** ✅
- **Source**: COPYING from `/anomaly-hunter/src/learning/autonomous_learner.py`
- **Adaptation**: Add Neo4j persistence on top
- **Not**: Building from scratch (saves 2-3 hours!)

### 5. **New Repo** ✅
- **Location**: `/Users/bledden/Documents/codeswarm/`
- **Not**: Inside existing repos
- **Fresh start**: Clean separation

---

**Ready to create the new repo and start Hour 1?** 🚀

Say "Create the repo" and I'll scaffold the entire structure!