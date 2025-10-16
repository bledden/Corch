# Agent-to-Agent Communication Protocols

## Overview

Comparison of protocols that enable agents to work **concurrently** and communicate results, rather than sequentially passing outputs.

---

## 1. **AutoGen (Microsoft Research)**

### Protocol Type: **Message-Based Multi-Agent Conversation**

```python
# Example: AutoGen architecture
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define specialized agents
architect = AssistantAgent(
    name="architect",
    system_message="You design system architecture",
    llm_config={"model": "gpt-4"}
)

coder = AssistantAgent(
    name="coder",
    system_message="You write production code",
    llm_config={"model": "deepseek-chat"}
)

reviewer = AssistantAgent(
    name="reviewer",
    system_message="You review code for quality",
    llm_config={"model": "claude-3.5-sonnet"}
)

# Create group chat for concurrent collaboration
groupchat = GroupChat(
    agents=[architect, coder, reviewer],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)

# Agents communicate via shared message bus
```

**Key Features:**
- ✅ **Asynchronous messaging** - Agents communicate via message passing
- ✅ **Group chat pattern** - Multiple agents can see all messages
- ✅ **Dynamic speaker selection** - Manager decides who speaks next
- ✅ **Nested chats** - Agents can spawn sub-conversations
- ❌ **Limited parallelism** - Still mostly turn-based

**Use Case:** Collaborative debugging, iterative refinement

---

## 2. **LangGraph (LangChain)**

### Protocol Type: **State Graph with Parallel Execution**

```python
# Example: LangGraph parallel agents
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

# Define state
class CollaborationState(TypedDict):
    task: str
    architecture: str
    implementation: str
    tests: str
    review: dict

# Define nodes (agents)
def architect_node(state):
    # Can run concurrently with test_writer
    return {"architecture": llm.invoke(state["task"])}

def coder_node(state):
    # Depends on architecture
    return {"implementation": llm.invoke(state["architecture"])}

def test_writer_node(state):
    # Can run concurrently with coder!
    return {"tests": llm.invoke(state["architecture"])}

def reviewer_node(state):
    # Waits for both implementation and tests
    return {"review": llm.invoke(state)}

# Build graph with parallel edges
workflow = StateGraph(CollaborationState)
workflow.add_node("architect", architect_node)
workflow.add_node("coder", coder_node)
workflow.add_node("test_writer", test_writer_node)
workflow.add_node("reviewer", reviewer_node)

# Define edges (dependencies)
workflow.set_entry_point("architect")
workflow.add_edge("architect", "coder")
workflow.add_edge("architect", "test_writer")  # Parallel!
workflow.add_edge("coder", "reviewer")
workflow.add_edge("test_writer", "reviewer")
workflow.add_edge("reviewer", END)
```

**Key Features:**
- ✅ **True parallelism** - Agents run simultaneously if no dependencies
- ✅ **DAG execution** - Clear dependency graph
- ✅ **State management** - Shared state accessible to all agents
- ✅ **Checkpointing** - Can pause/resume workflows
- ✅ **Conditional routing** - Dynamic branching based on results

**Use Case:** Complex workflows with clear dependencies

---

## 3. **CrewAI**

### Protocol Type: **Role-Based Hierarchical Delegation**

```python
# Example: CrewAI concurrent agents
from crewai import Agent, Task, Crew

# Define agents with roles
architect = Agent(
    role='Software Architect',
    goal='Design scalable system architecture',
    backstory='Expert in distributed systems',
    verbose=True
)

frontend_dev = Agent(
    role='Frontend Developer',
    goal='Build React UI components',
    backstory='Expert in TypeScript and React'
)

backend_dev = Agent(
    role='Backend Developer',
    goal='Build Python FastAPI backend',
    backstory='Expert in Python and databases'
)

# Define tasks that can run in parallel
design_task = Task(
    description='Design a todo app architecture',
    agent=architect
)

frontend_task = Task(
    description='Build React frontend',
    agent=frontend_dev,
    context=[design_task]  # Depends on architecture
)

backend_task = Task(
    description='Build FastAPI backend',
    agent=backend_dev,
    context=[design_task]  # Depends on architecture
)

# Create crew with parallel execution
crew = Crew(
    agents=[architect, frontend_dev, backend_dev],
    tasks=[design_task, frontend_task, backend_task],
    process=Process.parallel  # Frontend and backend run simultaneously!
)

result = crew.kickoff()
```

**Key Features:**
- ✅ **Process modes**: Sequential, Hierarchical, or **Parallel**
- ✅ **Task delegation** - Agents can delegate subtasks
- ✅ **Memory** - Agents remember past interactions
- ✅ **Tool integration** - Agents can use external tools
- ✅ **Human-in-the-loop** - Can pause for human input

**Use Case:** Multi-language systems (frontend + backend simultaneously)

---

## 4. **Multi-Agent Debate (MAD)**

### Protocol Type: **Concurrent Proposal → Debate → Consensus**

```python
# Example: Multi-Agent Debate protocol
class DebateProtocol:
    """
    Multiple agents generate solutions simultaneously,
    then debate to reach consensus.
    """

    async def execute(self, task: str):
        # Phase 1: PARALLEL generation
        proposals = await asyncio.gather(
            agent1.generate_solution(task),
            agent2.generate_solution(task),
            agent3.generate_solution(task)
        )

        # Phase 2: Debate rounds
        for round in range(3):
            critiques = await asyncio.gather(*[
                agent.critique(proposals)
                for agent in [agent1, agent2, agent3]
            ])

            # Refine based on critiques
            proposals = await asyncio.gather(*[
                agent.refine(proposal, critiques)
                for agent, proposal in zip(agents, proposals)
            ])

        # Phase 3: Consensus
        final_solution = await judge_agent.select_best(proposals)
        return final_solution
```

**Key Features:**
- ✅ **Concurrent generation** - All agents work simultaneously
- ✅ **Debate mechanism** - Agents critique each other's work
- ✅ **Diverse perspectives** - Multiple approaches explored
- ✅ **Consensus building** - Converge to best solution
- ❌ **High token cost** - 3-5x more LLM calls

**Use Case:** Critical systems requiring multiple perspectives (medical diagnosis, financial decisions)

---

## 5. **Message Queue Pattern (Custom)**

### Protocol Type: **Pub/Sub with Work Queues**

```python
# Example: RabbitMQ/Redis-based agent coordination
import asyncio
from redis import asyncio as aioredis

class MessageBusOrchestrator:
    """
    Agents subscribe to topics and publish results.
    Enables true concurrent, event-driven collaboration.
    """

    def __init__(self):
        self.redis = aioredis.from_url("redis://localhost")
        self.agents = {}

    async def register_agent(self, agent_id: str, topics: List[str]):
        """Register agent to listen on specific topics"""
        for topic in topics:
            await self.redis.sadd(f"subscribers:{topic}", agent_id)

    async def publish(self, topic: str, message: dict):
        """Publish message to all subscribers"""
        subscribers = await self.redis.smembers(f"subscribers:{topic}")
        for subscriber in subscribers:
            await self.redis.lpush(f"queue:{subscriber}", json.dumps({
                "topic": topic,
                "message": message
            }))

    async def run_agent(self, agent_id: str, agent_fn):
        """Agent listens on its queue and processes messages"""
        while True:
            # Block until message arrives
            msg = await self.redis.brpop(f"queue:{agent_id}", timeout=1)
            if msg:
                _, data = msg
                message = json.loads(data)

                # Process message
                result = await agent_fn(message)

                # Publish result
                await self.publish(f"result:{agent_id}", result)

# Usage
orchestrator = MessageBusOrchestrator()

# Register agents
await orchestrator.register_agent("architect", ["task:new"])
await orchestrator.register_agent("coder", ["architecture:done"])
await orchestrator.register_agent("reviewer", ["code:done"])
await orchestrator.register_agent("tester", ["architecture:done"])  # Parallel with coder!

# Start agents concurrently
await asyncio.gather(
    orchestrator.run_agent("architect", architect_fn),
    orchestrator.run_agent("coder", coder_fn),
    orchestrator.run_agent("reviewer", reviewer_fn),
    orchestrator.run_agent("tester", tester_fn)
)
```

**Key Features:**
- ✅ **True async** - Agents run completely independently
- ✅ **Event-driven** - React to events as they occur
- ✅ **Scalable** - Can distribute agents across machines
- ✅ **Fault-tolerant** - Messages persist in queue
- ✅ **Flexible** - Any agent can publish to any topic
- ❌ **Complex** - Requires infrastructure (Redis/RabbitMQ)

**Use Case:** Large-scale agent swarms, production systems

---

## 6. **Actor Model (Ray, Akka)**

### Protocol Type: **Distributed Actor System**

```python
# Example: Ray-based actor agents
import ray

@ray.remote
class ArchitectAgent:
    def __init__(self, model):
        self.model = model
        self.llm = LLM(model)

    def design(self, task):
        return self.llm.generate(f"Design architecture for: {task}")

@ray.remote
class CoderAgent:
    def __init__(self, model):
        self.model = model
        self.llm = LLM(model)

    def implement(self, architecture):
        return self.llm.generate(f"Implement: {architecture}")

@ray.remote
class TesterAgent:
    def __init__(self, model):
        self.model = model
        self.llm = LLM(model)

    def write_tests(self, architecture):
        return self.llm.generate(f"Write tests for: {architecture}")

# Create agents
architect = ArchitectAgent.remote("gpt-4")
coder = CoderAgent.remote("deepseek-chat")
tester = TesterAgent.remote("claude-3.5-sonnet")

# Execute with parallelism
architecture_future = architect.design.remote("Build a todo app")
architecture = ray.get(architecture_future)

# Coder and Tester run in parallel!
code_future = coder.implement.remote(architecture)
test_future = tester.write_tests.remote(architecture)

# Wait for both
code, tests = ray.get([code_future, test_future])
```

**Key Features:**
- ✅ **True distributed** - Agents run on different CPUs/machines
- ✅ **Location transparency** - Agents don't care where others are
- ✅ **Fault tolerance** - Can handle agent crashes
- ✅ **Resource management** - Ray handles scheduling
- ✅ **Low overhead** - Efficient message passing
- ❌ **Learning curve** - Requires understanding actor model

**Use Case:** Large-scale distributed agent systems

---

## Comparison Table

| Protocol | Parallelism | Complexity | Scalability | Token Efficiency | Best For |
|----------|-------------|------------|-------------|------------------|----------|
| **AutoGen** | Limited | Low | Small teams | Medium | Conversational agents |
| **LangGraph** | DAG-based | Medium | Medium | High | Complex workflows |
| **CrewAI** | Full | Low | Medium | High | Role-based teams |
| **MAD** | Full | Medium | Small | Low (3-5x calls) | Critical decisions |
| **Message Queue** | Full | High | Very High | High | Production systems |
| **Actor Model** | Full | High | Very High | High | Distributed systems |

---

## Recommendation for Facilitair

**For your hackathon demo, I recommend:**

### **LangGraph** (Best balance for demo)

**Why:**
1. ✅ True parallelism where it matters (coder + tester simultaneously)
2. ✅ Clear visual representation (show the DAG graph!)
3. ✅ Compatible with existing LLM integrations
4. ✅ Easy to explain to judges
5. ✅ Impressive speedup (2-3x faster)

**Architecture:**
```
         Task
          ↓
      Architect
       ↙   ↘
   Coder  Tester  ← PARALLEL!
       ↘   ↙
      Reviewer
          ↓
      Final Code
```

**Demo Impact:**
- "Instead of sequential (60 seconds), our agents work in parallel (25 seconds)"
- Show the graph with concurrent node execution
- Highlight: "Coder writes implementation while Tester writes tests simultaneously"

---

## Implementation Examples

See:
- [autogen_parallel_orchestrator.py](autogen_parallel_orchestrator.py) - AutoGen implementation
- [langgraph_parallel_orchestrator.py](langgraph_parallel_orchestrator.py) - LangGraph implementation (RECOMMENDED)
- [crewai_parallel_orchestrator.py](crewai_parallel_orchestrator.py) - CrewAI implementation
- [message_bus_orchestrator.py](message_bus_orchestrator.py) - Message queue implementation

---

## Next Steps

Want me to implement the LangGraph parallel orchestrator for Facilitair? It would:
1. Run coder + tester in parallel
2. Run documenter + integration tests in parallel
3. Show 2-3x speedup in benchmarks
4. Create visual DAG diagrams for your presentation

This would be a KILLER hackathon demo!
