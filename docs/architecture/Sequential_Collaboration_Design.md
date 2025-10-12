# Sequential Collaboration Architecture Design

## Problem Statement

The current system uses **consensus/voting** where agents work in parallel and vote on outputs. This has failed historically.

We need **sequential collaboration** where agents work in a chain, passing context forward, with each agent receiving input in their preferred format (XML, JSON, etc.).

## Current Issues

1. ❌ All agents run in parallel with same input ([collaborative_orchestrator.py:231-236](../weavehacks-collaborative/collaborative_orchestrator.py#L231-L236))
2. ❌ Consensus methods just vote/combine outputs ([lines 488-546](../weavehacks-collaborative/collaborative_orchestrator.py#L488-L546))
3. ❌ No agent-to-agent communication during execution
4. ❌ No format preference support (XML, JSON, etc.)
5. ❌ Original user request context gets lost

## Proposed Architecture

### 1. Agent Communication Preferences

Each agent should specify their preferred input/output formats:

```python
@dataclass
class AgentCommunicationProfile:
    """Communication preferences for an agent"""
    preferred_input_format: str  # "xml", "json", "markdown", "plain"
    preferred_output_format: str
    context_requirements: List[str]  # What info they need from previous agents
    provides: List[str]  # What info they provide to next agents
```

**Example Agent Profiles:**

```yaml
agents:
  architect:
    communication:
      input_format: "markdown"
      output_format: "xml"  # Structured design docs
      needs: ["user_request", "requirements"]
      provides: ["architecture", "components", "interfaces"]

  coder:
    communication:
      input_format: "xml"  # Reads architect's structured output
      output_format: "json"  # Code with metadata
      needs: ["architecture", "interfaces", "user_request"]
      provides: ["implementation", "tests", "dependencies"]

  reviewer:
    communication:
      input_format: "json"  # Reads coder's structured code
      output_format: "markdown"  # Human-readable review
      needs: ["implementation", "architecture", "user_request"]
      provides: ["issues", "suggestions", "approval_status"]

  documenter:
    communication:
      input_format: "markdown"  # Reads review + final code
      output_format: "markdown"  # Documentation
      needs: ["implementation", "architecture", "review", "user_request"]
      provides: ["documentation", "examples", "api_reference"]
```

### 2. Workflow Definition System

Instead of consensus methods, define **collaboration workflows**:

```python
@dataclass
class CollaborationWorkflow:
    """Defines how agents collaborate sequentially"""
    name: str
    description: str
    stages: List[WorkflowStage]

@dataclass
class WorkflowStage:
    """Single stage in collaboration workflow"""
    agent_id: str
    depends_on: List[str]  # Previous stage outputs this needs
    can_iterate: bool  # Can this stage request changes from previous?
    max_iterations: int  # Max back-and-forth rounds
```

**Example Workflows:**

```yaml
workflows:
  feature_development:
    description: "Full feature development with iteration"
    stages:
      - stage: design
        agent: architect
        depends_on: [user_request]
        can_iterate: false

      - stage: implementation
        agent: coder
        depends_on: [design, user_request]
        can_iterate: true  # Can ask architect for clarification
        max_iterations: 2

      - stage: review
        agent: reviewer
        depends_on: [implementation, design, user_request]
        can_iterate: true  # Can ask coder to fix issues
        max_iterations: 3

      - stage: documentation
        agent: documenter
        depends_on: [implementation, review, design, user_request]
        can_iterate: false

  quick_code_fix:
    description: "Fast coding without architecture"
    stages:
      - stage: implementation
        agent: coder
        depends_on: [user_request]
        can_iterate: false

      - stage: review
        agent: reviewer
        depends_on: [implementation, user_request]
        can_iterate: true
        max_iterations: 2

  architecture_review:
    description: "Design review and refinement"
    stages:
      - stage: initial_design
        agent: architect
        depends_on: [user_request]
        can_iterate: false

      - stage: security_review
        agent: reviewer
        depends_on: [initial_design, user_request]
        can_iterate: true
        max_iterations: 2

      - stage: final_design
        agent: architect
        depends_on: [security_review, initial_design, user_request]
        can_iterate: false
```

### 3. Context Passing System

Each agent receives:
1. **Original user request** (always preserved)
2. **Previous agent outputs** (formatted for their preference)
3. **Workflow context** (what stage we're at, what's expected)

```python
@dataclass
class AgentContext:
    """Context passed to each agent in workflow"""

    # Always included
    original_request: str  # User's original request
    workflow_name: str
    current_stage: str

    # Previous outputs (formatted for this agent)
    previous_outputs: Dict[str, str]  # stage_name -> formatted output

    # Iteration context (if applicable)
    iteration_number: int
    feedback_from: Optional[str]  # Which agent gave feedback
    feedback: Optional[str]  # What they want changed

    # Format preferences
    expected_output_format: str
```

**Format Conversion Layer:**

```python
class FormatConverter:
    """Converts between agent communication formats"""

    def convert(self, content: str, from_format: str, to_format: str) -> str:
        """Convert content between formats"""

        # Parse source format
        parsed = self._parse(content, from_format)

        # Convert to target format
        return self._serialize(parsed, to_format)

    def _parse(self, content: str, format: str) -> Dict:
        if format == "xml":
            return self._parse_xml(content)
        elif format == "json":
            return json.loads(content)
        elif format == "markdown":
            return self._parse_markdown(content)
        else:
            return {"content": content}

    def _serialize(self, data: Dict, format: str) -> str:
        if format == "xml":
            return self._to_xml(data)
        elif format == "json":
            return json.dumps(data, indent=2)
        elif format == "markdown":
            return self._to_markdown(data)
        else:
            return str(data.get("content", data))
```

### 4. Sequential Execution Engine

Replace `_reach_consensus()` with `_execute_workflow()`:

```python
async def _execute_workflow(
    self,
    workflow: CollaborationWorkflow,
    user_request: str
) -> WorkflowResult:
    """Execute agents sequentially according to workflow"""

    # Store outputs from each stage
    stage_outputs = {}

    # Track original request through entire workflow
    context = {
        "original_request": user_request,
        "workflow_name": workflow.name
    }

    for stage in workflow.stages:
        agent = self.agents[stage.agent_id]

        # Build context for this agent
        agent_context = self._build_agent_context(
            agent=agent,
            stage=stage,
            context=context,
            previous_outputs=stage_outputs
        )

        # Execute agent with their preferred format
        output = await self._execute_agent_with_context(
            agent=agent,
            context=agent_context
        )

        # Handle iteration if agent requests changes
        if stage.can_iterate and self._needs_iteration(output):
            output = await self._iterate_with_previous_stage(
                current_agent=agent,
                current_output=output,
                stage=stage,
                stage_outputs=stage_outputs,
                max_iterations=stage.max_iterations
            )

        # Store output for next stages
        stage_outputs[stage.stage] = output

    return WorkflowResult(
        workflow_name=workflow.name,
        original_request=user_request,
        stage_outputs=stage_outputs,
        final_output=stage_outputs[workflow.stages[-1].stage]
    )
```

### 5. Iteration Mechanism

When an agent requests changes from a previous agent:

```python
async def _iterate_with_previous_stage(
    self,
    current_agent: Agent,
    current_output: str,
    stage: WorkflowStage,
    stage_outputs: Dict[str, str],
    max_iterations: int
) -> str:
    """Handle iterative refinement between agents"""

    iteration = 0
    current_result = current_output

    while iteration < max_iterations:
        # Extract feedback/requests from current agent's output
        feedback = self._extract_feedback(current_result)

        if not feedback:
            break  # No more iterations needed

        # Find which previous agent needs to respond
        target_stage = feedback.get("target_stage")
        target_agent = self.agents[stage_outputs[target_stage]["agent_id"]]

        # Send feedback to previous agent
        revision_context = {
            "original_output": stage_outputs[target_stage],
            "feedback_from": current_agent.id,
            "feedback": feedback["message"],
            "iteration": iteration + 1
        }

        # Previous agent revises their output
        revised_output = await self._execute_agent_with_context(
            agent=target_agent,
            context=revision_context
        )

        # Update stage outputs with revision
        stage_outputs[target_stage] = revised_output

        # Current agent re-executes with revised input
        updated_context = self._build_agent_context(
            agent=current_agent,
            stage=stage,
            context={"iteration": iteration + 1},
            previous_outputs=stage_outputs
        )

        current_result = await self._execute_agent_with_context(
            agent=current_agent,
            context=updated_context
        )

        iteration += 1

    return current_result
```

## Implementation Plan

### Phase 1: Add Communication Profiles ✅
- [ ] Extend Agent dataclass with communication preferences
- [ ] Update config.yaml with format preferences
- [ ] Create FormatConverter class

### Phase 2: Build Workflow System ✅
- [ ] Create workflow data structures
- [ ] Define standard workflows (feature_development, quick_fix, etc.)
- [ ] Add workflow selection logic

### Phase 3: Replace Consensus with Sequential Execution ✅
- [ ] Implement _execute_workflow() method
- [ ] Add context building logic
- [ ] Implement iteration mechanism

### Phase 4: Format Conversion ✅
- [ ] Implement XML parser/serializer
- [ ] Implement JSON handling
- [ ] Implement Markdown parsing
- [ ] Test format conversions

### Phase 5: Testing ✅
- [ ] Test simple linear workflow (no iteration)
- [ ] Test iterative workflows (reviewer → coder → reviewer)
- [ ] Test format conversions between agents
- [ ] Test original request preservation

### Phase 6: Migration ✅
- [ ] Remove old consensus methods
- [ ] Update CollaborationResult to WorkflowResult
- [ ] Update evaluation scripts
- [ ] Update documentation

## Key Differences from Current System

| Current (Consensus) | New (Sequential Collaboration) |
|---------------------|--------------------------------|
| All agents run in parallel | Agents run in sequence |
| Vote/combine outputs | Pass context forward |
| Same input to all | Format converted per agent |
| No iteration | Agents can request changes |
| Context lost | Original request preserved |
| Consensus "methods" | Workflow "stages" |

## Example Execution Flow

**User Request:** "Build a REST API for user authentication"

**Workflow: feature_development**

1. **Architect** (receives markdown):
   ```markdown
   # Task
   Build a REST API for user authentication

   # Your Role
   Design the system architecture
   ```

   Outputs (XML):
   ```xml
   <architecture>
     <component name="AuthService">
       <endpoints>
         <endpoint path="/register" method="POST"/>
         <endpoint path="/login" method="POST"/>
         <endpoint path="/logout" method="POST"/>
       </endpoints>
     </component>
     <database schema="users">
       <field name="email" type="string" unique="true"/>
       <field name="password_hash" type="string"/>
     </database>
   </architecture>
   ```

2. **Coder** (receives XML → JSON):
   ```json
   {
     "task": "Build a REST API for user authentication",
     "architecture": {
       "AuthService": {
         "endpoints": [
           {"path": "/register", "method": "POST"},
           {"path": "/login", "method": "POST"}
         ]
       }
     }
   }
   ```

   Outputs (JSON):
   ```json
   {
     "implementation": {
       "file": "auth_service.py",
       "code": "class AuthService...",
       "tests": "test_auth.py",
       "dependencies": ["fastapi", "bcrypt"]
     }
   }
   ```

3. **Reviewer** (receives JSON → Markdown):
   ```markdown
   # Code Review

   **Implementation:** auth_service.py

   ## Issues Found
   - ⚠️ Password validation missing
   - ⚠️ No rate limiting on login endpoint

   ## Request Changes
   Please add input validation and rate limiting
   ```

   **→ Iteration: Coder receives feedback, revises code**

4. **Reviewer** (second iteration):
   ```markdown
   # Code Review - Iteration 2

   ✅ All issues resolved
   ✅ Ready to document
   ```

5. **Documenter** (receives all context):
   ```markdown
   # User Authentication API Documentation

   ## Overview
   [Based on original request + architecture + final implementation]

   ## Endpoints
   ...
   ```

**Result:** Comprehensive solution with context preserved throughout!

## Metrics to Track

Replace consensus metrics with collaboration metrics:

- **Stage completion time** (per agent)
- **Iteration counts** (how often agents requested changes)
- **Format conversion success rate**
- **Context preservation accuracy**
- **Agent satisfaction scores** (did they have info they needed?)
- **Workflow efficiency** (stages completed vs iterations needed)

## Next Steps

1. Review and approve this design
2. Begin Phase 1 implementation
3. Test with simple 2-agent workflow
4. Expand to full feature_development workflow
5. Remove all consensus code
6. Re-run evaluations with sequential collaboration
