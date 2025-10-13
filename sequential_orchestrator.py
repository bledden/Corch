"""
Sequential Collaborative Orchestrator for WeaveHacks 2
Port of Facilitair_v2's proven sequential collaboration architecture.

Instead of consensus/voting, models work sequentially:
  Architect → Implementer → Reviewer → Refiner (iterate) → Tester + Documenter

Each agent:
- Receives outputs from previous stages
- Has format preferences (JSON, XML, Markdown)
- Gets the original user request preserved throughout
"""

import weave
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from datetime import datetime
import time
import re
import uuid
import logging

logger = logging.getLogger(__name__)

# Timeouts from Facilitair_v2
TOTAL_BUDGET_S = 900.0  # 15 minutes total
STAGE_TIMEOUT_S = 180.0  # 3 minutes per stage
ARCHITECTURE_TIMEOUT_MULTIPLIER = 1.5
IMPLEMENTATION_TIMEOUT_MULTIPLIER = 1.2


class AgentRole(Enum):
    """Roles in sequential collaboration workflow"""
    ARCHITECT = "architect"      # High-level design (outputs Markdown)
    CODER = "coder"  # Code generation (outputs code)
    REVIEWER = "reviewer"        # Code review (outputs JSON)
    REFINER = "refiner"         # Fix issues (outputs code)
    TESTER = "tester"           # Test generation (outputs code)
    DOCUMENTER = "documenter"   # Documentation (outputs Markdown)


@dataclass
class AgentCommunicationProfile:
    """Communication preferences for an agent"""
    role: AgentRole
    model_id: str
    preferred_input_format: str   # "json", "xml", "markdown", "code"
    preferred_output_format: str
    context_requirements: List[str]  # What info from previous stages


@dataclass
class StageResult:
    """Result from a single workflow stage"""
    stage: str
    agent_role: AgentRole
    model_id: str
    timestamp: str
    input_context: Dict[str, Any]
    output: str
    format: str
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    run_id: str
    original_request: str
    workflow_name: str
    stages: List[StageResult]
    final_output: str
    iterations: int
    total_duration_seconds: float
    success: bool


class FormatConverter:
    """Converts between agent communication formats"""

    @staticmethod
    def to_json(data: Any) -> str:
        """Convert data to JSON format"""
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2)
            except:
                # Wrap string in JSON object
                return json.dumps({"content": data}, indent=2)
        return json.dumps(data, indent=2)

    @staticmethod
    def to_markdown(data: Any) -> str:
        """Convert data to Markdown format"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            lines = ["# Data\n"]
            for key, value in data.items():
                lines.append(f"## {key}\n")
                lines.append(f"{value}\n")
            return "\n".join(lines)
        return str(data)

    @staticmethod
    def to_xml(data: Any) -> str:
        """Convert data to XML format"""
        if isinstance(data, str):
            return f"<content>{data}</content>"
        elif isinstance(data, dict):
            lines = ["<data>"]
            for key, value in data.items():
                lines.append(f"  <{key}>{value}</{key}>")
            lines.append("</data>")
            return "\n".join(lines)
        return f"<content>{str(data)}</content>"

    @staticmethod
    def extract_code(text: str) -> str:
        """Extract code blocks from markdown/mixed content"""
        if not text:
            return text
        # Find fenced code blocks
        blocks = re.findall(r"```(?:[a-zA-Z0-9_+\-#]+)?\s*([\s\S]*?)```", text)
        if blocks:
            # Return the largest code block
            blocks_sorted = sorted(blocks, key=lambda b: len(b), reverse=True)
            return blocks_sorted[0].strip()
        # If looks like code but no fences, return as-is
        if any(tok in text for tok in ("def ", "class ", "import ", "from ", "function", "const ")):
            return text
        return text

    def convert(self, content: str, from_format: str, to_format: str) -> str:
        """Convert content between formats"""
        if from_format == to_format:
            return content

        # Parse source format
        if from_format == "json":
            try:
                parsed = json.loads(content)
            except:
                parsed = {"content": content}
        elif from_format == "code":
            parsed = {"code": self.extract_code(content)}
        else:  # markdown or plain text
            parsed = {"content": content}

        # Convert to target format
        if to_format == "json":
            return self.to_json(parsed)
        elif to_format == "xml":
            return self.to_xml(parsed)
        elif to_format == "markdown":
            return self.to_markdown(parsed)
        elif to_format == "code":
            if isinstance(parsed, dict) and "code" in parsed:
                return parsed["code"]
            return content
        else:
            return str(parsed)


class SequentialCollaborativeOrchestrator:
    """
    Sequential multi-agent collaboration orchestrator for WeaveHacks.
    Based on Facilitair_v2's proven architecture.
    """

    def __init__(self, llm_orchestrator, config: Optional[Dict] = None):
        """
        Initialize with LLM orchestrator from weavehacks-collaborative.

        Args:
            llm_orchestrator: MultiAgentLLMOrchestrator instance
            config: Optional agent configuration
        """
        self.llm = llm_orchestrator
        self.config = config or {}
        self.format_converter = FormatConverter()

        # Define agent communication profiles
        self.agent_profiles = self._setup_agent_profiles()

    def _setup_agent_profiles(self) -> Dict[AgentRole, AgentCommunicationProfile]:
        """Setup communication profiles for each agent"""

        # Get model assignments from config or use defaults
        architect_model = self.config.get("architect", {}).get("default_model", "anthropic/claude-3.5-sonnet")
        coder_model = self.config.get("coder", {}).get("default_model", "openai/gpt-4-turbo-preview")
        reviewer_model = self.config.get("reviewer", {}).get("default_model", "anthropic/claude-3.5-sonnet")
        refiner_model = coder_model  # Same as coder
        tester_model = self.config.get("coder", {}).get("default_model", "openai/gpt-3.5-turbo")
        doc_model = self.config.get("documenter", {}).get("default_model", "openai/gpt-3.5-turbo")

        return {
            AgentRole.ARCHITECT: AgentCommunicationProfile(
                role=AgentRole.ARCHITECT,
                model_id=architect_model,
                preferred_input_format="markdown",
                preferred_output_format="markdown",
                context_requirements=["original_request"]
            ),
            AgentRole.CODER: AgentCommunicationProfile(
                role=AgentRole.CODER,
                model_id=coder_model,
                preferred_input_format="markdown",
                preferred_output_format="code",
                context_requirements=["original_request", "architecture"]
            ),
            AgentRole.REVIEWER: AgentCommunicationProfile(
                role=AgentRole.REVIEWER,
                model_id=reviewer_model,
                preferred_input_format="code",
                preferred_output_format="json",
                context_requirements=["original_request", "architecture", "implementation"]
            ),
            AgentRole.DOCUMENTER: AgentCommunicationProfile(
                role=AgentRole.DOCUMENTER,
                model_id=doc_model,
                preferred_input_format="markdown",
                preferred_output_format="markdown",
                context_requirements=["original_request", "architecture", "final_implementation"]
            )
        }

    @weave.op()
    async def execute_workflow(
        self,
        task: str,
        max_iterations: int = 3,
        temperature: float = 0.2
    ) -> WorkflowResult:
        """
        Execute sequential collaborative workflow.

        Args:
            task: User's original request
            max_iterations: Max review-refine iterations
            temperature: LLM temperature

        Returns:
            WorkflowResult with all stage outputs
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()

        def budget_left() -> float:
            return max(0.1, TOTAL_BUDGET_S - (time.time() - start_time))

        stages: List[StageResult] = []
        context = {
            "original_request": task,
            "run_id": run_id
        }

        try:
            # Stage 1: Architecture
            logger.info(f"[{run_id}] Stage 1: Architecture")
            arch_timeout = min(STAGE_TIMEOUT_S * ARCHITECTURE_TIMEOUT_MULTIPLIER, budget_left())
            arch_result = await self._architect_stage(context, timeout=arch_timeout, temperature=temperature)
            stages.append(arch_result)
            context["architecture"] = arch_result.output

            # Stage 2: Implementation
            logger.info(f"[{run_id}] Stage 2: Implementation")
            impl_timeout = min(STAGE_TIMEOUT_S * IMPLEMENTATION_TIMEOUT_MULTIPLIER, budget_left())
            impl_result = await self._coder_stage(context, timeout=impl_timeout, temperature=temperature)
            stages.append(impl_result)
            context["implementation"] = impl_result.output

            # Stage 3: Review
            logger.info(f"[{run_id}] Stage 3: Review")
            review_timeout = min(STAGE_TIMEOUT_S, budget_left())
            review_result = await self._reviewer_stage(context, timeout=review_timeout, temperature=0.0)
            stages.append(review_result)
            context["review"] = review_result.output

            # Stage 4: Refinement (iterate if issues found)
            # REFINER = CODER agent with review context
            iterations = 0
            issues_found = self._parse_review_result(review_result.output)

            while issues_found and iterations < max_iterations and budget_left() > 10:
                iterations += 1
                logger.info(f"[{run_id}] Stage 4: Refinement (iteration {iterations}) - using CODER as refiner")

                refine_timeout = min(STAGE_TIMEOUT_S, budget_left())
                # Use coder_stage as refiner with review context
                refine_result = await self._coder_refine_stage(context, timeout=refine_timeout, temperature=temperature)
                stages.append(refine_result)
                context["implementation"] = refine_result.output  # Update implementation

                # Re-review
                review_timeout = min(STAGE_TIMEOUT_S, budget_left())
                review_result = await self._reviewer_stage(context, timeout=review_timeout, temperature=0.0)
                stages.append(review_result)
                context["review"] = review_result.output

                issues_found = self._parse_review_result(review_result.output)

            context["final_implementation"] = context["implementation"]

            # Stage 5: Documentation (no testing - can be requested post-delivery)
            logger.info(f"[{run_id}] Stage 5: Documentation")
            if budget_left() > 10:
                doc_timeout = min(STAGE_TIMEOUT_S, budget_left())
                doc_result = await self._documenter_stage(context, timeout=doc_timeout, temperature=temperature)
                stages.append(doc_result)
                context["documentation"] = doc_result.output

            total_duration = time.time() - start_time

            return WorkflowResult(
                run_id=run_id,
                original_request=task,
                workflow_name="feature_development",
                stages=stages,
                final_output=context.get("final_implementation", ""),
                iterations=iterations,
                total_duration_seconds=total_duration,
                success=True
            )

        except Exception as e:
            logger.error(f"[{run_id}] Workflow failed: {e}")
            total_duration = time.time() - start_time

            return WorkflowResult(
                run_id=run_id,
                original_request=task,
                workflow_name="feature_development",
                stages=stages,
                final_output="",
                iterations=0,
                total_duration_seconds=total_duration,
                success=False
            )

    async def _architect_stage(
        self,
        context: Dict[str, Any],
        timeout: float,
        temperature: float
    ) -> StageResult:
        """Architecture and design stage"""
        start = time.time()
        profile = self.agent_profiles[AgentRole.ARCHITECT]

        task = context["original_request"]

        prompt = f"""As a software architect, design a solution for this task:

Task: {task}

Provide:
1. High-level architecture
2. Component breakdown
3. Key design decisions
4. Data flow
5. Interface definitions

Return a structured design document in Markdown format."""

        try:
            output = await self._call_llm(
                agent_role=profile.role,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout
            )

            return StageResult(
                stage="architecture",
                agent_role=AgentRole.ARCHITECT,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"original_request": task},
                output=output,
                format="markdown",
                duration_seconds=time.time() - start,
                success=not output.startswith("[ERROR]")
            )
        except Exception as e:
            return StageResult(
                stage="architecture",
                agent_role=AgentRole.ARCHITECT,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"original_request": task},
                output=f"[ERROR] {str(e)}",
                format="markdown",
                duration_seconds=time.time() - start,
                success=False,
                error=str(e)
            )

    async def _coder_stage(
        self,
        context: Dict[str, Any],
        timeout: float,
        temperature: float
    ) -> StageResult:
        """Implementation stage"""
        start = time.time()
        profile = self.agent_profiles[AgentRole.CODER]

        task = context["original_request"]
        architecture = context.get("architecture", "")

        prompt = f"""Implement this solution based on the architecture:

Original Task: {task}

Architecture:
{architecture}

Generate complete, production-ready code that:
1. Follows the architectural design
2. Includes robust error handling
3. Is well-structured and maintainable
4. Follows best practices

Return ONLY the code, no explanations."""

        try:
            output = await self._call_llm(
                agent_role=profile.role,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout
            )

            # Extract code blocks
            code = self.format_converter.extract_code(output)

            return StageResult(
                stage="implementation",
                agent_role=AgentRole.CODER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"original_request": task, "architecture": architecture},
                output=code,
                format="code",
                duration_seconds=time.time() - start,
                success=not output.startswith("[ERROR]")
            )
        except Exception as e:
            return StageResult(
                stage="implementation",
                agent_role=AgentRole.CODER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"original_request": task, "architecture": architecture},
                output=f"[ERROR] {str(e)}",
                format="code",
                duration_seconds=time.time() - start,
                success=False,
                error=str(e)
            )

    async def _reviewer_stage(
        self,
        context: Dict[str, Any],
        timeout: float,
        temperature: float
    ) -> StageResult:
        """Code review stage"""
        start = time.time()
        profile = self.agent_profiles[AgentRole.REVIEWER]

        code = context.get("implementation", "")
        architecture = context.get("architecture", "")

        prompt = f"""Review this code implementation.

Architecture Intent:
{architecture}

Code to Review:
{code}

Return ONLY JSON with this exact schema:
{{
  "issues_found": true,
  "critical_issues": ["..."],
  "suggestions": ["..."],
  "code_quality_score": 0
}}"""

        try:
            output = await self._call_llm(
                agent_role=profile.role,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout
            )

            return StageResult(
                stage="review",
                agent_role=AgentRole.REVIEWER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"implementation": code[:200], "architecture": architecture[:200]},
                output=output,
                format="json",
                duration_seconds=time.time() - start,
                success=not output.startswith("[ERROR]")
            )
        except Exception as e:
            return StageResult(
                stage="review",
                agent_role=AgentRole.REVIEWER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"implementation": code[:200], "architecture": architecture[:200]},
                output=f"[ERROR] {str(e)}",
                format="json",
                duration_seconds=time.time() - start,
                success=False,
                error=str(e)
            )

    async def _coder_refine_stage(
        self,
        context: Dict[str, Any],
        timeout: float,
        temperature: float
    ) -> StageResult:
        """Refinement stage - REUSES CODER agent with review context"""
        start = time.time()
        profile = self.agent_profiles[AgentRole.CODER]

        code = context.get("implementation", "")
        review = context.get("review", "")
        task = context.get("original_request", "")

        prompt = f"""Fix the issues found in code review.

Original Task: {task}

Current Code:
{code}

Review Feedback:
{review}

Apply the suggested improvements and return the refined code.
Focus on:
1. Fixing identified issues
2. Improving performance
3. Enhancing readability
4. Following best practices

Return ONLY the complete refined code."""

        try:
            output = await self._call_llm(
                agent_role=profile.role,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout
            )

            # Extract code blocks
            refined_code = self.format_converter.extract_code(output)

            return StageResult(
                stage="refinement",
                agent_role=AgentRole.CODER,  # ← CODER acting as refiner
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"implementation": code[:200], "review": review[:200]},
                output=refined_code,
                format="code",
                duration_seconds=time.time() - start,
                success=not output.startswith("[ERROR]")
            )
        except Exception as e:
            return StageResult(
                stage="refinement",
                agent_role=AgentRole.CODER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"implementation": code[:200], "review": review[:200]},
                output=f"[ERROR] {str(e)}",
                format="code",
                duration_seconds=time.time() - start,
                success=False,
                error=str(e)
            )

    async def _documenter_stage(
        self,
        context: Dict[str, Any],
        timeout: float,
        temperature: float
    ) -> StageResult:
        """Documentation generation stage"""
        start = time.time()
        profile = self.agent_profiles[AgentRole.DOCUMENTER]

        architecture = context.get("architecture", "")
        code = context.get("final_implementation", "")

        prompt = f"""Create comprehensive documentation:

Architecture:
{architecture}

Implementation:
{code}

Generate:
1. README with overview and setup
2. API documentation
3. Usage examples
4. Architecture explanation

Return Markdown documentation."""

        try:
            output = await self._call_llm(
                agent_role=profile.role,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout
            )

            return StageResult(
                stage="documentation",
                agent_role=AgentRole.DOCUMENTER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"architecture": architecture[:200], "final_implementation": code[:200]},
                output=output,
                format="markdown",
                duration_seconds=time.time() - start,
                success=not output.startswith("[ERROR]")
            )
        except Exception as e:
            return StageResult(
                stage="documentation",
                agent_role=AgentRole.DOCUMENTER,
                model_id=profile.model_id,
                timestamp=datetime.now().isoformat(),
                input_context={"architecture": architecture[:200], "final_implementation": code[:200]},
                output=f"[ERROR] {str(e)}",
                format="markdown",
                duration_seconds=time.time() - start,
                success=False,
                error=str(e)
            )

    async def _call_llm(
        self,
        agent_role: AgentRole,
        prompt: str,
        temperature: float,
        timeout: float
    ) -> str:
        """Call LLM with timeout and error handling"""
        try:
            async def _call():
                # Convert AgentRole enum to agent_id string (e.g., AgentRole.CODER -> "coder")
                agent_id = agent_role.value
                # Note: temperature is configured in agent config, not passed as parameter
                return await self.llm.execute_agent_task(agent_id, prompt)

            output = await asyncio.wait_for(_call(), timeout=timeout)
            return output if isinstance(output, str) else str(output)

        except asyncio.TimeoutError:
            return "[ERROR] LLM timeout"
        except Exception as e:
            return f"[ERROR] LLM error: {e}"

    def _parse_review_result(self, review_output: str) -> bool:
        """Parse review output to determine if issues were found"""
        try:
            # Try to parse as JSON
            match = re.search(r"\{.*\}", review_output, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return bool(data.get("issues_found", False))
        except:
            pass

        # Fallback: keyword heuristic
        lower = review_output.lower()
        return ("critical" in lower) or ("bug" in lower) or ("issue" in lower)
