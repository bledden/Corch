"""
LLM Client for real agent execution
Supports OpenAI, Anthropic, and Google models
"""

import os
import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
import json
from dotenv import load_dotenv
import weave

# Load environment variables
load_dotenv()

# Import LLM libraries conditionally
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not available. Install with: pip install anthropic")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: Google AI not available. Install with: pip install google-generativeai")


@dataclass
class LLMResponse:
    """Response from an LLM"""
    content: str
    model: str
    tokens_used: int
    latency: float
    error: Optional[str] = None


class LLMClient:
    """Unified client for multiple LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize LLM clients based on available API keys"""

        # OpenAI (using new client API)
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        # Google
        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.google_client = genai

    @weave.op()
    async def execute_llm(
        self,
        agent_id: str,
        task: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        personality: str = "",
        expertise: list = None
    ) -> LLMResponse:
        """Execute task with specified LLM"""

        # Build the prompt
        prompt = self._build_prompt(agent_id, task, personality, expertise)

        # Track timing
        import time
        start_time = time.time()

        try:
            # Route to appropriate provider
            if "gpt" in model.lower() and self.openai_client:
                response = await self._execute_openai(prompt, model, temperature, max_tokens)
            elif "claude" in model.lower() and self.anthropic_client:
                response = await self._execute_anthropic(prompt, model, temperature, max_tokens)
            elif "gemini" in model.lower() and self.google_client:
                response = await self._execute_google(prompt, model, temperature, max_tokens)
            else:
                # Fallback to simulation if no real LLM available
                response = await self._simulate_response(agent_id, task, model)

            response.latency = time.time() - start_time

            # Log execution for demo
            if os.getenv("DEMO_MODE"):
                print(f"[LLM] {agent_id} using {model}: {response.content[:100]}...")

            return response

        except Exception as e:
            error_msg = f"LLM execution failed: {str(e)}"
            if os.getenv("DEMO_MODE"):
                print(f"[ERROR] {agent_id}: {error_msg}")

            # Return fallback response
            return LLMResponse(
                content=f"[{agent_id}] encountered an error but suggests: {task[:50]}...",
                model=model,
                tokens_used=0,
                latency=time.time() - start_time,
                error=error_msg
            )

    def _build_prompt(self, agent_id: str, task: str, personality: str, expertise: list) -> str:
        """Build prompt for LLM"""

        expertise_str = ", ".join(expertise) if expertise else "general tasks"

        prompt = f"""You are {agent_id}, an AI agent with the following characteristics:

Personality: {personality}
Expertise: {expertise_str}

Your task is: {task}

Please provide a detailed response that:
1. Leverages your specific expertise
2. Provides concrete, actionable insights
3. Considers potential challenges and solutions
4. Maintains your personality and perspective

Response:"""

        return prompt

    async def _execute_openai(self, prompt: str, model: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Execute with OpenAI"""

        if not self.openai_client:
            return await self._simulate_response("openai_agent", prompt, model)

        try:
            # Use the new OpenAI API v1.0+
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                tokens_used=response.usage.total_tokens,
                latency=0.0
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    async def _execute_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Execute with Anthropic"""

        if not self.anthropic_client:
            return await self._simulate_response("claude_agent", prompt, model)

        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            return LLMResponse(
                content=response.content[0].text,
                model=model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency=0.0
            )
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    async def _execute_google(self, prompt: str, model: str, temperature: float, max_tokens: int) -> LLMResponse:
        """Execute with Google AI"""

        if not self.google_client:
            return await self._simulate_response("gemini_agent", prompt, model)

        try:
            model_instance = genai.GenerativeModel(model)

            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            response = await asyncio.to_thread(
                model_instance.generate_content,
                prompt,
                generation_config=generation_config
            )

            # Estimate tokens (Google doesn't provide exact counts)
            estimated_tokens = len(prompt.split()) + len(response.text.split())

            return LLMResponse(
                content=response.text,
                model=model,
                tokens_used=estimated_tokens * 1.3,  # Rough token estimate
                latency=0.0
            )
        except Exception as e:
            raise Exception(f"Google AI API error: {str(e)}")

    async def _simulate_response(self, agent_id: str, task: str, model: str) -> LLMResponse:
        """Simulate response when no LLM is available"""

        await asyncio.sleep(0.5)  # Simulate latency

        # Generate response based on agent type
        responses = {
            "architect": f"For '{task[:50]}...', I recommend a modular architecture with clear separation of concerns. "
                       f"We should use microservices for scalability and implement proper API versioning.",

            "coder": f"I'll implement '{task[:50]}...' using best practices. "
                    f"Here's my approach:\n1. Set up the project structure\n2. Implement core functionality\n"
                    f"3. Add comprehensive error handling\n4. Write unit tests",

            "reviewer": f"Reviewing '{task[:50]}...': I've identified several areas for improvement:\n"
                       f"1. Add input validation\n2. Improve error messages\n3. Consider edge cases\n"
                       f"4. Enhance test coverage",

            "documenter": f"Documentation for '{task[:50]}...':\n"
                         f"## Overview\nThis component handles the specified task.\n\n"
                         f"## Usage\nProvide clear examples and API references.\n\n"
                         f"## Best Practices\nFollow established patterns.",

            "researcher": f"Research findings for '{task[:50]}...':\n"
                         f"Based on current best practices, I recommend:\n"
                         f"1. Consider industry standards\n2. Review similar implementations\n"
                         f"3. Analyze performance implications"
        }

        # Get response based on agent_id or use default
        agent_type = agent_id.split("_")[0] if "_" in agent_id else agent_id
        content = responses.get(agent_type, f"[{agent_id}] Processing: {task[:100]}...")

        return LLMResponse(
            content=content,
            model=f"{model} (simulated)",
            tokens_used=len(content.split()) * 1.3,  # Rough estimate
            latency=0.5
        )


class MultiAgentLLMOrchestrator:
    """Orchestrator for managing multiple LLM agents"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = LLMClient(config)
        self.agent_configs = config.get("agents", {})

    @weave.op()
    async def execute_agent_task(self, agent_id: str, task: str) -> str:
        """Execute a task with a specific agent using their configured LLM"""

        # Get agent configuration
        agent_config = self.agent_configs.get(agent_id, {})

        # Execute with LLM
        response = await self.llm_client.execute_llm(
            agent_id=agent_id,
            task=task,
            model=agent_config.get("model", "gpt-4"),
            temperature=agent_config.get("temperature", 0.7),
            max_tokens=agent_config.get("max_tokens", 2000),
            personality=agent_config.get("personality", ""),
            expertise=agent_config.get("expertise", [])
        )

        if response.error:
            # Log error but return content anyway
            if os.getenv("DEMO_MODE"):
                print(f"[WARN] {agent_id} error: {response.error}")

        return response.content

    async def execute_parallel_agents(self, agents: list, task: str) -> Dict[str, str]:
        """Execute task with multiple agents in parallel"""

        tasks = [
            self.execute_agent_task(agent_id, task)
            for agent_id in agents
        ]

        results = await asyncio.gather(*tasks)

        return {
            agent_id: result
            for agent_id, result in zip(agents, results)
        }