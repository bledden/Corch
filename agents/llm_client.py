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
import litellm

# Load environment variables
load_dotenv()


@dataclass
class LLMResponse:
    """Response from an LLM"""
    content: str
    model: str
    tokens_used: int
    latency: float
    error: Optional[str] = None


class LLMClient:
    """Unified client for multiple LLM providers using LiteLLM and OpenRouter"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Configure LiteLLM for OpenRouter
        litellm.api_key = os.getenv("OPENROUTER_API_KEY")
        litellm.api_base = "https://openrouter.ai/api/v1"

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
            # Use LiteLLM to call OpenRouter - prefix model with "openrouter/"
            openrouter_model = f"openrouter/{model}"
            response = await litellm.acompletion(
                model=openrouter_model,  # e.g., "openrouter/openai/gpt-4", "openrouter/anthropic/claude-sonnet-4.5"
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                tokens_used=response.usage.total_tokens,  # REAL tokens from OpenRouter
                latency=time.time() - start_time
            )

            # Log execution for demo
            if os.getenv("DEMO_MODE"):
                print(f"[LLM] {agent_id} using {model}: {llm_response.content[:100]}...")

            return llm_response

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

        # Execute with LLM using default_model from config
        response = await self.llm_client.execute_llm(
            agent_id=agent_id,
            task=task,
            model=agent_config.get("default_model", "qwen/qwen3-coder-plus"),  # Latest Qwen coding model as fallback
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