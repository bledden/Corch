"""
Thompson Sampling Strategy - Adapter for ThompsonSamplingSelector

Wraps the existing ThompsonSamplingSelector to conform to the BaseModelSelector interface.
This strategy uses reinforcement learning to learn which models perform best over time.
"""

from typing import Dict, Any, List
from agents.base_model_selector import (
    BaseModelSelector,
    SelectionContext,
    SelectionResult
)
from agents.model_selector import (
    ThompsonSamplingSelector,
    ModelPerformance
)
import logging

logger = logging.getLogger(__name__)


class ThompsonSamplingStrategy(BaseModelSelector):
    """
    Thompson Sampling-based model selection

    This strategy uses Bayesian reinforcement learning to balance exploration
    (trying new models) and exploitation (using known good models). It learns
    which models work best for different task types over time.

    Benefits:
    - Automatically learns optimal model selection
    - Balances exploration vs exploitation
    - Adapts to changing model performance
    - No manual configuration needed
    """

    def __init__(
        self,
        models: List[str] = None,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """
        Initialize Thompson Sampling strategy

        Args:
            models: List of model IDs to choose from
            alpha: Prior success parameter for Beta distribution
            beta: Prior failure parameter for Beta distribution
        """
        # Default models if none provided
        if models is None:
            models = [
                "anthropic/claude-sonnet-4.5",
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4o",
                "qwen/qwen-2.5-coder-32b-instruct",
                "deepseek/deepseek-chat"
            ]

        self.thompson_selector = ThompsonSamplingSelector(
            models=models,
            alpha=alpha,
            beta=beta
        )
        self.model_overrides = {}
        logger.info(f"Initialized ThompsonSamplingStrategy with {len(models)} models")

    def select_model(self, context: SelectionContext) -> SelectionResult:
        """
        Select model using Thompson Sampling

        Args:
            context: Selection context

        Returns:
            SelectionResult with chosen model and metadata
        """
        # Handle manual override first
        if context.force_model:
            return SelectionResult(
                model_id=context.force_model,
                strategy_used="MANUAL_OVERRIDE",
                confidence=1.0,
                estimated_cost=self._estimate_cost(context.force_model),
                estimated_quality=0.9,
                reasoning=f"Manual override specified: {context.force_model}",
                metadata={"override": True}
            )

        # Check agent-specific overrides
        if context.agent_type in self.model_overrides:
            override_model = self.model_overrides[context.agent_type]
            return SelectionResult(
                model_id=override_model,
                strategy_used="AGENT_OVERRIDE",
                confidence=1.0,
                estimated_cost=self._estimate_cost(override_model),
                estimated_quality=0.9,
                reasoning=f"Agent override for {context.agent_type}: {override_model}",
                metadata={"override": True, "agent_type": context.agent_type}
            )

        # Use Thompson Sampling to select model
        # Combine agent_type and task_type for task categorization
        task_key = f"{context.agent_type}:{context.task_type}"

        model_id = self.thompson_selector.select_model(
            task_type=task_key,
            generation=context.generation
        )

        # Get performance stats for confidence estimation
        perf = self.thompson_selector.performance[task_key][model_id]
        total_tries = perf.successes + perf.failures

        # Confidence increases with more data
        confidence = min(0.5 + (total_tries / 20.0), 0.95) if total_tries > 0 else 0.5

        # Determine if exploration or exploitation
        is_exploration = context.generation < 3
        strategy_phase = "exploration" if is_exploration else "exploitation"

        return SelectionResult(
            model_id=model_id,
            strategy_used=f"THOMPSON_SAMPLING_{strategy_phase.upper()}",
            confidence=confidence,
            estimated_cost=self._estimate_cost(model_id),
            estimated_quality=perf.avg_quality if perf.usage_count > 0 else 0.75,
            reasoning=(
                f"Thompson Sampling ({strategy_phase}): "
                f"Selected {model_id} for {task_key}. "
                f"Historical: {perf.successes} successes, {perf.failures} failures, "
                f"avg quality: {perf.avg_quality:.2f}"
            ),
            metadata={
                'task_key': task_key,
                'generation': context.generation,
                'phase': strategy_phase,
                'historical_success_rate': perf.success_rate,
                'historical_avg_quality': perf.avg_quality,
                'usage_count': perf.usage_count
            }
        )

    def update_performance(
        self,
        model_id: str,
        context: SelectionContext,
        success: bool,
        quality_score: float,
        latency_ms: float,
        cost_usd: float
    ):
        """
        Update Thompson Sampling with actual performance

        This is how the strategy learns over time.
        """
        task_key = f"{context.agent_type}:{context.task_type}"

        self.thompson_selector.update_performance(
            task_type=task_key,
            model=model_id,
            success=success,
            quality=quality_score,
            latency=latency_ms / 1000.0,  # Convert to seconds
            cost=cost_usd
        )

        logger.info(
            f"Updated Thompson Sampling: {model_id} on {task_key} - "
            f"success={success}, quality={quality_score:.2f}"
        )

    def get_available_models(self) -> list[str]:
        """Get list of models in the Thompson Sampling pool"""
        return self.thompson_selector.models.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get Thompson Sampling statistics"""
        # Aggregate stats across all task types
        total_tasks = 0
        total_successes = 0
        total_failures = 0
        task_type_stats = {}

        for task_type, model_perfs in self.thompson_selector.performance.items():
            task_stats = {
                'total_tasks': 0,
                'success_rate': 0.0,
                'avg_quality': 0.0,
                'model_usage': {}
            }

            for model_id, perf in model_perfs.items():
                task_total = perf.successes + perf.failures
                total_tasks += task_total
                total_successes += perf.successes
                total_failures += perf.failures

                task_stats['total_tasks'] += task_total
                task_stats['model_usage'][model_id] = {
                    'successes': perf.successes,
                    'failures': perf.failures,
                    'avg_quality': perf.avg_quality,
                    'usage_count': perf.usage_count
                }

            if task_stats['total_tasks'] > 0:
                task_stats['success_rate'] = total_successes / total_tasks if total_tasks > 0 else 0.0

            task_type_stats[task_type] = task_stats

        return {
            'strategy_type': 'thompson_sampling',
            'total_tasks': total_tasks,
            'total_successes': total_successes,
            'total_failures': total_failures,
            'overall_success_rate': total_successes / total_tasks if total_tasks > 0 else 0.0,
            'task_type_stats': task_type_stats,
            'available_models': len(self.thompson_selector.models),
            'overrides_active': len(self.model_overrides)
        }

    def set_model_override(self, agent_type: str, model_id: str):
        """Set manual override for specific agent type"""
        self.model_overrides[agent_type] = model_id
        logger.info(f"Set model override: {agent_type} -> {model_id}")

    def clear_overrides(self):
        """Clear all manual overrides"""
        self.model_overrides.clear()
        logger.info("Cleared all model overrides")

    def _estimate_cost(self, model_id: str) -> float:
        """
        Estimate cost for a model

        This is a simplified estimation. In production, you'd want to use
        actual pricing data from the provider.
        """
        # Rough cost estimates (USD per 1M tokens)
        cost_map = {
            "anthropic/claude-sonnet-4.5": 0.015,
            "anthropic/claude-3.5-sonnet": 0.015,
            "openai/gpt-4o": 0.015,
            "qwen/qwen-2.5-coder-32b-instruct": 0.0,  # Free
            "deepseek/deepseek-chat": 0.001,
        }

        # Default cost for unknown models
        base_cost = cost_map.get(model_id, 0.01)

        # Assume average task uses ~2000 tokens (1k input, 1k output)
        estimated_tokens = 2000
        return (base_cost / 1_000_000) * estimated_tokens
