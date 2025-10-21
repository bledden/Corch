"""
Granular Matching Strategy - Adapter for GranularModelSelector

Wraps the existing GranularModelSelector to conform to the BaseModelSelector interface.
This strategy selects models based on language, framework, and task-specific strengths.
"""

from typing import Dict, Any, List
from agents.base_model_selector import (
    BaseModelSelector,
    SelectionContext,
    SelectionResult
)
from agents.granular_model_selector import (
    GranularModelSelector,
    TaskContext
)
import logging

logger = logging.getLogger(__name__)


class GranularMatchingStrategy(BaseModelSelector):
    """
    Granular context-based model selection

    This strategy picks models based on their measured strengths in specific
    scenarios: programming languages, frameworks, task types (debugging, review, etc.).

    Benefits:
    - Matches models to their actual strengths
    - Language-specific optimization (e.g., Claude for Python, GPT for JS)
    - Framework-aware (React, FastAPI, Django, etc.)
    - Task-type aware (debugging vs coding vs review)
    """

    def __init__(self):
        """Initialize granular matching strategy"""
        self.granular_selector = GranularModelSelector()
        self.model_overrides = {}
        logger.info("Initialized GranularMatchingStrategy")

    def select_model(self, context: SelectionContext) -> SelectionResult:
        """
        Select model based on granular context matching

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

        # Convert unified context to GranularModelSelector's context
        granular_context = TaskContext(
            task_type=context.task_type,
            primary_language=context.primary_language,
            frameworks=context.frameworks or [],
            complexity=context.complexity
        )

        # Get available models (use all models from strength matrix)
        available_models = list(self.granular_selector.strength_matrix.LANGUAGE_STRENGTHS.keys())

        # Select best model
        model_id, confidence = self.granular_selector.select_best_model(
            granular_context,
            available_models,
            generation=context.generation
        )

        # Build reasoning
        reasons = []
        if context.primary_language:
            reasons.append(f"language={context.primary_language}")
        if context.frameworks:
            reasons.append(f"frameworks={','.join(context.frameworks)}")
        reasons.append(f"task_type={context.task_type}")
        reasons.append(f"complexity={context.complexity}")

        reasoning = (
            f"Granular matching selected {model_id} based on: {', '.join(reasons)}. "
            f"Confidence: {confidence:.2f}"
        )

        return SelectionResult(
            model_id=model_id,
            strategy_used="GRANULAR_MATCHING",
            confidence=confidence,
            estimated_cost=self._estimate_cost(model_id),
            estimated_quality=confidence,  # Confidence is based on strength scores
            reasoning=reasoning,
            metadata={
                'primary_language': context.primary_language,
                'frameworks': context.frameworks,
                'task_type': context.task_type,
                'complexity': context.complexity,
                'agent_type': context.agent_type
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
        Update granular selector's contextual performance tracking

        This allows the selector to learn which models work best in
        specific language/framework/task combinations.
        """
        # Build context key
        context_key = (
            context.primary_language or "general",
            ",".join(sorted(context.frameworks)) if context.frameworks else "none",
            context.task_type
        )

        # Update performance
        perf = self.granular_selector.contextual_performance[context_key][model_id]
        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1

        # Update rolling average quality
        total = perf["successes"] + perf["failures"]
        perf["avg_quality"] = (
            (perf["avg_quality"] * (total - 1) + quality_score) / total
            if total > 0 else quality_score
        )

        logger.info(
            f"Updated Granular Matching: {model_id} on {context_key} - "
            f"success={success}, quality={quality_score:.2f}"
        )

    def get_available_models(self) -> list[str]:
        """Get list of models in the strength matrix"""
        return list(self.granular_selector.strength_matrix.LANGUAGE_STRENGTHS.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get granular matching statistics"""
        # Aggregate contextual performance stats
        context_stats = {}
        total_contexts = len(self.granular_selector.contextual_performance)

        for context_key, model_perfs in self.granular_selector.contextual_performance.items():
            lang, frameworks, task = context_key
            context_name = f"{lang}/{frameworks}/{task}"

            stats = {
                'total_tasks': 0,
                'success_rate': 0.0,
                'model_performance': {}
            }

            total_successes = 0
            total_tasks = 0

            for model_id, perf in model_perfs.items():
                tasks = perf["successes"] + perf["failures"]
                total_tasks += tasks
                total_successes += perf["successes"]

                stats['model_performance'][model_id] = {
                    'successes': perf["successes"],
                    'failures': perf["failures"],
                    'avg_quality': perf["avg_quality"]
                }

            stats['total_tasks'] = total_tasks
            stats['success_rate'] = total_successes / total_tasks if total_tasks > 0 else 0.0

            context_stats[context_name] = stats

        return {
            'strategy_type': 'granular_matching',
            'total_contexts_tracked': total_contexts,
            'context_stats': context_stats,
            'available_models': len(self.get_available_models()),
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
        """Estimate cost for a model (simplified)"""
        cost_map = {
            "gpt-4-turbo-2025-01": 0.015,
            "claude-3.5-sonnet-20241022": 0.015,
            "anthropic/claude-sonnet-4.5": 0.015,
            "qwen-2.5-coder": 0.0,
            "deepseek-coder-v2": 0.001,
        }

        base_cost = cost_map.get(model_id, 0.01)
        estimated_tokens = 2000
        return (base_cost / 1_000_000) * estimated_tokens
