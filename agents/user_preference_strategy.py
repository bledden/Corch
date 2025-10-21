"""
User Preference Strategy - Adapter for StrategySelector

Wraps the existing StrategySelector to conform to the BaseModelSelector interface.
This allows strategy-based selection (QUALITY_FIRST, COST_FIRST, BALANCED, etc.)
to work with the unified model selection system.
"""

from typing import Dict, Any
from agents.base_model_selector import (
    BaseModelSelector,
    SelectionContext,
    SelectionResult
)
from agents.strategy_selector import (
    StrategySelector,
    ModelSelectionContext,
    Strategy
)
import logging

logger = logging.getLogger(__name__)


class UserPreferenceStrategy(BaseModelSelector):
    """
    User preference-based model selection

    This strategy lets users choose between QUALITY_FIRST, COST_FIRST,
    BALANCED, SPEED_FIRST, and PRIVACY_FIRST approaches. Each strategy
    defines preferences for which models to use for different agents.
    """

    def __init__(self, config_path: str = "config/model_strategy_config.yaml"):
        """
        Initialize with strategy configuration

        Args:
            config_path: Path to strategy configuration YAML
        """
        self.strategy_selector = StrategySelector(config_path)
        self.model_overrides = {}
        logger.info("Initialized UserPreferenceStrategy")

    def select_model(self, context: SelectionContext) -> SelectionResult:
        """
        Select model using user's preferred strategy

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
                estimated_cost=self._estimate_cost(context.force_model, context),
                estimated_quality=0.9,  # Assume high quality for manual override
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
                estimated_cost=self._estimate_cost(override_model, context),
                estimated_quality=0.9,
                reasoning=f"Agent override for {context.agent_type}: {override_model}",
                metadata={"override": True, "agent_type": context.agent_type}
            )

        # Convert unified context to StrategySelector's context
        strategy_context = ModelSelectionContext(
            task_type=context.task_type,
            task_complexity=context.task_complexity,
            remaining_budget=context.remaining_budget,
            sensitive_data=context.sensitive_data,
            required_latency=context.required_latency,
            user_waiting=context.user_waiting
        )

        # Call original StrategySelector
        model_id, selection_info = self.strategy_selector.select_model(
            context.agent_type,
            strategy_context
        )

        # Convert to unified SelectionResult
        return SelectionResult(
            model_id=model_id,
            strategy_used=selection_info['strategy_used'],
            confidence=selection_info.get('quality_score', 0.8),
            estimated_cost=selection_info['estimated_cost'],
            estimated_quality=selection_info.get('quality_score', 0.8),
            reasoning=selection_info['reason'],
            metadata={
                'original_strategy': selection_info['original_strategy'],
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
        Update performance tracking

        StrategySelector doesn't learn from performance (it's rule-based),
        but we track stats for observability.
        """
        # StrategySelector is rule-based, not learning-based
        # We could add optional performance tracking here for analytics
        logger.debug(
            f"Performance update: {model_id} - "
            f"success={success}, quality={quality_score:.2f}, "
            f"latency={latency_ms:.0f}ms, cost=${cost_usd:.4f}"
        )

    def get_available_models(self) -> list[str]:
        """Get list of models available in the strategy configuration"""
        # Extract unique models from all strategies
        models = set()
        for strategy_name, strategy_config in self.strategy_selector.config['strategies'].items():
            for agent_type, model_list in strategy_config['model_preferences'].items():
                models.update(model_list)
        return sorted(list(models))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about strategy usage"""
        return {
            'strategy_type': 'user_preference',
            'current_strategy': self.strategy_selector.current_strategy.value,
            'total_tasks': self.strategy_selector.task_count,
            'total_cost': self.strategy_selector.total_cost,
            'available_models': len(self.get_available_models()),
            'overrides_active': len(self.model_overrides)
        }

    def set_model_override(self, agent_type: str, model_id: str):
        """Set manual override for specific agent type"""
        self.model_overrides[agent_type] = model_id
        self.strategy_selector.set_model_override(agent_type, model_id)
        logger.info(f"Set model override: {agent_type} -> {model_id}")

    def clear_overrides(self):
        """Clear all manual overrides"""
        self.model_overrides.clear()
        self.strategy_selector.model_overrides.clear()
        logger.info("Cleared all model overrides")

    def set_user_strategy(self, strategy: Strategy):
        """Change the active user strategy"""
        self.strategy_selector.set_user_strategy(strategy)
        logger.info(f"Changed strategy to: {strategy.value}")

    def get_user_strategy(self) -> Strategy:
        """Get the current user strategy"""
        return self.strategy_selector.get_user_strategy()

    def _estimate_cost(self, model_id: str, context: SelectionContext) -> float:
        """Estimate cost for a model (simplified)"""
        # Use StrategySelector's cost estimation
        strategy_context = ModelSelectionContext(
            task_type=context.task_type,
            task_complexity=context.task_complexity,
            remaining_budget=context.remaining_budget
        )
        return self.strategy_selector._estimate_cost(model_id, strategy_context)
