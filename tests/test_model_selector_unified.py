"""
Comprehensive tests for unified model selector system

Tests all three strategies (UserPreference, ThompsonSampling, GranularMatching)
and the factory pattern for creating selectors.
"""

import pytest
from agents.model_selector_factory import ModelSelectorFactory, reset_model_selector
from agents.base_model_selector import SelectionStrategy, SelectionContext
from agents.user_preference_strategy import UserPreferenceStrategy
from agents.thompson_sampling_strategy import ThompsonSamplingStrategy
from agents.granular_matching_strategy import GranularMatchingStrategy
from agents.strategy_selector import Strategy


class TestModelSelectorFactory:
    """Test the factory pattern for creating selectors"""

    def teardown_method(self):
        """Reset singleton after each test"""
        reset_model_selector()

    def test_create_user_preference_strategy(self):
        """Test creating UserPreferenceStrategy"""
        selector = ModelSelectorFactory.create(SelectionStrategy.USER_PREFERENCE)
        assert isinstance(selector, UserPreferenceStrategy)
        assert len(selector.get_available_models()) > 0

    def test_create_thompson_sampling_strategy(self):
        """Test creating ThompsonSamplingStrategy"""
        models = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o"]
        selector = ModelSelectorFactory.create(
            SelectionStrategy.THOMPSON_SAMPLING,
            models=models,
            alpha=2.0,
            beta=1.5
        )
        assert isinstance(selector, ThompsonSamplingStrategy)
        assert selector.get_available_models() == models
        assert selector.thompson_selector.alpha == 2.0
        assert selector.thompson_selector.beta == 1.5

    def test_create_granular_matching_strategy(self):
        """Test creating GranularMatchingStrategy"""
        selector = ModelSelectorFactory.create(SelectionStrategy.GRANULAR_MATCHING)
        assert isinstance(selector, GranularMatchingStrategy)
        assert len(selector.get_available_models()) > 0

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError"""
        with pytest.raises(ValueError, match="Unknown selection strategy"):
            ModelSelectorFactory.create("invalid_strategy")


class TestUserPreferenceStrategy:
    """Test UserPreferenceStrategy functionality"""

    def test_select_model_basic(self):
        """Test basic model selection"""
        selector = UserPreferenceStrategy()
        context = SelectionContext(
            agent_type="architect",
            task_type="architecture",
            task_complexity=0.7
        )

        result = selector.select_model(context)

        assert result.model_id is not None
        assert result.confidence > 0.0
        assert result.estimated_cost >= 0.0
        assert len(result.reasoning) > 0

    def test_manual_override(self):
        """Test manual model override"""
        selector = UserPreferenceStrategy()
        context = SelectionContext(
            agent_type="coder",
            task_type="coding",
            force_model="anthropic/claude-sonnet-4.5"
        )

        result = selector.select_model(context)

        assert result.model_id == "anthropic/claude-sonnet-4.5"
        assert result.strategy_used == "MANUAL_OVERRIDE"

    def test_agent_override(self):
        """Test agent-specific override"""
        selector = UserPreferenceStrategy()
        selector.set_model_override("reviewer", "openai/gpt-4o")

        context = SelectionContext(
            agent_type="reviewer",
            task_type="review"
        )

        result = selector.select_model(context)

        assert result.model_id == "openai/gpt-4o"
        assert result.strategy_used == "AGENT_OVERRIDE"

    def test_clear_overrides(self):
        """Test clearing overrides"""
        selector = UserPreferenceStrategy()
        selector.set_model_override("coder", "test-model")

        assert len(selector.model_overrides) == 1

        selector.clear_overrides()

        assert len(selector.model_overrides) == 0

    def test_get_stats(self):
        """Test getting selector statistics"""
        selector = UserPreferenceStrategy()
        stats = selector.get_stats()

        assert stats['strategy_type'] == 'user_preference'
        assert 'current_strategy' in stats
        assert 'available_models' in stats


class TestThompsonSamplingStrategy:
    """Test ThompsonSamplingStrategy functionality"""

    def test_select_model_exploration_phase(self):
        """Test model selection during exploration phase (generation < 3)"""
        models = ["model1", "model2", "model3"]
        selector = ThompsonSamplingStrategy(models=models)

        context = SelectionContext(
            agent_type="coder",
            task_type="coding",
            generation=0  # Exploration phase
        )

        result = selector.select_model(context)

        assert result.model_id in models
        assert "EXPLORATION" in result.strategy_used
        assert result.confidence > 0.0

    def test_select_model_exploitation_phase(self):
        """Test model selection during exploitation phase (generation >= 3)"""
        models = ["model1", "model2"]
        selector = ThompsonSamplingStrategy(models=models)

        context = SelectionContext(
            agent_type="coder",
            task_type="coding",
            generation=5  # Exploitation phase
        )

        result = selector.select_model(context)

        assert result.model_id in models
        assert "EXPLOITATION" in result.strategy_used

    def test_update_performance(self):
        """Test updating performance metrics"""
        selector = ThompsonSamplingStrategy()

        context = SelectionContext(
            agent_type="coder",
            task_type="coding",
            generation=0
        )

        # Select a model
        result = selector.select_model(context)
        model_id = result.model_id

        # Update performance
        selector.update_performance(
            model_id=model_id,
            context=context,
            success=True,
            quality_score=0.9,
            latency_ms=1500,
            cost_usd=0.02
        )

        # Check that performance was recorded
        task_key = f"{context.agent_type}:{context.task_type}"
        perf = selector.thompson_selector.performance[task_key][model_id]

        assert perf.successes == 1
        assert perf.failures == 0
        assert perf.avg_quality > 0.0

    def test_get_stats(self):
        """Test getting selector statistics"""
        selector = ThompsonSamplingStrategy()
        stats = selector.get_stats()

        assert stats['strategy_type'] == 'thompson_sampling'
        assert 'total_tasks' in stats
        assert 'available_models' in stats


class TestGranularMatchingStrategy:
    """Test GranularMatchingStrategy functionality"""

    def test_select_model_with_language(self):
        """Test model selection with specific language"""
        selector = GranularMatchingStrategy()

        context = SelectionContext(
            agent_type="coder",
            task_type="implementation",
            primary_language="python",
            complexity="medium"
        )

        result = selector.select_model(context)

        assert result.model_id is not None
        assert result.strategy_used == "GRANULAR_MATCHING"
        assert "python" in result.reasoning.lower()
        assert result.confidence > 0.0

    def test_select_model_with_frameworks(self):
        """Test model selection with frameworks"""
        selector = GranularMatchingStrategy()

        context = SelectionContext(
            agent_type="coder",
            task_type="implementation",
            primary_language="python",
            frameworks=["fastapi", "pydantic"],
            complexity="complex"
        )

        result = selector.select_model(context)

        assert result.model_id is not None
        assert "frameworks" in result.reasoning.lower()

    def test_update_performance_tracks_context(self):
        """Test that performance updates are tracked by context"""
        selector = GranularMatchingStrategy()

        context = SelectionContext(
            agent_type="coder",
            task_type="debugging",
            primary_language="rust",
            frameworks=["tokio"]
        )

        result = selector.select_model(context)

        # Update performance
        selector.update_performance(
            model_id=result.model_id,
            context=context,
            success=True,
            quality_score=0.85,
            latency_ms=2000,
            cost_usd=0.03
        )

        # Check contextual performance was recorded
        context_key = ("rust", "tokio", "debugging")
        assert context_key in selector.granular_selector.contextual_performance
        perf = selector.granular_selector.contextual_performance[context_key][result.model_id]
        assert perf["successes"] == 1

    def test_get_stats(self):
        """Test getting selector statistics"""
        selector = GranularMatchingStrategy()
        stats = selector.get_stats()

        assert stats['strategy_type'] == 'granular_matching'
        assert 'total_contexts_tracked' in stats
        assert 'available_models' in stats


class TestUnifiedInterface:
    """Test that all strategies conform to the unified interface"""

    @pytest.fixture(params=[
        SelectionStrategy.USER_PREFERENCE,
        SelectionStrategy.THOMPSON_SAMPLING,
        SelectionStrategy.GRANULAR_MATCHING
    ])
    def selector(self, request):
        """Fixture that provides each strategy"""
        reset_model_selector()
        return ModelSelectorFactory.create(request.param)

    def test_all_strategies_have_select_model(self, selector):
        """Test that all strategies implement select_model"""
        context = SelectionContext(
            agent_type="coder",
            task_type="coding"
        )

        result = selector.select_model(context)

        assert result.model_id is not None
        assert result.strategy_used is not None
        assert result.confidence >= 0.0
        assert result.estimated_cost >= 0.0

    def test_all_strategies_have_get_available_models(self, selector):
        """Test that all strategies implement get_available_models"""
        models = selector.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

    def test_all_strategies_have_get_stats(self, selector):
        """Test that all strategies implement get_stats"""
        stats = selector.get_stats()

        assert isinstance(stats, dict)
        assert 'strategy_type' in stats

    def test_all_strategies_support_overrides(self, selector):
        """Test that all strategies support set_model_override"""
        # Should not raise an error
        selector.set_model_override("coder", "test-model")
        selector.clear_overrides()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
