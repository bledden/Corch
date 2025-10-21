"""
Model Selector Factory - Create and configure model selectors

This factory allows selecting between different model selection strategies
via configuration, making it easy to experiment with different approaches.
"""

import yaml
import os
from pathlib import Path
from typing import Optional
import logging

from agents.base_model_selector import (
    BaseModelSelector,
    SelectionStrategy
)
from agents.user_preference_strategy import UserPreferenceStrategy
from agents.thompson_sampling_strategy import ThompsonSamplingStrategy
from agents.granular_matching_strategy import GranularMatchingStrategy

logger = logging.getLogger(__name__)


class ModelSelectorFactory:
    """
    Factory for creating model selectors

    Usage:
        # Via config file
        selector = ModelSelectorFactory.create_from_config()

        # Explicitly
        selector = ModelSelectorFactory.create(
            strategy=SelectionStrategy.THOMPSON_SAMPLING
        )
    """

    DEFAULT_CONFIG_PATH = "config/model_selector.yaml"

    @staticmethod
    def create(
        strategy: SelectionStrategy = SelectionStrategy.USER_PREFERENCE,
        **kwargs
    ) -> BaseModelSelector:
        """
        Create a model selector with the specified strategy

        Args:
            strategy: Which selection strategy to use
            **kwargs: Strategy-specific configuration

        Returns:
            BaseModelSelector instance

        Raises:
            ValueError: If strategy is unknown
        """
        logger.info(f"Creating model selector with strategy: {strategy.value}")

        if strategy == SelectionStrategy.USER_PREFERENCE:
            config_path = kwargs.get('config_path', 'config/model_strategy_config.yaml')
            return UserPreferenceStrategy(config_path=config_path)

        elif strategy == SelectionStrategy.THOMPSON_SAMPLING:
            models = kwargs.get('models', None)
            alpha = kwargs.get('alpha', 1.0)
            beta = kwargs.get('beta', 1.0)
            return ThompsonSamplingStrategy(
                models=models,
                alpha=alpha,
                beta=beta
            )

        elif strategy == SelectionStrategy.GRANULAR_MATCHING:
            return GranularMatchingStrategy()

        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    @staticmethod
    def create_from_config(config_path: Optional[str] = None) -> BaseModelSelector:
        """
        Create a model selector from configuration file

        Args:
            config_path: Path to config file (defaults to config/model_selector.yaml)

        Returns:
            BaseModelSelector instance configured from file
        """
        config_path = config_path or ModelSelectorFactory.DEFAULT_CONFIG_PATH

        # Check if config file exists
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(
                f"Config file not found: {config_path}, "
                f"using default USER_PREFERENCE strategy"
            )
            return ModelSelectorFactory.create(SelectionStrategy.USER_PREFERENCE)

        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Get strategy from config
        strategy_name = config.get('model_selection', {}).get('strategy', 'user_preference')

        try:
            strategy = SelectionStrategy(strategy_name)
        except ValueError:
            logger.warning(
                f"Unknown strategy '{strategy_name}' in config, "
                f"falling back to USER_PREFERENCE"
            )
            strategy = SelectionStrategy.USER_PREFERENCE

        # Get strategy-specific config
        strategy_config = config.get('model_selection', {}).get('config', {})

        # Create selector
        return ModelSelectorFactory.create(strategy, **strategy_config)

    @staticmethod
    def create_from_env() -> BaseModelSelector:
        """
        Create a model selector from environment variables

        Environment variables:
            MODEL_SELECTION_STRATEGY: user_preference, thompson_sampling, or granular_matching
            MODEL_SELECTION_CONFIG_PATH: Path to config file

        Returns:
            BaseModelSelector instance
        """
        strategy_name = os.getenv('MODEL_SELECTION_STRATEGY', 'user_preference')
        config_path = os.getenv('MODEL_SELECTION_CONFIG_PATH', None)

        try:
            strategy = SelectionStrategy(strategy_name)
        except ValueError:
            logger.warning(
                f"Invalid MODEL_SELECTION_STRATEGY: '{strategy_name}', "
                f"falling back to user_preference"
            )
            strategy = SelectionStrategy.USER_PREFERENCE

        if config_path:
            return ModelSelectorFactory.create_from_config(config_path)
        else:
            return ModelSelectorFactory.create(strategy)


# Singleton instance for convenience
_default_selector: Optional[BaseModelSelector] = None


def get_model_selector(
    strategy: Optional[SelectionStrategy] = None,
    config_path: Optional[str] = None
) -> BaseModelSelector:
    """
    Get or create the default model selector instance

    This provides a singleton pattern for convenience. If you need multiple
    selectors or want to reset the selector, use ModelSelectorFactory directly.

    Args:
        strategy: Strategy to use (only used if creating new instance)
        config_path: Config path (only used if creating new instance)

    Returns:
        BaseModelSelector instance
    """
    global _default_selector

    if _default_selector is None:
        if config_path:
            _default_selector = ModelSelectorFactory.create_from_config(config_path)
        elif strategy:
            _default_selector = ModelSelectorFactory.create(strategy)
        else:
            # Try env vars, then config file, then default
            env_strategy = os.getenv('MODEL_SELECTION_STRATEGY')
            if env_strategy:
                _default_selector = ModelSelectorFactory.create_from_env()
            else:
                _default_selector = ModelSelectorFactory.create_from_config()

    return _default_selector


def reset_model_selector():
    """Reset the default model selector (useful for testing)"""
    global _default_selector
    _default_selector = None
