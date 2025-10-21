"""
Base Model Selector - Unified Interface for Model Selection Strategies

This module provides an abstract base class that all model selectors must implement,
enabling swappable selection strategies via configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SelectionStrategy(str, Enum):
    """Available model selection strategies"""
    USER_PREFERENCE = "user_preference"  # Strategy-based (QUALITY_FIRST, COST_FIRST, etc.)
    THOMPSON_SAMPLING = "thompson_sampling"  # Reinforcement learning
    GRANULAR_MATCHING = "granular_matching"  # Language/framework/task specific


@dataclass
class SelectionContext:
    """
    Unified context for model selection decisions

    This combines elements from all selection strategies to provide
    a comprehensive context for decision-making.
    """
    # Core identification
    agent_type: str  # architect, coder, reviewer, refiner, documenter
    task_type: str  # coding, review, architecture, debugging, etc.

    # Task details (for granular matching)
    primary_language: Optional[str] = None  # Python, JavaScript, Rust, etc.
    frameworks: list = None  # React, FastAPI, Django, etc.
    complexity: str = "medium"  # simple, medium, complex

    # Budget/performance constraints (for strategy selection)
    task_complexity: float = 0.5  # 0.0 to 1.0
    remaining_budget: float = 10.0  # in USD
    sensitive_data: bool = False
    required_latency: Optional[float] = None  # in seconds
    user_waiting: bool = False

    # Learning context (for Thompson Sampling)
    generation: int = 0  # Task iteration number

    # Manual overrides
    force_model: Optional[str] = None  # Force specific model ID

    def __post_init__(self):
        if self.frameworks is None:
            self.frameworks = []


@dataclass
class SelectionResult:
    """
    Standardized result from model selection

    All selectors return this format for consistency and observability.
    """
    model_id: str  # The selected model ID (e.g., "anthropic/claude-3.5-sonnet")
    strategy_used: str  # Which strategy was used
    confidence: float  # 0.0 to 1.0 - how confident the selector is
    estimated_cost: float  # Estimated cost in USD
    estimated_quality: float  # Estimated quality score 0.0 to 1.0
    reasoning: str  # Human-readable explanation of why this model was chosen
    metadata: Dict[str, Any] = None  # Strategy-specific additional data

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModelSelector(ABC):
    """
    Abstract base class for all model selection strategies

    All model selectors must implement this interface to be swappable
    via configuration. This enables experimentation with different
    selection strategies without changing orchestrator code.
    """

    @abstractmethod
    def select_model(self, context: SelectionContext) -> SelectionResult:
        """
        Select the best model for the given context

        Args:
            context: SelectionContext with all relevant information

        Returns:
            SelectionResult with the chosen model and metadata
        """
        pass

    @abstractmethod
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
        Update the selector's knowledge based on actual performance

        This allows selectors to learn and improve over time.

        Args:
            model_id: The model that was used
            context: The context it was used in
            success: Whether the task succeeded
            quality_score: Quality score (0.0 to 1.0)
            latency_ms: Actual latency in milliseconds
            cost_usd: Actual cost in USD
        """
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """
        Get list of models this selector can choose from

        Returns:
            List of model IDs
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about this selector's performance

        Returns:
            Dictionary with selector-specific stats
        """
        pass

    def set_model_override(self, agent_type: str, model_id: str):
        """
        Set a manual model override for a specific agent type

        Optional to implement - some selectors may not support overrides.

        Args:
            agent_type: The agent type to override
            model_id: The model ID to force
        """
        pass

    def clear_overrides(self):
        """
        Clear all manual model overrides

        Optional to implement - some selectors may not support overrides.
        """
        pass
