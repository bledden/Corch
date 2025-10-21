"""
WeaveHacks Self-Improving Orchestrator
A production-ready learning orchestrator for complex multi-step workflows
Built for the Wandb Weave Hackathon

Key Features:
- Self-improving strategy selection through reinforcement learning
- Full Weave observability integration
- Adaptive execution based on learned patterns
- Real-time learning visualization
- Production-grade error handling and retry logic
"""

__version__ = "1.0.0"
__author__ = "Facilitair Team"

from .core.config import OrchestratorConfig
from .core.weave_integration import WeaveOrchestrator
from .engine.adaptive_engine import AdaptiveExecutionEngine
from .learning.strategy_learner import StrategyLearner
from .learning.feedback_loop import LearningFeedbackLoop

__all__ = [
    "OrchestratorConfig",
    "WeaveOrchestrator",
    "AdaptiveExecutionEngine",
    "StrategyLearner",
    "LearningFeedbackLoop",
]
