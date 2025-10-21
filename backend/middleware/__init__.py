"""
Middleware for FastAPI application
"""

from .logging_middleware import LoggingMiddleware

__all__ = ['LoggingMiddleware']
