"""
Logging middleware for FastAPI

Adds:
- Correlation ID to all requests
- Request/response logging with timing
- Automatic error logging
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from src.logging import (
    set_correlation_id,
    clear_correlation_id,
    log_request,
    log_response,
    log_error_with_context
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation IDs and log all requests/responses

    Features:
    - Generates or extracts correlation ID from X-Correlation-ID header
    - Logs all incoming requests with method, path, and headers
    - Logs all responses with status code and duration
    - Logs errors with full context
    - Adds correlation ID to response headers
    """

    def __init__(self, app: ASGIApp, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response"""

        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        # Start timer
        start_time = time.time()

        # Log incoming request
        log_request(
            self.logger,
            method=request.method,
            path=str(request.url.path),
            correlation_id=correlation_id,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            log_response(
                self.logger,
                status_code=response.status_code,
                duration_ms=duration_ms,
                path=str(request.url.path)
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            log_error_with_context(
                self.logger,
                error=e,
                context={
                    "method": request.method,
                    "path": str(request.url.path),
                    "duration_ms": round(duration_ms, 2)
                },
                message=f"Request failed: {request.method} {request.url.path}"
            )

            raise

        finally:
            # Clean up correlation ID
            clear_correlation_id()
