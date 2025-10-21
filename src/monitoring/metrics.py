"""
Performance Monitoring and Metrics Collection

Provides:
- Prometheus metrics integration
- Request/response metrics (latency, status codes, throughput)
- Agent execution metrics (per-stage timing, success/failure rates)
- LLM metrics (tokens, cost, latency per model)
- System metrics (memory, CPU)
- Custom business metrics
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from typing import Dict, Any, Optional
import time
import psutil
import os
from contextlib import contextmanager
from functools import wraps


# Create custom registry (allows testing without singleton issues)
registry = CollectorRegistry()

# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry
)

http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=registry
)

# ============================================================================
# Agent Execution Metrics
# ============================================================================

agent_executions_total = Counter(
    'agent_executions_total',
    'Total agent executions',
    ['agent', 'stage', 'status'],
    registry=registry
)

agent_execution_duration_seconds = Histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration in seconds',
    ['agent', 'stage'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry
)

agent_iterations_total = Counter(
    'agent_iterations_total',
    'Total agent iterations (refinements)',
    ['agent'],
    registry=registry
)

# ============================================================================
# LLM Metrics
# ============================================================================

llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status'],
    registry=registry
)

llm_request_duration_seconds = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=registry
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens consumed',
    ['model', 'type'],  # type: prompt or completion
    registry=registry
)

llm_cost_usd_total = Counter(
    'llm_cost_usd_total',
    'Total LLM cost in USD',
    ['model'],
    registry=registry
)

# ============================================================================
# Task Metrics
# ============================================================================

tasks_total = Counter(
    'tasks_total',
    'Total tasks processed',
    ['status'],  # success, failure
    registry=registry
)

task_duration_seconds = Histogram(
    'task_duration_seconds',
    'Task execution duration in seconds',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0),
    registry=registry
)

task_quality_score = Histogram(
    'task_quality_score',
    'Task quality scores',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# ============================================================================
# System Metrics
# ============================================================================

system_memory_bytes = Gauge(
    'system_memory_bytes',
    'System memory usage in bytes',
    ['type'],  # total, available, used
    registry=registry
)

system_cpu_percent = Gauge(
    'system_cpu_percent',
    'System CPU usage percentage',
    registry=registry
)

active_tasks = Gauge(
    'active_tasks',
    'Number of currently active tasks',
    registry=registry
)

# ============================================================================
# Stream Metrics
# ============================================================================

active_streams = Gauge(
    'active_streams',
    'Number of active SSE streams',
    registry=registry
)

stream_events_total = Counter(
    'stream_events_total',
    'Total stream events sent',
    ['event_type'],
    registry=registry
)


class MetricsCollector:
    """
    Central metrics collection and management

    Provides convenient methods for recording metrics throughout the application.
    """

    def __init__(self, registry=registry):
        self.registry = registry

    # HTTP Metrics

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ):
        """Record HTTP request metrics"""
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()

        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)

        if request_size is not None:
            http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)

        if response_size is not None:
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)

    # Agent Metrics

    def record_agent_execution(
        self,
        agent: str,
        stage: str,
        status: str,  # success, failure, timeout
        duration_seconds: float
    ):
        """Record agent execution metrics"""
        agent_executions_total.labels(
            agent=agent,
            stage=stage,
            status=status
        ).inc()

        agent_execution_duration_seconds.labels(
            agent=agent,
            stage=stage
        ).observe(duration_seconds)

    def record_agent_iteration(self, agent: str):
        """Record agent iteration (refinement)"""
        agent_iterations_total.labels(agent=agent).inc()

    # LLM Metrics

    def record_llm_request(
        self,
        model: str,
        status: str,  # success, failure, timeout
        duration_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        """Record LLM request metrics"""
        llm_requests_total.labels(
            model=model,
            status=status
        ).inc()

        llm_request_duration_seconds.labels(
            model=model
        ).observe(duration_seconds)

        llm_tokens_total.labels(
            model=model,
            type='prompt'
        ).inc(prompt_tokens)

        llm_tokens_total.labels(
            model=model,
            type='completion'
        ).inc(completion_tokens)

        llm_cost_usd_total.labels(
            model=model
        ).inc(cost_usd)

    # Task Metrics

    def record_task_completion(
        self,
        status: str,  # success, failure
        duration_seconds: float,
        quality_score: Optional[float] = None
    ):
        """Record task completion metrics"""
        tasks_total.labels(status=status).inc()
        task_duration_seconds.observe(duration_seconds)

        if quality_score is not None:
            task_quality_score.observe(quality_score)

    # System Metrics

    def update_system_metrics(self):
        """Update system resource metrics"""
        memory = psutil.virtual_memory()
        system_memory_bytes.labels(type='total').set(memory.total)
        system_memory_bytes.labels(type='available').set(memory.available)
        system_memory_bytes.labels(type='used').set(memory.used)

        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_cpu_percent.set(cpu_percent)

    def set_active_tasks(self, count: int):
        """Set number of active tasks"""
        active_tasks.set(count)

    def increment_active_tasks(self):
        """Increment active task counter"""
        active_tasks.inc()

    def decrement_active_tasks(self):
        """Decrement active task counter"""
        active_tasks.dec()

    # Stream Metrics

    def set_active_streams(self, count: int):
        """Set number of active streams"""
        active_streams.set(count)

    def record_stream_event(self, event_type: str):
        """Record stream event"""
        stream_events_total.labels(event_type=event_type).inc()

    # Export

    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics = MetricsCollector()


# ============================================================================
# Convenience Decorators and Context Managers
# ============================================================================

@contextmanager
def track_agent_execution(agent: str, stage: str):
    """
    Context manager to track agent execution

    Usage:
        with track_agent_execution("architect", "design"):
            result = execute_agent()
    """
    start_time = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "failure"
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_agent_execution(agent, stage, status, duration)


@contextmanager
def track_llm_request(model: str):
    """
    Context manager to track LLM requests

    Returns a dict that should be populated with: prompt_tokens, completion_tokens, cost_usd

    Usage:
        with track_llm_request("gpt-4") as llm_stats:
            result = call_llm()
            llm_stats['prompt_tokens'] = result.prompt_tokens
            llm_stats['completion_tokens'] = result.completion_tokens
            llm_stats['cost_usd'] = result.cost
    """
    start_time = time.time()
    status = "success"
    stats = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'cost_usd': 0.0
    }

    try:
        yield stats
    except Exception:
        status = "failure"
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_llm_request(
            model=model,
            status=status,
            duration_seconds=duration,
            prompt_tokens=stats.get('prompt_tokens', 0),
            completion_tokens=stats.get('completion_tokens', 0),
            cost_usd=stats.get('cost_usd', 0.0)
        )


@contextmanager
def track_task_execution():
    """
    Context manager to track task execution

    Returns a dict that should be populated with quality_score

    Usage:
        with track_task_execution() as task_stats:
            result = execute_task()
            task_stats['quality_score'] = result.quality_score
    """
    start_time = time.time()
    status = "success"
    stats = {'quality_score': None}

    metrics.increment_active_tasks()

    try:
        yield stats
    except Exception:
        status = "failure"
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_task_completion(
            status=status,
            duration_seconds=duration,
            quality_score=stats.get('quality_score')
        )
        metrics.decrement_active_tasks()


def track_http_request(func):
    """
    Decorator to track HTTP endpoint metrics

    Usage:
        @track_http_request
        async def my_endpoint(request: Request):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request from args (assumes Request is first arg)
        request = None
        for arg in args:
            if hasattr(arg, 'method') and hasattr(arg, 'url'):
                request = arg
                break

        start_time = time.time()

        try:
            response = await func(*args, **kwargs)

            if request:
                duration = time.time() - start_time
                metrics.record_http_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=getattr(response, 'status_code', 200),
                    duration_seconds=duration
                )

            return response

        except Exception as e:
            if request:
                duration = time.time() - start_time
                metrics.record_http_request(
                    method=request.method,
                    endpoint=str(request.url.path),
                    status_code=500,
                    duration_seconds=duration
                )
            raise

    return wrapper


# Background task to update system metrics periodically
def update_system_metrics_periodically():
    """
    Update system metrics (to be called periodically)

    This should be run in a background task every 10-30 seconds
    """
    metrics.update_system_metrics()
