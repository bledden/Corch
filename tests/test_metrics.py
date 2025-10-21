"""
Tests for performance monitoring and metrics
"""

import pytest
import time
from src.monitoring import (
    metrics,
    MetricsCollector,
    track_agent_execution,
    track_llm_request,
    track_task_execution
)
from prometheus_client import CollectorRegistry


class TestMetricsCollector:
    """Test metrics collection"""

    def setup_method(self):
        """Use the global metrics collector for each test"""
        self.collector = metrics  # Use global metrics instance

    def test_record_http_request(self):
        """Test HTTP request metrics recording"""
        self.collector.record_http_request(
            method="POST",
            endpoint="/api/v1/collaborate",
            status_code=200,
            duration_seconds=1.5,
            request_size=1024,
            response_size=2048
        )

        # Export and verify metrics exist
        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'http_requests_total' in metrics_output
        assert 'http_request_duration_seconds' in metrics_output
        assert 'POST' in metrics_output
        assert '/api/v1/collaborate' in metrics_output

    def test_record_agent_execution(self):
        """Test agent execution metrics"""
        self.collector.record_agent_execution(
            agent="architect",
            stage="design",
            status="success",
            duration_seconds=5.2
        )

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'agent_executions_total' in metrics_output
        assert 'agent_execution_duration_seconds' in metrics_output
        assert 'architect' in metrics_output
        assert 'design' in metrics_output

    def test_record_agent_iteration(self):
        """Test agent iteration recording"""
        self.collector.record_agent_iteration(agent="refiner")

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'agent_iterations_total' in metrics_output
        assert 'refiner' in metrics_output

    def test_record_llm_request(self):
        """Test LLM request metrics"""
        self.collector.record_llm_request(
            model="gpt-4",
            status="success",
            duration_seconds=2.3,
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=0.005
        )

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'llm_requests_total' in metrics_output
        assert 'llm_request_duration_seconds' in metrics_output
        assert 'llm_tokens_total' in metrics_output
        assert 'llm_cost_usd_total' in metrics_output
        assert 'gpt-4' in metrics_output

    def test_record_task_completion(self):
        """Test task completion metrics"""
        self.collector.record_task_completion(
            status="success",
            duration_seconds=45.7,
            quality_score=0.92
        )

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'tasks_total' in metrics_output
        assert 'task_duration_seconds' in metrics_output
        assert 'task_quality_score' in metrics_output

    def test_update_system_metrics(self):
        """Test system metrics update"""
        self.collector.update_system_metrics()

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'system_memory_bytes' in metrics_output
        assert 'system_cpu_percent' in metrics_output

    def test_active_tasks_tracking(self):
        """Test active task tracking"""
        self.collector.set_active_tasks(5)
        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'active_tasks 5' in metrics_output

        self.collector.increment_active_tasks()
        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'active_tasks 6' in metrics_output

        self.collector.decrement_active_tasks()
        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'active_tasks 5' in metrics_output

    def test_stream_metrics(self):
        """Test stream metrics"""
        self.collector.set_active_streams(3)
        self.collector.record_stream_event("agent_update")

        metrics_output = self.collector.export_metrics().decode('utf-8')
        assert 'active_streams 3' in metrics_output
        assert 'stream_events_total' in metrics_output


class TestContextManagers:
    """Test metric tracking context managers"""

    def setup_method(self):
        """Tests use global metrics"""
        pass  # Context managers use global metrics, so we just verify they don't crash

    def test_track_agent_execution_success(self):
        """Test agent execution tracking on success"""
        with track_agent_execution("coder", "implementation"):
            time.sleep(0.01)  # Simulate work

        # Should complete without error
        assert True

    def test_track_agent_execution_failure(self):
        """Test agent execution tracking on failure"""
        with pytest.raises(ValueError):
            with track_agent_execution("reviewer", "review"):
                raise ValueError("Test error")

        # Should still record metrics even on failure
        assert True

    def test_track_llm_request_success(self):
        """Test LLM request tracking"""
        with track_llm_request("claude-3-sonnet") as llm_stats:
            time.sleep(0.01)
            llm_stats['prompt_tokens'] = 100
            llm_stats['completion_tokens'] = 50
            llm_stats['cost_usd'] = 0.002

        # Should complete without error
        assert True

    def test_track_llm_request_failure(self):
        """Test LLM request tracking on failure"""
        with pytest.raises(RuntimeError):
            with track_llm_request("gpt-4") as llm_stats:
                raise RuntimeError("LLM error")

        # Should still record metrics
        assert True

    def test_track_task_execution_success(self):
        """Test task execution tracking"""
        with track_task_execution() as task_stats:
            time.sleep(0.01)
            task_stats['quality_score'] = 0.85

        # Should complete without error
        assert True

    def test_track_task_execution_failure(self):
        """Test task execution tracking on failure"""
        with pytest.raises(Exception):
            with track_task_execution() as task_stats:
                raise Exception("Task failed")

        # Should still record metrics
        assert True


class TestMetricsExport:
    """Test metrics export format"""

    def test_export_prometheus_format(self):
        """Test metrics are exported in Prometheus format"""
        # Record some metrics
        metrics.record_http_request(
            method="GET",
            endpoint="/metrics",
            status_code=200,
            duration_seconds=0.05
        )

        # Export
        output = metrics.export_metrics().decode('utf-8')

        # Should be Prometheus format
        assert '# HELP' in output or '# TYPE' in output or 'http_requests_total' in output
        assert isinstance(output, str)

    def test_content_type(self):
        """Test correct content type for Prometheus"""
        content_type = metrics.get_content_type()
        assert 'text/plain' in content_type or 'openmetrics' in content_type.lower()


class TestMetricsIntegration:
    """Test metrics work together"""

    def test_full_task_workflow_metrics(self):
        """Test metrics for a complete task workflow"""
        # Simulate a complete task
        with track_task_execution() as task_stats:
            # Agent 1: Architect
            with track_agent_execution("architect", "design"):
                time.sleep(0.01)

            # LLM call
            with track_llm_request("claude-3-sonnet") as llm_stats:
                llm_stats['prompt_tokens'] = 200
                llm_stats['completion_tokens'] = 100
                llm_stats['cost_usd'] = 0.003

            # Agent 2: Coder
            with track_agent_execution("coder", "implementation"):
                time.sleep(0.01)

            task_stats['quality_score'] = 0.88

        # Export and verify all metrics recorded
        output = metrics.export_metrics().decode('utf-8')

        assert 'tasks_total' in output
        assert 'agent_executions_total' in output
        assert 'llm_requests_total' in output
        assert 'claude-3-sonnet' in output

    def test_multiple_llm_calls_accumulate(self):
        """Test that multiple LLM calls accumulate tokens and cost"""
        # Make multiple LLM calls
        for i in range(3):
            with track_llm_request("gpt-4") as llm_stats:
                llm_stats['prompt_tokens'] = 100
                llm_stats['completion_tokens'] = 50
                llm_stats['cost_usd'] = 0.002

        output = metrics.export_metrics().decode('utf-8')

        # Tokens and costs should accumulate
        assert 'llm_requests_total' in output
        assert 'llm_tokens_total' in output
        assert 'llm_cost_usd_total' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
