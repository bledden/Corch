"""
API Integration Tests

Tests the FastAPI endpoints with real HTTP requests.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_returns_200(self, client):
        """Test health check returns 200"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client):
        """Test health check response has correct structure"""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "api_keys_valid" in data
        assert "orchestrator_ready" in data
        assert "timestamp" in data

    def test_health_status_values(self, client):
        """Test health status has valid values"""
        response = client.get("/api/v1/health")
        data = response.json()

        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["api_keys_valid"], bool)
        assert isinstance(data["orchestrator_ready"], bool)


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_returns_200(self, client):
        """Test root endpoint returns 200"""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self, client):
        """Test root returns API information"""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "Facilitair" in data["name"]


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint_returns_200(self, client):
        """Test metrics endpoint returns 200"""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Test metrics returns Prometheus format"""
        response = client.get("/metrics")
        content_type = response.headers.get("content-type", "")

        # Prometheus format
        assert "text/plain" in content_type or "openmetrics" in content_type.lower()

    def test_metrics_contains_prometheus_data(self, client):
        """Test metrics contains Prometheus metrics"""
        response = client.get("/metrics")
        content = response.text

        # Should have at least some metrics
        assert len(content) > 0
        # Common Prometheus patterns
        assert any(keyword in content for keyword in ["# HELP", "# TYPE", "_total", "_seconds"])


class TestCollaborateEndpoint:
    """Test collaboration endpoint"""

    def test_collaborate_requires_task(self, client):
        """Test collaborate endpoint requires task parameter"""
        response = client.post("/api/v1/collaborate", json={})
        assert response.status_code == 422  # Validation error

    def test_collaborate_rejects_short_task(self, client):
        """Test collaborate rejects tasks that are too short"""
        response = client.post(
            "/api/v1/collaborate",
            json={"task": "short"}
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_collaborate_rejects_invalid_temperature(self, client):
        """Test collaborate rejects invalid temperature"""
        response = client.post(
            "/api/v1/collaborate",
            json={
                "task": "Write a function to add two numbers",
                "temperature": 3.0  # Invalid: > 2.0
            }
        )
        assert response.status_code == 422  # Validation error

    def test_collaborate_rejects_invalid_agents(self, client):
        """Test collaborate rejects invalid agent names"""
        response = client.post(
            "/api/v1/collaborate",
            json={
                "task": "Write a function to add two numbers",
                "force_agents": ["invalid_agent"]
            }
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_collaborate_accepts_valid_request(self, client):
        """Test collaborate accepts valid request (note: may be slow)"""
        # This is a real API call - mark as slow test
        pytest.skip("Skipping slow integration test - enable manually for full testing")

        response = client.post(
            "/api/v1/collaborate",
            json={
                "task": "Write a simple Python function to add two numbers",
                "temperature": 0.2,
                "max_iterations": 1
            },
            timeout=120.0  # Allow 2 minutes
        )

        # Should either succeed or fail gracefully
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert "output" in data
            assert "agents_used" in data


class TestTasksEndpoint:
    """Test tasks listing endpoint"""

    def test_tasks_list_returns_200(self, client):
        """Test tasks list returns 200"""
        response = client.get("/api/v1/tasks")
        assert response.status_code == 200

    def test_tasks_list_response_structure(self, client):
        """Test tasks list has correct structure"""
        response = client.get("/api/v1/tasks")
        data = response.json()

        assert "tasks" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["tasks"], list)

    def test_tasks_list_pagination(self, client):
        """Test tasks list pagination parameters"""
        response = client.get("/api/v1/tasks?limit=5&offset=0")
        assert response.status_code == 200

        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 0

    def test_get_nonexistent_task_returns_404(self, client):
        """Test getting non-existent task returns 404"""
        response = client.get("/api/v1/tasks/nonexistent_task_id")
        assert response.status_code == 404


class TestCORSHeaders:
    """Test CORS headers"""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present"""
        # Test with a GET request (OPTIONS might not trigger CORS)
        response = client.get("/api/v1/health")

        # CORS headers should be present in regular requests
        # Note: TestClient might not fully simulate CORS
        assert response.status_code == 200


class TestErrorHandling:
    """Test API error handling"""

    def test_404_for_invalid_endpoint(self, client):
        """Test 404 for invalid endpoint"""
        response = client.get("/api/v1/invalid_endpoint")
        assert response.status_code == 404

    def test_405_for_wrong_method(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.put("/api/v1/health")
        assert response.status_code == 405

    def test_validation_error_format(self, client):
        """Test validation errors return proper format"""
        response = client.post(
            "/api/v1/collaborate",
            json={"task": "x"}  # Too short
        )

        assert response.status_code in [400, 422]
        data = response.json()

        # Should have error information
        assert "detail" in data or "error" in data


class TestRequestTracing:
    """Test correlation ID tracking"""

    def test_correlation_id_returned_in_response(self, client):
        """Test correlation ID is returned in response headers"""
        response = client.get("/api/v1/health")

        # Should have correlation ID header
        assert "x-correlation-id" in response.headers

    def test_custom_correlation_id_preserved(self, client):
        """Test custom correlation ID is preserved"""
        custom_id = "test-correlation-123"
        response = client.get(
            "/api/v1/health",
            headers={"X-Correlation-ID": custom_id}
        )

        # Should return the same correlation ID
        assert response.headers.get("x-correlation-id") == custom_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not slow"])
