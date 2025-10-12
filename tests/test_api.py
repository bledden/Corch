"""
Tests for Facilitair REST API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

# Import API
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from api import app, get_orchestrator


class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Facilitair API"
        assert "docs" in data

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "api_keys_valid" in data
        assert "orchestrator_ready" in data

    def test_list_agents(self):
        """Test list agents endpoint"""
        response = self.client.get("/api/v1/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) > 0

        # Check agent structure
        agent = data["agents"][0]
        assert "id" in agent
        assert "name" in agent
        assert "description" in agent

    def test_list_tasks_empty(self):
        """Test list tasks endpoint when no tasks"""
        response = self.client.get("/api/v1/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert "total" in data

    def test_get_task_not_found(self):
        """Test getting non-existent task"""
        response = self.client.get("/api/v1/tasks/nonexistent")
        assert response.status_code == 404

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_tasks" in data
        assert "successful_tasks" in data
        assert "failed_tasks" in data


class TestCollaborateEndpoint:
    """Test suite for collaboration endpoint"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    @pytest.mark.asyncio
    async def test_collaborate_valid_request(self):
        """Test collaboration with valid request"""
        # Mock the orchestrator
        mock_result = Mock()
        mock_result.success = True
        mock_result.agents_used = ["architect", "coder", "reviewer", "documenter"]
        mock_result.final_output = "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
        mock_result.metrics = {"quality": 0.9, "efficiency": 0.85}
        mock_result.consensus_method = "sequential"

        with patch('api.get_orchestrator') as mock_get_orch:
            mock_orch = Mock()
            mock_orch.collaborate = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orch

            response = self.client.post("/api/v1/collaborate", json={
                "task": "Write a factorial function",
                "use_sequential": True,
                "max_iterations": 3,
                "temperature": 0.2
            })

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "task_id" in data
            assert len(data["agents_used"]) > 0
            assert "output" in data

    def test_collaborate_invalid_request(self):
        """Test collaboration with invalid request"""
        response = self.client.post("/api/v1/collaborate", json={
            "task": "",  # Empty task should fail
            "use_sequential": True
        })
        assert response.status_code == 422  # Validation error

    def test_collaborate_missing_required_field(self):
        """Test collaboration with missing required field"""
        response = self.client.post("/api/v1/collaborate", json={
            "use_sequential": True
            # Missing "task" field
        })
        assert response.status_code == 422  # Validation error


class TestEvaluationEndpoint:
    """Test suite for evaluation endpoint"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_run_evaluation(self):
        """Test running evaluation"""
        with patch('api.BackgroundTasks') as mock_bg:
            response = self.client.post("/api/v1/evaluate", json={
                "num_tasks": 10,
                "compare_baseline": True
            })

            assert response.status_code == 200
            data = response.json()
            assert "eval_id" in data
            assert data["status"] == "started"

    def test_run_evaluation_invalid_num_tasks(self):
        """Test evaluation with invalid number of tasks"""
        response = self.client.post("/api/v1/evaluate", json={
            "num_tasks": 0,  # Should be at least 1
            "compare_baseline": True
        })
        assert response.status_code == 422  # Validation error


class TestAPIIntegration:
    """Integration tests for API"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    @pytest.mark.integration
    def test_full_workflow(self):
        """Test full workflow: health -> collaborate -> get task"""
        # 1. Check health
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200

        # 2. List agents
        response = self.client.get("/api/v1/agents")
        assert response.status_code == 200

        # 3. Collaborate (mocked)
        with patch('api.get_orchestrator') as mock_get_orch:
            mock_result = Mock()
            mock_result.success = True
            mock_result.agents_used = ["architect", "coder"]
            mock_result.final_output = "test output"
            mock_result.metrics = {"quality": 0.9}
            mock_result.consensus_method = "sequential"

            mock_orch = Mock()
            mock_orch.collaborate = AsyncMock(return_value=mock_result)
            mock_get_orch.return_value = mock_orch

            response = self.client.post("/api/v1/collaborate", json={
                "task": "Test task"
            })
            assert response.status_code == 200
            task_id = response.json()["task_id"]

            # 4. Get task by ID
            response = self.client.get(f"/api/v1/tasks/{task_id}")
            assert response.status_code == 200
            assert response.json()["task_id"] == task_id


class TestAPIDocumentation:
    """Test API documentation endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_openapi_schema(self):
        """Test OpenAPI schema is available"""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Facilitair API"

    def test_docs_available(self):
        """Test Swagger docs are available"""
        response = self.client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self):
        """Test ReDoc is available"""
        response = self.client.get("/redoc")
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
