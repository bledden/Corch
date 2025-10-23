"""
Integration tests for GitHub deployment functionality
Tests the GitHub client and API deployment endpoint
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.github_client import GitHubClient
from fastapi.testclient import TestClient


class TestGitHubClient:
    """Test GitHub client functionality"""

    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess for gh CLI commands"""
        with patch('src.integrations.github_client.subprocess') as mock:
            yield mock

    def test_github_client_checks_authentication(self, mock_subprocess):
        """Test that GitHubClient checks for gh CLI and authentication"""

        # Mock successful gh CLI check
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="github.com\n  ✓ Logged in",
            stderr=""
        )

        client = GitHubClient()

        # Should call gh --version and gh auth status
        assert mock_subprocess.run.call_count >= 2
        assert client.gh_available is True
        assert client.is_authenticated() is True

    def test_github_client_handles_missing_gh_cli(self, mock_subprocess):
        """Test graceful handling when gh CLI not installed"""

        # Mock FileNotFoundError (gh not in PATH)
        mock_subprocess.run.side_effect = FileNotFoundError("gh not found")

        client = GitHubClient()

        assert client.gh_available is False
        assert client.is_authenticated() is False

    def test_github_client_handles_not_authenticated(self, mock_subprocess):
        """Test detection of unauthenticated state"""

        # Mock gh --version success but gh auth status failure
        def mock_run(*args, **kwargs):
            command = args[0]
            if "auth" in command:
                return MagicMock(returncode=1, stdout="", stderr="not logged in")
            return MagicMock(returncode=0, stdout="gh version 2.0.0", stderr="")

        mock_subprocess.run.side_effect = mock_run

        client = GitHubClient()

        assert client.gh_available is False  # Considered unavailable if not authenticated

    @pytest.mark.asyncio
    async def test_deploy_code_creates_new_repo(self, mock_subprocess):
        """Test deploying code to new repository"""

        # Mock successful gh CLI and git commands
        mock_subprocess.run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/user/test-repo",
            stderr=""
        )

        client = GitHubClient()
        client.gh_available = True
        client.authenticated_user = "testuser"

        result = await client.deploy_code(
            repo_name="test-repo",
            files={"main.py": "print('hello')"},
            description="Test repository",
            task="Create test repo",
            private=False
        )

        assert result["success"] is True
        assert "github.com" in (result.get("url") or "")
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_deploy_code_fails_without_authentication(self):
        """Test that deployment fails without authentication"""

        client = GitHubClient()
        client.gh_available = False

        result = await client.deploy_code(
            repo_name="test-repo",
            files={"main.py": "print('hello')"},
            description="Test repository"
        )

        assert result["success"] is False
        assert "not available" in result["error"].lower()
        assert result["url"] is None

    @pytest.mark.asyncio
    async def test_deploy_code_validates_file_paths(self, mock_subprocess):
        """Test that file path validation works"""

        mock_subprocess.run.return_value = MagicMock(returncode=0)

        client = GitHubClient()
        client.gh_available = True
        client.authenticated_user = "testuser"

        # Valid file paths should work
        files = {
            "src/main.py": "code",
            "README.md": "docs",
            "tests/test_main.py": "tests"
        }

        result = await client.deploy_code(
            repo_name="test-repo",
            files=files,
            description="Test repository"
        )

        # Should not raise validation errors
        assert result is not None

    def test_get_status_returns_correct_info(self):
        """Test that get_status returns authentication info"""

        client = GitHubClient()
        client.gh_available = True
        client.authenticated_user = "testuser"

        status = client.get_status()

        assert status["available"] is True
        assert status["authenticated"] is True
        assert status["username"] == "testuser"


class TestGitHubAPIEndpoint:
    """Test GitHub deployment API endpoint"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from api import app
        return TestClient(app)

    @pytest.fixture
    def mock_github_client(self):
        """Mock GitHub client"""
        with patch('src.integrations.github_client.GitHubClient') as mock:
            instance = MagicMock()
            instance.is_authenticated.return_value = True
            instance.authenticated_user = "testuser"

            async def mock_deploy(**kwargs):
                return {
                    "success": True,
                    "url": "https://github.com/testuser/test-repo",
                    "pr_url": None,
                    "branch": "main",
                    "error": None
                }

            instance.deploy_code = AsyncMock(side_effect=mock_deploy)
            instance.get_status.return_value = {
                "available": True,
                "authenticated": True,
                "username": "testuser"
            }

            mock.return_value = instance
            yield instance

    def test_deploy_status_endpoint(self, client, mock_github_client):
        """Test /api/v1/deploy/status endpoint"""

        with patch('src.integrations.get_github_client', return_value=mock_github_client):
            response = client.get("/api/v1/deploy/status")

            assert response.status_code == 200
            data = response.json()
            assert data["available"] is True
            assert data["authenticated"] is True
            assert data["username"] == "testuser"

    def test_deploy_endpoint_validates_repo_name(self, client):
        """Test that deploy endpoint validates repository name"""

        response = client.post("/api/v1/deploy", json={
            "repo_name": "invalid/repo/name/with/slashes",
            "files": {"main.py": "code"},
            "description": "Test repository"
        })

        assert response.status_code == 422  # Validation error

    def test_deploy_endpoint_validates_files(self, client):
        """Test that deploy endpoint validates files"""

        # Empty files should fail
        response = client.post("/api/v1/deploy", json={
            "repo_name": "test-repo",
            "files": {},
            "description": "Test repository"
        })

        assert response.status_code == 422  # Validation error

    def test_deploy_endpoint_validates_file_paths(self, client):
        """Test that deploy endpoint rejects dangerous file paths"""

        response = client.post("/api/v1/deploy", json={
            "repo_name": "test-repo",
            "files": {
                "../etc/passwd": "malicious"
            },
            "description": "Test repository"
        })

        assert response.status_code == 422  # Validation error

    def test_deploy_endpoint_success(self, client, mock_github_client):
        """Test successful deployment via API"""

        with patch('src.integrations.get_github_client', return_value=mock_github_client):
            response = client.post("/api/v1/deploy", json={
                "repo_name": "test-repo",
                "files": {
                    "main.py": "print('hello')",
                    "README.md": "# Test"
                },
                "description": "Test repository created by Facilitair",
                "task": "Create test repo",
                "private": False
            })

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "github.com" in data["url"]
            assert data["branch"] == "main"
            assert data["error"] is None

    def test_deploy_endpoint_handles_authentication_error(self, client):
        """Test that deploy endpoint handles unauthenticated users"""

        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = False

        with patch('src.integrations.get_github_client', return_value=mock_client):
            response = client.post("/api/v1/deploy", json={
                "repo_name": "test-repo",
                "files": {"main.py": "code"},
                "description": "Test repository"
            })

            assert response.status_code == 401  # Unauthorized
            data = response.json()
            assert "not authenticated" in data["detail"]["error"].lower()


class TestGitHubSingleton:
    """Test GitHub client singleton pattern"""

    def test_get_github_client_returns_singleton(self):
        """Test that get_github_client returns same instance"""
        from src.integrations import get_github_client

        # Reset singleton for test
        import src.integrations.github_client
        src.integrations.github_client._github_client = None

        client1 = get_github_client()
        client2 = get_github_client()

        assert client1 is client2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
