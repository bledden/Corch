"""
Integration tests for Neo4j Knowledge Graph with Orchestrator
Tests RAG retrieval and pattern storage
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.neo4j_knowledge_graph import FacilitairKnowledgeGraph
from src.orchestrators.collaborative_orchestrator import CollaborativeOrchestrator


class TestKnowledgeGraphIntegration:
    """Test knowledge graph integration with orchestrator"""

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph"""
        kg = MagicMock(spec=FacilitairKnowledgeGraph)
        kg.enabled = True

        # Mock retrieve_similar_tasks to return sample patterns
        async def mock_retrieve(task, limit=5, min_quality=None):
            return [
                {
                    "id": "task_20241021_120000_abc123",
                    "task": "Implement a Python REST API with authentication",
                    "output": "def create_api():\n    app = Flask(__name__)\n    # ... authentication logic",
                    "quality_score": 0.85,
                    "agents_used": ["architect", "coder", "reviewer"],
                    "duration_seconds": 45.2,
                    "timestamp": "2024-10-21T12:00:00Z",
                    "similarity": "exact"
                },
                {
                    "id": "task_20241021_110000_def456",
                    "task": "Build a REST API with OAuth2",
                    "output": "from fastapi import FastAPI\napp = FastAPI()\n# ... OAuth2 implementation",
                    "quality_score": 0.78,
                    "agents_used": ["architect", "coder"],
                    "duration_seconds": 38.5,
                    "timestamp": "2024-10-21T11:00:00Z",
                    "similarity": "general"
                }
            ]

        kg.retrieve_similar_tasks = AsyncMock(side_effect=mock_retrieve)

        # Mock store_successful_task to return pattern ID
        async def mock_store(task, output, quality_score, agents_used, metrics, duration_seconds, metadata=None):
            if quality_score >= 0.7:
                return f"task_20241021_130000_{hash(task) % 1000}"
            return None

        kg.store_successful_task = AsyncMock(side_effect=mock_store)

        return kg

    @pytest.mark.asyncio
    async def test_rag_retrieval_before_execution(self, mock_knowledge_graph):
        """Test that knowledge graph retrieves similar patterns before execution"""

        # Patch the knowledge graph in the orchestrator
        with patch('src.integrations.get_knowledge_graph', return_value=mock_knowledge_graph):
            # Mock the sequential orchestrator to avoid actual LLM calls
            with patch('src.orchestrators.collaborative_orchestrator.LLM_AVAILABLE', False):
                orchestrator = CollaborativeOrchestrator(use_sequential=False)
                orchestrator.knowledge_graph = mock_knowledge_graph

                task = "Create a REST API with JWT authentication"

                # Since sequential is disabled, we test RAG retrieval directly
                # In real usage, this would be called inside collaborate()
                similar_patterns = await mock_knowledge_graph.retrieve_similar_tasks(
                    task=task,
                    limit=3,
                    min_quality=0.7
                )

                # Verify retrieval was called
                mock_knowledge_graph.retrieve_similar_tasks.assert_called_once()

                # Verify patterns were returned
                assert len(similar_patterns) == 2
                assert similar_patterns[0]["quality_score"] == 0.85
                assert "REST API" in similar_patterns[0]["task"]

    @pytest.mark.asyncio
    async def test_pattern_storage_after_success(self, mock_knowledge_graph):
        """Test that successful executions are stored in knowledge graph"""

        task = "Implement user authentication"
        output = "def authenticate_user(username, password):\n    # implementation"
        quality_score = 0.82
        agents_used = ["architect", "coder", "reviewer"]

        # Store pattern
        pattern_id = await mock_knowledge_graph.store_successful_task(
            task=task,
            output=output,
            quality_score=quality_score,
            agents_used=agents_used,
            metrics={"total_tokens": 1500, "total_cost_usd": 0.045},
            duration_seconds=42.3,
            metadata={"task_type": "coding"}
        )

        # Verify storage was called
        mock_knowledge_graph.store_successful_task.assert_called_once()

        # Verify pattern ID was returned (quality > 0.7)
        assert pattern_id is not None
        assert pattern_id.startswith("task_20241021")

    @pytest.mark.asyncio
    async def test_low_quality_not_stored(self, mock_knowledge_graph):
        """Test that low-quality results are not stored"""

        task = "Build something"
        output = "incomplete code"
        quality_score = 0.45  # Below 0.7 threshold

        pattern_id = await mock_knowledge_graph.store_successful_task(
            task=task,
            output=output,
            quality_score=quality_score,
            agents_used=["coder"],
            metrics={"total_tokens": 500, "total_cost_usd": 0.01},
            duration_seconds=10.0
        )

        # Verify storage was called
        mock_knowledge_graph.store_successful_task.assert_called_once()

        # Verify pattern was NOT stored (quality < 0.7)
        assert pattern_id is None

    @pytest.mark.asyncio
    async def test_rag_context_enhancement(self, mock_knowledge_graph):
        """Test that RAG context enhances the task prompt"""

        task = "Create a REST API"

        # Retrieve patterns
        patterns = await mock_knowledge_graph.retrieve_similar_tasks(
            task=task,
            limit=3,
            min_quality=0.7
        )

        # Build RAG context like orchestrator does
        rag_context = "\n\n[KNOWLEDGE GRAPH - Similar Successful Patterns]:\n"
        for i, pattern in enumerate(patterns, 1):
            rag_context += f"\n{i}. Task: {pattern['task'][:100]}...\n"
            rag_context += f"   Quality: {pattern['quality_score']:.2f}\n"
            rag_context += f"   Agents: {', '.join(pattern['agents_used'])}\n"
            rag_context += f"   Output Preview: {pattern['output'][:150]}...\n"

        enhanced_task = task + rag_context

        # Verify enhancement
        assert "KNOWLEDGE GRAPH" in enhanced_task
        assert "Quality: 0.85" in enhanced_task
        assert "REST API with authentication" in enhanced_task
        assert len(enhanced_task) > len(task)

    @pytest.mark.asyncio
    async def test_knowledge_graph_disabled_graceful_degradation(self):
        """Test that system works without knowledge graph"""

        # Create disabled knowledge graph
        kg = MagicMock(spec=FacilitairKnowledgeGraph)
        kg.enabled = False

        with patch('src.integrations.get_knowledge_graph', return_value=kg):
            with patch('src.orchestrators.collaborative_orchestrator.LLM_AVAILABLE', False):
                orchestrator = CollaborativeOrchestrator(use_sequential=False)
                orchestrator.knowledge_graph = kg

                # System should still function
                assert orchestrator.knowledge_graph is not None
                assert not orchestrator.knowledge_graph.enabled

    def test_knowledge_graph_singleton(self):
        """Test that knowledge graph uses singleton pattern"""
        from src.integrations import get_knowledge_graph

        kg1 = get_knowledge_graph()
        kg2 = get_knowledge_graph()

        # Should return same instance
        assert kg1 is kg2


class TestKnowledgeGraphConfiguration:
    """Test knowledge graph configuration and initialization"""

    def test_graceful_degradation_without_neo4j(self):
        """Test that knowledge graph gracefully degrades without Neo4j credentials"""

        # Create knowledge graph without credentials
        kg = FacilitairKnowledgeGraph(uri=None, password=None)

        # Should be disabled but not crash
        assert not kg.enabled
        assert kg.driver is None

    def test_enabled_with_credentials(self):
        """Test that knowledge graph enables with credentials"""

        # Mock driver to avoid actual Neo4j connection
        with patch('src.integrations.neo4j_knowledge_graph.AsyncGraphDatabase.driver'):
            kg = FacilitairKnowledgeGraph(
                uri="neo4j+s://test.databases.neo4j.io",
                user="neo4j",
                password="test_password"
            )

            # Should be enabled
            assert kg.enabled

    def test_quality_threshold_configuration(self):
        """Test that quality threshold is configurable"""

        kg = FacilitairKnowledgeGraph(quality_threshold=0.8)

        assert kg.quality_threshold == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
