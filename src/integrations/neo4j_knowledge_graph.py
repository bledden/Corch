"""
Neo4j Knowledge Graph for Facilitair
Stores successful task patterns for RAG retrieval and quality improvement

Adapted from CodeSwarm's Neo4j client
"""

import os
from typing import Dict, Any, List, Optional
from neo4j import AsyncGraphDatabase
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class FacilitairKnowledgeGraph:
    """
    Neo4j knowledge graph for storing and retrieving successful task patterns

    Features:
    - Store tasks with quality score > threshold (default 0.7)
    - RAG retrieval of similar past tasks
    - Track agent performance per task type
    - Documentation effectiveness tracking
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        quality_threshold: float = 0.7
    ):
        """
        Initialize knowledge graph client

        Args:
            uri: Neo4j URI (e.g., neo4j+s://xxxxx.databases.neo4j.io)
            user: Neo4j username
            password: Neo4j password
            quality_threshold: Minimum quality score to store (0.0-1.0)
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.quality_threshold = quality_threshold

        # Neo4j is optional - gracefully degrade if not configured
        self.enabled = bool(self.uri and self.password)

        if not self.enabled:
            logger.warning(
                "[KNOWLEDGE_GRAPH] Neo4j not configured - knowledge graph disabled. "
                "Set NEO4J_URI and NEO4J_PASSWORD to enable pattern learning."
            )
            self.driver = None
            return

        # Validate URI format
        if self.uri == "bolt://localhost:7687":
            logger.warning(
                "[KNOWLEDGE_GRAPH] Using local Neo4j. "
                "For production, use Neo4j Aura (neo4j+s://...)"
            )

        # Create async driver
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info(f"[KNOWLEDGE_GRAPH] Connected to Neo4j: {self.uri}")
        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Failed to connect: {e}")
            self.enabled = False
            self.driver = None

    async def close(self):
        """Close the driver"""
        if self.driver:
            await self.driver.close()

    async def __aenter__(self):
        """Context manager entry"""
        if self.enabled:
            await self.verify_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()

    async def verify_connection(self) -> bool:
        """Verify connection to Neo4j"""
        if not self.enabled:
            return False

        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 AS test")
                record = await result.single()
                if record and record["test"] == 1:
                    logger.info("[KNOWLEDGE_GRAPH] Connection verified")
                    return True
        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Connection failed: {e}")
            self.enabled = False
            return False

        return False

    def _task_hash(self, task: str) -> str:
        """Create a hash for task similarity matching"""
        # Normalize task for consistent hashing
        normalized = task.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    async def store_successful_task(
        self,
        task: str,
        output: str,
        quality_score: float,
        agents_used: List[str],
        metrics: Dict[str, Any],
        duration_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store successful task pattern

        Args:
            task: Original task description
            output: Final generated output
            quality_score: Overall quality score (0.0-1.0)
            agents_used: List of agents that executed
            metrics: Performance metrics (tokens, cost, etc.)
            duration_seconds: Total execution time
            metadata: Optional additional metadata

        Returns:
            Pattern ID if stored, None if skipped
        """
        if not self.enabled:
            return None

        if quality_score < self.quality_threshold:
            logger.debug(
                f"[KNOWLEDGE_GRAPH] Quality {quality_score:.2f} < {self.quality_threshold}, skipping storage"
            )
            return None

        pattern_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._task_hash(task)}"

        try:
            async with self.driver.session() as session:
                # Create task pattern node
                query = """
                CREATE (t:TaskPattern {
                    id: $pattern_id,
                    task: $task,
                    output: $output,
                    quality_score: $quality_score,
                    timestamp: datetime(),
                    agents_used: $agents_used,
                    agent_count: $agent_count,
                    duration_seconds: $duration_seconds,
                    total_tokens: $total_tokens,
                    total_cost_usd: $total_cost_usd,
                    task_hash: $task_hash
                })
                RETURN t.id AS id
                """

                await session.run(
                    query,
                    pattern_id=pattern_id,
                    task=task[:1000],  # Limit length
                    output=output[:5000],  # Limit output length
                    quality_score=quality_score,
                    agents_used=agents_used,
                    agent_count=len(agents_used),
                    duration_seconds=duration_seconds,
                    total_tokens=metrics.get("total_tokens", 0),
                    total_cost_usd=metrics.get("total_cost_usd", 0.0),
                    task_hash=self._task_hash(task)
                )

                # Create agent performance nodes
                for agent in agents_used:
                    agent_query = """
                    MATCH (t:TaskPattern {id: $pattern_id})
                    CREATE (a:AgentExecution {
                        agent_name: $agent_name,
                        task_id: $pattern_id
                    })
                    CREATE (t)-[:EXECUTED_BY]->(a)
                    """

                    await session.run(
                        agent_query,
                        pattern_id=pattern_id,
                        agent_name=agent
                    )

                logger.info(
                    f"[KNOWLEDGE_GRAPH] Stored pattern {pattern_id} "
                    f"(quality: {quality_score:.2f}, agents: {len(agents_used)})"
                )
                return pattern_id

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Failed to store pattern: {e}")
            return None

    async def retrieve_similar_tasks(
        self,
        task: str,
        limit: int = 5,
        min_quality: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar successful tasks for RAG context

        Args:
            task: Current task to find similar patterns for
            limit: Maximum number of patterns to retrieve
            min_quality: Minimum quality score (defaults to threshold)

        Returns:
            List of similar task patterns with outputs and metadata
        """
        if not self.enabled:
            return []

        min_quality = min_quality or self.quality_threshold
        task_hash = self._task_hash(task)

        try:
            async with self.driver.session() as session:
                # First try exact hash match (very similar tasks)
                query = """
                MATCH (t:TaskPattern)
                WHERE t.task_hash = $task_hash
                  AND t.quality_score >= $min_quality
                RETURN t.id AS id,
                       t.task AS task,
                       t.output AS output,
                       t.quality_score AS quality_score,
                       t.agents_used AS agents_used,
                       t.duration_seconds AS duration_seconds,
                       t.timestamp AS timestamp
                ORDER BY t.quality_score DESC, t.timestamp DESC
                LIMIT $limit
                """

                result = await session.run(
                    query,
                    task_hash=task_hash,
                    min_quality=min_quality,
                    limit=limit
                )

                patterns = []
                async for record in result:
                    patterns.append({
                        "id": record["id"],
                        "task": record["task"],
                        "output": record["output"],
                        "quality_score": record["quality_score"],
                        "agents_used": record["agents_used"],
                        "duration_seconds": record["duration_seconds"],
                        "timestamp": str(record["timestamp"]),
                        "similarity": "exact"
                    })

                # If no exact matches, get high-quality patterns as general context
                if not patterns:
                    general_query = """
                    MATCH (t:TaskPattern)
                    WHERE t.quality_score >= $min_quality
                    RETURN t.id AS id,
                           t.task AS task,
                           t.output AS output,
                           t.quality_score AS quality_score,
                           t.agents_used AS agents_used,
                           t.duration_seconds AS duration_seconds,
                           t.timestamp AS timestamp
                    ORDER BY t.quality_score DESC, t.timestamp DESC
                    LIMIT $limit
                    """

                    result = await session.run(
                        general_query,
                        min_quality=min_quality,
                        limit=min(limit, 3)  # Fewer general examples
                    )

                    async for record in result:
                        patterns.append({
                            "id": record["id"],
                            "task": record["task"],
                            "output": record["output"],
                            "quality_score": record["quality_score"],
                            "agents_used": record["agents_used"],
                            "duration_seconds": record["duration_seconds"],
                            "timestamp": str(record["timestamp"]),
                            "similarity": "general"
                        })

                if patterns:
                    logger.info(
                        f"[KNOWLEDGE_GRAPH] Retrieved {len(patterns)} similar patterns "
                        f"for RAG context"
                    )

                return patterns

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Failed to retrieve patterns: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics

        Returns:
            Statistics about stored patterns
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            async with self.driver.session() as session:
                query = """
                MATCH (t:TaskPattern)
                RETURN count(t) AS total_patterns,
                       avg(t.quality_score) AS avg_quality,
                       max(t.quality_score) AS max_quality,
                       min(t.quality_score) AS min_quality
                """

                result = await session.run(query)
                record = await result.single()

                if record:
                    return {
                        "enabled": True,
                        "total_patterns": record["total_patterns"],
                        "avg_quality": round(record["avg_quality"] or 0, 3),
                        "max_quality": round(record["max_quality"] or 0, 3),
                        "min_quality": round(record["min_quality"] or 0, 3)
                    }

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Failed to get statistics: {e}")

        return {"enabled": True, "error": "Failed to get statistics"}

    async def clear_low_quality_patterns(
        self,
        quality_threshold: float = 0.6,
        keep_recent: int = 100
    ) -> int:
        """
        Clean up low-quality or old patterns to maintain graph quality

        Args:
            quality_threshold: Remove patterns below this score
            keep_recent: Always keep this many most recent patterns

        Returns:
            Number of patterns deleted
        """
        if not self.enabled:
            return 0

        try:
            async with self.driver.session() as session:
                query = """
                MATCH (t:TaskPattern)
                WHERE t.quality_score < $quality_threshold
                WITH t
                ORDER BY t.timestamp DESC
                SKIP $keep_recent
                DETACH DELETE t
                RETURN count(t) AS deleted
                """

                result = await session.run(
                    query,
                    quality_threshold=quality_threshold,
                    keep_recent=keep_recent
                )

                record = await result.single()
                deleted = record["deleted"] if record else 0

                if deleted > 0:
                    logger.info(
                        f"[KNOWLEDGE_GRAPH] Cleaned up {deleted} low-quality patterns"
                    )

                return deleted

        except Exception as e:
            logger.error(f"[KNOWLEDGE_GRAPH] Failed to clean up patterns: {e}")
            return 0


# Global knowledge graph instance (singleton)
_knowledge_graph: Optional[FacilitairKnowledgeGraph] = None


def get_knowledge_graph() -> FacilitairKnowledgeGraph:
    """Get global knowledge graph instance (singleton pattern)"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = FacilitairKnowledgeGraph()
    return _knowledge_graph
