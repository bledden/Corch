"""
Integrations for Facilitair
External services and knowledge systems
"""
from .neo4j_knowledge_graph import (
    FacilitairKnowledgeGraph,
    get_knowledge_graph
)
from .github_client import (
    GitHubClient,
    get_github_client
)

__all__ = [
    'FacilitairKnowledgeGraph',
    'get_knowledge_graph',
    'GitHubClient',
    'get_github_client'
]
