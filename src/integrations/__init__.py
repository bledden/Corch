"""
Integrations for Facilitair
External services and knowledge systems
"""
from .neo4j_knowledge_graph import (
    FacilitairKnowledgeGraph,
    get_knowledge_graph
)

__all__ = [
    'FacilitairKnowledgeGraph',
    'get_knowledge_graph'
]
