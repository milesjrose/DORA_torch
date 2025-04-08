# nodes/sets/node_representations/__init__.py
"""
Package: node_representations

Provides classes for representing single nodes.
Used for adding nodes to set tensors, and referencing single nodes in any set.

- New_Token: Represents a new token.
- New_Semantic: Represents a new semantic.
- Ref_Node: Represents a reference to a node in any set.
- Ref_Semantic: Represents a reference to a semantic in any set.
"""

from .new_nodes import New_Token, New_Semantic
from .reference_nodes import Ref_Node, Ref_Semantic

__all__ = [
    "New_Token",
    "New_Semantic",
    "Ref_Node",
    "Ref_Semantic"
]

