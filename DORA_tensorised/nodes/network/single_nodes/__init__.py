# nodes/network/single_nodes/__init__.py
"""
Package: single_nodes

Provides classes for representing single nodes.
Used for adding nodes to set tensors, and referencing single nodes in any set.

- Token: Represents a token.
- Semantic: Represents a semantic.
- Ref_Token: Reference to a token in any set tensor.
- Ref_Semantic: Reference to a semantic in semantics tensor.
"""

from .token import Token, Ref_Token
from .semantic import Semantic, Ref_Semantic

__all__ = [
    "Token",
    "Semantic",
    "Ref_Token",
    "Ref_Semantic"
]

