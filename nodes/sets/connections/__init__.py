# nodes/sets/connections/__init__.py
"""
Package: connections

Provides classes for representing connections between nodes in different sets.

- Links: Represents weighted connections between tokens and semantics.
- Mappings: Represents mappings between nodes and semantics.
"""

from .links import Links
from .mappings import Mappings

__all__ = [
    "Links",
    "Mappings"
]
