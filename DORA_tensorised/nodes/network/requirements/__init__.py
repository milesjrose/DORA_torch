# nodes/network/requirements/__init__.py
# Initialises requirements object

"""
Package for requirment checks.

Requirements object:
- predication() -> Checks requirements for predication
- rel_form() -> Checks requirements for relation formation 
- schema() -> Checks requirements for schematisation 
- rel_gen() -> Checks requirements for relation generalisation )
"""

from .requirements import Requirements

__all__ = ['Requirements']