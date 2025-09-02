# nodes/network/routines/__init__.py
# Initialises routines object

"""
Package for routines.

- RetrievalOperations: Handles retrieval routines
- RelFormOperations: Handles relation formation routines
- SchematisationOperations: Handles schematisation routines
- RelGenOperations: Handles relation generalisation routines
- PredicationOperations: Handles predication routines
"""

from .retrieval import RetrievalOperations
from .rel_form import RelFormOperations
from .schematisation import SchematisationOperations
from .rel_gen import RelGenOperations
from .predication import PredicationOperations
from .routines import Routines

__all__ = [
    'RetrievalOperations',
    'RelFormOperations', 
    'SchematisationOperations', 
    'RelGenOperations', 
    'PredicationOperations',
    'Routines'
    ]