# nodes/tests/test_semantics.py
# Tests for semantics operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import Set, SF, Type, B, null
from nodes.network.sets.semantics import Semantics
from nodes.network.single_nodes import Semantic, Ref_Semantic
from nodes.network.connections import Links

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps


@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()


@pytest.fixture
def basic_semantics():
    """Create a basic Semantics object for testing."""
    nodes = torch.zeros(5, len(SF))
    connections = torch.zeros(5, 5)
    IDs = {1: 0, 2: 1, 3: 2}
    names = {1: "test1", 2: "test2", 3: "test3"}
    
    # Set up some basic semantic data
    nodes[0, SF.ID] = 1
    nodes[0, SF.TYPE] = Type.SEMANTIC
    nodes[0, SF.ACT] = 0.5
    nodes[0, SF.INPUT] = 0.3
    
    nodes[1, SF.ID] = 2
    nodes[1, SF.TYPE] = Type.SEMANTIC
    nodes[1, SF.ACT] = 0.7
    nodes[1, SF.INPUT] = 0.4
    
    nodes[2, SF.ID] = 3
    nodes[2, SF.TYPE] = Type.SEMANTIC
    nodes[2, SF.ACT] = 0.2
    nodes[2, SF.INPUT] = 0.1
    
    return Semantics(nodes, connections, IDs, names)
