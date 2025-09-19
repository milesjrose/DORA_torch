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


class TestComparativeSemantics:
    """Test comparative semantics initialization and checking."""
    
    def test_init_comparative_semantics_empty(self):
        """Test initializing comparative semantics when none exist."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = {1: "test1", 2: "test2", 3: "test3"}
        
        semantics = Semantics(nodes, connections, IDs, names)
        
        # Initially should be None
        assert semantics.more is None
        assert semantics.less is None
        assert semantics.same is None
        
        # Initialize comparative semantics
        semantics.init_comparative_semantics()
        
        # Should now have semantic objects
        assert semantics.more is not None
        assert semantics.less is not None
        assert semantics.same is not None
        
        # Should be Ref_Semantic objects
        assert hasattr(semantics.more, 'ID')
        assert hasattr(semantics.less, 'ID')
        assert hasattr(semantics.same, 'ID')
        assert hasattr(semantics.more, 'name')
        assert hasattr(semantics.less, 'name')
        assert hasattr(semantics.same, 'name')
        
        # Should be added to names
        assert "more" in semantics.names.values()
        assert "less" in semantics.names.values()
        assert "same" in semantics.names.values()
    
    def test_init_comparative_semantics_existing_names(self):
        """Test initializing when comparative semantics already exist in names."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = {1: "test1", 2: "more", 3: "test3"}  # "more" already exists
        
        semantics = Semantics(nodes, connections, IDs, names)
        semantics.init_comparative_semantics()
        
        # Should not create new "more" semantic
        assert semantics.more is None
        # But should create "less" and "same"
        assert semantics.less is not None
        assert semantics.same is not None
        
        # Should be Ref_Semantic objects
        assert hasattr(semantics.less, 'ID')
        assert hasattr(semantics.same, 'ID')
        assert hasattr(semantics.less, 'name')
        assert hasattr(semantics.same, 'name')
    
    def test_check_comps(self):
        """Test the check_comps method."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = {1: "test1", 2: "test2", 3: "test3"}
        
        semantics = Semantics(nodes, connections, IDs, names)
        
        # Initially should return False (not all are set)
        assert not semantics.check_comps()
        
        # After initialization, should return True
        semantics.init_comparative_semantics()
        assert semantics.check_comps()
