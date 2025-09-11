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


class TestSemanticsInit:
    """Test the Semantics.__init__ method."""
    
    def test_init_valid_inputs(self):
        """Test initialization with valid inputs."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = {1: "test1", 2: "test2", 3: "test3"}
        
        semantics = Semantics(nodes, connections, IDs, names)
        
        assert semantics.nodes.shape == (3, len(SF))
        assert semantics.connections.shape == (3, 3)
        assert semantics.IDs == IDs
        assert semantics.names == names
        assert semantics.links is None
        assert semantics.params is None
        assert semantics.expansion_factor == 1.1
    
    def test_init_without_names(self):
        """Test initialization without names dictionary."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        
        semantics = Semantics(nodes, connections, IDs)
        
        assert semantics.names is None
        assert semantics.IDs == IDs
    
    def test_init_mismatched_sizes(self):
        """Test initialization with mismatched tensor sizes."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(4, 4)  # Different size
        IDs = {1: 0, 2: 1, 3: 2}
        
        with pytest.raises(ValueError, match="nodes and connections must have the same number of semantics"):
            Semantics(nodes, connections, IDs)
    
    def test_init_wrong_feature_count(self):
        """Test initialization with wrong number of features."""
        nodes = torch.zeros(3, 5)  # Wrong number of features
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        
        with pytest.raises(ValueError, match="nodes must have number of features listed in SF enum"):
            Semantics(nodes, connections, IDs)
    
    def test_init_invalid_names_type(self):
        """Test initialization with invalid names type."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = "invalid"  # Should be dict
        
        with pytest.raises(ValueError, match="names must be a dictionary"):
            Semantics(nodes, connections, IDs, names)
    
    def test_init_invalid_names_values(self):
        """Test initialization with invalid names values."""
        nodes = torch.zeros(3, len(SF))
        connections = torch.zeros(3, 3)
        IDs = {1: 0, 2: 1, 3: 2}
        names = {1: "test1", 2: 123, 3: "test3"}  # Non-string value
        
        with pytest.raises(ValueError, match="names must be a dictionary of strings"):
            Semantics(nodes, connections, IDs, names)
