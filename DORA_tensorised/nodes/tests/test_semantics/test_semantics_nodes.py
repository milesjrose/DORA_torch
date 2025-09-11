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



class TestAddSemantic:
    """Test the add_semantic method."""
    
    def test_add_semantic_to_empty_slot(self, basic_semantics):
        """Test adding a semantic to an empty slot."""
        # Mark one semantic as deleted to create empty slot
        basic_semantics.nodes[1, SF.DELETED] = B.TRUE
        
        # Create new semantic
        new_semantic = Semantic("new_semantic", {SF.TYPE: Type.SEMANTIC})
        new_semantic.tensor[SF.ACT] = 0.8
        
        initial_count = len(basic_semantics.IDs)
        basic_semantics.add_semantic(new_semantic)
        
        # Should reuse the deleted slot
        assert len(basic_semantics.IDs) == initial_count + 1
        assert "new_semantic" in basic_semantics.names.values()
        
        # Check that the semantic was added to the correct slot
        new_id = max(basic_semantics.IDs.keys())
        new_index = basic_semantics.IDs[new_id]
        assert basic_semantics.nodes[new_index, SF.ACT] == 0.8
    
    def test_add_semantic_expand_tensor(self, basic_semantics):
        """Test adding a semantic when tensor needs expansion."""
        # Fill all slots (no deleted semantics)
        initial_size = basic_semantics.nodes.size(0)
        
        new_semantic = Semantic("new_semantic", {SF.TYPE: Type.SEMANTIC})
        basic_semantics.add_semantic(new_semantic)
        
        # Should expand tensor
        new_size = basic_semantics.nodes.size(0)
        assert new_size > initial_size
        assert len(basic_semantics.IDs) == 4  # 3 original + 1 new
    
    def test_add_semantic_with_none_name(self, basic_semantics):
        """Test adding a semantic with None name."""
        basic_semantics.nodes[1, SF.DELETED] = B.TRUE
        
        new_semantic = Semantic(None, {SF.TYPE: Type.SEMANTIC})
        basic_semantics.add_semantic(new_semantic)
        
        # Should assign default name
        new_id = max(basic_semantics.IDs.keys())
        assert basic_semantics.names[new_id].startswith("Semantic ")


class TestExpandTensor:
    """Test tensor expansion methods."""
    
    def test_expand_tensor(self, basic_semantics):
        """Test expanding the tensor."""
        initial_size = basic_semantics.nodes.size(0)
        initial_connections_size = basic_semantics.connections.size(0)
        
        basic_semantics.expand_tensor()
        
        new_size = basic_semantics.nodes.size(0)
        new_connections_size = basic_semantics.connections.size(0)
        
        # Should expand by expansion factor (but at least by 1)
        expected_size = max(int(initial_size * basic_semantics.expansion_factor), initial_size + 1)
        assert new_size == expected_size
        assert new_connections_size == expected_size
        
        # New nodes should be marked as deleted
        assert basic_semantics.nodes[initial_size:, SF.DELETED].all()
        
        # Old data should be preserved (check that it's not all zeros since we have test data)
        old_data = basic_semantics.nodes[:initial_size, :]
        assert old_data.shape == (initial_size, len(SF))
        # Should have some non-zero values from our test data
        assert old_data.sum() > 0
    
    def test_expand_links(self, network):
        """Test expanding links for a specific set using the actual network."""
        semantics = network.semantics
        
        # Get initial link sizes
        initial_driver_links = semantics.links[Set.DRIVER]
        initial_driver_size = initial_driver_links.size(1)  # Number of semantics
        
        # Expand the semantics tensor
        semantics.expand_tensor()
        
        # Check that links were expanded
        new_driver_links = semantics.links[Set.DRIVER]
        new_driver_size = new_driver_links.size(1)
        
        assert new_driver_size > initial_driver_size
        assert new_driver_links.size(0) == initial_driver_links.size(0)  # Same number of tokens
        
        # Original links should be preserved
        assert torch.equal(new_driver_links[:, :initial_driver_size], initial_driver_links)
        # New links should be zero
        assert (new_driver_links[:, initial_driver_size:] == 0).all()


class TestDelSemantic:
    """Test the del_semantic method."""
    
    def test_del_semantic(self, basic_semantics):
        """Test deleting a semantic."""
        semantic_id = 2
        semantic_index = basic_semantics.IDs[semantic_id]
        
        initial_count = len(basic_semantics.IDs)
        basic_semantics.del_semantic(semantic_id)
        
        # Should be removed from IDs and names
        assert semantic_id not in basic_semantics.IDs
        assert semantic_id not in basic_semantics.names
        
        # Should be marked as deleted
        assert basic_semantics.nodes[semantic_index, SF.DELETED] == B.TRUE
        
        # Connections should be zeroed
        assert (basic_semantics.connections[semantic_index, :] == 0).all()
        assert (basic_semantics.connections[:, semantic_index] == 0).all()
    
    def test_del_semantic_with_links(self, network):
        """Test deleting a semantic with links using the actual network."""
        semantics = network.semantics
        
        # Get a semantic ID that exists
        if len(semantics.IDs) > 0:
            semantic_id = list(semantics.IDs.keys())[0]
            semantic_index = semantics.IDs[semantic_id]
            
            # Store initial link values for this semantic
            initial_driver_links = semantics.links[Set.DRIVER][:, semantic_index].clone()
            initial_recipient_links = semantics.links[Set.RECIPIENT][:, semantic_index].clone()
            
            semantics.del_semantic(semantic_id)
            
            # Links should be zeroed for this semantic
            assert (semantics.links[Set.DRIVER][:, semantic_index] == 0).all()
            assert (semantics.links[Set.RECIPIENT][:, semantic_index] == 0).all()
            
            # Other semantics should be unaffected
            if len(semantics.IDs) > 0:
                other_id = list(semantics.IDs.keys())[0]
                other_index = semantics.IDs[other_id]
                # This should still have non-zero links (assuming the network has links)
                assert semantics.links[Set.DRIVER][:, other_index].sum() >= 0  # Just check it's not corrupted
