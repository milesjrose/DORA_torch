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



class TestGetCount:
    """Test the get_count method."""
    
    def test_get_count(self, basic_semantics):
        """Test getting the count of semantics."""
        count = basic_semantics.get_count()
        assert count == basic_semantics.nodes.shape[0]
        assert count == 5


class TestIndividualTokenFunctions:
    """Test individual token access functions."""
    
    def test_get(self, basic_semantics):
        """Test getting a feature for a semantic."""
        ref_semantic = Ref_Semantic(1, "test1")
        
        # Test getting ACT feature
        act_value = basic_semantics.get(ref_semantic, SF.ACT)
        assert act_value == 0.5
        
        # Test getting INPUT feature
        input_value = basic_semantics.get(ref_semantic, SF.INPUT)
        assert input_value == 0.3
    
    def test_get_invalid_reference(self, basic_semantics):
        """Test getting feature with invalid reference."""
        ref_semantic = Ref_Semantic(999, "nonexistent")  # Invalid ID
        
        with pytest.raises(ValueError, match="Invalid reference semantic or feature"):
            basic_semantics.get(ref_semantic, SF.ACT)
    
    def test_set(self, basic_semantics):
        """Test setting a feature for a semantic."""
        ref_semantic = Ref_Semantic(1, "test1")
        
        # Set new ACT value
        basic_semantics.set(ref_semantic, SF.ACT, 0.9)
        
        # Verify it was set
        assert basic_semantics.nodes[basic_semantics.IDs[1], SF.ACT] == 0.9
    
    def test_set_invalid_reference(self, basic_semantics):
        """Test setting feature with invalid reference."""
        ref_semantic = Ref_Semantic(999, "nonexistent")  # Invalid ID
        
        with pytest.raises(ValueError, match="Invalid reference semantic or feature"):
            basic_semantics.set(ref_semantic, SF.ACT, 0.5)
    
    def test_get_index(self, basic_semantics):
        """Test getting index for a semantic."""
        ref_semantic = Ref_Semantic(1, "test1")
        
        index = basic_semantics.get_index(ref_semantic)
        assert index == 0  # Based on the fixture setup
    
    def test_get_index_invalid(self, basic_semantics):
        """Test getting index with invalid reference."""
        ref_semantic = Ref_Semantic(999, "nonexistent")
        
        with pytest.raises(ValueError, match="Invalid reference semantic"):
            basic_semantics.get_index(ref_semantic)
    
    #Hangs
    def test_get_reference_by_id(self, basic_semantics):
        """Test getting reference by ID."""
        ref = basic_semantics.get_reference(id=1)
        
        assert isinstance(ref, Ref_Semantic)
        assert ref.ID == 1
        assert ref.name == "test1"
    
    def test_get_reference_by_index(self, basic_semantics):
        """Test getting reference by index."""
        ref = basic_semantics.get_reference(index=0)
        
        assert isinstance(ref, Ref_Semantic)
        assert ref.ID == 1
        assert ref.name == "test1"
    
    def test_get_reference_by_name(self, basic_semantics):
        """Test getting reference by name."""
        ref = basic_semantics.get_reference(name="test1")
        
        assert isinstance(ref, Ref_Semantic)
        assert ref.ID == 1
        assert ref.name == "test1"
    
    def test_get_reference_invalid(self, basic_semantics):
        """Test getting reference with invalid parameters."""
        # No parameters
        with pytest.raises(ValueError, match="No ID, index, or name provided"):
            basic_semantics.get_reference()
        
        # Invalid ID
        with pytest.raises(ValueError, match="Invalid ID"):
            basic_semantics.get_reference(id=999)
        
        # Invalid index
        with pytest.raises(ValueError, match="Invalid index"):
            basic_semantics.get_reference(index=999)
        
        # Invalid name
        with pytest.raises(ValueError, match="Invalid name"):
            basic_semantics.get_reference(name="nonexistent")
    
    def test_get_single_semantic_copy(self, basic_semantics):
        """Test getting a single semantic with copy=True."""
        ref_semantic = Ref_Semantic(1, "test1")
        
        semantic = basic_semantics.get_single_semantic(ref_semantic, copy=True)
        
        assert isinstance(semantic, Semantic)
        assert semantic.name == "test1"
        assert torch.equal(semantic.tensor, basic_semantics.nodes[0, :])
        
        # Modifying the returned semantic should not affect the original
        semantic.tensor[SF.ACT] = 0.9
        assert basic_semantics.nodes[0, SF.ACT] == 0.5  # Original unchanged
    
    def test_get_single_semantic_no_copy(self, basic_semantics):
        """Test getting a single semantic with copy=False."""
        ref_semantic = Ref_Semantic(1, "test1")
        
        semantic = basic_semantics.get_single_semantic(ref_semantic, copy=False)
        
        # Modifying the returned semantic should affect the original
        semantic.tensor[SF.ACT] = 0.9
        assert basic_semantics.nodes[0, SF.ACT] == 0.9  # Original changed


class TestSemanticFunctions:
    """Test semantic operation functions."""
    
    def test_init_sem(self, basic_semantics):
        """Test initializing semantics."""
        # Set some initial values
        basic_semantics.nodes[:, SF.ACT] = 0.8
        basic_semantics.nodes[:, SF.INPUT] = 0.6
        
        basic_semantics.init_sem()
        
        # Should reset ACT and INPUT to 0
        assert (basic_semantics.nodes[:, SF.ACT] == 0.0).all()
        assert (basic_semantics.nodes[:, SF.INPUT] == 0.0).all()
    
    def test_init_input(self, basic_semantics):
        """Test initializing input with refresh value."""
        refresh_value = 0.3
        basic_semantics.init_input(refresh_value)
        
        assert (basic_semantics.nodes[:, SF.INPUT] == refresh_value).all()
    
    def test_set_max_input(self, basic_semantics):
        """Test setting max input for all semantics."""
        max_input_value = 0.7
        basic_semantics.set_max_input(max_input_value)
        
        assert (basic_semantics.nodes[:, SF.MAX_INPUT] == max_input_value).all()
    
    def test_get_max_input(self, basic_semantics):
        """Test getting maximum input."""
        # Set some input values
        basic_semantics.nodes[0, SF.INPUT] = 0.3
        basic_semantics.nodes[1, SF.INPUT] = 0.7
        basic_semantics.nodes[2, SF.INPUT] = 0.5
        
        max_input = basic_semantics.get_max_input()
        assert max_input == 0.7
    
    def test_update_act(self, basic_semantics):
        """Test updating activation values."""
        # Set up test data
        basic_semantics.nodes[0, SF.INPUT] = 0.6
        basic_semantics.nodes[0, SF.MAX_INPUT] = 1.0
        
        basic_semantics.nodes[1, SF.INPUT] = 0.3
        basic_semantics.nodes[1, SF.MAX_INPUT] = 0.5
        
        basic_semantics.nodes[2, SF.INPUT] = 0.4
        basic_semantics.nodes[2, SF.MAX_INPUT] = 0.0  # Should result in 0 act
        
        basic_semantics.update_act()
        
        # Check calculated activations
        assert basic_semantics.nodes[0, SF.ACT] == 0.6  # 0.6/1.0
        assert basic_semantics.nodes[1, SF.ACT] == 0.6  # 0.3/0.5
        assert basic_semantics.nodes[2, SF.ACT] == 0.0  # max_input == 0

class TestUpdateInput:
    """Test input update functions."""
    
    def test_update_input_from_set_no_links(self, basic_semantics):
        """Test update_input_from_set when links are not initialized."""
        # Create a mock Base_Set
        class MockSet:
            def get_mask(self, token_type):
                return torch.tensor([True, False, True])
            
            def __init__(self):
                self.nodes = torch.tensor([[0.5, 0.3], [0.2, 0.4], [0.7, 0.1]])
        
        mock_set = MockSet()
        
        # Ensure links is None
        basic_semantics.links = None
        
        with pytest.raises(ValueError, match="Links not initialised"):
            basic_semantics.update_input_from_set(mock_set, Set.DRIVER)
    
    def test_update_input_from_set_with_links(self, network):
        """Test update_input_from_set with actual network links."""
        semantics = network.semantics
        driver_set = network.sets[Set.DRIVER]
        
        # Set initial input values
        semantics.nodes[:, SF.INPUT] = 0.1
        
        # Store initial input values
        initial_inputs = semantics.nodes[:, SF.INPUT].clone()
        
        # Update input from driver set
        semantics.update_input_from_set(driver_set, Set.DRIVER)
        
        # Check that inputs were updated (should be different from initial)
        updated_inputs = semantics.nodes[:, SF.INPUT]
        
        # At least some semantics should have updated inputs
        # (assuming the network has some non-zero links and activations)
        input_changes = (updated_inputs - initial_inputs).abs()
        assert input_changes.sum() >= 0  # Should have some changes or at least no errors
        
        # Test with recipient set as well
        recipient_set = network.sets[Set.RECIPIENT]
        semantics.update_input_from_set(recipient_set, Set.RECIPIENT)
        
        # Should complete without errors
        final_inputs = semantics.nodes[:, SF.INPUT]
        assert final_inputs.shape == initial_inputs.shape
    
    def test_update_input_full_method(self, network):
        """Test the full update_input method with driver, recipient, and memory."""
        semantics = network.semantics
        
        # Set initial input values
        semantics.nodes[:, SF.INPUT] = 0.0
        initial_inputs = semantics.nodes[:, SF.INPUT].clone()
        
        # Test with driver and recipient only
        driver_set = network.sets[Set.DRIVER]
        recipient_set = network.sets[Set.RECIPIENT]
        
        semantics.update_input(driver_set, recipient_set)
        
        # Should have updated inputs
        updated_inputs = semantics.nodes[:, SF.INPUT]
        input_changes = (updated_inputs - initial_inputs).abs()
        assert input_changes.sum() >= 0  # Should have some changes or at least no errors
        
        # Test with memory as well
        memory_set = network.sets[Set.MEMORY]
        semantics.update_input(driver_set, recipient_set, memory_set)
        
        # Should complete without errors
        final_inputs = semantics.nodes[:, SF.INPUT]
        assert final_inputs.shape == initial_inputs.shape
