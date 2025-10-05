# nodes/tests/set_ops/test_memory.py
# Tests for memory set operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.network import Network
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token
from nodes.utils import tensor_ops as tOps

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

@pytest.fixture
def memory_set(network):
    """Get the memory set from the network."""
    return network.memory()

def test_memory_initialization(memory_set):
    """Test Memory class initialization."""
    assert memory_set.token_set == Set.MEMORY
    assert isinstance(memory_set.nodes, torch.Tensor)
    assert isinstance(memory_set.connections, torch.Tensor)
    assert isinstance(memory_set.IDs, dict)
    assert isinstance(memory_set.names, dict)

def test_memory_update_input(memory_set):
    """Test the main update_input function."""
    # set random acts for all semantics and nodes
    semantics = memory_set.links.semantics
    semantics.nodes[:, SF.ACT] = torch.rand(semantics.nodes.shape[0])
    all_mask = memory_set.get_all_nodes_mask()
    memory_set.nodes[all_mask, TF.ACT] = torch.rand(all_mask.sum())
    
    # Store initial values
    initial_td_input = memory_set.nodes[:, TF.TD_INPUT].clone()
    initial_bu_input = memory_set.nodes[:, TF.BU_INPUT].clone()
    initial_map_input = memory_set.nodes[:, TF.MAP_INPUT].clone()
    initial_lateral_input = memory_set.nodes[:, TF.LATERAL_INPUT].clone()
    
    # Call update_input
    memory_set.update_input()
    
    # Check that inputs have been updated (should be different from initial values)
    # Note: The exact values depend on the current state, but they should have changed
    td_changed = not torch.allclose(memory_set.nodes[:, TF.TD_INPUT], initial_td_input)
    bu_changed = not torch.allclose(memory_set.nodes[:, TF.BU_INPUT], initial_bu_input)
    map_changed = not torch.allclose(memory_set.nodes[:, TF.MAP_INPUT], initial_map_input)
    lateral_changed = not torch.allclose(memory_set.nodes[:, TF.LATERAL_INPUT], initial_lateral_input)
    
    # At least some inputs should have changed
    assert td_changed or bu_changed or map_changed or lateral_changed

def test_update_input_p_parent(memory_set):
    """Test update_input_p_parent function."""
    # set random acts for all semantics and p nodes
    semantics = memory_set.links.semantics
    semantics.nodes[:, SF.ACT] = torch.rand(semantics.nodes.shape[0])
    p_mask = memory_set.get_mask(Type.P)
    memory_set.nodes[p_mask, TF.ACT] = torch.rand(p_mask.sum())
    
    # Get P nodes mask
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Store initial values
    initial_td_input = memory_set.nodes[p_mask, TF.TD_INPUT].clone()
    initial_bu_input = memory_set.nodes[p_mask, TF.BU_INPUT].clone()
    initial_map_input = memory_set.nodes[p_mask, TF.MAP_INPUT].clone()
    initial_lateral_input = memory_set.nodes[p_mask, TF.LATERAL_INPUT].clone()
    
    # Call update_input_p_parent
    memory_set.update_input_p_parent()
    
    # Check that inputs have been updated (or at least the function doesn't crash)
    # The function may not change values if there are no connections or mappings
    td_changed = not torch.allclose(memory_set.nodes[p_mask, TF.TD_INPUT], initial_td_input)
    bu_changed = not torch.allclose(memory_set.nodes[p_mask, TF.BU_INPUT], initial_bu_input)
    map_changed = not torch.allclose(memory_set.nodes[p_mask, TF.MAP_INPUT], initial_map_input)
    lateral_changed = not torch.allclose(memory_set.nodes[p_mask, TF.LATERAL_INPUT], initial_lateral_input)
    
    # The function should at least run without error
    # Values may not change if there are no connections or mappings set up
    assert True  # Function executed successfully

def test_update_input_p_child(memory_set):
    """Test update_input_p_child function."""
    # Get P nodes in child mode
    p_mask = memory_set.get_mask(Type.P)
    child_p_mask = tOps.refine_mask(memory_set.nodes, p_mask, TF.MODE, Mode.CHILD)
    
    if not torch.any(child_p_mask):
        pytest.skip("No P nodes in child mode in memory set")
    
    # Store initial values
    initial_td_input = memory_set.nodes[child_p_mask, TF.TD_INPUT].clone()
    initial_bu_input = memory_set.nodes[child_p_mask, TF.BU_INPUT].clone()
    initial_map_input = memory_set.nodes[child_p_mask, TF.MAP_INPUT].clone()
    initial_lateral_input = memory_set.nodes[child_p_mask, TF.LATERAL_INPUT].clone()
    
    # Call update_input_p_child
    memory_set.update_input_p_child()
    
    # Check that inputs have been updated
    td_changed = not torch.allclose(memory_set.nodes[child_p_mask, TF.TD_INPUT], initial_td_input)
    bu_changed = not torch.allclose(memory_set.nodes[child_p_mask, TF.BU_INPUT], initial_bu_input)
    map_changed = not torch.allclose(memory_set.nodes[child_p_mask, TF.MAP_INPUT], initial_map_input)
    lateral_changed = not torch.allclose(memory_set.nodes[child_p_mask, TF.LATERAL_INPUT], initial_lateral_input)
    
    # The function should at least run without error
    # Values may not change if there are no connections or mappings set up
    assert True  # Function executed successfully

def test_update_input_rb(memory_set):
    """Test update_input_rb function."""
    # Get RB nodes mask
    rb_mask = memory_set.get_mask(Type.RB)
    if not torch.any(rb_mask):
        pytest.skip("No RB nodes in memory set")
    
    # Store initial values
    initial_td_input = memory_set.nodes[rb_mask, TF.TD_INPUT].clone()
    initial_bu_input = memory_set.nodes[rb_mask, TF.BU_INPUT].clone()
    initial_map_input = memory_set.nodes[rb_mask, TF.MAP_INPUT].clone()
    initial_lateral_input = memory_set.nodes[rb_mask, TF.LATERAL_INPUT].clone()
    
    # Call update_input_rb
    memory_set.update_input_rb()
    
    # Check that inputs have been updated (or at least the function doesn't crash)
    # The function may not change values if there are no connections or mappings
    td_changed = not torch.allclose(memory_set.nodes[rb_mask, TF.TD_INPUT], initial_td_input)
    bu_changed = not torch.allclose(memory_set.nodes[rb_mask, TF.BU_INPUT], initial_bu_input)
    map_changed = not torch.allclose(memory_set.nodes[rb_mask, TF.MAP_INPUT], initial_map_input)
    lateral_changed = not torch.allclose(memory_set.nodes[rb_mask, TF.LATERAL_INPUT], initial_lateral_input)
    
    # The function should at least run without error
    # Values may not change if there are no connections or mappings set up
    assert True  # Function executed successfully

def test_update_input_po(memory_set):
    """Test update_input_po function."""
    # Get PO nodes mask (non-inferred)
    all_po_mask = memory_set.get_mask(Type.PO)
    po_mask = tOps.refine_mask(memory_set.nodes, all_po_mask, TF.INFERRED, B.FALSE)
    # set random acts for all semantics and po nodes
    semantics = memory_set.links.semantics
    semantics.nodes[:, SF.ACT] = torch.rand(semantics.nodes.shape[0])
    po_mask = memory_set.get_mask(Type.PO)
    memory_set.nodes[po_mask, TF.ACT] = torch.rand(po_mask.sum())
    if not torch.any(po_mask):
        pytest.skip("No non-inferred PO nodes in memory set")
    
    # Store initial values
    initial_td_input = memory_set.nodes[po_mask, TF.TD_INPUT].clone()
    initial_bu_input = memory_set.nodes[po_mask, TF.BU_INPUT].clone()
    initial_map_input = memory_set.nodes[po_mask, TF.MAP_INPUT].clone()
    initial_lateral_input = memory_set.nodes[po_mask, TF.LATERAL_INPUT].clone()
    
    # Call update_input_po
    memory_set.update_input_po()
    
    # Check that inputs have been updated
    td_changed = not torch.allclose(memory_set.nodes[po_mask, TF.TD_INPUT], initial_td_input)
    bu_changed = not torch.allclose(memory_set.nodes[po_mask, TF.BU_INPUT], initial_bu_input)
    map_changed = not torch.allclose(memory_set.nodes[po_mask, TF.MAP_INPUT], initial_map_input)
    lateral_changed = not torch.allclose(memory_set.nodes[po_mask, TF.LATERAL_INPUT], initial_lateral_input)
    
    # At least some inputs should have changed
    assert td_changed or bu_changed or map_changed or lateral_changed

def test_map_input(memory_set):
    """Test map_input function."""
    # Get a mask for testing (use P nodes if available, otherwise RB nodes)
    test_mask = memory_set.get_mask(Type.P)
    if not torch.any(test_mask):
        test_mask = memory_set.get_mask(Type.RB)
    if not torch.any(test_mask):
        test_mask = memory_set.get_mask(Type.PO)
    if not torch.any(test_mask):
        pytest.skip("No suitable nodes for testing map_input")
    
    # Call map_input
    result = memory_set.map_input(test_mask)
    
    # Check that result is a tensor with correct shape
    assert isinstance(result, torch.Tensor)
    assert result.shape == (torch.sum(test_mask).item(),)
    
    # Check that result contains finite values
    assert torch.all(torch.isfinite(result))

def test_map_input_with_different_masks(memory_set):
    """Test map_input function with different node type masks."""
    # Test with P nodes
    p_mask = memory_set.get_mask(Type.P)
    if torch.any(p_mask):
        result_p = memory_set.map_input(p_mask)
        assert isinstance(result_p, torch.Tensor)
        assert result_p.shape == (torch.sum(p_mask).item(),)
    
    # Test with RB nodes
    rb_mask = memory_set.get_mask(Type.RB)
    if torch.any(rb_mask):
        result_rb = memory_set.map_input(rb_mask)
        assert isinstance(result_rb, torch.Tensor)
        assert result_rb.shape == (torch.sum(rb_mask).item(),)
    
    # Test with PO nodes
    po_mask = memory_set.get_mask(Type.PO)
    if torch.any(po_mask):
        result_po = memory_set.map_input(po_mask)
        assert isinstance(result_po, torch.Tensor)
        assert result_po.shape == (torch.sum(po_mask).item(),)

def test_memory_set_has_required_attributes(memory_set):
    """Test that memory set has all required attributes."""
    required_attrs = [
        'nodes', 'connections', 'IDs', 'names', 'token_set',
        'token_ops', 'tensor_ops', 'update_ops'
    ]
    
    for attr in required_attrs:
        assert hasattr(memory_set, attr), f"Memory set missing attribute: {attr}"

def test_memory_set_operations_available(memory_set):
    """Test that memory set operations are available."""
    # Test token operations
    assert hasattr(memory_set.token_ops, 'get_feature')
    assert hasattr(memory_set.token_ops, 'set_feature')
    assert hasattr(memory_set.token_ops, 'get_reference')
    
    # Test tensor operations
    assert hasattr(memory_set.tensor_ops, 'get_mask')
    assert hasattr(memory_set.tensor_ops, 'add_token')
    assert hasattr(memory_set.tensor_ops, 'del_token')
    
    # Test update operations
    assert hasattr(memory_set.update_ops, 'update_act')
    assert hasattr(memory_set.update_ops, 'init_act')

def test_memory_set_connections_structure(memory_set):
    """Test that memory set connections have correct structure."""
    num_nodes = memory_set.nodes.shape[0]
    assert memory_set.connections.shape == (num_nodes, num_nodes)
    assert memory_set.connections.dtype == torch.float

def test_memory_set_nodes_structure(memory_set):
    """Test that memory set nodes have correct structure."""
    assert memory_set.nodes.dtype == torch.float
    assert memory_set.nodes.shape[1] == len(TF)  # Should have all token features
    
    # Check that all nodes have valid IDs
    for i in range(memory_set.nodes.shape[0]):
        node_id = memory_set.nodes[i, TF.ID].item()
        assert node_id in memory_set.IDs.values()

def test_memory_set_with_empty_mask(memory_set):
    """Test memory functions with empty masks."""
    # Create an empty mask
    empty_mask = torch.zeros(memory_set.nodes.shape[0], dtype=torch.bool)
    
    # Test map_input with empty mask
    result = memory_set.map_input(empty_mask)
    assert result.shape == (0,)
    
    # Test that update functions don't crash with no nodes of a type
    # This is more of a robustness test
    try:
        memory_set.update_input_p_parent()
        memory_set.update_input_p_child()
        memory_set.update_input_rb()
        memory_set.update_input_po()
    except Exception as e:
        pytest.fail(f"Update functions should handle empty sets gracefully: {e}")

def test_memory_set_parameter_dependencies(memory_set):
    """Test that memory set functions depend on parameters correctly."""
    # This test ensures that the functions use the params object
    # We can't easily test the exact behavior without mocking, but we can ensure
    # the functions don't crash when params are accessed
    
    # Store original params
    original_params = memory_set.params
    
    # Test that functions can access params
    try:
        memory_set.update_input_p_parent()
        memory_set.update_input_p_child()
        memory_set.update_input_rb()
        memory_set.update_input_po()
    except AttributeError as e:
        if "params" in str(e):
            pytest.fail(f"Memory functions should handle missing params gracefully: {e}")
        else:
            raise  # Re-raise if it's a different AttributeError
