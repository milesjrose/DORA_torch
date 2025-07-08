# nodes/tests/test_firing_ops.py
# Tests for FiringOperations class

import pytest
import torch
import random

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.operations.firing_ops import FiringOperations

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

@pytest.fixture
def firing_ops(network):
    """Create a FiringOperations object."""
    return FiringOperations(network)

def test_init(firing_ops, network):
    """Test FiringOperations initialization."""
    assert firing_ops.network == network

def test_by_top_random_with_groups(firing_ops):
    """Test by_top_random when highest token type is Group."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if there are any groups in the driver
    highest_type = driver.token_ops.get_highest_token_type()
    
    if highest_type == Type.GROUP:
        result = firing_ops.by_top_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # Should contain groups, Ps, and RBs in order
        if result:
            # Verify the order: groups first, then Ps, then RBs
            group_indices = [i for i in result if driver.nodes[i, TF.TYPE] == Type.GROUP]
            p_indices = [i for i in result if driver.nodes[i, TF.TYPE] == Type.P]
            rb_indices = [i for i in result if driver.nodes[i, TF.TYPE] == Type.RB]
            
            # Groups should come first
            if group_indices and p_indices:
                first_group = result.index(group_indices[0])
                first_p = result.index(p_indices[0])
                assert first_group < first_p
            
            # Ps should come before RBs
            if p_indices and rb_indices:
                first_p = result.index(p_indices[0])
                first_rb = result.index(rb_indices[0])
                assert first_p < first_rb

def test_by_top_random_with_ps(firing_ops):
    """Test by_top_random when highest token type is P."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if the highest type is P
    highest_type = driver.token_ops.get_highest_token_type()
    
    if highest_type == Type.P:
        result = firing_ops.by_top_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # Should contain Ps and RBs in order
        if result:
            p_indices = [i for i in result if driver.nodes[i, TF.TYPE] == Type.P]
            rb_indices = [i for i in result if driver.nodes[i, TF.TYPE] == Type.RB]
            
            # Ps should come before RBs
            if p_indices and rb_indices:
                first_p = result.index(p_indices[0])
                first_rb = result.index(rb_indices[0])
                assert first_p < first_rb

def test_by_top_random_with_rbs(firing_ops):
    """Test by_top_random when highest token type is RB."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if the highest type is RB
    highest_type = driver.token_ops.get_highest_token_type()
    
    if highest_type == Type.RB:
        result = firing_ops.by_top_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # Should only contain RBs
        if result:
            for i in result:
                assert driver.nodes[i, TF.TYPE] == Type.RB

def test_by_top_random_with_pos(firing_ops):
    """Test by_top_random when highest token type is PO."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if the highest type is PO
    highest_type = driver.token_ops.get_highest_token_type()
    
    if highest_type == Type.PO:
        result = firing_ops.by_top_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # Should only contain POs
        if result:
            for i in result:
                assert driver.nodes[i, TF.TYPE] == Type.PO

def test_by_top_random_no_tokens(firing_ops):
    """Test by_top_random when no tokens exist."""
    # Create an empty network for this test
    empty_builder = NetworkBuilder(symProps=[])
    empty_network = empty_builder.build_network()
    empty_firing_ops = FiringOperations(empty_network)
    
    result = empty_firing_ops.by_top_random()
    
    # Should return empty list
    assert result == []

def test_get_all_children_firing_order(firing_ops):
    """Test get_all_children_firing_order method."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Get some indices that have children
    all_nodes_mask = driver.tensor_op.get_all_nodes_mask()
    parent_indices = torch.where(all_nodes_mask)[0][:3].tolist()  # Take first 3 nodes
    
    result = firing_ops.get_all_children_firing_order(parent_indices)
    
    # Should return a list of indices
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)
    
    # Each child should be a child of one of the parent indices
    for child_index in result:
        is_child = False
        for parent_index in parent_indices:
            if driver.connections[parent_index, child_index] == B.TRUE:
                is_child = True
                break
        assert is_child, f"Index {child_index} is not a child of any parent"

def test_get_random_order_of_type(firing_ops):
    """Test get_random_order_of_type method."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Test with RB type
    result = firing_ops.get_random_order_of_type(Type.RB)
    
    # Should return a list of indices
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)
    
    # All returned indices should be of type RB
    for i in result:
        assert driver.nodes[i, TF.TYPE] == Type.RB

def test_totally_random_with_rbs(firing_ops):
    """Test totally_random when RB nodes exist."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if there are RBs
    rb_mask = driver.tensor_op.get_mask(Type.RB)
    if rb_mask.sum() > 0:
        result = firing_ops.totally_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # All returned indices should be of type RB
        for i in result:
            assert driver.nodes[i, TF.TYPE] == Type.RB

def test_totally_random_with_pos(firing_ops):
    """Test totally_random when no RBs but PO nodes exist."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Check if there are no RBs but POs exist
    rb_mask = driver.tensor_op.get_mask(Type.RB)
    po_mask = driver.tensor_op.get_mask(Type.PO)
    
    if rb_mask.sum() == 0 and po_mask.sum() > 0:
        result = firing_ops.totally_random()
        
        # Should return a list of indices
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)
        
        # All returned indices should be of type PO
        for i in result:
            assert driver.nodes[i, TF.TYPE] == Type.PO

def test_totally_random_no_nodes(firing_ops):
    """Test totally_random when no RB or PO nodes exist."""
    # Create an empty network for this test
    empty_builder = NetworkBuilder(symProps=[])
    empty_network = empty_builder.build_network()
    empty_firing_ops = FiringOperations(empty_network)
    
    result = empty_firing_ops.totally_random()
    
    # Should return empty list
    assert result == []

def test_get_children_firing_order(firing_ops):
    """Test get_children_firing_order method."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Find a node that has children
    all_nodes_mask = driver.tensor_op.get_all_nodes_mask()
    for index in torch.where(all_nodes_mask)[0]:
        children = driver.token_ops.get_child_indices(index.item())
        if children:
            result = firing_ops.get_children_firing_order(index.item())
            
            # Should return the same children
            assert result == children
            break
    else:
        # If no node has children, test with empty result
        result = firing_ops.get_children_firing_order(0)
        assert result == []

def test_integration_with_real_network(firing_ops):
    """Test integration with a real network setup."""
    # Test that all required methods exist
    assert hasattr(firing_ops, 'network')
    assert hasattr(firing_ops, 'by_top_random')
    assert hasattr(firing_ops, 'totally_random')
    assert hasattr(firing_ops, 'get_random_order_of_type')
    assert hasattr(firing_ops, 'get_all_children_firing_order')
    assert hasattr(firing_ops, 'get_children_firing_order')

def test_error_handling(firing_ops):
    """Test error handling in various scenarios."""
    # Test with invalid token type
    driver = firing_ops.network.driver()
    
    # Temporarily modify the highest token type to an invalid value
    original_nodes = driver.nodes.clone()
    driver.nodes[:, TF.TYPE] = 999  # Invalid type
    
    result = firing_ops.by_top_random()
    assert result == []
    
    # Restore original nodes
    driver.nodes = original_nodes

def test_random_seed_consistency(firing_ops):
    """Test that random operations are called correctly."""
    # Set a fixed seed for reproducible results
    random.seed(42)
    
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Test multiple calls to ensure they work
    result1 = firing_ops.by_top_random()
    result2 = firing_ops.by_top_random()
    
    # Both should return valid results
    assert isinstance(result1, list)
    assert isinstance(result2, list)
    
    # Reset seed
    random.seed()

def test_firing_order_contains_all_tokens(firing_ops):
    """Test that firing order contains all tokens of the highest type."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Get the highest token type
    highest_type = driver.token_ops.get_highest_token_type()
    if highest_type is None:
        return  # Skip if no tokens
    
    # Get firing order
    firing_order = firing_ops.by_top_random()
    
    # Get all tokens of the highest type
    type_mask = driver.tensor_op.get_mask(highest_type)
    all_tokens_of_type = torch.where(type_mask)[0].tolist()
    
    # All tokens of the highest type should be in the firing order
    for token_index in all_tokens_of_type:
        assert token_index in firing_order, f"Token {token_index} of type {highest_type} not in firing order"

def test_firing_order_structure(firing_ops):
    """Test that firing order follows the correct hierarchical structure."""
    # Get the driver set
    driver = firing_ops.network.driver()
    
    # Get firing order
    firing_order = firing_ops.by_top_random()
    
    if not firing_order:
        return  # Skip if no firing order
    
    # Check that the order follows hierarchy: Groups -> Ps -> RBs -> POs
    type_order = [Type.GROUP, Type.P, Type.RB, Type.PO]
    
    for i, token_index in enumerate(firing_order):
        token_type = driver.nodes[token_index, TF.TYPE]
        
        # Find the position of this type in the hierarchy
        type_position = type_order.index(token_type)
        
        # Check that no lower types come after higher types
        for j in range(i + 1, len(firing_order)):
            later_token_type = driver.nodes[firing_order[j], TF.TYPE]
            later_type_position = type_order.index(later_token_type)
            
            # Higher types should come before lower types
            assert type_position <= later_type_position, \
                f"Token of type {token_type} at position {i} comes after token of type {later_token_type} at position {j}"
