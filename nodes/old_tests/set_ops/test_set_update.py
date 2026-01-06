# nodes/tests/set_ops/test_set_update.py
# Tests for update operations.

import pytest
import torch

from nodes.network.sets.base_set import Base_Set
from nodes.builder import NetworkBuilder
from nodes.network import Network
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token, Memory
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

@pytest.fixture
def driver_set(network):
    """Get the driver set from the network."""
    return network.driver()

def test_init_float(memory_set):
    """Test init_float function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.3
    
    # Call init_float
    memory_set.update_ops.init_float([Type.P], [TF.ACT, TF.TD_INPUT])
    
    # Check that values were set to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.ACT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.TD_INPUT], torch.zeros(torch.sum(p_mask).item()))

def test_init_float_multiple_types(memory_set):
    """Test init_float function with multiple node types."""
    # Get masks for different types
    p_mask = memory_set.get_mask(Type.P)
    rb_mask = memory_set.get_mask(Type.RB)
    
    if not torch.any(p_mask) or not torch.any(rb_mask):
        pytest.skip("Need both P and RB nodes for this test")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[rb_mask, TF.ACT] = 0.7
    
    # Call init_float with multiple types
    memory_set.update_ops.init_float([Type.P, Type.RB], [TF.ACT])
    
    # Check that values were set to 0.0 for both types
    assert torch.allclose(memory_set.nodes[p_mask, TF.ACT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[rb_mask, TF.ACT], torch.zeros(torch.sum(rb_mask).item()))

def test_init_input(memory_set):
    """Test init_input function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.5
    memory_set.nodes[p_mask, TF.BU_INPUT] = 0.3
    memory_set.nodes[p_mask, TF.LATERAL_INPUT] = 0.2
    memory_set.nodes[p_mask, TF.MAP_INPUT] = 0.1
    memory_set.nodes[p_mask, TF.NET_INPUT] = 0.4
    
    # Call init_input with refresh value
    refresh_value = 0.8
    memory_set.update_ops.init_input([Type.P], refresh_value)
    
    # Check that td_input was set to refresh value
    assert torch.allclose(memory_set.nodes[p_mask, TF.TD_INPUT], 
                         torch.full((torch.sum(p_mask).item(),), refresh_value))
    
    # Check that other inputs were set to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.BU_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.LATERAL_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.MAP_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.NET_INPUT], torch.zeros(torch.sum(p_mask).item()))

def test_init_act(memory_set):
    """Test init_act function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.3
    memory_set.nodes[p_mask, TF.BU_INPUT] = 0.2
    
    # Call init_act
    memory_set.update_ops.init_act([Type.P])
    
    # Check that act was set to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.ACT], torch.zeros(torch.sum(p_mask).item()))
    
    # Check that inputs were initialized (td_input = 0.0, others = 0.0)
    assert torch.allclose(memory_set.nodes[p_mask, TF.TD_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.BU_INPUT], torch.zeros(torch.sum(p_mask).item()))

def test_init_state(memory_set):
    """Test init_state function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[p_mask, TF.RETRIEVED] = B.TRUE
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.3
    
    # Call init_state
    memory_set.update_ops.init_state([Type.P])
    
    # Check that act was set to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.ACT], torch.zeros(torch.sum(p_mask).item()))
    
    # Check that retrieved was set to 0.0 (False)
    assert torch.allclose(memory_set.nodes[p_mask, TF.RETRIEVED], torch.zeros(torch.sum(p_mask).item()))
    
    # Check that inputs were initialized
    assert torch.allclose(memory_set.nodes[p_mask, TF.TD_INPUT], torch.zeros(torch.sum(p_mask).item()))

def test_update_act(memory_set):
    """Test update_act function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set initial activations and inputs
    memory_set.nodes[p_mask, TF.ACT] = 0.2
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.3
    memory_set.nodes[p_mask, TF.BU_INPUT] = 0.1
    memory_set.nodes[p_mask, TF.LATERAL_INPUT] = -0.1
    memory_set.nodes[p_mask, TF.MAP_INPUT] = 0.2
    
    # Store initial activations
    initial_acts = memory_set.nodes[p_mask, TF.ACT].clone()
    
    # Call update_act
    memory_set.update_ops.update_act()
    
    # Check that activations have changed
    final_acts = memory_set.nodes[p_mask, TF.ACT]
    assert not torch.allclose(final_acts, initial_acts)
    
    # Check that activations are within valid range [0.0, 1.0]
    assert torch.all(final_acts >= 0.0)
    assert torch.all(final_acts <= 1.0)

def test_update_act_with_clipping(memory_set):
    """Test update_act function with values that should be clipped."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set very high inputs to test clipping
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[p_mask, TF.TD_INPUT] = 10.0  # Very high input
    memory_set.nodes[p_mask, TF.BU_INPUT] = 0.0
    memory_set.nodes[p_mask, TF.LATERAL_INPUT] = 0.0
    memory_set.nodes[p_mask, TF.MAP_INPUT] = 0.0
    
    # Call update_act multiple times to ensure clipping works
    for _ in range(10):
        memory_set.update_ops.update_act()
    
    # Check that activations are clipped to [0.0, 1.0]
    final_acts = memory_set.nodes[p_mask, TF.ACT]
    assert torch.all(final_acts >= 0.0)
    assert torch.all(final_acts <= 1.0)

def test_zero_lateral_input(memory_set):
    """Test zero_lateral_input function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial lateral input
    memory_set.nodes[p_mask, TF.LATERAL_INPUT] = 0.5
    
    # Call zero_lateral_input
    memory_set.update_ops.zero_lateral_input([Type.P])
    
    # Check that lateral input was set to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.LATERAL_INPUT], torch.zeros(torch.sum(p_mask).item()))

def test_update_inhibitor_input(memory_set):
    """Test update_inhibitor_input function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.3
    memory_set.nodes[p_mask, TF.INHIBITOR_INPUT] = 0.1
    
    # Store initial inhibitor input
    initial_inhibitor_input = memory_set.nodes[p_mask, TF.INHIBITOR_INPUT].clone()
    
    # Call update_inhibitor_input
    memory_set.update_ops.update_inhibitor_input([Type.P])
    
    # Check that inhibitor input was increased by activation values
    expected_inhibitor_input = initial_inhibitor_input + memory_set.nodes[p_mask, TF.ACT]
    assert torch.allclose(memory_set.nodes[p_mask, TF.INHIBITOR_INPUT], expected_inhibitor_input)

def test_reset_inhibitor(memory_set):
    """Test reset_inhibitor function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial values
    memory_set.nodes[p_mask, TF.INHIBITOR_INPUT] = 0.5
    memory_set.nodes[p_mask, TF.INHIBITOR_ACT] = 0.8
    
    # Call reset_inhibitor
    memory_set.update_ops.reset_inhibitor([Type.P])
    
    # Check that inhibitor values were reset to 0.0
    assert torch.allclose(memory_set.nodes[p_mask, TF.INHIBITOR_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.INHIBITOR_ACT], torch.zeros(torch.sum(p_mask).item()))

def test_update_inhibitor_act(memory_set):
    """Test update_inhibitor_act function."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set up test values
    memory_set.nodes[p_mask, TF.INHIBITOR_INPUT] = 0.8
    memory_set.nodes[p_mask, TF.INHIBITOR_THRESHOLD] = 0.5
    memory_set.nodes[p_mask, TF.INHIBITOR_ACT] = 0.0
    
    # Call update_inhibitor_act
    memory_set.update_ops.update_inhibitor_act([Type.P])
    
    # Check that inhibitor act was set to 1.0 where input >= threshold
    # Since input (0.8) >= threshold (0.5), inhibitor_act should be 1.0
    # Use the actual mask to get the correct values
    p_indices = torch.where(p_mask)[0]
    assert torch.allclose(memory_set.nodes[p_indices, TF.INHIBITOR_ACT], torch.ones(len(p_indices)))

def test_update_inhibitor_act_below_threshold(memory_set):
    """Test update_inhibitor_act function with input below threshold."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set up test values where input < threshold
    memory_set.nodes[p_mask, TF.INHIBITOR_INPUT] = 0.3
    memory_set.nodes[p_mask, TF.INHIBITOR_THRESHOLD] = 0.5
    memory_set.nodes[p_mask, TF.INHIBITOR_ACT] = 0.0
    
    # Call update_inhibitor_act
    memory_set.update_ops.update_inhibitor_act([Type.P])
    
    # Check that inhibitor act remains 0.0 where input < threshold
    # Use the actual mask to get the correct values
    p_indices = torch.where(p_mask)[0]
    assert torch.allclose(memory_set.nodes[p_indices, TF.INHIBITOR_ACT], torch.zeros(len(p_indices)))

def test_p_initialise_mode(memory_set):
    """Test p_initialise_mode function."""
    # Get P nodes
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set some initial mode values
    memory_set.nodes[p_mask, TF.MODE] = Mode.PARENT
    
    # Call p_initialise_mode
    memory_set.update_ops.p_initialise_mode()
    
    # Check that all P nodes have mode set to NEUTRAL
    # Use the actual mask to get the correct values
    p_indices = torch.where(p_mask)[0]
    assert torch.allclose(memory_set.nodes[p_indices, TF.MODE], 
                         torch.full((len(p_indices),), Mode.NEUTRAL, dtype=torch.float))

def test_p_get_mode(memory_set):
    """Test p_get_mode function."""
    # Get P and RB nodes
    p_mask = memory_set.get_mask(Type.P)
    rb_mask = memory_set.get_mask(Type.RB)
    
    if not torch.any(p_mask) or not torch.any(rb_mask):
        pytest.skip("Need both P and RB nodes for this test")
    
    # Set up some connections and activations
    # Create a simple connection pattern: P[0] -> RB[0], RB[1] -> P[1]
    if torch.sum(p_mask).item() >= 2 and torch.sum(rb_mask).item() >= 2:
        p_indices = torch.where(p_mask)[0]
        rb_indices = torch.where(rb_mask)[0]
        
        # Set up connections: P[0] -> RB[0], RB[1] -> P[1]
        memory_set.connections[p_indices[0], rb_indices[0]] = 1.0
        memory_set.connections[rb_indices[1], p_indices[1]] = 1.0
        
        # Set activations: RB[0] = 0.8, RB[1] = 0.2
        memory_set.nodes[rb_indices[0], TF.ACT] = 0.8
        memory_set.nodes[rb_indices[1], TF.ACT] = 0.2
        
        # Call p_get_mode
        memory_set.update_ops.p_get_mode()
        
        # Check that modes were set appropriately
        # P[0] should be in PARENT mode (child RB[0] has higher activation)
        # P[1] should be in CHILD mode (parent RB[1] has lower activation)
        assert memory_set.nodes[p_indices[0], TF.MODE] == Mode.PARENT
        assert memory_set.nodes[p_indices[1], TF.MODE] == Mode.CHILD

def test_po_get_weight_length(memory_set):
    """Test po_get_weight_length function."""
    # Get PO nodes
    po_mask = memory_set.get_mask(Type.PO)
    if not torch.any(po_mask):
        pytest.skip("No PO nodes in memory set")
    
    # Check if links are available
    if memory_set.links is None:
        pytest.skip("Links not available for this test")
    
    # Call po_get_weight_length
    memory_set.update_ops.po_get_weight_length()
    
    # Check that SEM_COUNT was set (should be non-negative)
    # Use the actual mask to get the correct values
    po_indices = torch.where(po_mask)[0]
    sem_counts = memory_set.nodes[po_indices, TF.SEM_COUNT]
    assert torch.all(sem_counts >= 0.0)

def test_po_get_max_semantic_weight(memory_set):
    """Test po_get_max_semantic_weight function."""
    # Get PO nodes
    po_mask = memory_set.get_mask(Type.PO)
    if not torch.any(po_mask):
        pytest.skip("No PO nodes in memory set")
    
    # Check if links are available
    if memory_set.links is None:
        pytest.skip("Links not available for this test")
    
    # Call po_get_max_semantic_weight
    memory_set.update_ops.po_get_max_semantic_weight()
    
    # Check that MAX_SEM_WEIGHT was set (should be non-negative)
    # Use the actual mask to get the correct values
    po_indices = torch.where(po_mask)[0]
    max_weights = memory_set.nodes[po_indices, TF.MAX_SEM_WEIGHT]
    assert torch.all(max_weights >= 0.0)

def test_po_functions_without_links(memory_set):
    """Test PO functions raise ValueError when links are not available."""
    # Temporarily set links to None
    original_links = memory_set.links
    memory_set.links = None
    
    try:
        # Test po_get_weight_length
        with pytest.raises(ValueError, match="Links is not initialised"):
            memory_set.update_ops.po_get_weight_length()
        
        # Test po_get_max_semantic_weight
        with pytest.raises(ValueError, match="Links is not initialised"):
            memory_set.update_ops.po_get_max_semantic_weight()
    
    finally:
        # Restore original links
        memory_set.links = original_links

def test_update_ops_parameter_dependencies(memory_set):
    """Test that update operations depend on parameters correctly."""
    # Store original params
    original_params = memory_set.params
    
    # Test that update_act can access params
    try:
        memory_set.update_ops.update_act()
    except AttributeError as e:
        if "params" in str(e):
            pytest.fail(f"Update operations should handle missing params gracefully: {e}")
        else:
            raise  # Re-raise if it's a different AttributeError

def test_update_ops_with_different_sets(network):
    """Test update operations work with different sets."""
    memory = network.memory()
    driver = network.driver()
    
    # Test that update operations work on both sets
    for test_set in [memory, driver]:
        p_mask = test_set.get_mask(Type.P)
        if torch.any(p_mask):
            # Test basic operations
            test_set.update_ops.init_float([Type.P], [TF.ACT])
            test_set.update_ops.init_input([Type.P], 0.5)
            
            # Verify init operations worked
            p_indices = torch.where(p_mask)[0]
            assert torch.allclose(test_set.nodes[p_indices, TF.ACT], torch.zeros(len(p_indices)))
            assert torch.allclose(test_set.nodes[p_indices, TF.TD_INPUT], 
                                 torch.full((len(p_indices),), 0.5))
            
            # Test update_act (this might change values due to other inputs)
            test_set.update_ops.update_act()
            
            # Just verify the function ran without error
            assert True

def test_update_ops_chain_operations(memory_set):
    """Test chaining multiple update operations."""
    # Get some nodes to test with
    p_mask = memory_set.get_mask(Type.P)
    if not torch.any(p_mask):
        pytest.skip("No P nodes in memory set")
    
    # Set initial values
    memory_set.nodes[p_mask, TF.ACT] = 0.5
    memory_set.nodes[p_mask, TF.TD_INPUT] = 0.3
    memory_set.nodes[p_mask, TF.LATERAL_INPUT] = 0.2
    
    # Chain multiple operations
    memory_set.update_ops.init_state([Type.P])  # This calls init_act and init_float
    memory_set.update_ops.zero_lateral_input([Type.P])
    memory_set.update_ops.update_act()
    
    # Verify final state
    assert torch.allclose(memory_set.nodes[p_mask, TF.ACT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.LATERAL_INPUT], torch.zeros(torch.sum(p_mask).item()))
    assert torch.allclose(memory_set.nodes[p_mask, TF.RETRIEVED], torch.zeros(torch.sum(p_mask).item()))

def test_update_act(memory_set: 'Base_Set'):
    """Test update_act function."""
    # check number of nodes
    assert torch.any(memory_set.tensor_op.get_all_nodes_mask())
    # set some initial values
    memory_set.nodes[:5, TF.ACT] = torch.tensor([0.1, 0.5, 0.8, 0.3, 0.9])
    memory_set.nodes[:5, TF.ANALOG] = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0])
    memory_set.nodes[:5, TF.DELETED] = B.FALSE
    memory_set.tensor_ops.cache_masks()
    # call update_act
    memory_set.update_ops.update_act()
    # check that activations have changed
    assert not torch.allclose(memory_set.nodes[:5, TF.ACT], torch.tensor([0.1, 0.5, 0.8, 0.3, 0.9]))
    # check for nan values
    assert not torch.any(torch.isnan(memory_set.nodes[:5, TF.ACT]))
    # check for inf values
    assert not torch.any(torch.isinf(memory_set.nodes[:5, TF.ACT]))
    # check for values out of range
    assert torch.all(memory_set.nodes[:5, TF.ACT] >= 0.0)
    assert torch.all(memory_set.nodes[:5, TF.ACT] <= 1.0)

def test_update_input_and_act(memory_set: 'Memory'):
    """Test update_input and update_act function."""
    # check number of nodes
    assert torch.any(memory_set.tensor_op.get_all_nodes_mask())
    # set some initial values
    memory_set.nodes[:5, TF.ACT] = torch.tensor([0.1, 0.5, 0.8, 0.3, 0.9])
    memory_set.nodes[:5, TF.ANALOG] = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0])
    memory_set.nodes[:5, TF.DELETED] = B.FALSE
    memory_set.tensor_ops.cache_masks()
    # call update_input, then update_act
    memory_set.update_input()
    memory_set.update_ops.update_act()
    # check that activations have changed
    assert not torch.allclose(memory_set.nodes[:5, TF.ACT], torch.tensor([0.1, 0.5, 0.8, 0.3, 0.9]))
    # check for nan values
    assert not torch.any(torch.isnan(memory_set.nodes[:5, TF.ACT]))
    # check for inf values
    assert not torch.any(torch.isinf(memory_set.nodes[:5, TF.ACT]))
    # check for values out of range
    assert torch.all(memory_set.nodes[:5, TF.ACT] >= 0.0)
    assert torch.all(memory_set.nodes[:5, TF.ACT] <= 1.0)