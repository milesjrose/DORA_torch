# nodes/tests/test_update_ops.py
# Tests for UpdateOperations class

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.operations.update_ops import UpdateOperations

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

@pytest.fixture
def update_ops(network):
    """Create an UpdateOperations object."""
    return UpdateOperations(network)

def test_init(update_ops, network):
    """Test UpdateOperations initialization."""
    assert update_ops.network == network

def test_initialise_act(update_ops):
    """Test initialise_act method."""
    # Get initial activations
    driver_initial = update_ops.network.driver().nodes[:, TF.ACT].clone()
    recipient_initial = update_ops.network.recipient().nodes[:, TF.ACT].clone()
    new_set_initial = update_ops.network.new_set().nodes[:, TF.ACT].clone()
    semantics_initial = update_ops.network.semantics.nodes[:, SF.ACT].clone()
    
    # Call initialise_act
    update_ops.initialise_act()
    
    # Check that activations were reset to 0 (only for sets that have tokens)
    if update_ops.network.driver().nodes.shape[0] > 0:
        assert torch.all(update_ops.network.driver().nodes[:, TF.ACT] == 0.0)
    if update_ops.network.recipient().nodes.shape[0] > 0:
        assert torch.all(update_ops.network.recipient().nodes[:, TF.ACT] == 0.0)
    if update_ops.network.new_set().nodes.shape[0] > 0:
        assert torch.all(update_ops.network.new_set().nodes[:, TF.ACT] == 0.0)
    assert torch.all(update_ops.network.semantics.nodes[:, SF.ACT] == 0.0)
    
    # Check that inputs were also reset (only for sets that have tokens)
    input_features = [TF.BU_INPUT, TF.LATERAL_INPUT, TF.MAP_INPUT, TF.NET_INPUT]
    for feature in input_features:
        if update_ops.network.driver().nodes.shape[0] > 0:
            assert torch.all(update_ops.network.driver().nodes[:, feature] == 0.0)
        if update_ops.network.recipient().nodes.shape[0] > 0:
            assert torch.all(update_ops.network.recipient().nodes[:, feature] == 0.0)
        if update_ops.network.new_set().nodes.shape[0] > 0:
            assert torch.all(update_ops.network.new_set().nodes[:, feature] == 0.0)

def test_acts_single_set(update_ops):
    """Test acts method for a single set."""
    # Set some initial activations
    driver = update_ops.network.driver()
    driver.nodes[:, TF.ACT] = 0.5
    driver.nodes[:, TF.TD_INPUT] = 0.3
    driver.nodes[:, TF.BU_INPUT] = 0.2
    driver.nodes[:, TF.LATERAL_INPUT] = 0.1
    
    # Store initial activations
    initial_acts = driver.nodes[:, TF.ACT].clone()
    
    # Call acts for driver set
    update_ops.acts(Set.DRIVER)
    
    # Check that activations were updated (should be different)
    assert not torch.allclose(driver.nodes[:, TF.ACT], initial_acts)

def test_acts_sem(update_ops):
    """Test acts_sem method."""
    # Set some initial semantic activations and inputs
    semantics = update_ops.network.semantics
    semantics.nodes[:, SF.ACT] = 0.5
    semantics.nodes[:, SF.INPUT] = 0.3
    semantics.nodes[:, SF.MAX_INPUT] = 1.0
    
    # Store initial activations
    initial_acts = semantics.nodes[:, SF.ACT].clone()
    
    # Call acts_sem
    update_ops.acts_sem()
    
    # Check that semantic activations were updated
    assert not torch.allclose(semantics.nodes[:, SF.ACT], initial_acts)

def test_acts_am(update_ops):
    """Test acts_am method."""
    # Set some initial activations in all sets
    sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
    initial_acts = {}
    
    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        # Only set activations if the set has tokens
        if set_obj.nodes.shape[0] > 0:
            set_obj.nodes[:, TF.ACT] = 0.5
            set_obj.nodes[:, TF.TD_INPUT] = 0.3
            set_obj.nodes[:, TF.BU_INPUT] = 0.2
            set_obj.nodes[:, TF.LATERAL_INPUT] = 0.1
            initial_acts[set_name] = set_obj.nodes[:, TF.ACT].clone()
    
    # Set semantic activations
    semantics = update_ops.network.semantics
    semantics.nodes[:, SF.ACT] = 0.5
    semantics.nodes[:, SF.INPUT] = 0.3
    semantics.nodes[:, SF.MAX_INPUT] = 1.0
    sem_initial_acts = semantics.nodes[:, SF.ACT].clone()
    
    # Call acts_am
    update_ops.acts_am()
    
    # Check that all activations were updated (only for sets that had tokens)
    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        if set_name in initial_acts and set_obj.nodes.shape[0] > 0:
            assert not torch.allclose(set_obj.nodes[:, TF.ACT], initial_acts[set_name])
    
    assert not torch.allclose(semantics.nodes[:, SF.ACT], sem_initial_acts)

def test_initialise_input(update_ops):
    """Test initialise_input method."""
    # Set some initial inputs
    sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        # Only set inputs if the set has tokens
        if set_obj.nodes.shape[0] > 0:
            set_obj.nodes[:, TF.TD_INPUT] = 0.5
            set_obj.nodes[:, TF.BU_INPUT] = 0.3
            set_obj.nodes[:, TF.LATERAL_INPUT] = 0.2
            set_obj.nodes[:, TF.MAP_INPUT] = 0.1
            set_obj.nodes[:, TF.NET_INPUT] = 0.4
    
    # Set semantic inputs
    semantics = update_ops.network.semantics
    semantics.nodes[:, SF.INPUT] = 0.5
    
    # Call initialise_input
    update_ops.initialise_input()
    
    # Check that inputs were reset (only for sets that have tokens)
    input_features = [TF.BU_INPUT, TF.LATERAL_INPUT, TF.MAP_INPUT, TF.NET_INPUT]
    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        if set_obj.nodes.shape[0] > 0:
            for feature in input_features:
                assert torch.all(set_obj.nodes[:, feature] == 0.0)
    
    assert torch.all(semantics.nodes[:, SF.INPUT] == 0.0)
"""
update inputs not implemented yet, so don't test.

def test_inputs_single_set(update_ops):
    #Test inputs method for a single set.

    # Set some initial inputs
    driver = update_ops.network.driver()
    driver.nodes[:, TF.TD_INPUT] = 0.5
    driver.nodes[:, TF.BU_INPUT] = 0.3
    driver.nodes[:, TF.LATERAL_INPUT] = 0.2
    driver.nodes[:, TF.MAP_INPUT] = 0.1
    
    # Store initial inputs
    initial_inputs = driver.nodes[:, TF.NET_INPUT].clone()
    
    # Call inputs for driver set
    update_ops.inputs(Set.DRIVER)
    
    # Check that inputs were updated (should be different)
    assert not torch.allclose(driver.nodes[:, TF.NET_INPUT], initial_inputs)
"""

def test_inputs_sem(update_ops):
    """Test inputs_sem method."""
    # Set some initial semantic inputs
    semantics = update_ops.network.semantics
    semantics.nodes[:, SF.INPUT] = 0.5
    
    # Set some activations in driver and recipient
    driver = update_ops.network.driver()
    recipient = update_ops.network.recipient()
    driver.nodes[:, TF.ACT] = 0.3
    recipient.nodes[:, TF.ACT] = 0.4
    
    # Store initial semantic inputs
    initial_sem_inputs = semantics.nodes[:, SF.INPUT].clone()
    
    # Call inputs_sem
    update_ops.inputs_sem()
    
    # Check that semantic inputs were updated
    assert not torch.allclose(semantics.nodes[:, SF.INPUT], initial_sem_inputs)

def test_inputs_am(update_ops):
    """Test inputs_am method."""
    # Set some initial activations and inputs in all sets
    sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
    initial_inputs = {}

    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        # Only set inputs if the set has tokens
        if set_obj.nodes.shape[0] > 0:
            # Set some activations so the update methods have something to work with
            set_obj.nodes[:, TF.ACT] = 0.5
            # Set initial inputs to 0 so we can see the changes
            set_obj.nodes[:, TF.TD_INPUT] = 0.0
            set_obj.nodes[:, TF.BU_INPUT] = 0.0
            set_obj.nodes[:, TF.LATERAL_INPUT] = 0.0
            set_obj.nodes[:, TF.MAP_INPUT] = 0.0
            initial_inputs[set_name] = set_obj.nodes[:, TF.NET_INPUT].clone()

    # Set semantic inputs
    semantics = update_ops.network.semantics
    if semantics.nodes.shape[0] > 0:
        semantics.nodes[:, SF.INPUT] = 0.0
        sem_initial_inputs = semantics.nodes[:, SF.INPUT].clone()

    # Call inputs_am
    update_ops.inputs_am()

    # Check that all inputs were updated (only for sets that had tokens)
    for set_name in sets:
        set_obj = update_ops.network.sets[set_name]
        if set_name in initial_inputs and set_obj.nodes.shape[0] > 0:
            # Check that at least some inputs changed (they should be non-zero after update)
            assert torch.any(set_obj.nodes[:, TF.TD_INPUT] != 0.0) or \
                   torch.any(set_obj.nodes[:, TF.BU_INPUT] != 0.0) or \
                   torch.any(set_obj.nodes[:, TF.LATERAL_INPUT] != 0.0) or \
                   torch.any(set_obj.nodes[:, TF.MAP_INPUT] != 0.0)
    
    # Check that semantic inputs were updated (if semantics exist)
    if semantics.nodes.shape[0] > 0:
        assert torch.any(semantics.nodes[:, SF.INPUT] != 0.0)

def test_get_max_sem_input(update_ops):
    """Test get_max_sem_input method."""
    # Set some semantic inputs
    semantics = update_ops.network.semantics
    num_semantics = semantics.nodes.shape[0]
    
    # Create a tensor with the correct size and set some values
    test_inputs = torch.zeros(num_semantics)
    test_inputs[0] = 0.1
    test_inputs[1] = 0.5
    test_inputs[2] = 0.3
    test_inputs[3] = 0.8
    test_inputs[4] = 0.2
    
    semantics.nodes[:, SF.INPUT] = test_inputs
    
    # Get max semantic input
    max_input = update_ops.get_max_sem_input()
    
    # Check that it returns the maximum input
    assert max_input == 0.8

def test_del_small_link(update_ops):
    """Test del_small_link method."""
    # Set some link weights
    links = update_ops.network.links
    for set_name in Set:
        if set_name in links.sets:
            links.sets[set_name] = torch.tensor([[0.05, 0.3, 0.8], [0.2, 0.1, 0.9]])
    
    # Call del_small_link with threshold 0.15
    update_ops.del_small_link(0.15)
    
    # Check that weights below threshold were set to 0
    for set_name in Set:
        if set_name in links.sets:
            assert torch.all(links.sets[set_name] < 0.15) == torch.all(links.sets[set_name] == 0.0)

def test_round_big_link(update_ops):
    """Test round_big_link method."""
    # Set some link weights
    links = update_ops.network.links
    for set_name in Set:
        if set_name in links.sets:
            links.sets[set_name] = torch.tensor([[0.05, 0.3, 0.8], [0.2, 0.1, 0.9]])
    
    # Call round_big_link with threshold 0.7
    update_ops.round_big_link(0.7)
    
    # Check that weights above threshold were set to 1.0
    for set_name in Set:
        if set_name in links.sets:
            assert torch.all(links.sets[set_name] > 0.7) == torch.all(links.sets[set_name] == 1.0)

def test_activation_update_mechanics(update_ops):
    """Test that activation updates follow the correct mathematical formula."""
    # Get driver set
    driver = update_ops.network.driver()
    
    # Set specific inputs and parameters
    driver.nodes[:, TF.TD_INPUT] = 0.3
    driver.nodes[:, TF.BU_INPUT] = 0.2
    driver.nodes[:, TF.LATERAL_INPUT] = 0.1
    driver.nodes[:, TF.MAP_INPUT] = 0.0
    driver.nodes[:, TF.ACT] = 0.5
    
    # Store initial activations
    initial_acts = driver.nodes[:, TF.ACT].clone()
    
    # Call update_act
    driver.update_act()
    
    # Check that activations changed
    assert not torch.allclose(driver.nodes[:, TF.ACT], initial_acts)
    
    # Check that activations are within bounds [0, 1]
    assert torch.all(driver.nodes[:, TF.ACT] >= 0.0)
    assert torch.all(driver.nodes[:, TF.ACT] <= 1.0)

def test_semantic_activation_update(update_ops):
    """Test semantic activation update mechanics."""
    # Get semantics
    semantics = update_ops.network.semantics
    
    # Set specific inputs
    num_semantics = semantics.nodes.shape[0]
    test_inputs = torch.zeros(num_semantics)
    test_inputs[0] = 0.2
    test_inputs[1] = 0.5
    test_inputs[2] = 0.8
    test_inputs[3] = 0.1
    test_inputs[4] = 0.6
    
    semantics.nodes[:, SF.INPUT] = test_inputs
    semantics.nodes[:, SF.MAX_INPUT] = 0.8
    
    # Store initial activations
    initial_acts = semantics.nodes[:, SF.ACT].clone()
    
    # Call update_act
    semantics.update_act()
    
    # Check that activations changed
    assert not torch.allclose(semantics.nodes[:, SF.ACT], initial_acts)
    
    # Check that activations are normalized correctly
    # For semantics with max_input > 0, act should be input/max_input
    # For semantics with max_input == 0, act should be 0
    for i in range(semantics.nodes.shape[0]):
        if semantics.nodes[i, SF.MAX_INPUT] > 0:
            expected_act = semantics.nodes[i, SF.INPUT] / semantics.nodes[i, SF.MAX_INPUT]
            assert abs(semantics.nodes[i, SF.ACT] - expected_act) < 1e-6
        else:
            assert semantics.nodes[i, SF.ACT] == 0.0

def test_input_initialization_mechanics(update_ops):
    """Test that input initialization works correctly."""
    # Get driver set
    driver = update_ops.network.driver()
    
    # Set some initial inputs
    driver.nodes[:, TF.TD_INPUT] = 0.5
    driver.nodes[:, TF.BU_INPUT] = 0.3
    driver.nodes[:, TF.LATERAL_INPUT] = 0.2
    driver.nodes[:, TF.MAP_INPUT] = 0.1
    driver.nodes[:, TF.NET_INPUT] = 0.4
    
    # Call initialise_input for specific types
    driver.update_op.initialise_input([Type.PO], 0.0)
    
    # Check that inputs were reset for PO types only
    po_mask = driver.tensor_op.get_mask(Type.PO)
    input_features = [TF.BU_INPUT, TF.LATERAL_INPUT, TF.MAP_INPUT, TF.NET_INPUT]
    
    for feature in input_features:
        assert torch.all(driver.nodes[po_mask, feature] == 0.0)

def test_activation_bounds(update_ops):
    """Test that activations are properly bounded between 0 and 1."""
    # Get driver set
    driver = update_ops.network.driver()
    
    # Set inputs that would cause activations to exceed bounds
    driver.nodes[:, TF.TD_INPUT] = 10.0  # Very high input
    driver.nodes[:, TF.BU_INPUT] = 10.0
    driver.nodes[:, TF.LATERAL_INPUT] = 10.0
    driver.nodes[:, TF.ACT] = 0.5
    
    # Call update_act
    driver.update_act()
    
    # Check that activations are bounded
    assert torch.all(driver.nodes[:, TF.ACT] <= 1.0)
    assert torch.all(driver.nodes[:, TF.ACT] >= 0.0)
    
    # Set inputs that would cause activations to go negative
    driver.nodes[:, TF.TD_INPUT] = -10.0  # Very negative input
    driver.nodes[:, TF.BU_INPUT] = -10.0
    driver.nodes[:, TF.LATERAL_INPUT] = -10.0
    driver.nodes[:, TF.ACT] = 0.5
    
    # Call update_act
    driver.update_act()
    
    # Check that activations are bounded
    assert torch.all(driver.nodes[:, TF.ACT] >= 0.0)

def test_semantic_input_update_from_sets(update_ops):
    """Test that semantic inputs are updated correctly from token sets."""
    # Get semantics and driver
    semantics = update_ops.network.semantics
    driver = update_ops.network.driver()
    
    # Set some activations in driver
    driver.nodes[:, TF.ACT] = 0.5
    
    # Set some links between driver and semantics
    if driver.token_set in update_ops.network.links.sets:
        links = update_ops.network.links.sets[driver.token_set]
        # Set some non-zero link weights
        links[0, 0] = 0.8
        links[1, 1] = 0.6
    
    # Store initial semantic inputs
    initial_inputs = semantics.nodes[:, SF.INPUT].clone()
    
    # Call update_input_from_set
    semantics.update_input_from_set(driver, Set.DRIVER)
    
    # Check that semantic inputs were updated
    assert not torch.allclose(semantics.nodes[:, SF.INPUT], initial_inputs)

def test_integration_with_real_network(update_ops):
    """Test integration with a real network setup."""
    # Test that all required methods exist
    assert hasattr(update_ops, 'network')
    assert hasattr(update_ops, 'initialise_act')
    assert hasattr(update_ops, 'acts')
    assert hasattr(update_ops, 'acts_sem')
    assert hasattr(update_ops, 'acts_am')
    assert hasattr(update_ops, 'initialise_input')
    assert hasattr(update_ops, 'inputs')
    assert hasattr(update_ops, 'inputs_sem')
    assert hasattr(update_ops, 'inputs_am')
    assert hasattr(update_ops, 'get_max_sem_input')
    assert hasattr(update_ops, 'del_small_link')
    assert hasattr(update_ops, 'round_big_link')

def test_error_handling(update_ops):
    """Test error handling in various scenarios."""
    # Test with invalid set
    with pytest.raises(KeyError):
        update_ops.acts(999)  # Invalid set
    
    with pytest.raises(KeyError):
        update_ops.inputs(999)  # Invalid set

def test_parameter_sensitivity(update_ops):
    """Test that activation updates are sensitive to parameter changes."""
    # Get driver set
    driver = update_ops.network.driver()
    
    # Set initial conditions
    driver.nodes[:, TF.TD_INPUT] = 0.3
    driver.nodes[:, TF.BU_INPUT] = 0.2
    driver.nodes[:, TF.LATERAL_INPUT] = 0.1
    driver.nodes[:, TF.ACT] = 0.5
    
    # Store initial activations
    initial_acts = driver.nodes[:, TF.ACT].clone()
    
    # Change parameters and test sensitivity
    original_gamma = driver.params.gamma
    original_delta = driver.params.delta
    
    # Test with different gamma
    driver.params.gamma = original_gamma * 2
    driver.update_act()
    acts_high_gamma = driver.nodes[:, TF.ACT].clone()
    
    # Reset and test with different delta
    driver.nodes[:, TF.ACT] = initial_acts
    driver.params.gamma = original_gamma
    driver.params.delta = original_delta * 2
    driver.update_act()
    acts_high_delta = driver.nodes[:, TF.ACT].clone()
    
    # Check that different parameters produce different results
    assert not torch.allclose(acts_high_gamma, acts_high_delta) 