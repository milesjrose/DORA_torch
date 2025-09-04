# nodes/tests/test_node_ops.py
# Tests for node operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import Set, TF, Type, B

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_get_most_active_token(network):
    """Test getting the most active token."""
    # work with DRIVER set
    driver_set = network.sets[Set.DRIVER]
    
    # get all non-deleted tokens in the driver set
    driver_mask = driver_set.tensor_op.get_all_nodes_mask()
    driver_indices = torch.where(driver_mask)[0]
    
    # ensure we have tokens to work with
    assert len(driver_indices) > 0, "No tokens in DRIVER set to test."
    
    # set all activations to 0
    driver_set.nodes[driver_indices, TF.ACT] = 0.0
    
    # pick a token to be the most active
    target_index = driver_indices[0]
    target_id = driver_set.nodes[target_index, TF.ID].item()
    
    # set its activation to a high value
    high_activation_value = 0.9
    driver_set.nodes[target_index, TF.ACT] = high_activation_value
    
    masks = {Set.DRIVER: driver_mask}
    
    # Test with id=False
    most_active_tokens = network.node_ops.get_most_active_token(masks, id=False)
    
    assert Set.DRIVER in most_active_tokens
    
    retrieved_token_ref = most_active_tokens[Set.DRIVER]
    
    assert retrieved_token_ref.ID == target_id, "The retrieved token ID does not match the most active token's ID."

    # Test with id=True
    most_active_tokens_ids = network.node_ops.get_most_active_token(masks, id=True)
    assert isinstance(most_active_tokens_ids, list)
    retrieved_id = most_active_tokens_ids[0]

    assert retrieved_id == target_id, "The retrieved ID should be the ID of the most active token."
