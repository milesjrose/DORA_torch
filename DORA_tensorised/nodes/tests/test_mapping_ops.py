# nodes/tests/test_mappings.py
# Tests the mappings class.

import pytest
import torch

from ..enums import MappingFields, Set, Type, TF
from ..builder import NetworkBuilder
from ..network import Network, Token


# Import the symProps from sim.py
from .sims.sim import symProps


@pytest.fixture
def network():
    builder = NetworkBuilder(symProps=symProps)
    net = builder.build_network()
    # Ensure all sets have some tokens for testing
    for s_enum in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY]:
        s = net.sets[s_enum]
        while s.nodes.shape[0] < 3:
            s.tensor_ops.add_token(Token(Type.PO))
    return net

def test_get_max_maps_updates_all_sets_correctly(network: Network):
    """
    Tests that get_max_maps in mapping_ops correctly updates the max_map feature for all sets.
    - Updates "from" sets (recipient, memory) based on their outgoing connections.
    - Updates "to" set (driver) based on its strongest incoming connection from ANY source.
    """
    driver = network.driver()
    recipient = network.recipient()
    memory = network.memory()
    
    # --- Setup Recipient -> Driver Weights ---
    rec_mappings = network.mappings[Set.RECIPIENT]
    rec_weights = rec_mappings[MappingFields.WEIGHT]
    # zero out all weights
    rec_weights.fill_(0)
    # set some weights
    rec_weights[0, 1] = 0.8
    rec_weights[1, 2] = 0.5
    rec_weights[2, 0] = 0.2

    # --- Setup Memory -> Driver Weights ---
    mem_mappings = network.mappings[Set.MEMORY]
    mem_weights = mem_mappings[MappingFields.WEIGHT]
    # zero out all weights
    mem_weights.fill_(0)
    # set some weights
    mem_weights[0, 1] = 0.7
    mem_weights[1, 2] = 0.9
    mem_weights[2, 0] = 0.3

    # --- Run the Operation ---
    network.mapping_ops.get_max_maps(Set.RECIPIENT)
    network.mapping_ops.get_max_maps(Set.MEMORY)
    network.mapping_ops.get_max_maps(Set.DRIVER)

    # --- Assertions ---

    # 1. Check Recipient 
    expected_recipient_max = torch.zeros([1, rec_weights.shape[0]])
    expected_recipient_max[0, 0] = 0.8
    expected_recipient_max[0, 1] = 0.5
    expected_recipient_max[0, 2] = 0.2
    assert torch.allclose(recipient.nodes[:, TF.MAX_MAP], expected_recipient_max)
    
    # 2. Check Memory 
    expected_memory_max = torch.zeros([1, mem_weights.shape[0]])
    expected_memory_max[0, 0] = 0.7
    expected_memory_max[0, 1] = 0.9
    expected_memory_max[0, 2] = 0.3
    assert torch.allclose(memory.nodes[:, TF.MAX_MAP], expected_memory_max)

    # 3. Check Driver 
    expected_driver_max = torch.zeros([1, rec_weights.shape[1]])
    expected_driver_max[0, 0] = 0.2
    expected_driver_max[0, 1] = 0.8
    expected_driver_max[0, 2] = 0.5
    assert torch.allclose(driver.nodes[:, TF.MAX_MAP], expected_driver_max)

def test_get_max_maps_am_updates_am_sets_correctly(network: Network):
    """
    Tests that get_max_maps_am in mapping_ops correctly updates the max_map feature for driver and recipient.
    """
    driver = network.driver()
    recipient = network.recipient()
    
    # --- Setup Recipient -> Driver Weights ---
    rec_mappings = network.mappings[Set.RECIPIENT]
    rec_weights = rec_mappings[MappingFields.WEIGHT]
    # zero out all weights
    rec_weights.fill_(0)
    # set some weights
    rec_weights[0, 1] = 0.3
    rec_weights[1, 2] = 0.8
    rec_weights[2, 0] = 0.4

    # --- Run the Operation ---
    network.mapping_ops.get_max_maps_am()

    # --- Assertions ---

    # 1. Check Recipient
    expected_recipient_max = torch.zeros([1, rec_weights.shape[0]])
    expected_recipient_max[0, 0] = 0.3
    expected_recipient_max[0, 1] = 0.8
    expected_recipient_max[0, 2] = 0.4
    assert torch.allclose(recipient.nodes[:, TF.MAX_MAP], expected_recipient_max)

    # 2. Check Driver
    expected_driver_max = torch.zeros([1, rec_weights.shape[1]])
    expected_driver_max[0, 0] = 0.4
    expected_driver_max[0, 1] = 0.3
    expected_driver_max[0, 2] = 0.8
    assert torch.allclose(driver.nodes[:, TF.MAX_MAP], expected_driver_max)
