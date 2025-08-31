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

def test_get_max_maps_memory(network: Network):
    """
    Tests that get_max_maps in mapping_ops correctly updates the max_map feature for all sets.
    - Updates "from" sets (recipient, memory) based on their outgoing connections.
    - Updates "to" set (driver) based on its strongest incoming connection from ANY source.
    """
    memory = network.memory()
    
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
    network.mapping_ops.get_max_map_memory()

    # --- Assertions ---

    # Check Memory 
    expected_memory_max = torch.zeros([1, mem_weights.shape[0]])
    expected_memory_max[0, 0] = 0.7
    expected_memory_max[0, 1] = 0.9
    expected_memory_max[0, 2] = 0.3
    assert torch.allclose(memory.nodes[:, TF.MAX_MAP], expected_memory_max)


def test_get_max_maps(network: Network):
    """
    Tests that get_max_maps in mapping_ops correctly updates the max_map and max_map_unit feature for driver and recipient.
    """
    driver = network.driver()
    recipient = network.recipient()
    
    # --- Setup Recipient -> Driver Weights ---
    rec_mappings = network.mappings[Set.RECIPIENT]
    rec_weights = rec_mappings[MappingFields.WEIGHT]
    # zero out all weights
    rec_weights.fill_(0)

    # set some weights
    #rec_weights = [[0.2, 0.3, 0.2, ...],
    #               [0.2, 0.2, 0.8, ...],
    #               [0.4, 0.2, 0.1, ...],
    #               ...]
    rec_weights[0, 0] = 0.2
    rec_weights[0, 1] = 0.3
    rec_weights[0, 2] = 0.2
    rec_weights[1, 0] = 0.2
    rec_weights[1, 1] = 0.2
    rec_weights[1, 2] = 0.8
    rec_weights[2, 0] = 0.4
    rec_weights[2, 1] = 0.2
    rec_weights[2, 2] = 0.1

    # --- Run the Operation ---
    network.mapping_ops.get_max_maps()

    # --- Assertions ---

    # 1. Check Recipient
    expected_recipient_max = torch.zeros([1, rec_weights.shape[0]])
    expected_recipient_max[0, 0] = 0.3
    expected_recipient_max[0, 1] = 0.8
    expected_recipient_max[0, 2] = 0.4
    expected_recipient_max_unit = torch.zeros([1, recipient.nodes.shape[0]])
    expected_recipient_max_unit[0, 0] = 1
    expected_recipient_max_unit[0, 1] = 2
    expected_recipient_max_unit[0, 2] = 0
    assert torch.allclose(recipient.nodes[:, TF.MAX_MAP], expected_recipient_max)
    assert torch.allclose(recipient.nodes[:, TF.MAX_MAP_UNIT], expected_recipient_max_unit)

    # 2. Check Driver
    expected_driver_max = torch.zeros([1, rec_weights.shape[1]])
    expected_driver_max[0, 0] = 0.4
    expected_driver_max[0, 1] = 0.3
    expected_driver_max[0, 2] = 0.8
    expected_driver_max_unit = torch.zeros([1, driver.nodes.shape[0]])
    expected_driver_max_unit[0, 0] = 2
    expected_driver_max_unit[0, 1] = 0
    expected_driver_max_unit[0, 2] = 1
    assert torch.allclose(driver.nodes[:, TF.MAX_MAP], expected_driver_max)
    assert torch.allclose(driver.nodes[:, TF.MAX_MAP_UNIT], expected_driver_max_unit)


def test_reset_mapping_units(network: Network):
    """
    Tests that reset_mapping_units zeros out hypotheses, max_hyp, and connections for driver and recipient.
    """
    for s_enum in [Set.DRIVER, Set.RECIPIENT]:
        mappings = network.mappings[s_enum]
        mappings[MappingFields.HYPOTHESIS].fill_(1.0)
        mappings[MappingFields.MAX_HYP].fill_(1.0)
        mappings[MappingFields.CONNECTIONS].fill_(1.0)

    network.mapping_ops.reset_mapping_units()

    for s_enum in [Set.DRIVER, Set.RECIPIENT]:
        mappings = network.mappings[s_enum]
        assert torch.all(mappings[MappingFields.HYPOTHESIS] == 0)
        assert torch.all(mappings[MappingFields.MAX_HYP] == 0)
        assert torch.all(mappings[MappingFields.CONNECTIONS] == 0)


def test_reset_mappings(network: Network):
    """
    Tests that reset_mappings zeros out all mapping fields for all sets.
    """
    for s_enum in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY]:
        mappings = network.mappings[s_enum]
        for field in MappingFields:
            mappings[field].fill_(1.0)

    network.mapping_ops.reset_mappings()

    for s_enum in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY]:
        mappings = network.mappings[s_enum]
        for field in MappingFields:
            if field != MappingFields.WEIGHT: # WEIGHT is not reset by this op
                assert torch.all(mappings[field] == 0)


def test_update_mapping_hyps(network: Network):
    """
    Tests that update_mapping_hyps correctly updates hypotheses based on token activations.
    """
    driver = network.driver()
    recipient = network.recipient()

    # --- Setup Activations ---
    driver.nodes[0, TF.ACT] = 0.5
    driver.nodes[1, TF.ACT] = 0.8
    recipient.nodes[0, TF.ACT] = 0.7
    recipient.nodes[1, TF.ACT] = 0.9
    
    # Set node types to allow for hypothesis update
    driver.nodes[0, TF.TYPE] = Type.P
    driver.nodes[1, TF.TYPE] = Type.P
    recipient.nodes[0, TF.TYPE] = Type.P
    recipient.nodes[1, TF.TYPE] = Type.P


    network.mapping_ops.update_mapping_hyps()

    rec_mappings = network.mappings[Set.RECIPIENT]
    hypotheses = rec_mappings[MappingFields.HYPOTHESIS]

    assert torch.isclose(hypotheses[0, 0], torch.tensor(0.5 * 0.7))
    assert torch.isclose(hypotheses[0, 1], torch.tensor(0.8 * 0.7))
    assert torch.isclose(hypotheses[1, 0], torch.tensor(0.5 * 0.9))
    assert torch.isclose(hypotheses[1, 1], torch.tensor(0.8 * 0.9))


def test_reset_mapping_hyps(network: Network):
    """
    Tests that reset_mapping_hyps zeros out hypotheses and max_hyp for driver and recipient.
    """
    for s_enum in [Set.DRIVER, Set.RECIPIENT]:
        mappings = network.mappings[s_enum]
        mappings[MappingFields.HYPOTHESIS].fill_(1.0)
        mappings[MappingFields.MAX_HYP].fill_(1.0)

    network.mapping_ops.reset_mapping_hyps()

    for s_enum in [Set.DRIVER, Set.RECIPIENT]:
        mappings = network.mappings[s_enum]
        assert torch.all(mappings[MappingFields.HYPOTHESIS] == 0)
        assert torch.all(mappings[MappingFields.MAX_HYP] == 0)


def test_update_mapping_connections(network: Network):
    """
    Tests that update_mapping_connections correctly updates connection weights.
    """
    eta = network.params.eta
    maps = network.mappings[Set.RECIPIENT]

    # --- Setup Hypotheses ---
    hyp = maps[MappingFields.HYPOTHESIS]
    hyp.fill_(0)
    hyp[0, 0] = 0.5
    hyp[1, 1] = 0.9

    weight = maps[MappingFields.WEIGHT]
    weight.fill_(0)
    weight[0, 0] = 0.2
    weight[1, 1] = 0.8

    # -- Run the Operation --

    network.mapping_ops.update_mapping_connections()

    weight = maps[MappingFields.WEIGHT]

    # Expected values are calculated based on the logic in mappings.py update_connections
    # 1. get_max_hypothesis
    max_hyp_00 = max(hyp[0,:].max(), hyp[:,0].max()) # 0.5
    max_hyp_11 = max(hyp[1,:].max(), hyp[:,1].max()) # 0.9
    
    # 2. Divisive normalisation
    hyp_norm_div_00 = 0.5 / max_hyp_00 # 1.0
    hyp_norm_div_11 = 0.9 / max_hyp_11 # 1.0
    
    # 3. get_max_hypothesis again (on divisively normalised hyps)
    max_hyp_norm_00 = 1.0
    max_hyp_norm_11 = 1.0
    
    # 4. Subtractive normalisation
    hyp_norm_sub_00 = hyp_norm_div_00 - max_hyp_norm_00 # 0.0
    hyp_norm_sub_11 = hyp_norm_div_11 - max_hyp_norm_11 # 0.0
    
    # 5. Update connections
    expected_con_00 = eta * (1.1 - 0.0) * hyp_norm_sub_00 # 0.0
    expected_con_11 = eta * (1.1 - 0.0) * hyp_norm_sub_11 # 0.0
    
    assert torch.isclose(weight[0, 0], torch.tensor(expected_con_00))
    assert torch.isclose(weight[1, 1], torch.tensor(expected_con_11))

