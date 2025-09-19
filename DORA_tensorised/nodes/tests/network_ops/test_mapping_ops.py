# nodes/tests/test_mappings.py
# Tests the mappings class.

import pytest
import torch
import pandas as pd

from nodes.enums import MappingFields, Set, Type, TF
from nodes.builder import NetworkBuilder
from nodes.utils import nodePrinter
from nodes.network import Network, Token



# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps


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
    mappings = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.HYPOTHESIS].fill_(1.0)
    mappings[MappingFields.MAX_HYP].fill_(1.0)
    mappings[MappingFields.WEIGHT].fill_(1.0)

    network.mapping_ops.reset_mapping_units()

    assert torch.all(mappings[MappingFields.HYPOTHESIS] == 0)
    assert torch.all(mappings[MappingFields.MAX_HYP] == 0)
    assert torch.all(mappings[MappingFields.WEIGHT] == 0)


def test_reset_mappings(network: Network):
    """
    Tests that reset_mappings zeros out all mapping fields for all sets.
    """
    mappings = network.mappings[Set.RECIPIENT]
    for field in MappingFields:
        mappings[field].fill_(1.0)

    network.mapping_ops.reset_mappings()

    assert torch.all(mappings[MappingFields.HYPOTHESIS] == 0)
    assert torch.all(mappings[MappingFields.MAX_HYP] == 0)
    assert torch.all(mappings[MappingFields.WEIGHT] == 0)



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
    network.recipient().cache_masks()
    network.driver().cache_masks()

    printer = nodePrinter(print_to_file=False)
    printer.print_tk_tensor(driver.nodes, headers=["Driver"], types=[TF.ID, TF.DELETED, TF.TYPE, TF.MODE, TF.ACT])
    printer.print_tk_tensor(recipient.nodes, headers=["Recipient"], types=[TF.ID, TF.DELETED, TF.TYPE, TF.MODE, TF.ACT])
    print(network.recipient().get_mask(Type.P))
    print(network.recipient().nodes[:, TF.TYPE] == Type.P)
    network.mappings[Set.RECIPIENT].print(MappingFields.HYPOTHESIS)


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

    mappings = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.HYPOTHESIS].fill_(1.0)
    mappings[MappingFields.MAX_HYP].fill_(1.0)

    network.mapping_ops.reset_mapping_hyps()

    assert torch.all(mappings[MappingFields.HYPOTHESIS] == 0)
    assert torch.all(mappings[MappingFields.MAX_HYP] == 0)

def import_tensor(filename: str):
    data = pd.read_csv(filename, header=None, index_col=None)
    return torch.tensor(data.values, dtype=torch.float32)

def test_update_mapping_connections(network: Network):
    """
    Tests that update_mapping_connections correctly updates connection weights.
    """
    prefix = 'scripts/'
    weight = import_tensor(prefix + 'weight_data.csv')
    hyp = import_tensor(prefix + 'hyp_data.csv')
    weight_updated = import_tensor(prefix + 'weight_updated.csv')

    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT] = weight
    network.mappings[Set.RECIPIENT][MappingFields.HYPOTHESIS] = hyp
    print(network.params.eta)
    network.mapping_ops.update_mapping_connections()

    network.mappings[Set.RECIPIENT].print(MappingFields.WEIGHT)
    printer = nodePrinter(print_to_file=False)
    printer.print_weight_tensor(weight_updated, headers=["Weight Updated"])

    assert torch.allclose(network.mappings[Set.RECIPIENT][MappingFields.WEIGHT], weight_updated)


