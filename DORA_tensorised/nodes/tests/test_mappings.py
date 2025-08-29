# nodes/tests/test_mappings.py
# Tests the mappings class.

import pytest
import torch

from ..enums import MappingFields, Set, Type, B, TF
from ..network.connections import Mappings
from ..builder import NetworkBuilder
from ..network import Network, Token, Ref_Token


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

def test_mappings_initialization(network: Network):
    # Create test tensors
    size = 3
    fields = {}

    driver = network.sets[Set.DRIVER]
    size = driver.nodes.shape[0]
    for field in MappingFields:
        fields[field] = torch.ones(size, size)
    
    # Test initialization
    mappings = Mappings(driver, fields)
    
    # Test tensor shape
    assert mappings.adj_matrix.shape == (size, size, len(MappingFields))

    assert mappings.adj_matrix.dtype == torch.float32
    
    # Test accessor methods
    for field in MappingFields:
        assert torch.all(mappings[field] == fields[field])

    # Test size method
    assert mappings.size(0) == size
    assert mappings.size(1) == size
    assert mappings.size(2) == len(MappingFields)

def test_mappings_invalid_input(network: Network):
    # Test with mismatched tensor shapes
    driver = network.sets[Set.DRIVER]
    size = driver.nodes.shape[0]

    # Should raise error for invalid shapes
    fields = {}
    for i, field in enumerate(MappingFields):
        fields[field] = torch.ones(size, size + i)
    with pytest.raises(ValueError):
        Mappings(driver, fields)

    # Should raise error for driver nodes mismatch
    fields = {}
    for field in MappingFields:
        fields[field] = torch.ones(size, size + 1)
    with pytest.raises(ValueError):
        Mappings(driver, fields)

def test_mappings_update_hypotheses(network: Network):
    driver = network.sets[Set.DRIVER]
    recipient = network.sets[Set.RECIPIENT]
    mappings = network.mappings[Set.RECIPIENT]

    # First update nodes to have known nodes values (and make pairs of nodes that can map to each other - same mode/type)
    driver_node = Token(Type.PO, {TF.PRED: B.TRUE, TF.ACT: 0.6, TF.SET: Set.DRIVER}) # Pred PO for driver
    driver_node.name = "Driver Node"

    recipient_node = Token(Type.PO, {TF.PRED: B.TRUE, TF.ACT: 0.7, TF.SET: Set.RECIPIENT}) # Pred PO for recipient
    recipient_node.name = "Recipient Node"

    # First make space for nodes
    
    ref = driver.token_op.get_reference(id=2)
    network.del_token(ref)
    ref = recipient.token_op.get_reference(id=2)
    network.del_token(ref)
    # Add nodes to network
    driver_node: Ref_Token = driver.add_token(driver_node)
    driver_index = driver.token_op.get_index(driver_node)
    recipient_node: Ref_Token = recipient.add_token(recipient_node)
    recipient_index = recipient.token_op.get_index(recipient_node)

    # Add mapping connections between nodes
    mappings[MappingFields.CONNECTIONS][driver_index, recipient_index] = 1.0
    mappings[MappingFields.HYPOTHESIS][driver_index, recipient_index] = 0.5

    # Update hypotheses
    mappings.update_hypotheses()

    # New values should be hyp += (driver_act * recipient_act)
    expected_hyp = 0.5 + (0.6 * 0.7)
    assert mappings[MappingFields.HYPOTHESIS][driver_index, recipient_index] == expected_hyp

def test_mappings_update_hypotheses_multiple(network: Network):
    driver = network.sets[Set.DRIVER]
    recipient = network.sets[Set.RECIPIENT]
    mappings = network.mappings[Set.RECIPIENT]

    # First update nodes to have known nodes values (and make pairs of nodes that can map to each other - same mode/type)
    driver_node = Token(Type.PO, {TF.PRED: B.TRUE, TF.ACT: 0.6, TF.SET: Set.DRIVER}) # Pred PO for driver
    driver_node.name = "Driver Node"

    recipient_node = Token(Type.PO, {TF.PRED: B.TRUE, TF.ACT: 0.7, TF.SET: Set.RECIPIENT}) # Pred PO for recipient
    recipient_node.name = "Recipient Node"

    # First make space for nodes
    
    ref = driver.token_op.get_reference(id=2)
    network.del_token(ref)
    ref = recipient.token_op.get_reference(id=2)
    network.del_token(ref)
    # Add nodes to network
    driver_node: Ref_Token = driver.add_token(driver_node)
    driver_index = driver.token_op.get_index(driver_node)
    recipient_node: Ref_Token = recipient.add_token(recipient_node)
    recipient_index = recipient.token_op.get_index(recipient_node)

    # Add mapping connections between nodes
    mappings[MappingFields.CONNECTIONS][driver_index, recipient_index] = 1.0
    mappings[MappingFields.HYPOTHESIS][driver_index, recipient_index] = 0.5

    # Update hypotheses
    mappings.update_hypotheses()

    # New values should be hyp += (driver_act * recipient_act)
    expected_hyp = 0.5 + (0.6 * 0.7)
    assert mappings[MappingFields.HYPOTHESIS][driver_index, recipient_index] == expected_hyp

    for i in range(10):
        expected_hyp += (0.6 * 0.7)
        mappings.update_hypotheses()
        assert mappings[MappingFields.HYPOTHESIS][driver_index, recipient_index] - expected_hyp < 1e-6 # Ingore floating point errors
