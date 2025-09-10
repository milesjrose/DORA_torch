# nodes/tests/test_memory_ops.py
# Tests for memory operations from the network.

import pytest

from nodes.builder import NetworkBuilder
from nodes.network import Network
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token

# Import the symProps from sim.py
from .sims.sim import symProps

import torch

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_reset_inferences(network: Network):
    """Test reset inferences clears fields"""
    #add token
    ref_tk = network.add_token(Token(Type.RB, {
        TF.SET:Set.RECIPIENT, 
        TF.INFERRED:B.TRUE,
        TF.MAKER_UNIT:1.0,
        TF.MAKER_SET:Set.DRIVER,
        TF.MADE_UNIT:1.0,
        TF.MADE_SET:Set.NEW_SET
        }))
    # clear inference
    network.tensor_ops.reset_inferences()
    # check cleared
    assert network.get_value(ref_tk, TF.INFERRED) == False
    assert network.get_value(ref_tk, TF.MAKER_UNIT) == null
    assert network.get_value(ref_tk, TF.MADE_UNIT) == null

def test_swap_driver_recipient(network: Network):
    """Test swapping contents of recipient and driver"""
    old_driver = network.driver()
    old_driver_links = network.links.sets[Set.DRIVER]
    old_rec = network.recipient()
    old_rec_links = network.links.sets[Set.RECIPIENT]
    network.tensor_ops.swap_driver_recipient()
    new_driver = network.driver()
    new_rec = network.recipient()
    # Set tensors and data structures
    assert new_driver.nodes.all() == old_rec.nodes.all()
    assert new_rec.nodes.all() == old_driver.nodes.all()
    assert new_driver.connections.all() == old_rec.connections.all()
    assert new_rec.connections.all() == old_driver.connections.all()
    assert new_driver.IDs == old_rec.IDs
    assert new_rec.IDs == old_driver.IDs
    assert new_driver.names == old_rec.names
    assert new_rec.names == old_driver.names
    # Mapping object
    # TODO: Implement this.
    # Links object
    assert network.links.sets[Set.DRIVER].all() == old_rec_links.all()
    assert network.links.sets[Set.RECIPIENT].all() == old_driver_links.all()
