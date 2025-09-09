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

