# nodes/tests/test_links.py
# Tests for links..

import pytest
import torch
import logging
logger = logging.getLogger(__name__)

from nodes.network import Network
from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network.single_nodes import Ref_Analog, Analog

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_calibrate_weights(network: Network):
    """Test calibrating most active links to driver"""
    # Set link weights for driver token
    sets = network.links.sets
    po_mask = network.driver().get_mask(Type.PO)
    network.links.sets[Set.DRIVER][po_mask, :] = 0.0
    network.links.sets[Set.DRIVER][po_mask, 1] = 0.9
    logger.debug(f"po_links:{sets[Set.DRIVER][po_mask,:]}")
    #Check weight
    failed = network.links[Set.DRIVER][po_mask, 1] != 0.9
    assert failed.nonzero().shape[0] == 0
    # calibrate weights
    network.links.calibrate_weights()
    logger.debug(f"po_links:{sets[Set.DRIVER][po_mask,:]}")
    # Check new weights
    failed = network.links[Set.DRIVER][po_mask, 1] != 1.0
    assert failed.nonzero().shape[0] == 0