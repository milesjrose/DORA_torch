# nodes/tests/set_ops/test_update_input.py
# Tests for update_input operations across all sets.

import pytest
from sympy.logic import true
import torch

import logging
logger = logging.getLogger(__name__)

from nodes.builder import NetworkBuilder
from nodes.network import Network
from nodes.network.sets import Driver, Memory, Recipient, New_Set, Semantics
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token
from nodes.utils import tensor_ops as tOps

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_update_input_memory(network: Network):
    """Test the update_input function for the memory set."""
    mem: Memory = network.memory()
    is_nan = False
    for update_function in [mem.update_input_p_parent, mem.update_input_rb, mem.update_input_po]:
        update_function()
        if not check_nan_values(mem, update_function.__name__):
            is_nan = True
    assert not is_nan

def check_nan_values(mem, update_function: str):
    """
    Check for nan values in the nodes after an update function is called
    """
    for input_field in [TF.TD_INPUT, TF.BU_INPUT, TF.MAP_INPUT, TF.LATERAL_INPUT]:
        if torch.isnan(mem.nodes[:, input_field]).any():
            logger.critical(f"NaN values in {input_field.name} for {update_function}")
            return False
    return True

def test_update_input_driver(network: Network):
    """Test the update_input function for the driver set."""
    driver: Driver = network.driver()
    is_nan = False
    for update_function in [driver.update_input_p_parent, driver.update_input_p_child, driver.update_input_rb, driver.update_input_po]:
        update_function()
        if not check_nan_values(driver, update_function.__name__):
            is_nan = True
    assert not is_nan