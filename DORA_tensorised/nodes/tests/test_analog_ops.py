# nodes/tests/test_analog_ops.py
# Tests for analog operations.

import pytest

from nodes.builder import NetworkBuilder
from nodes.enums import *

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_copy_analog(network):
    """Test copying an analog."""
    # get driver analog
    driver_analog = network.sets[Set.DRIVER].get_analog(0)
    
    # check that the analog has tokens (otherwise nothing is being tested)
    assert driver_analog.tokens.shape[0] > 0, "Driver analog should have tokens"
    
    # get initial size of recipient set
    initial_recipient_size = network.sets[Set.RECIPIENT].get_count()
    
    # copy driver analog to recipient
    analog_number = network.analog.copy(0, Set.DRIVER, Set.RECIPIENT)
    
    # get final size of recipient set
    final_recipient_size = network.sets[Set.RECIPIENT].get_count()
    
    # check that recipient size increased by the number of tokens in the analog
    expected_size = initial_recipient_size + driver_analog.tokens.shape[0]
    assert final_recipient_size == expected_size, f"Recipient size should be {expected_size}, got {final_recipient_size}"
    
    # now get recipient analog
    recipient_analog = network.sets[Set.RECIPIENT].get_analog(analog_number)
    
    # check that recipient analog is a copy of driver analog
    assert recipient_analog.tokens.shape == driver_analog.tokens.shape
    assert recipient_analog.connections.shape == driver_analog.connections.shape
    assert recipient_analog.links.shape == driver_analog.links.shape
    
    # check that names dict contain the same names, but not the same keys
    assert set(driver_analog.name_dict.values()) == set(recipient_analog.name_dict.values()), "Names should be the same"
    
