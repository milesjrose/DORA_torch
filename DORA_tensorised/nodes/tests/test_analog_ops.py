# nodes/tests/test_analog_ops.py
# Tests for analog operations.

import pytest
import torch

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

def test_copy_analog(network):
    """Test copying an analog."""
    # get driver analog
    ref_analog = Ref_Analog(0, Set.DRIVER)
    driver_analog = network.analog.get_analog(ref_analog)
    
    # check that the analog has tokens (otherwise nothing is being tested)
    assert driver_analog.tokens.shape[0] > 0, "Driver analog should have tokens"
    
    # get initial size of recipient set
    initial_recipient_size = network.sets[Set.RECIPIENT].get_count()
    
    # copy driver analog to recipient
    recipient_ref_analog = network.analog.copy(ref_analog, Set.RECIPIENT)
    
    # get final size of recipient set
    final_recipient_size = network.sets[Set.RECIPIENT].get_count()
    
    # check that recipient size increased by the number of tokens in the analog
    expected_size = initial_recipient_size + driver_analog.tokens.shape[0]
    assert final_recipient_size == expected_size, f"Recipient size should be {expected_size}, got {final_recipient_size}"
    
    # now get recipient analog
    recipient_analog = network.analog.get_analog(recipient_ref_analog)
    
    # check that recipient analog is a copy of driver analog
    assert recipient_analog.tokens.shape == driver_analog.tokens.shape
    assert recipient_analog.connections.shape == driver_analog.connections.shape
    assert recipient_analog.links.shape == driver_analog.links.shape
    
    # check that names dict contain the same names, but not the same keys
    assert set(driver_analog.name_dict.values()) == set(recipient_analog.name_dict.values()), "Names should be the same"

def test_delete_analog(network):
    """Test deleting an analog."""
    # Get driver analog
    ref_analog = Ref_Analog(0, Set.DRIVER)
    driver_analog = network.analog.get_analog(ref_analog)
    
    # Check that the analog has tokens (otherwise nothing is being tested)
    assert driver_analog.tokens.shape[0] > 0, "Driver analog should have tokens"
    
    # Get initial count of tokens in driver set
    initial_driver_count = network.sets[Set.DRIVER].get_count()
    
    # Delete the analog from driver set
    network.analog.delete(ref_analog)
    
    # Get final count of tokens in driver set
    final_driver_count = network.sets[Set.DRIVER].get_count()
    
    # Check that driver count decreased by the number of tokens in the analog
    expected_count = initial_driver_count - driver_analog.tokens.shape[0]
    assert final_driver_count == expected_count, f"Driver count should be {expected_count}, got {final_driver_count}"
    
    # Verify that the analog no longer exists in the driver set
    try:
        network.analog.get_analog(ref_analog)
        assert False, f"Analog {ref_analog.analog_number} should have been deleted"
    except ValueError:
        # Expected - analog should not exist
        pass

def test_move_analog(network):
    """Test moving an analog."""
    # Get driver analog
    ref_analog = Ref_Analog(0, Set.DRIVER)
    driver_analog = network.analog.get_analog(ref_analog)
    
    # Check that the analog has tokens (otherwise nothing is being tested)
    assert driver_analog.tokens.shape[0] > 0, "Driver analog should have tokens"
    
    # Get initial counts
    initial_driver_count = network.sets[Set.DRIVER].get_count()
    initial_recipient_count = network.sets[Set.RECIPIENT].get_count()
    
    # Move the analog from driver to recipient
    recipient_ref_analog = network.analog.move(ref_analog, Set.RECIPIENT)
    
    # Get final counts
    final_driver_count = network.sets[Set.DRIVER].get_count()
    final_recipient_count = network.sets[Set.RECIPIENT].get_count()
    
    # Check that driver count decreased by the number of tokens in the analog
    expected_driver_count = initial_driver_count - driver_analog.tokens.shape[0]
    assert final_driver_count == expected_driver_count, f"Driver count should be {expected_driver_count}, got {final_driver_count}"
    
    # Check that recipient count increased by the number of tokens in the analog
    expected_recipient_count = initial_recipient_count + driver_analog.tokens.shape[0]
    assert final_recipient_count == expected_recipient_count, f"Recipient count should be {expected_recipient_count}, got {final_recipient_count}"
    
    # Verify that the analog no longer exists in the driver set
    try:
        network.sets[Set.DRIVER].get_analog(ref_analog)
        assert False, "Analog should have been deleted from driver"
    except ValueError:
        # Expected - analog should not exist in driver
        pass
    
    # Verify that the analog now exists in the recipient set
    try:
        recipient_analog = network.sets[Set.RECIPIENT].get_analog(recipient_ref_analog)
        # Check that the analog has the same structure
        assert recipient_analog.tokens.shape == driver_analog.tokens.shape
        assert recipient_analog.connections.shape == driver_analog.connections.shape
        assert recipient_analog.links.shape == driver_analog.links.shape
    except ValueError:
        assert False, "Analog should exist in recipient"

def test_get_analog_from_memory(network):
    """
    Set a memory token to a non-memory set, then try make AM copy.

    Check the analog is not in memory, then check if it is in the set it was moved to.
    """
    # Get memory analog
    ref_analog = Ref_Analog(0, Set.MEMORY)
    memory_analog = network.analog.get_analog(ref_analog)
    
    # Check that the analog has tokens (otherwise nothing is being tested)
    assert memory_analog.tokens.shape[0] > 0, "Memory analog should have tokens"
    
    # Get initial counts
    initial_memory_count = network.sets[Set.MEMORY].get_count()
    initial_driver_count = network.sets[Set.DRIVER].get_count()
    
    # Set some tokens in the memory analog to driver set
    # Get indices of tokens in the analog
    analog_indices = network.analog.get_analog_indices(ref_analog)
    
    # Set the first few tokens to driver set
    num_tokens_to_move = min(2, len(analog_indices))  # Move at most 2 tokens
    tokens_to_move = analog_indices[:num_tokens_to_move]
    network.sets[Set.MEMORY].token_op.set_features(tokens_to_move, TF.SET, Set.DRIVER)
    
    # Check that the analog now has set != memory
    updated_analog = network.analog.get_analog(ref_analog)
    assert updated_analog.set != Set.MEMORY, "Analog should not be in memory after setting tokens to driver"
    
    # Make AM copy
    copied_analogs = network.analog.make_AM_copy()
    
    # Check that the analog was moved to the driver set
    try:
        driver_analog = network.analog.get_analog(copied_analogs[0])
        # Verify the analog exists in driver with the same structure
        network.sets[Set.DRIVER].tensor_op.print_tokens()
        network.sets[Set.MEMORY].tensor_op.print_tokens()
        # Get number of undeleted tokens in memory analog
        new_mem_analog: Analog = network.analog.get_analog(ref_analog)
        new_mem_analog.retrieve_lower_tokens()
        new_mem_analog.remove_memory_tokens()
        num_tk = sum(new_mem_analog.tokens[:, TF.DELETED] == B.FALSE)
        assert driver_analog.tokens.shape[0] == num_tk, f"Driver analog should have {num_tk} tokens, got {driver_analog.tokens.shape[0]}"
        assert driver_analog.connections.shape[0] == num_tk, f"Driver analog should have {num_tk} connections, got {driver_analog.connections.shape[0]}"
        assert driver_analog.links.shape[0] == num_tk, f"Driver analog should have {num_tk} links, got {driver_analog.links.shape[0]}"
    except ValueError:
        assert False, "Analog should exist in driver after make_AM_copy"
    
    # Check that memory count decreased
    # final_memory_count = network.sets[Set.MEMORY].get_count()
    # assert final_memory_count < initial_memory_count, "Memory count should have decreased after make_AM_copy"
    
    # Check that driver count increased
    final_driver_count = network.sets[Set.DRIVER].get_count()
    assert final_driver_count > initial_driver_count, "Driver count should have increased after make_AM_copy"
    
def test_find_recip_analog(network):
    """Test finding the analog in the recipient that is mapped to."""
    # add tokens to driver rand recipient
    driver_token = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE})) 
    idx_d_token = network.get_index(driver_token) 
    rec_token = network.recipient().add_token(Token(Type.PO, {TF.PRED: B.FALSE}))  
    idx_rec_token = network.get_index(rec_token)
    # set mapping between nodes
    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][idx_rec_token, idx_d_token] = 1.0
    ref_analog = network.analog.find_mapped_analog(Set.RECIPIENT)

    assert ref_analog is not None, "Recipient analog should be found"
    assert ref_analog.set == Set.RECIPIENT, "Recipient analog should be in recipient set"
    assert ref_analog.analog_number is not None, "Recipient analog should have an analog number"
    assert ref_analog.analog_number == network.get_value(rec_token, TF.ANALOG)
    
    # Check that the analog has tokens (otherwise nothing is being tested)
    analog = network.analog.get_analog(ref_analog)
    assert analog.tokens.shape[0] > 0, "Recipient analog should have tokens"

def test_find_driver_analog(network):
    """Test finding the analog in the driver that is mapped to."""
    # add tokens to driver rand recipient
    driver_token = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE})) 
    idx_d_token = network.get_index(driver_token) 
    rec_token = network.recipient().add_token(Token(Type.PO, {TF.PRED: B.FALSE}))  
    idx_rec_token = network.get_index(rec_token)
    # set mapping between nodes
    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][idx_rec_token, idx_d_token] = 1.0
    ref_analog = network.analog.find_mapped_analog(Set.DRIVER)
    
    assert ref_analog is not None, " analog should be found"
    assert ref_analog.set == Set.DRIVER, " analog should be in driver set"
    assert ref_analog.analog_number is not None, " analog should have an analog number"
    assert ref_analog.analog_number == network.get_value(rec_token, TF.ANALOG)
    
    # Check that the analog has tokens (otherwise nothing is being tested)
    analog = network.analog.get_analog(ref_analog)
    assert analog.tokens.shape[0] > 0, " analog should have tokens"

def test_new_set_to_analog(network: Network):
    new_set = network.new_set()
    token = Token(Type.RB, {TF.SET:Set.NEW_SET, TF.ANALOG: 3.0})
    for i in range(5):
        network.node_ops.add_token(token)
    # Check tokens are all in analog 3
    analogs = new_set.nodes[:, TF.ANALOG] != 3.0
    print(new_set.nodes[:,TF.ANALOG])
    assert analogs.nonzero().shape[0] == 0

    # set analog
    network.analog.new_set_to_analog()

    #check tokens are all in analog 1
    analogs = new_set.nodes[:, TF.ANALOG] != 1.0
    assert analogs.nonzero().shape[0] == 0

def test_print_analog(network, capsys):
    """Test printing an analog."""
    # Get driver analog
    ref_analog = Ref_Analog(0, Set.DRIVER)
    network.analog.print_analog(ref_analog)
    idxs = network.sets[Set.DRIVER].get_analog_indices(ref_analog)
    for idx in idxs:
        ref_token = network.sets[Set.DRIVER].token_op.get_reference(index=idx)
        network.sets[Set.DRIVER].token_op.set_name(ref_token, f"test_{idx}")
    
    # Capture the output
    captured = capsys.readouterr()
    print_output = captured.out
    
    # Verify the output contains expected content
    assert "analog" in print_output.lower() or len(print_output) > 0, f"Expected print output, got: {print_output}"
    print(f"Captured output: {print_output}")

