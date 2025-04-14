# nodes/tests/test_delete_nodes.py
# Tests for deleting nodes from the network.

import pytest

from nodes.builder import NetworkBuilder
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
    return builder.build_nodes()

def test_delete_token_from_driver(network):
    """Test deleting a token from the driver tensor."""
    # Create a new PO token
    token = Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    
    # Add the token to the driver tensor
    token_reference: Ref_Token = network.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.DRIVER
    
    # Get the token from the tensor
    token = network.driver.get_single_token(token_reference)
    
    # Verify the token properties
    assert token.tensor[TF.TYPE] == Type.PO
    assert token.tensor[TF.SET] == Set.DRIVER
    
    # Get index to check if deleted later
    index = network.driver.get_index(token_reference)

    # Delete the token
    network.del_token(token_reference)
    
    # Verify the token was deleted
    # The token should be marked as deleted in the tensor
    token = network.driver.nodes[index, :]
    assert token[TF.DELETED] == B.TRUE
    
    # The token should not be in the IDs dictionary
    assert token_reference.ID not in network.driver.IDs

def test_delete_token_from_recipient(network):
    """Test deleting a token from the recipient tensor."""
    # Create a new RB token
    token = Token(type=Type.RB, features={
        TF.SET: Set.RECIPIENT,
        TF.ANALOG: 0
    })
    
    # Add the token to the recipient tensor
    token_reference: Ref_Token = network.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.RECIPIENT
    
    # Get the token from the tensor
    token = network.recipient.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.RB
    assert token[TF.SET] == Set.RECIPIENT
    
    # Get index to check if deleted later
    index = network.recipient.get_index(token_reference)

    # Delete the token
    network.del_token(token_reference)
    
    # Verify the token was deleted
    # The token should be marked as deleted in the tensor
    token = network.recipient.nodes[index, :]
    assert token[TF.DELETED] == B.TRUE
    
    # The token should not be in the IDs dictionary
    assert token_reference.ID not in network.recipient.IDs

def test_delete_token_from_memory(network):
    """Test deleting a token from the memory tensor."""
    # Create a new P token
    token = Token(type=Type.P, features={
        TF.SET: Set.MEMORY,
        TF.ANALOG: 0
    })
    
    # Add the token to the memory tensor
    token_reference: Ref_Token = network.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.MEMORY
    
    # Get the token from the tensor
    token = network.memory.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.P
    assert token[TF.SET] == Set.MEMORY
    
    # Get index to check if deleted later
    index = network.memory.get_index(token_reference)

    # Delete the token
    network.del_token(token_reference)
    
    # Verify the token was deleted
    # The token should be marked as deleted in the tensor
    token = network.memory.nodes[index, :]
    assert token[TF.DELETED] == B.TRUE
    
    # The token should not be in the IDs dictionary
    assert token_reference.ID not in network.memory.IDs

def test_delete_multiple_tokens(network):
    """Test deleting multiple tokens from the new_set tensor."""
    # Create multiple tokens
    tokens = [
        Token(type=Type.PO, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0, TF.PRED: True}),
        Token(type=Type.RB, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0}),
        Token(type=Type.P, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0})
    ]
    
    # Add the tokens to the new_set tensor
    token_references: list[Ref_Token] = [network.add_token(token) for token in tokens]
    
    # Verify all tokens were added correctly
    assert len(token_references) == 3
    assert all(token_reference.set == Set.NEW_SET for token_reference in token_references)
    indices = [network.new_set.get_index(token_reference) for token_reference in token_references]
    # Delete all tokens
    for token_reference in token_references:
        network.del_token(token_reference)
    
    # Verify all tokens were deleted
    for i, token_reference in enumerate(token_references):
        # The token should be marked as deleted in the tensor
        token = network.new_set.nodes[indices[i], :]
        assert token[TF.ID] == null
        assert token[TF.DELETED] == B.TRUE
        
        # The token should not be in the IDs dictionary
        assert token_reference.ID not in network.new_set.IDs

def test_delete_and_reuse_token_slot(network):
    """Test that a deleted token slot can be reused."""
    # Create a token
    token = Token(type=Type.PO, features={
        TF.SET: Set.NEW_SET,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    
    # Add the token to the new_set tensor
    token_reference: Ref_Token = network.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.NEW_SET
    
    # Get the token from the tensor
    token = network.new_set.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.PO
    assert token[TF.SET] == Set.NEW_SET
    
    # Get index to check if deleted later
    index = network.new_set.get_index(token_reference)

    # Delete the token
    network.del_token(token_reference)
    
    # Verify the token was deleted
    # The token should be marked as deleted in the tensor
    token = network.new_set.nodes[index, :]
    assert token[TF.ID] == null
    assert token[TF.DELETED] == B.TRUE
    
    # The token should not be in the IDs dictionary
    assert token_reference.ID not in network.new_set.IDs
    
    # Create a new token
    new_token = Token(type=Type.RB, features={
        TF.SET: Set.NEW_SET,
        TF.ANALOG: 0
    })
    
    # Add the new token to the new_set tensor
    new_token_reference: Ref_Token = network.add_token(new_token)
    
    # Verify the new token was added correctly
    assert new_token_reference is not None
    assert new_token_reference.set == Set.NEW_SET
    
    # Get the new token from the tensor
    new_token = network.new_set.nodes[index, :]
    
    # Verify the new token properties
    assert new_token[TF.TYPE] == Type.RB
    assert new_token[TF.SET] == Set.NEW_SET
    assert new_token[TF.DELETED] == B.FALSE
    
    # The new token should be in the IDs dictionary
    assert new_token_reference.ID in network.new_set.IDs 