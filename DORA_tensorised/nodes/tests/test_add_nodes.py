# nodes/tests/test_add_nodes.py
# Tests for adding nodes to the network.

import pytest

from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token
from nodes.enums import *

# Import the symProps from sim.py
from .sims.sim import symProps

@pytest.fixture
def nodes():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_nodes()

def test_add_po_token_to_driver(nodes):
    """Test adding a PO token to the driver tensor."""
    # Create a new PO token
    token = New_Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    
    # Add the token to the driver tensor
    token_id = nodes.driver.add_token(token)
    
    # Verify the token was added correctly
    assert token_id is not None
    assert token_id in nodes.driver.IDs
    
    # Get the token from the tensor
    token_index = nodes.driver.IDs[token_id]
    token_tensor = nodes.driver.nodes[token_index]
    
    # Verify the token properties
    assert token_tensor[TF.TYPE] == Type.PO
    assert token_tensor[TF.SET] == Set.DRIVER
    assert token_tensor[TF.ANALOG] == 0
    assert token_tensor[TF.PRED] == B.TRUE
    assert token_tensor[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_rb_token_to_recipient(nodes):
    """Test adding a RB token to the recipient tensor."""
    # Create a new RB token
    token = New_Token(type=Type.RB, features={
        TF.SET: Set.RECIPIENT,
        TF.ANALOG: 0
    })
    
    # Add the token to the recipient tensor
    token_id = nodes.recipient.add_token(token)
    
    # Verify the token was added correctly
    assert token_id is not None
    assert token_id in nodes.recipient.IDs
    
    # Get the token from the tensor
    token_index = nodes.recipient.IDs[token_id]
    token_tensor = nodes.recipient.nodes[token_index]
    
    # Verify the token properties
    assert token_tensor[TF.TYPE] == Type.RB
    assert token_tensor[TF.SET] == Set.RECIPIENT
    assert token_tensor[TF.ANALOG] == 0
    assert token_tensor[TF.INHIBITOR_THRESHOLD] == 220  # Default for RB tokens

def test_add_p_token_to_memory(nodes):
    """Test adding a P token to the memory tensor."""
    # Create a new P token
    token = New_Token(type=Type.P, features={
        TF.SET: Set.MEMORY,
        TF.ANALOG: 0
    })
    
    # Add the token to the memory tensor
    token_id = nodes.memory.add_token(token)
    
    # Verify the token was added correctly
    assert token_id is not None
    assert token_id in nodes.memory.IDs
    
    # Get the token from the tensor
    token_index = nodes.memory.IDs[token_id]
    token_tensor = nodes.memory.nodes[token_index]
    
    # Verify the token properties
    assert token_tensor[TF.TYPE] == Type.P
    assert token_tensor[TF.SET] == Set.MEMORY
    assert token_tensor[TF.ANALOG] == 0
    assert token_tensor[TF.INHIBITOR_THRESHOLD] == 440  # Default for P tokens

def test_add_multiple_tokens_new_set(nodes):
    """Test adding multiple tokens to the new_set tensor."""
    # Create multiple tokens
    tokens = [
        New_Token(type=Type.PO, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0, TF.PRED: True}),
        New_Token(type=Type.RB, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0}),
        New_Token(type=Type.P, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0})
    ]
    
    nodes.new_set.print()
    # Add the tokens to the new_set tensor
    token_ids = [nodes.new_set.add_token(token) for token in tokens]
    
    # Verify all tokens were added correctly
    assert len(token_ids) == 3
    assert all(token_id in nodes.new_set.IDs for token_id in token_ids)
    
    # Verify the tokens have different IDs
    assert len(set(token_ids)) == 3

def test_add_token_with_custom_features(nodes):
    """Test adding a token with custom features to the driver tensor."""
    # Create a new PO token with custom features
    token = New_Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 1,
        TF.PRED: True,
        TF.ACT: 0.5,
        TF.MAX_ACT: 1.0,
        TF.NET_INPUT: 0.3
    })
    
    # Add the token to the driver tensor
    token_id = nodes.driver.add_token(token)
    
    # Verify the token was added correctly
    assert token_id is not None
    assert token_id in nodes.driver.IDs
    
    # Get the token from the tensor
    token_index = nodes.driver.IDs[token_id]
    token_tensor = nodes.driver.nodes[token_index]
    
    # Verify the token properties
    assert token_tensor[TF.TYPE] == Type.PO
    assert token_tensor[TF.SET] == Set.DRIVER
    assert token_tensor[TF.ANALOG] == 1
    assert token_tensor[TF.PRED] == B.TRUE
    assert token_tensor[TF.ACT] == 0.5
    assert token_tensor[TF.MAX_ACT] == 1.0
    assert token_tensor[TF.NET_INPUT] == 0.3
    assert token_tensor[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_token_to_full_tensor(nodes):
    """Test adding a token when the tensor is full."""
    # First, fill up the new_set tensor
    initial_size = nodes.new_set.nodes.shape[0]
    
    # Add tokens until the tensor is full
    for i in range(initial_size):
        token = New_Token(type=Type.PO, features={
            TF.SET: Set.NEW_SET,
            TF.ANALOG: 0,
            TF.PRED: True
        })
        nodes.new_set.add_token(token)
    
    # Now add one more token, which should trigger tensor expansion
    token = New_Token(type=Type.PO, features={
        TF.SET: Set.NEW_SET,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    token_id = nodes.new_set.add_token(token)
    
    # Verify the token was added correctly
    assert token_id is not None
    assert token_id in nodes.new_set.IDs
    
    # Verify the tensor was expanded
    assert nodes.new_set.nodes.shape[0] > initial_size

    