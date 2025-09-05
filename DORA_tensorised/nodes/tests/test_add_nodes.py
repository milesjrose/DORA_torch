# nodes/tests/test_add_nodes.py
# Tests for adding nodes to the network.

import pytest

from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network import Ref_Token, Network

# Import the symProps from sim.py
from .sims.sim import symProps

import torch

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_add_po_token_to_driver(network):
    """Test adding a PO token to the driver tensor."""
    # Create a new PO token
    token = Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    
    # Add the token to the driver tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.DRIVER
    
    # Get the token from the tensor
    token = network.sets[Set.DRIVER].token_op.get_single_token(token_reference)
    
    
    # Verify the token properties
    assert token.tensor[TF.TYPE] == Type.PO
    assert token.tensor[TF.SET] == Set.DRIVER
    assert token.tensor[TF.ANALOG] == 0
    assert token.tensor[TF.PRED] == B.TRUE
    assert token.tensor[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_rb_token_to_recipient(network):
    """Test adding a RB token to the recipient tensor."""
    # Create a new RB token
    token = Token(type=Type.RB, features={
        TF.SET: Set.RECIPIENT,
        TF.ANALOG: 0
    })
    
    # Add the token to the recipient tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.RECIPIENT
    
    # Get the token from the tensor
    token = network.sets[Set.RECIPIENT].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.RB
    assert token[TF.SET] == Set.RECIPIENT
    assert token[TF.ANALOG] == 0
    assert token[TF.INHIBITOR_THRESHOLD] == 220  # Default for RB tokens

def test_add_p_token_to_memory(network):
    """Test adding a P token to the memory tensor."""
    # Create a new P token
    token = Token(type=Type.P, features={
        TF.SET: Set.MEMORY,
        TF.ANALOG: 0
    })
    
    # Add the token to the memory tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.MEMORY
    
    # Get the token from the tensor
    token = network.sets[Set.MEMORY].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.P
    assert token[TF.SET] == Set.MEMORY
    assert token[TF.ANALOG] == 0
    assert token[TF.INHIBITOR_THRESHOLD] == 440  # Default for P tokens

def test_add_multiple_tokens_new_set(network):
    """Test adding multiple tokens to the new_set tensor."""
    # Create multiple tokens
    tokens = [
        Token(type=Type.PO, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0, TF.PRED: True}),
        Token(type=Type.RB, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0}),
        Token(type=Type.P, features={TF.SET: Set.NEW_SET, TF.ANALOG: 0})
    ]
    
    network[Set.NEW_SET].print()
    # Add the tokens to the new_set tensor
    token_references: list[Ref_Token] = [network.node.add_token(token) for token in tokens]
    
    # Verify all tokens were added correctly
    assert len(token_references) == 3
    assert all(token_reference.set == Set.NEW_SET for token_reference in token_references)
    
    # Verify the tokens have different IDs
    assert len(set(token_reference.ID for token_reference in token_references)) == 3

def test_add_token_with_custom_features(network):
    """Test adding a token with custom features to the driver tensor."""
    # Create a new PO token with custom features
    token = Token(type=Type.PO, features={
        TF.SET: Set.DRIVER,
        TF.ANALOG: 1,
        TF.PRED: True,
        TF.ACT: 0.5,
        TF.MAX_ACT: 1.0,
        TF.NET_INPUT: 0.3
    })
    
    # Add the token to the driver tensor
    token_reference: Ref_Token = network.node.add_token(token)
    
    network.sets[Set.DRIVER].print(f_types=[TF.ID, TF.TYPE, TF.SET, TF.ANALOG, TF.PRED, TF.ACT, TF.MAX_ACT, TF.NET_INPUT, TF.INHIBITOR_THRESHOLD])
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.DRIVER
    
    # Get the token from the tensor
    token = network.sets[Set.DRIVER].token_op.get_single_token(token_reference)
    
    # Verify the token properties
    assert token[TF.TYPE] == Type.PO
    assert token[TF.SET] == Set.DRIVER
    assert token[TF.ANALOG] == 1
    assert token[TF.PRED] == B.TRUE
    assert token[TF.ACT] == 0.5
    assert token[TF.MAX_ACT] == 1.0
    assert token[TF.NET_INPUT] == 0.3
    assert token[TF.INHIBITOR_THRESHOLD] == 110  # Default for PO tokens

def test_add_token_to_full_tensor(network):
    """Test adding a token when the tensor is full."""
    # First, fill up the new_set tensor
    initial_count = network[Set.NEW_SET].nodes.size(dim=0)

    new_count = int(initial_count * network[Set.NEW_SET].expansion_factor) 
    if new_count < 5:                                                   # minimum expansion is 5
            new_count = 5

    # Add tokens until the tensor is full
    references: list[Ref_Token] = []
    tokens: list[Token] = []
    for i in range(initial_count):
        token = Token(type=Type.PO, features={
            TF.SET: Set.NEW_SET,
            TF.ANALOG: 0,
            TF.PRED: True
        })
        reference: Ref_Token = network.node.add_token(token)
        references.append(reference)
        tokens.append(network.token_op.get_single_token(reference))
    
    # Now add one more token, which should trigger tensor expansion
    token = Token(type=Type.PO, features={
        TF.SET: Set.NEW_SET,
        TF.ANALOG: 0,
        TF.PRED: True
    })
    token_reference: Ref_Token = network.node.add_token(token)
    
    # Verify the token was added correctly
    assert token_reference is not None
    assert token_reference.set == Set.NEW_SET
    
    network[Set.NEW_SET].print()
    # Verify the tensor was expanded
    assert network[Set.NEW_SET].nodes.size(dim=0) > initial_count
    assert network[Set.NEW_SET].nodes.size(dim=0) == new_count
    print(network[Set.NEW_SET].nodes.size(dim=0))
    print(network[Set.NEW_SET].links[Set.NEW_SET].size(dim=0), network[Set.NEW_SET].links[Set.NEW_SET])
    assert network[Set.NEW_SET].nodes.size(dim=0) == network[Set.NEW_SET].links[Set.NEW_SET].size(dim=0)

    # Verify references still give the same tokens
    for reference, token in zip(references, tokens):
        new_token = network.token_op.get_single_token(reference)
        assert torch.equal(new_token.tensor, token.tensor)
        assert new_token.ID == token.ID
        assert new_token.set == token.set

def test_type_after_expansion(network):
    """Test that the type of a nodes tensor is the same after expansion."""
    network.sets[Set.DRIVER].print()
    assert type(network.sets[Set.DRIVER].nodes[0, TF.ACT].item()) == float
    network.sets[Set.DRIVER].expand_tensor()
    network.sets[Set.DRIVER].print()
    assert type(network.sets[Set.DRIVER].nodes[0, TF.ACT].item()) == float

def test_add_token_to_new_set(network):
    """Test adding a token to the new_set tensor."""
    new_token = Token(Type.P, 
        {
            TF.SET: Set.NEW_SET, 
            TF.INFERRED: B.TRUE, 
            TF.MODE: Mode.PARENT, 
            TF.ACT: 1.0,
            TF.ANALOG: null
        })
    token_reference: Ref_Token = network.node.add_token(new_token)
    assert token_reference is not None
    assert token_reference.set == Set.NEW_SET
    token = network.sets[Set.NEW_SET].token_op.get_single_token(token_reference)
    assert token[TF.TYPE] == Type.P
    assert token[TF.SET] == Set.NEW_SET
    assert token[TF.INFERRED] == B.TRUE
    assert token[TF.MODE] == Mode.PARENT
    assert token[TF.ACT] == 1.0
    assert token[TF.ANALOG] == null

def test_mask_update_after_add_token_new_set(network):
    """Test that the mask is updated after adding a token to the new_set tensor."""
    new_token = Token(Type.P, 
        {
            TF.SET: Set.NEW_SET, 
            TF.INFERRED: B.TRUE, 
            TF.MODE: Mode.PARENT, 
            TF.ACT: 1.0,
            TF.ANALOG: null
        })
    old_mask_p = network.sets[Set.NEW_SET].get_mask(Type.P)
    old_mask_rb = network.sets[Set.NEW_SET].get_mask(Type.RB)
    old_mask_po = network.sets[Set.NEW_SET].get_mask(Type.PO)

    token_reference: Ref_Token = network.node.add_token(new_token)
    assert token_reference is not None
    assert token_reference.set == Set.NEW_SET
    
    new_mask_p = network.sets[Set.NEW_SET].get_mask(Type.P)
    new_mask_rb = network.sets[Set.NEW_SET].get_mask(Type.RB)
    new_mask_po = network.sets[Set.NEW_SET].get_mask(Type.PO)
    
    diff_mask_p = new_mask_p.sum() - old_mask_p.sum()
    diff_mask_rb = new_mask_rb.sum() - old_mask_rb.sum()
    diff_mask_po = new_mask_po.sum() - old_mask_po.sum()
    
    assert diff_mask_p == 1
    assert diff_mask_rb == 0
    assert diff_mask_po == 0

def test_driver_map_expansion(network: Network):
    """ Test that the mapping objects are expanded correctly when the driver set is expanded."""
    # get number of driver tokens in mapping tensor for each set
    for set in [Set.RECIPIENT, Set.MEMORY]:
        assert network.mappings[set].adj_matrix is not None
        initial_d = network.sets[Set.DRIVER].nodes.shape[0]
        assert network.mappings[set].adj_matrix.shape[0] == network.sets[set].nodes.shape[0]
        assert network.mappings[set].adj_matrix.shape[1] == network.sets[Set.DRIVER].nodes.shape[0]
        while network.sets[Set.DRIVER].nodes.shape[0] == initial_d:
            new_token = Token(Type.P, {TF.SET: Set.DRIVER, TF.INFERRED: B.TRUE, TF.MODE: Mode.PARENT, TF.ACT: 1.0, TF.ANALOG: null})
            token_reference: Ref_Token = network.add_token(new_token)
        assert network.mappings[set].adj_matrix.shape[0] == network.sets[set].nodes.shape[0]
        assert network.mappings[set].adj_matrix.shape[1] == network.sets[Set.DRIVER].nodes.shape[0]

def test_sets_map_expansion(network: Network):
    """ Test that the mapping tensors are expanded correctly when the sets are expanded."""
    for set in [Set.RECIPIENT, Set.MEMORY]:
        assert network.mappings[set].adj_matrix is not None
        initial_set_size = network.sets[set].nodes.shape[0]
        assert network.mappings[set].adj_matrix.shape[0] == initial_set_size
        while network.sets[set].nodes.shape[0] == initial_set_size:
            new_token = Token(Type.P, {TF.SET: set, TF.INFERRED: B.TRUE, TF.MODE: Mode.PARENT, TF.ACT: 1.0, TF.ANALOG: null})
            token_reference: Ref_Token = network.add_token(new_token)
        assert network.mappings[set].adj_matrix.shape[0] == network.sets[set].nodes.shape[0]

def test_sets_link_expansion(network: Network):
    """ Test that the link tensors are expanded correctly when the sets are expanded."""
    for set in Set:
        assert network.links[set] is not None
        initial_set_size = network.sets[set].nodes.shape[0]
        assert network.links[set].shape[0] == initial_set_size
        while network.sets[set].nodes.shape[0] == initial_set_size:
            new_token = Token(Type.P, {TF.SET: set, TF.INFERRED: B.TRUE, TF.MODE: Mode.PARENT, TF.ACT: 1.0, TF.ANALOG: null})
            token_reference: Ref_Token = network.add_token(new_token)
        assert network.links[set].shape[0] == network.sets[set].nodes.shape[0]
