# nodes/tests/set_ops/test_set_token.py
# Tests for set token operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.network import Network
from nodes.network.single_nodes import Token, Pairs
from nodes.enums import *
from nodes.network import Ref_Token

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_get_set_feature(network: Network):
    """Test get_feature and set_feature."""
    driver = network.driver()
    ref_token = driver.token_ops.get_reference(name='lover')
    
    # Check initial activation
    initial_act = driver.token_ops.get_feature(ref_token, TF.ACT)
    assert isinstance(initial_act, float)
    
    # Set new activation
    new_act = 0.5
    driver.token_ops.set_feature(ref_token, TF.ACT, new_act)
    
    # Check new activation
    updated_act = driver.token_ops.get_feature(ref_token, TF.ACT)
    assert updated_act == pytest.approx(new_act)

def test_get_set_name(network: Network):
    """Test get_name and set_name."""
    driver = network.driver()
    ref_token = driver.token_ops.get_reference(name='lover')

    # Check initial name
    initial_name = driver.token_ops.get_name(ref_token)
    assert initial_name == 'lover'
    
    # Set new name
    new_name = 'hater'
    driver.token_ops.set_name(ref_token, new_name)

    # Check new name
    updated_name = driver.token_ops.get_name(ref_token)
    assert updated_name == new_name

    # Check if we can get the token by its new name
    ref_by_new_name = driver.token_ops.get_reference(name=new_name)
    assert ref_by_new_name.ID == ref_token.ID

def test_get_index(network: Network):
    """Test get_index."""
    driver = network.driver()
    ref_token = driver.token_ops.get_reference(name='lover')
    
    index = driver.token_ops.get_index(ref_token)
    assert isinstance(index, int)
    
    # Verify that the index points to the correct token
    token_from_tensor = driver.nodes[index]
    token_id = token_from_tensor[TF.ID].item()
    assert token_id == ref_token.ID

def test_get_reference(network: Network):
    """Test get_reference by id, index, and name."""
    driver = network.driver()
    
    # Get by name
    ref_by_name = driver.token_ops.get_reference(name='lover')
    assert ref_by_name is not None

    # Get by ID
    ref_by_id = driver.token_ops.get_reference(id=ref_by_name.ID)
    assert ref_by_id.ID == ref_by_name.ID
    
    # Get by index
    index = driver.token_ops.get_index(ref_by_name)
    ref_by_index = driver.token_ops.get_reference(index=index)
    assert ref_by_index.ID == ref_by_name.ID

def test_get_single_token(network: Network):
    """Test get_single_token."""
    driver = network.driver()
    ref_token = driver.token_ops.get_reference(name='lover')

    # Get a copy
    token_copy = driver.token_ops.get_single_token(ref_token, copy=True)
    assert isinstance(token_copy, Token)
    assert token_copy[TF.ID] == ref_token.ID
    
    # Modify copy and check that original is unchanged
    token_copy[TF.ACT] = 0.99
    original_act = driver.token_ops.get_feature(ref_token, TF.ACT)
    assert original_act != 0.99

    # Get a reference (not a copy)
    token_ref = driver.token_ops.get_single_token(ref_token, copy=False)
    token_ref[TF.ACT] = 0.88
    original_act_after_ref_change = driver.token_ops.get_feature(ref_token, TF.ACT)
    assert original_act_after_ref_change == pytest.approx(0.88)

def test_connect_and_get_connected(network: Network):
    """Test connect and get_connected_tokens."""
    driver = network.driver()
    lover = driver.token_ops.get_reference(name='lover')
    mary = driver.token_ops.get_reference(name='Mary')
    
    # Connect lover to Mary
    driver.token_ops.connect(lover, mary)
    
    # Check if Mary is in lover's connections
    connected_tokens = driver.token_ops.get_connected_tokens(lover)
    connected_ids = [tk.ID for tk in connected_tokens]
    assert mary.ID in connected_ids

def test_get_most_active_token(network: Network):
    """Test get_most_active_token."""
    driver = network.driver()
    lover = driver.token_ops.get_reference(name='lover')
    mary = driver.token_ops.get_reference(name='Mary')

    # Set activations
    driver.token_ops.set_feature(lover, TF.ACT, 0.7)
    driver.token_ops.set_feature(mary, TF.ACT, 0.9)

    most_active = driver.token_ops.get_most_active_token()
    assert most_active.ID == mary.ID

    most_active_id = driver.token_ops.get_most_active_token(id=True)
    assert most_active_id == mary.ID

def test_get_reference_multiple(network: Network):
    """Test get_reference_multiple."""
    driver = network.driver()
    
    # Get all PO tokens
    po_tokens = driver.token_ops.get_reference_multiple(types=[Type.PO])
    
    # From sim.py, we have 'lover', 'beloved', 'jealous_act', 'jealous_pat' which are POs in driver
    # But NetworkBuilder creates them. Let's check how many there are.
    
    po_mask = driver.tensor_op.get_mask(Type.PO)
    num_po_tokens = torch.sum(po_mask).item()

    assert len(po_tokens) == num_po_tokens
    
    for token_ref in po_tokens:
        token = driver.token_ops.get_single_token(token_ref)
        assert token[TF.TYPE] == Type.PO

def test_get_pred_rb_no_ps(network: Network):
    """Test get_pred_rb_no_ps."""
    driver = network.driver()
    
    # Add tokens: 2 preds, 1 RB, 0 Ps
    pred1_ref = driver.tensor_op.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.PRED: B.TRUE}))
    pred2_ref = driver.tensor_op.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.PRED: B.TRUE}))
    rb_ref = driver.tensor_op.add_token(Token(Type.RB, {TF.SET: Set.DRIVER}))

    # Connect preds to RB
    driver.token_ops.connect(pred1_ref, rb_ref)
    driver.token_ops.connect(pred2_ref, rb_ref)

    # The function should find the pair (pred1, pred2)
    pairs = driver.token_ops.get_pred_rb_no_ps(Pairs())
    pair_list = pairs.get_list()
    
    pred1_idx = driver.token_ops.get_index(pred1_ref)
    pred2_idx = driver.token_ops.get_index(pred2_ref)

    assert len(pair_list) == 1
    found_pair = pair_list[0]
    assert (pred1_idx in found_pair and pred2_idx in found_pair)

def test_get_pred_rb_shared_p(network: Network):
    """Test get_pred_rb_shared_p."""
    driver = network.driver()

    # Add tokens: 2 preds, 2 RBs, 1 P
    pred1_ref = driver.tensor_op.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.PRED: B.TRUE}))
    pred2_ref = driver.tensor_op.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.PRED: B.TRUE}))
    rb1_ref = driver.tensor_op.add_token(Token(Type.RB, {TF.SET: Set.DRIVER}))
    rb2_ref = driver.tensor_op.add_token(Token(Type.RB, {TF.SET: Set.DRIVER}))
    p_ref = driver.tensor_op.add_token(Token(Type.P, {TF.SET: Set.DRIVER}))

    # Connect structure:
    # pred1 -> rb1 -> p
    # pred2 -> rb2 -> p
    driver.token_ops.connect(pred1_ref, rb1_ref)
    driver.token_ops.connect(rb1_ref, p_ref)
    driver.token_ops.connect(pred2_ref, rb2_ref)
    driver.token_ops.connect(rb2_ref, p_ref)

    # The function should find the pair (pred1, pred2)
    pairs = driver.token_ops.get_pred_rb_shared_p(Pairs())
    pair_list = pairs.get_list()

    pred1_idx = driver.token_ops.get_index(pred1_ref)
    pred2_idx = driver.token_ops.get_index(pred2_ref)
    
    assert len(pair_list) == 1
    found_pair = pair_list[0]
    assert (pred1_idx in found_pair and pred2_idx in found_pair)