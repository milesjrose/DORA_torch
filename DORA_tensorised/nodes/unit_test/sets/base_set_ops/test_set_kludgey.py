# nodes/unit_test/sets/base_set_ops/test_set_kludgey.py
# Tests for KludgeyOperations class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, null, tensor_type
from nodes.network.single_nodes import Pairs


@pytest.fixture
def mock_tensor_with_preds():
    """
    Create a mock tensor with preds, RBs, and Ps for testing kludgey operations.
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all tokens
    tensor[:, TF.DELETED] = B.FALSE
    
    # DRIVER set: tokens 0-9
    tensor[0:9, TF.SET] = Set.DRIVER
    # Preds (PO with PRED=TRUE): tokens 0-2
    tensor[0:3, TF.TYPE] = Type.PO
    tensor[0:3, TF.PRED] = B.TRUE
    # RBs: tokens 3-5
    tensor[3:6, TF.TYPE] = Type.RB
    # Ps: tokens 6-8
    tensor[6:9, TF.TYPE] = Type.P
    
    # RECIPIENT set: tokens 10-19
    tensor[10:19, TF.SET] = Set.RECIPIENT
    # Preds: tokens 10-12
    tensor[10:13, TF.TYPE] = Type.PO
    tensor[10:13, TF.PRED] = B.TRUE
    # RBs: tokens 13-15
    tensor[13:16, TF.TYPE] = Type.RB
    # Ps: tokens 16-18
    tensor[16:19, TF.TYPE] = Type.P
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    return connections


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(20)}


@pytest.fixture
def mock_params():
    """Create a mock Params object."""
    from nodes.network.default_parameters import parameters
    return Params(parameters)


@pytest.fixture
def token_tensor(mock_tensor_with_preds, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections)
    return Token_Tensor(mock_tensor_with_preds, connections_tensor, mock_names)


@pytest.fixture
def driver_set(token_tensor, mock_params):
    """Create a Base_Set instance for DRIVER set."""
    return Base_Set(token_tensor, Set.DRIVER, mock_params)


@pytest.fixture
def recipient_set(token_tensor, mock_params):
    """Create a Base_Set instance for RECIPIENT set."""
    return Base_Set(token_tensor, Set.RECIPIENT, mock_params)


# =====================[ get_pred_rb_no_ps tests ]======================

def test_get_pred_rb_no_ps_no_connections(driver_set):
    """Test get_pred_rb_no_ps with no connections."""
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # With no connections, no RBs should be connected to Ps
    # So all RBs should be "no_p" RBs
    # But if preds aren't connected to RBs either, no pairs should be found
    assert len(result.get_list()) == 0


def test_get_pred_rb_no_ps_preds_connected_to_rb_no_p(driver_set):
    """Test get_pred_rb_no_ps when preds are connected to RBs that aren't connected to P."""
    # Set up connections:
    # Pred 0 -> RB 3 (RB 3 not connected to any P)
    # Pred 1 -> RB 4 (RB 4 not connected to any P)
    # RB 3 and RB 4 are not connected to any P
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # Preds 0 and 1 should form a pair since they're both connected to RBs with no P connections
    pairs_list = result.get_list()
    assert len(pairs_list) == 1
    # Check that the pair contains preds 0 and 1 (order may vary)
    pair = pairs_list[0]
    assert set(pair) == {0, 1}


def test_get_pred_rb_no_ps_some_rbs_connected_to_p(driver_set):
    """Test get_pred_rb_no_ps when some RBs are connected to P."""
    # Set up connections:
    # Pred 0 -> RB 3 (RB 3 not connected to P)
    # Pred 1 -> RB 4 (RB 4 connected to P 6)
    # RB 3 not connected to P
    # RB 4 -> P 6
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # Only pred 0 should be in pairs (connected to RB with no P)
    # Pred 1 is connected to RB 4 which is connected to P 6, so it shouldn't be included
    pairs_list = result.get_list()
    assert len(pairs_list) == 0  # Need at least 2 preds to form a pair


def test_get_pred_rb_no_ps_multiple_preds(driver_set):
    """Test get_pred_rb_no_ps with multiple preds connected to RBs with no P."""
    # Set up connections:
    # Pred 0 -> RB 3
    # Pred 1 -> RB 4
    # Pred 2 -> RB 5
    # None of the RBs are connected to P
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[2, 5] = True  # Pred 2 -> RB 5
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # Should have pairs: (0,1), (0,2), (1,2) = 3 pairs
    pairs_list = result.get_list()
    assert len(pairs_list) == 3
    pairs_set = {tuple(sorted(p)) for p in pairs_list}
    assert pairs_set == {(0, 1), (0, 2), (1, 2)}


def test_get_pred_rb_no_ps_rb_connected_to_multiple_ps(driver_set):
    """Test get_pred_rb_no_ps when RB is connected to multiple Ps (should still be excluded)."""
    # Set up connections:
    # Pred 0 -> RB 3
    # RB 3 -> P 6 and P 7
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[3, 7] = True  # RB 3 -> P 7
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # RB 3 is connected to P, so pred 0 shouldn't be included
    pairs_list = result.get_list()
    assert len(pairs_list) == 0


def test_get_pred_rb_no_ps_only_one_pred(driver_set):
    """Test get_pred_rb_no_ps with only one pred connected to RB with no P."""
    # Set up connections:
    # Pred 0 -> RB 3 (RB 3 not connected to P)
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_no_ps(pairs)
    
    # Need at least 2 preds to form a pair
    pairs_list = result.get_list()
    assert len(pairs_list) == 0


def test_get_pred_rb_no_ps_different_sets(driver_set, recipient_set):
    """Test that get_pred_rb_no_ps only considers tokens in the same set."""
    # Set up connections in DRIVER set
    driver_set.glbl.connections.connections[0, 3] = True  # DRIVER Pred 0 -> DRIVER RB 3
    
    # Set up connections in RECIPIENT set
    recipient_set.glbl.connections.connections[10, 13] = True  # RECIPIENT Pred 10 -> RECIPIENT RB 13
    
    # DRIVER set test
    pairs_driver = Pairs()
    result_driver = driver_set.kludgey_op.get_pred_rb_no_ps(pairs_driver)
    assert len(result_driver.get_list()) == 0  # Only one pred in DRIVER
    
    # RECIPIENT set test
    pairs_recipient = Pairs()
    result_recipient = recipient_set.kludgey_op.get_pred_rb_no_ps(pairs_recipient)
    assert len(result_recipient.get_list()) == 0  # Only one pred in RECIPIENT


# =====================[ get_pred_rb_shared_p tests ]======================

def test_get_pred_rb_shared_p_no_connections(driver_set):
    """Test get_pred_rb_shared_p with no connections."""
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # With no connections, no pairs should be found
    assert len(result.get_list()) == 0


def test_get_pred_rb_shared_p_preds_share_p(driver_set):
    """Test get_pred_rb_shared_p when preds are connected to RBs that share the same P."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6
    # Pred 1 -> RB 4 -> P 6
    # Both RBs share the same P (P 6)
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Preds 0 and 1 should form a pair since their RBs share P 6
    pairs_list = result.get_list()
    assert len(pairs_list) == 1
    pair = pairs_list[0]
    assert set(pair) == {0, 1}


def test_get_pred_rb_shared_p_preds_dont_share_p(driver_set):
    """Test get_pred_rb_shared_p when preds are connected to RBs that don't share the same P."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6
    # Pred 1 -> RB 4 -> P 7
    # RBs don't share the same P
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[4, 7] = True  # RB 4 -> P 7
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Preds shouldn't form a pair since their RBs don't share a P
    pairs_list = result.get_list()
    assert len(pairs_list) == 0


def test_get_pred_rb_shared_p_multiple_preds_multiple_ps(driver_set):
    """Test get_pred_rb_shared_p with multiple preds and multiple Ps."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6
    # Pred 1 -> RB 4 -> P 6, P 7
    # Pred 2 -> RB 5 -> P 7
    # Preds 0 and 1 share P 6, Preds 1 and 2 share P 7
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[2, 5] = True  # Pred 2 -> RB 5
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    driver_set.glbl.connections.connections[4, 7] = True  # RB 4 -> P 7
    driver_set.glbl.connections.connections[5, 7] = True  # RB 5 -> P 7
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Should have pairs: (0,1) share P 6, (1,2) share P 7
    pairs_list = result.get_list()
    assert len(pairs_list) == 2
    pairs_set = {tuple(sorted(p)) for p in pairs_list}
    assert pairs_set == {(0, 1), (1, 2)}


def test_get_pred_rb_shared_p_rb_not_connected_to_p(driver_set):
    """Test get_pred_rb_shared_p when RB is not connected to any P (should be excluded)."""
    # Set up connections:
    # Pred 0 -> RB 3 (RB 3 not connected to P)
    # Pred 1 -> RB 4 -> P 6
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # RB 3 is not connected to P, so pred 0 shouldn't be considered
    # Only pred 1 is connected to RB with P, so no pairs
    pairs_list = result.get_list()
    assert len(pairs_list) == 0


def test_get_pred_rb_shared_p_only_one_pred(driver_set):
    """Test get_pred_rb_shared_p with only one pred connected to RB with P."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Need at least 2 preds to form a pair
    pairs_list = result.get_list()
    assert len(pairs_list) == 0


def test_get_pred_rb_shared_p_rb_connected_to_multiple_ps(driver_set):
    """Test get_pred_rb_shared_p when RB is connected to multiple Ps."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6, P 7
    # Pred 1 -> RB 4 -> P 6, P 7
    # Both RBs share both Ps
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[3, 7] = True  # RB 3 -> P 7
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    driver_set.glbl.connections.connections[4, 7] = True  # RB 4 -> P 7
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Preds 0 and 1 should form a pair since their RBs share at least one P
    pairs_list = result.get_list()
    assert len(pairs_list) == 1
    pair = pairs_list[0]
    assert set(pair) == {0, 1}


def test_get_pred_rb_shared_p_different_sets(driver_set, recipient_set):
    """Test that get_pred_rb_shared_p only considers tokens in the same set."""
    # Set up connections in DRIVER set
    driver_set.glbl.connections.connections[0, 3] = True  # DRIVER Pred 0 -> DRIVER RB 3
    driver_set.glbl.connections.connections[3, 6] = True  # DRIVER RB 3 -> DRIVER P 6
    
    # Set up connections in RECIPIENT set
    recipient_set.glbl.connections.connections[10, 13] = True  # RECIPIENT Pred 10 -> RECIPIENT RB 13
    recipient_set.glbl.connections.connections[13, 16] = True  # RECIPIENT RB 13 -> RECIPIENT P 16
    
    # DRIVER set test
    pairs_driver = Pairs()
    result_driver = driver_set.kludgey_op.get_pred_rb_shared_p(pairs_driver)
    assert len(result_driver.get_list()) == 0  # Only one pred in DRIVER
    
    # RECIPIENT set test
    pairs_recipient = Pairs()
    result_recipient = recipient_set.kludgey_op.get_pred_rb_shared_p(pairs_recipient)
    assert len(result_recipient.get_list()) == 0  # Only one pred in RECIPIENT


def test_get_pred_rb_shared_p_complex_scenario(driver_set):
    """Test get_pred_rb_shared_p with a complex scenario."""
    # Set up connections:
    # Pred 0 -> RB 3 -> P 6
    # Pred 1 -> RB 4 -> P 6, P 7
    # Pred 2 -> RB 5 -> P 7, P 8
    # Expected pairs: (0,1) share P 6, (1,2) share P 7
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[2, 5] = True  # Pred 2 -> RB 5
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    driver_set.glbl.connections.connections[4, 7] = True  # RB 4 -> P 7
    driver_set.glbl.connections.connections[5, 7] = True  # RB 5 -> P 7
    driver_set.glbl.connections.connections[5, 8] = True  # RB 5 -> P 8
    
    pairs = Pairs()
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Should have pairs: (0,1) share P 6, (1,2) share P 7
    pairs_list = result.get_list()
    assert len(pairs_list) == 2
    pairs_set = {tuple(sorted(p)) for p in pairs_list}
    assert pairs_set == {(0, 1), (1, 2)}


def test_get_pred_rb_shared_p_empty_pairs_input(driver_set):
    """Test that get_pred_rb_shared_p works with an empty Pairs object."""
    pairs = Pairs()
    assert len(pairs.get_list()) == 0
    
    # Set up connections
    driver_set.glbl.connections.connections[0, 3] = True  # Pred 0 -> RB 3
    driver_set.glbl.connections.connections[1, 4] = True  # Pred 1 -> RB 4
    driver_set.glbl.connections.connections[3, 6] = True  # RB 3 -> P 6
    driver_set.glbl.connections.connections[4, 6] = True  # RB 4 -> P 6
    
    result = driver_set.kludgey_op.get_pred_rb_shared_p(pairs)
    
    # Should add pairs to the input object
    pairs_list = result.get_list()
    assert len(pairs_list) == 1
    assert result is pairs  # Should return the same object

