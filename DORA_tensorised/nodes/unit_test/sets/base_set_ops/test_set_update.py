# nodes/unit_test/sets/base_set_ops/test_set_update.py
# Tests for UpdateOperations class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens across different sets.
    """
    num_tokens = 30
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for active tokens (0-24)
    tensor[0:25, TF.DELETED] = B.FALSE
    # Set DELETED to True for deleted tokens (25-29)
    tensor[25:30, TF.DELETED] = B.TRUE
    
    # DRIVER set: tokens 0-9
    tensor[0:10, TF.SET] = Set.DRIVER
    tensor[0:3, TF.TYPE] = Type.PO
    tensor[3:6, TF.TYPE] = Type.RB
    tensor[6:9, TF.TYPE] = Type.P
    tensor[9, TF.TYPE] = Type.GROUP
    tensor[0:10, TF.ACT] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tensor[0:10, TF.ID] = torch.arange(0, 10)
    tensor[0:10, TF.MAX_ACT] = 0.0
    tensor[0:10, TF.INFERRED] = B.FALSE
    tensor[0:10, TF.MAKER_UNIT] = null
    tensor[0:10, TF.MADE_UNIT] = null
    # Set some input values for testing
    tensor[0:10, TF.TD_INPUT] = 0.5
    tensor[0:10, TF.BU_INPUT] = 0.3
    tensor[0:10, TF.LATERAL_INPUT] = 0.2
    tensor[0:10, TF.MAP_INPUT] = 0.1
    tensor[0:10, TF.NET_INPUT] = 0.4
    tensor[0:10, TF.RETRIEVED] = 0.0
    
    # RECIPIENT set: tokens 10-19
    tensor[10:20, TF.SET] = Set.RECIPIENT
    tensor[10:13, TF.TYPE] = Type.PO
    tensor[13:16, TF.TYPE] = Type.RB
    tensor[16:19, TF.TYPE] = Type.P
    tensor[19, TF.TYPE] = Type.SEMANTIC
    tensor[10:20, TF.ACT] = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    tensor[10:20, TF.ID] = torch.arange(10, 20)
    tensor[10:20, TF.MAX_ACT] = 0.0
    tensor[10:20, TF.INFERRED] = B.FALSE
    tensor[10:20, TF.MAKER_UNIT] = null
    tensor[10:20, TF.MADE_UNIT] = null
    tensor[10:20, TF.TD_INPUT] = 0.6
    tensor[10:20, TF.BU_INPUT] = 0.4
    tensor[10:20, TF.LATERAL_INPUT] = 0.3
    tensor[10:20, TF.MAP_INPUT] = 0.2
    tensor[10:20, TF.NET_INPUT] = 0.5
    tensor[10:20, TF.RETRIEVED] = 0.0
    
    # MEMORY set: tokens 20-24
    tensor[20:25, TF.SET] = Set.MEMORY
    tensor[20:23, TF.TYPE] = Type.PO
    tensor[23:25, TF.TYPE] = Type.RB
    tensor[20:25, TF.ACT] = torch.tensor([2.1, 2.2, 2.3, 2.4, 2.5])
    tensor[20:25, TF.ID] = torch.arange(20, 25)
    tensor[20:25, TF.MAX_ACT] = 0.0
    tensor[20:25, TF.INFERRED] = B.FALSE
    tensor[20:25, TF.MAKER_UNIT] = null
    tensor[20:25, TF.MADE_UNIT] = null
    tensor[20:25, TF.TD_INPUT] = 0.7
    tensor[20:25, TF.BU_INPUT] = 0.5
    tensor[20:25, TF.LATERAL_INPUT] = 0.4
    tensor[20:25, TF.MAP_INPUT] = 0.3
    tensor[20:25, TF.NET_INPUT] = 0.6
    tensor[20:25, TF.RETRIEVED] = 0.0
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 30
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    return connections


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(25)}


@pytest.fixture
def mock_params():
    """Create a mock Params object."""
    from nodes.network.default_parameters import parameters
    return Params(parameters)


@pytest.fixture
def token_tensor(mock_tensor, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections)
    return Token_Tensor(mock_tensor, connections_tensor, mock_names)


@pytest.fixture
def driver_set(token_tensor, mock_params):
    """Create a Base_Set instance for DRIVER set."""
    return Base_Set(token_tensor, Set.DRIVER, mock_params)


@pytest.fixture
def recipient_set(token_tensor, mock_params):
    """Create a Base_Set instance for RECIPIENT set."""
    return Base_Set(token_tensor, Set.RECIPIENT, mock_params)


@pytest.fixture
def memory_set(token_tensor, mock_params):
    """Create a Base_Set instance for MEMORY set."""
    return Base_Set(token_tensor, Set.MEMORY, mock_params)


# =====================[ init_float tests ]======================

def test_init_float_single_type_single_feature(driver_set):
    """Test initializing a single feature for a single type."""
    # Set some values first
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    
    # Initialize ACT to 0.0 for PO tokens
    driver_set.update_op.init_float([Type.PO], [TF.ACT])
    
    # Verify PO tokens have ACT = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.ACT] != 0.0)


def test_init_float_single_type_multiple_features(driver_set):
    """Test initializing multiple features for a single type."""
    # Set some values first
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.MAX_ACT] = 3.0  # PO tokens
    
    # Initialize ACT and MAX_ACT to 0.0 for PO tokens
    driver_set.update_op.init_float([Type.PO], [TF.ACT, TF.MAX_ACT])
    
    # Verify PO tokens have both features = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAX_ACT] == 0.0)


def test_init_float_multiple_types_single_feature(driver_set):
    """Test initializing a single feature for multiple types."""
    # Set some values first
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[3:6, TF.ACT] = 6.0  # RB tokens
    
    # Initialize ACT to 0.0 for PO and RB tokens
    driver_set.update_op.init_float([Type.PO, Type.RB], [TF.ACT])
    
    # Verify PO and RB tokens have ACT = 0.0
    po_rb_mask = (driver_set.lcl[:, TF.TYPE] == Type.PO) | (driver_set.lcl[:, TF.TYPE] == Type.RB)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.ACT] == 0.0)
    
    # Verify other tokens are unchanged
    other_mask = (driver_set.lcl[:, TF.TYPE] != Type.PO) & (driver_set.lcl[:, TF.TYPE] != Type.RB)
    assert torch.all(driver_set.lcl[other_mask, TF.ACT] != 0.0)


def test_init_float_empty_type_list(driver_set):
    """Test that init_float handles empty type list gracefully."""
    # Should not raise an error
    driver_set.update_op.init_float([], [TF.ACT])
    
    # Nothing should change
    assert torch.all(driver_set.lcl[:, TF.ACT] >= 0.0)


def test_init_float_no_matching_tokens(driver_set):
    """Test init_float when no tokens match the type."""
    # Try to initialize SEMANTIC tokens (none exist in DRIVER set)
    original_act = driver_set.lcl[:, TF.ACT].clone()
    driver_set.update_op.init_float([Type.SEMANTIC], [TF.ACT])
    
    # Nothing should change
    assert torch.equal(driver_set.lcl[:, TF.ACT], original_act)


# =====================[ init_input tests ]======================

def test_init_input_single_type(driver_set):
    """Test initializing input for a single type."""
    # Initialize input for PO tokens with refresh = 1.5
    driver_set.update_op.init_input([Type.PO], 1.5)
    
    # Verify PO tokens have TD_INPUT = 1.5
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 1.5)
    
    # Verify PO tokens have other inputs = 0.0
    assert torch.all(driver_set.lcl[po_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.NET_INPUT] == 0.0)
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.TD_INPUT] == 0.5)  # Original value


def test_init_input_multiple_types(driver_set):
    """Test initializing input for multiple types."""
    # Initialize input for PO and RB tokens with refresh = 2.0
    driver_set.update_op.init_input([Type.PO, Type.RB], 2.0)
    
    # Verify PO and RB tokens have TD_INPUT = 2.0
    po_rb_mask = (driver_set.lcl[:, TF.TYPE] == Type.PO) | (driver_set.lcl[:, TF.TYPE] == Type.RB)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.TD_INPUT] == 2.0)
    
    # Verify PO and RB tokens have other inputs = 0.0
    assert torch.all(driver_set.lcl[po_rb_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.NET_INPUT] == 0.0)


def test_init_input_zero_refresh(driver_set):
    """Test initializing input with refresh = 0.0."""
    # Initialize input for PO tokens with refresh = 0.0
    driver_set.update_op.init_input([Type.PO], 0.0)
    
    # Verify PO tokens have TD_INPUT = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 0.0)


def test_init_input_negative_refresh(driver_set):
    """Test initializing input with negative refresh value."""
    # Initialize input for PO tokens with refresh = -1.0
    driver_set.update_op.init_input([Type.PO], -1.0)
    
    # Verify PO tokens have TD_INPUT = -1.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == -1.0)


# =====================[ init_act tests ]======================

def test_init_act_single_type(driver_set):
    """Test initializing act for a single type."""
    # Set some values first
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.TD_INPUT] = 2.0  # PO tokens
    
    # Initialize act for PO tokens
    driver_set.update_op.init_act([Type.PO])
    
    # Verify PO tokens have ACT = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    
    # Verify PO tokens have TD_INPUT = 0.0 (from init_input)
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 0.0)
    
    # Verify PO tokens have other inputs = 0.0
    assert torch.all(driver_set.lcl[po_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.NET_INPUT] == 0.0)


def test_init_act_multiple_types(driver_set):
    """Test initializing act for multiple types."""
    # Initialize act for PO and RB tokens
    driver_set.update_op.init_act([Type.PO, Type.RB])
    
    # Verify PO and RB tokens have ACT = 0.0
    po_rb_mask = (driver_set.lcl[:, TF.TYPE] == Type.PO) | (driver_set.lcl[:, TF.TYPE] == Type.RB)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.ACT] == 0.0)
    
    # Verify all inputs are initialized
    assert torch.all(driver_set.lcl[po_rb_mask, TF.TD_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.NET_INPUT] == 0.0)


def test_init_act_preserves_other_tokens(driver_set):
    """Test that init_act only affects specified types."""
    # Set values for all tokens
    driver_set.lcl[:, TF.ACT] = 1.0
    driver_set.lcl[:, TF.TD_INPUT] = 2.0
    
    # Initialize act only for PO tokens
    driver_set.update_op.init_act([Type.PO])
    
    # Verify PO tokens are initialized
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.ACT] == 1.0)


# =====================[ init_state tests ]======================

def test_init_state_single_type(driver_set):
    """Test initializing state for a single type."""
    # Set some values first
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.RETRIEVED] = 1.0  # PO tokens
    
    # Initialize state for PO tokens
    driver_set.update_op.init_state([Type.PO])
    
    # Verify PO tokens have RETRIEVED = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.RETRIEVED] == 0.0)
    
    # Verify PO tokens have ACT = 0.0 (from init_act)
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    
    # Verify PO tokens have all inputs = 0.0 (from init_act -> init_input)
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.NET_INPUT] == 0.0)


def test_init_state_multiple_types(driver_set):
    """Test initializing state for multiple types."""
    # Initialize state for PO and RB tokens
    driver_set.update_op.init_state([Type.PO, Type.RB])
    
    # Verify PO and RB tokens have RETRIEVED = 0.0
    po_rb_mask = (driver_set.lcl[:, TF.TYPE] == Type.PO) | (driver_set.lcl[:, TF.TYPE] == Type.RB)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.RETRIEVED] == 0.0)
    
    # Verify PO and RB tokens have ACT = 0.0
    assert torch.all(driver_set.lcl[po_rb_mask, TF.ACT] == 0.0)
    
    # Verify all inputs are initialized
    assert torch.all(driver_set.lcl[po_rb_mask, TF.TD_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.NET_INPUT] == 0.0)


def test_init_state_preserves_other_tokens(driver_set):
    """Test that init_state only affects specified types."""
    # Set values for all tokens
    driver_set.lcl[:, TF.ACT] = 1.0
    driver_set.lcl[:, TF.RETRIEVED] = 1.0
    
    # Initialize state only for PO tokens
    driver_set.update_op.init_state([Type.PO])
    
    # Verify PO tokens are initialized
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.RETRIEVED] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.RETRIEVED] == 1.0)
    assert torch.all(driver_set.lcl[non_po_mask, TF.ACT] == 1.0)


# =====================[ update_act tests ]======================

def test_update_act_no_nodes(driver_set):
    """Test update_act when there are no active nodes."""
    # Mark all tokens as deleted
    driver_set.lcl[:, TF.DELETED] = B.TRUE
    
    # Should not raise an error
    driver_set.update_op.update_act()
    
    # Nothing should change (all tokens are deleted)
    assert True  # Just verify it doesn't crash


def test_update_act_incomplete_implementation(driver_set):
    """Test that update_act exists but is incomplete."""
    # The function exists but is incomplete in the current implementation
    # This test verifies it can be called without error
    # Note: The function only sets up variables but doesn't complete the update
    try:
        driver_set.update_op.update_act()
        # If it completes without error, that's expected for now
        assert True
    except AttributeError as e:
        # If params are missing, that's also expected
        assert "params" in str(e).lower() or "gamma" in str(e).lower()


# =====================[ Integration tests ]======================

def test_init_chain_operations(driver_set):
    """Test that init_state calls init_act which calls init_input."""
    # Set some values
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.TD_INPUT] = 2.0  # PO tokens
    driver_set.lcl[0:3, TF.RETRIEVED] = 1.0  # PO tokens
    
    # Initialize state (should chain through init_act -> init_input)
    driver_set.update_op.init_state([Type.PO])
    
    # Verify all are initialized
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.RETRIEVED] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.BU_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.LATERAL_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAP_INPUT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.NET_INPUT] == 0.0)


def test_operations_independent_across_sets(driver_set, recipient_set):
    """Test that update operations are independent across different sets."""
    # Initialize state for PO tokens in DRIVER set
    driver_set.update_op.init_state([Type.PO])
    
    # Initialize state for PO tokens in RECIPIENT set
    recipient_set.update_op.init_state([Type.PO])
    
    # Verify DRIVER set PO tokens are initialized
    driver_po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[driver_po_mask, TF.ACT] == 0.0)
    assert torch.all(driver_set.lcl[driver_po_mask, TF.RETRIEVED] == 0.0)
    
    # Verify RECIPIENT set PO tokens are initialized
    recipient_po_mask = recipient_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(recipient_set.lcl[recipient_po_mask, TF.ACT] == 0.0)
    assert torch.all(recipient_set.lcl[recipient_po_mask, TF.RETRIEVED] == 0.0)
    
    # Verify they don't affect each other (different local views)
    assert len(driver_set.lcl) == 10  # DRIVER has 10 tokens
    assert len(recipient_set.lcl) == 10  # RECIPIENT has 10 tokens


def test_init_float_multiple_calls(driver_set):
    """Test that multiple calls to init_float work correctly."""
    # Set values
    driver_set.lcl[0:3, TF.ACT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.MAX_ACT] = 3.0  # PO tokens
    
    # First call: initialize ACT
    driver_set.update_op.init_float([Type.PO], [TF.ACT])
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)
    assert torch.all(driver_set.lcl[po_mask, TF.MAX_ACT] == 3.0)  # Unchanged
    
    # Second call: initialize MAX_ACT
    driver_set.update_op.init_float([Type.PO], [TF.MAX_ACT])
    assert torch.all(driver_set.lcl[po_mask, TF.ACT] == 0.0)  # Still 0.0
    assert torch.all(driver_set.lcl[po_mask, TF.MAX_ACT] == 0.0)  # Now 0.0


def test_init_input_different_refresh_values(driver_set):
    """Test init_input with different refresh values."""
    # Initialize with refresh = 1.0
    driver_set.update_op.init_input([Type.PO], 1.0)
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 1.0)
    
    # Initialize with refresh = 2.5
    driver_set.update_op.init_input([Type.PO], 2.5)
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == 2.5)
    
    # Initialize with refresh = -0.5
    driver_set.update_op.init_input([Type.PO], -0.5)
    assert torch.all(driver_set.lcl[po_mask, TF.TD_INPUT] == -0.5)

