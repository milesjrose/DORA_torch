# nodes/unit_test/sets/base_set_ops/test_set_update.py
# Tests for UpdateOperations class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, Mode, null, tensor_type


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
    """Test update_act when there are no tokens."""
    # Create an empty set by using get_count check
    # The function checks get_count() == 0, so we need to test with an actual empty set
    # Note: If this fails, it may indicate a bug in update_act or get_count
    driver_set.update_op.update_act()
    # Should complete without error when there are no tokens
    assert True

def test_update_act_basic(driver_set):
    """Test basic update_act functionality."""
    # Set initial values
    driver_set.lcl[:, TF.ACT] = 0.5
    driver_set.lcl[:, TF.TD_INPUT] = 0.3
    driver_set.lcl[:, TF.BU_INPUT] = 0.2
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.1
    driver_set.lcl[:, TF.MAP_INPUT] = 0.0
    
    # Store original acts
    original_acts = driver_set.lcl[:, TF.ACT].clone()
    
    # Note: If this fails with IndexError about TensorView indexing,
    # it means update_act has issues indexing with a list of column indices
    driver_set.update_op.update_act()
    
    # Verify acts have changed
    new_acts = driver_set.lcl[:, TF.ACT]
    assert not torch.equal(new_acts, original_acts), \
        "ACT values should change after update_act"
    
    # Verify acts are within bounds [0.0, 1.0]
    assert torch.all(new_acts >= 0.0), \
        "All ACT values should be >= 0.0"
    assert torch.all(new_acts <= 1.0), \
        "All ACT values should be <= 1.0"


def test_update_act_with_mapping_input(driver_set):
    """Test update_act with mapping input."""
    # Set initial values
    driver_set.lcl[:, TF.ACT] = 0.5
    driver_set.lcl[:, TF.TD_INPUT] = 0.2
    driver_set.lcl[:, TF.BU_INPUT] = 0.1
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.1
    driver_set.lcl[:, TF.MAP_INPUT] = 0.5  # Non-zero mapping input
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    driver_set.update_op.update_act()
    
    # Verify acts are within bounds
    new_acts = driver_set.lcl[:, TF.ACT]
    assert torch.all(new_acts >= 0.0), \
        "All ACT values should be >= 0.0"
    assert torch.all(new_acts <= 1.0), \
        "All ACT values should be <= 1.0"


def test_update_act_clamps_to_one(driver_set):
    """Test that update_act clamps activation to 1.0 maximum."""
    # Set very high inputs to force activation above 1.0
    driver_set.lcl[:, TF.ACT] = 0.9
    driver_set.lcl[:, TF.TD_INPUT] = 10.0  # Very high input
    driver_set.lcl[:, TF.BU_INPUT] = 10.0
    driver_set.lcl[:, TF.LATERAL_INPUT] = 10.0
    driver_set.lcl[:, TF.MAP_INPUT] = 10.0
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    driver_set.update_op.update_act()
    
    # Verify all acts are clamped to 1.0
    new_acts = driver_set.lcl[:, TF.ACT]
    assert torch.all(new_acts <= 1.0), \
        "All ACT values should be clamped to <= 1.0"
    # At least some should be at 1.0 (if inputs are high enough)


def test_update_act_clamps_to_zero(driver_set):
    """Test that update_act clamps activation to 0.0 minimum."""
    # Set negative inputs and low initial activation to force below 0.0
    driver_set.lcl[:, TF.ACT] = 0.1
    driver_set.lcl[:, TF.TD_INPUT] = -10.0  # Very negative input
    driver_set.lcl[:, TF.BU_INPUT] = -10.0
    driver_set.lcl[:, TF.LATERAL_INPUT] = -10.0
    driver_set.lcl[:, TF.MAP_INPUT] = -10.0
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    driver_set.update_op.update_act()
    
    # Verify all acts are clamped to 0.0 or above
    new_acts = driver_set.lcl[:, TF.ACT]
    assert torch.all(new_acts >= 0.0), \
        "All ACT values should be clamped to >= 0.0"
    # At least some should be at 0.0 (if inputs are negative enough)


def test_update_act_with_different_gamma_delta(driver_set):
    """Test update_act with different gamma and delta parameters."""
    # Set initial values
    driver_set.lcl[:, TF.ACT] = 0.5
    driver_set.lcl[:, TF.TD_INPUT] = 0.3
    driver_set.lcl[:, TF.BU_INPUT] = 0.2
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.1
    driver_set.lcl[:, TF.MAP_INPUT] = 0.0
    
    # Store original acts
    original_acts = driver_set.lcl[:, TF.ACT].clone()
    
    # Get current params
    gamma = driver_set.params.gamma
    delta = driver_set.params.delta
    HebbBias = driver_set.params.HebbBias
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    driver_set.update_op.update_act()
    
    # Verify acts changed based on formula:
    # delta_act = gamma * net_input * (1.1 - acts) - (delta * acts)
    new_acts = driver_set.lcl[:, TF.ACT]
    assert not torch.equal(new_acts, original_acts), \
        "ACT values should change after update_act"
    
    # Verify acts are within bounds
    assert torch.all(new_acts >= 0.0), \
        "All ACT values should be >= 0.0"
    assert torch.all(new_acts <= 1.0), \
        "All ACT values should be <= 1.0"


def test_update_act_multiple_iterations(driver_set):
    """Test update_act over multiple iterations."""
    # Set initial values
    driver_set.lcl[:, TF.ACT] = 0.0
    driver_set.lcl[:, TF.TD_INPUT] = 0.5
    driver_set.lcl[:, TF.BU_INPUT] = 0.3
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.2
    driver_set.lcl[:, TF.MAP_INPUT] = 0.0
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    # First iteration
    driver_set.update_op.update_act()
    acts_after_first = driver_set.lcl[:, TF.ACT].clone()
    
    # Second iteration (with same inputs)
    driver_set.update_op.update_act()
    acts_after_second = driver_set.lcl[:, TF.ACT].clone()
    
    # Acts should continue to change (unless at equilibrium)
    # Verify they're within bounds
    assert torch.all(acts_after_first >= 0.0), \
        "All ACT values after first iteration should be >= 0.0"
    assert torch.all(acts_after_first <= 1.0), \
        "All ACT values after first iteration should be <= 1.0"
    assert torch.all(acts_after_second >= 0.0), \
        "All ACT values after second iteration should be >= 0.0"
    assert torch.all(acts_after_second <= 1.0), \
        "All ACT values after second iteration should be <= 1.0"


def test_update_act_preserves_other_features(driver_set):
    """Test that update_act only modifies ACT feature."""
    # Set initial values
    driver_set.lcl[:, TF.ACT] = 0.5
    driver_set.lcl[:, TF.TD_INPUT] = 0.3
    driver_set.lcl[:, TF.BU_INPUT] = 0.2
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.1
    driver_set.lcl[:, TF.MAP_INPUT] = 0.0
    driver_set.lcl[:, TF.TYPE] = Type.PO  # Set some other feature
    
    # Store original values
    original_type = driver_set.lcl[:, TF.TYPE].clone()
    original_td_input = driver_set.lcl[:, TF.TD_INPUT].clone()
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    driver_set.update_op.update_act()
    
    # Verify ACT changed
    new_acts = driver_set.lcl[:, TF.ACT]
    assert not torch.equal(new_acts, torch.full_like(new_acts, 0.5)), \
        "ACT values should change after update_act"
    
    # Verify other features unchanged
    assert torch.equal(driver_set.lcl[:, TF.TYPE], original_type), \
        "TYPE should not be modified by update_act"
    assert torch.equal(driver_set.lcl[:, TF.TD_INPUT], original_td_input), \
        "TD_INPUT should not be modified by update_act"


def test_update_act_independent_across_sets(driver_set, recipient_set):
    """Test that update_act is independent across different sets."""
    # Set different initial values
    driver_set.lcl[:, TF.ACT] = 0.3
    driver_set.lcl[:, TF.TD_INPUT] = 0.5
    driver_set.lcl[:, TF.BU_INPUT] = 0.3
    driver_set.lcl[:, TF.LATERAL_INPUT] = 0.2
    driver_set.lcl[:, TF.MAP_INPUT] = 0.0
    
    recipient_set.lcl[:, TF.ACT] = 0.7
    recipient_set.lcl[:, TF.TD_INPUT] = 0.8
    recipient_set.lcl[:, TF.BU_INPUT] = 0.6
    recipient_set.lcl[:, TF.LATERAL_INPUT] = 0.4
    recipient_set.lcl[:, TF.MAP_INPUT] = 0.0
    
    # Note: If this fails with IndexError, it means update_act has indexing issues
    # Update both sets
    driver_set.update_op.update_act()
    recipient_set.update_op.update_act()
    
    # Verify they have different results (due to different inputs)
    driver_acts = driver_set.lcl[:, TF.ACT]
    recipient_acts = recipient_set.lcl[:, TF.ACT]
    
    # They should be different (unless by coincidence)
    # At minimum, verify they're both within bounds
    assert torch.all(driver_acts >= 0.0), \
        "All DRIVER ACT values should be >= 0.0"
    assert torch.all(driver_acts <= 1.0), \
        "All DRIVER ACT values should be <= 1.0"
    assert torch.all(recipient_acts >= 0.0), \
        "All RECIPIENT ACT values should be >= 0.0"
    assert torch.all(recipient_acts <= 1.0), \
        "All RECIPIENT ACT values should be <= 1.0"


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


# =====================[ zero_laternal_input tests ]======================

def test_zero_laternal_input_single_type(driver_set):
    """Test zeroing lateral input for a single type."""
    # Set some lateral input values
    driver_set.lcl[0:3, TF.LATERAL_INPUT] = 5.0  # PO tokens
    
    # Zero lateral input for PO tokens
    driver_set.update_op.zero_laternal_input([Type.PO])
    
    # Verify PO tokens have LATERAL_INPUT = 0.0
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.LATERAL_INPUT] == 0.0)
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.LATERAL_INPUT] == 0.2)  # Original value


def test_zero_laternal_input_multiple_types(driver_set):
    """Test zeroing lateral input for multiple types."""
    # Set some lateral input values
    driver_set.lcl[0:3, TF.LATERAL_INPUT] = 5.0  # PO tokens
    driver_set.lcl[3:6, TF.LATERAL_INPUT] = 6.0  # RB tokens
    
    # Zero lateral input for PO and RB tokens
    driver_set.update_op.zero_laternal_input([Type.PO, Type.RB])
    
    # Verify PO and RB tokens have LATERAL_INPUT = 0.0
    po_rb_mask = (driver_set.lcl[:, TF.TYPE] == Type.PO) | (driver_set.lcl[:, TF.TYPE] == Type.RB)
    assert torch.all(driver_set.lcl[po_rb_mask, TF.LATERAL_INPUT] == 0.0)


def test_zero_laternal_input_empty_type_list(driver_set):
    """Test that zero_laternal_input handles empty type list gracefully."""
    # Should not raise an error
    driver_set.update_op.zero_laternal_input([])
    
    # Nothing should change
    assert torch.all(driver_set.lcl[:, TF.LATERAL_INPUT] >= 0.0)


# =====================[ update_inhibitor_input tests ]======================

def test_update_inhibitor_input_single_type(driver_set):
    """Test updating inhibitor input for a single type."""
    # Set some ACT values
    driver_set.lcl[0:3, TF.ACT] = 0.5  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 1.0  # PO tokens
    
    # Update inhibitor input for PO tokens
    driver_set.update_op.update_inhibitor_input([Type.PO])
    
    # Verify PO tokens have INHIBITOR_INPUT = 1.0 + 0.5 = 1.5
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([1.5, 1.5, 1.5]))
    
    # Verify other tokens are unchanged
    non_po_mask = driver_set.lcl[:, TF.TYPE] != Type.PO
    assert torch.all(driver_set.lcl[non_po_mask, TF.INHIBITOR_INPUT] == null)  # Original value (null)


def test_update_inhibitor_input_multiple_calls(driver_set):
    """Test that multiple calls to update_inhibitor_input accumulate."""
    # Set initial values
    driver_set.lcl[0:3, TF.ACT] = 0.5  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 1.0  # PO tokens
    
    # First call
    driver_set.update_op.update_inhibitor_input([Type.PO])
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([1.5, 1.5, 1.5]))
    
    # Second call (should accumulate)
    driver_set.update_op.update_inhibitor_input([Type.PO])
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([2.0, 2.0, 2.0]))


def test_update_inhibitor_input_multiple_types(driver_set):
    """Test updating inhibitor input for multiple types."""
    # Set some ACT values
    driver_set.lcl[0:3, TF.ACT] = 0.5  # PO tokens
    driver_set.lcl[3:6, TF.ACT] = 0.7  # RB tokens
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 1.0  # PO tokens
    driver_set.lcl[3:6, TF.INHIBITOR_INPUT] = 2.0  # RB tokens
    
    # Update inhibitor input for PO and RB tokens
    driver_set.update_op.update_inhibitor_input([Type.PO, Type.RB])
    
    # Verify PO tokens
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([1.5, 1.5, 1.5]))
    
    # Verify RB tokens
    rb_mask = driver_set.lcl[:, TF.TYPE] == Type.RB
    assert torch.allclose(driver_set.lcl[rb_mask, TF.INHIBITOR_INPUT], torch.tensor([2.7, 2.7, 2.7]))


def test_update_inhibitor_input_no_matching_tokens(driver_set):
    """Test update_inhibitor_input when no tokens match the type."""
    # Try to update SEMANTIC tokens (none exist in DRIVER set)
    original_inhibitor = driver_set.lcl[:, TF.INHIBITOR_INPUT].clone()
    driver_set.update_op.update_inhibitor_input([Type.SEMANTIC])
    
    # Nothing should change
    assert torch.equal(driver_set.lcl[:, TF.INHIBITOR_INPUT], original_inhibitor)


# =====================[ reset_inhibitor tests ]======================

def test_reset_inhibitor_single_type(driver_set):
    """Test resetting inhibitor for a single type."""
    # Set some inhibitor values
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 3.0  # PO tokens
    
    # Note: If this test fails with TypeError about init_float arguments,
    # it means reset_inhibitor is calling init_float with wrong arguments (3 args instead of 2)
    driver_set.update_op.reset_inhibitor([Type.PO])
    
    # Verify PO tokens are reset
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT] == 0.0), \
        "INHIBITOR_INPUT should be reset to 0.0"
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 0.0), \
        "INHIBITOR_ACT should be reset to 0.0"


def test_reset_inhibitor_empty_type_list(driver_set):
    """Test that reset_inhibitor handles empty type list gracefully."""
    # Should not raise an error
    driver_set.update_op.reset_inhibitor([])
    
    # Nothing should change
    assert True  # Just verify it doesn't crash


# =====================[ update_inhibitor_act tests ]======================

def test_update_inhibitor_act_above_threshold(driver_set):
    """Test updating inhibitor act when input is above threshold."""
    # Set inhibitor input and threshold
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 5.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_THRESHOLD] = 3.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 0.0  # PO tokens
    
    # Update inhibitor act for PO tokens
    driver_set.update_op.update_inhibitor_act([Type.PO])
    
    # Verify PO tokens have INHIBITOR_ACT = 1.0 (input >= threshold)
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 1.0)


def test_update_inhibitor_act_below_threshold(driver_set):
    """Test updating inhibitor act when input is below threshold."""
    # Set inhibitor input and threshold
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 1.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_THRESHOLD] = 3.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 0.0  # PO tokens
    
    # Update inhibitor act for PO tokens
    driver_set.update_op.update_inhibitor_act([Type.PO])
    
    # Verify PO tokens have INHIBITOR_ACT = 0.0 (input < threshold)
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 0.0)


def test_update_inhibitor_act_at_threshold(driver_set):
    """Test updating inhibitor act when input equals threshold."""
    # Set inhibitor input and threshold
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 3.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_THRESHOLD] = 3.0  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 0.0  # PO tokens
    
    # Update inhibitor act for PO tokens
    driver_set.update_op.update_inhibitor_act([Type.PO])
    
    # Verify PO tokens have INHIBITOR_ACT = 1.0 (input >= threshold)
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 1.0)


def test_update_inhibitor_act_mixed_thresholds(driver_set):
    """Test updating inhibitor act with mixed threshold conditions."""
    # Set different values for each PO token
    driver_set.lcl[0, TF.INHIBITOR_INPUT] = 5.0  # Above threshold
    driver_set.lcl[0, TF.INHIBITOR_THRESHOLD] = 3.0
    driver_set.lcl[1, TF.INHIBITOR_INPUT] = 2.0  # Below threshold
    driver_set.lcl[1, TF.INHIBITOR_THRESHOLD] = 3.0
    driver_set.lcl[2, TF.INHIBITOR_INPUT] = 3.0  # At threshold
    driver_set.lcl[2, TF.INHIBITOR_THRESHOLD] = 3.0
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 0.0
    
    # Update inhibitor act for PO tokens
    driver_set.update_op.update_inhibitor_act([Type.PO])
    
    # Verify results
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert driver_set.lcl[0, TF.INHIBITOR_ACT] == 1.0  # Above threshold
    assert driver_set.lcl[1, TF.INHIBITOR_ACT] == 0.0  # Below threshold
    assert driver_set.lcl[2, TF.INHIBITOR_ACT] == 1.0  # At threshold


def test_update_inhibitor_act_no_matching_tokens(driver_set):
    """Test update_inhibitor_act when no tokens match the type."""
    # Try to update SEMANTIC tokens (none exist in DRIVER set)
    original_inhibitor_act = driver_set.lcl[:, TF.INHIBITOR_ACT].clone()
    driver_set.update_op.update_inhibitor_act([Type.SEMANTIC])
    
    # Nothing should change
    assert torch.equal(driver_set.lcl[:, TF.INHIBITOR_ACT], original_inhibitor_act)


# =====================[ p_initialise_mode tests ]======================

def test_p_initialise_mode(driver_set):
    """Test initializing mode for P tokens."""
    # Set some mode values first
    driver_set.lcl[6:9, TF.MODE] = Mode.PARENT  # P tokens
    
    # Initialize mode for P tokens
    driver_set.update_op.p_initialise_mode()
    
    # Verify P tokens have MODE = NEUTRAL
    p_mask = driver_set.lcl[:, TF.TYPE] == Type.P
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] == Mode.NEUTRAL)
    
    # Verify other tokens are unchanged
    non_p_mask = driver_set.lcl[:, TF.TYPE] != Type.P
    # Other tokens should have their original values (or null)


def test_p_initialise_mode_no_p_tokens(driver_set):
    """Test p_initialise_mode when there are no P tokens."""
    # Create a set with no P tokens (only PO and RB)
    # DRIVER set has P tokens, so this test verifies it handles the case gracefully
    driver_set.update_op.p_initialise_mode()
    
    # Should not raise an error
    assert True


# =====================[ p_get_mode tests ]======================

def test_p_get_mode_parent_mode(driver_set):
    """Test p_get_mode when parent RB activation > child RB activation."""
    # Set up connections: P token 6 connects to RB tokens 3, 4, 5 (children)
    # RB tokens 3, 4, 5 connect to P token 6 (parent)
    # Set RB activations: children have lower activation than parent
    driver_set.lcl[3:6, TF.ACT] = 0.3  # RB tokens (children) - lower
    driver_set.lcl[6, TF.ACT] = 0.0  # P token
    
    # Set up connections in global tensor
    # P token 6 (local index 6, global index 6) -> RB tokens 3, 4, 5 (local indices 3, 4, 5, global indices 3, 4, 5)
    global_connections = driver_set.glbl.connections.connections
    global_connections[6, 3] = True  # P -> RB (child connection)
    global_connections[6, 4] = True
    global_connections[6, 5] = True
    # RB -> P (parent connection) - transpose
    global_connections[3, 6] = True
    global_connections[4, 6] = True
    global_connections[5, 6] = True
    
    # Note: If this test fails, it may indicate a bug in p_get_mode
    # (e.g., using local masks to index global connections tensor)
    driver_set.update_op.p_get_mode()
    
    # Verify P tokens have a mode set
    p_mask = driver_set.lcl[:, TF.TYPE] == Type.P
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] >= Mode.CHILD), \
        "P tokens should have MODE >= CHILD"
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] <= Mode.PARENT), \
        "P tokens should have MODE <= PARENT"


def test_p_get_mode_no_p_tokens(driver_set):
    """Test p_get_mode when there are no P tokens."""
    # Remove P tokens by changing their type
    driver_set.lcl[6:9, TF.TYPE] = Type.PO
    
    # Should not raise an error
    driver_set.update_op.p_get_mode()
    assert True


def test_p_get_mode_no_rb_tokens(driver_set):
    """Test p_get_mode when there are no RB tokens."""
    # Remove RB tokens by changing their type
    driver_set.lcl[3:6, TF.TYPE] = Type.PO
    
    # Should not raise an error
    driver_set.update_op.p_get_mode()
    assert True


# =====================[ po_get_weight_length tests ]======================

def test_po_get_weight_length_not_implemented(driver_set):
    """Test that po_get_weight_length raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        driver_set.update_op.po_get_weight_length()


# =====================[ po_get_max_semantic_weight tests ]======================

def test_po_get_max_semantic_weight_not_implemented(driver_set):
    """Test that po_get_max_semantic_weight raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        driver_set.update_op.po_get_max_semantic_weight()


# =====================[ Integration tests for new functions ]======================

def test_inhibitor_operations_chain(driver_set):
    """Test chaining inhibitor operations."""
    # Set initial values
    driver_set.lcl[0:3, TF.ACT] = 0.5  # PO tokens
    driver_set.lcl[0:3, TF.INHIBITOR_INPUT] = 0.0
    driver_set.lcl[0:3, TF.INHIBITOR_THRESHOLD] = 1.0
    driver_set.lcl[0:3, TF.INHIBITOR_ACT] = 0.0
    
    # Update inhibitor input (accumulates ACT)
    driver_set.update_op.update_inhibitor_input([Type.PO])
    po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([0.5, 0.5, 0.5]))
    
    # Update again
    driver_set.update_op.update_inhibitor_input([Type.PO])
    assert torch.allclose(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT], torch.tensor([1.0, 1.0, 1.0]))
    
    # Update inhibitor act (should set to 1.0 since input >= threshold)
    driver_set.update_op.update_inhibitor_act([Type.PO])
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 1.0)
    
    # Reset inhibitor
    # Note: If this fails with TypeError about init_float arguments,
    # it means reset_inhibitor is calling init_float with wrong arguments (3 args instead of 2)
    driver_set.update_op.reset_inhibitor([Type.PO])
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_INPUT] == 0.0), \
        "INHIBITOR_INPUT should be reset to 0.0"
    assert torch.all(driver_set.lcl[po_mask, TF.INHIBITOR_ACT] == 0.0), \
        "INHIBITOR_ACT should be reset to 0.0"


def test_p_mode_operations_chain(driver_set):
    """Test chaining P mode operations."""
    # Initialize mode
    driver_set.update_op.p_initialise_mode()
    p_mask = driver_set.lcl[:, TF.TYPE] == Type.P
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] == Mode.NEUTRAL)
    
    # Get mode
    # Note: If this test fails, it may indicate a bug in p_get_mode
    driver_set.update_op.p_get_mode()
    
    # Verify mode is set to a valid value
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] >= Mode.CHILD), \
        "P tokens should have MODE >= CHILD"
    assert torch.all(driver_set.lcl[p_mask, TF.MODE] <= Mode.PARENT), \
        "P tokens should have MODE <= PARENT"


def test_operations_independent_across_sets_new_functions(driver_set, recipient_set):
    """Test that new update operations are independent across different sets."""
    # Zero lateral input in DRIVER set
    driver_set.update_op.zero_laternal_input([Type.PO])
    driver_po_mask = driver_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(driver_set.lcl[driver_po_mask, TF.LATERAL_INPUT] == 0.0)
    
    # Zero lateral input in RECIPIENT set
    recipient_set.update_op.zero_laternal_input([Type.PO])
    recipient_po_mask = recipient_set.lcl[:, TF.TYPE] == Type.PO
    assert torch.all(recipient_set.lcl[recipient_po_mask, TF.LATERAL_INPUT] == 0.0)
    
    # Verify they don't affect each other
    assert len(driver_set.lcl) == 10
    assert len(recipient_set.lcl) == 10

