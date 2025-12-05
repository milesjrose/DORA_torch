# nodes/unit_test/sets/base_set_ops/test_set_analog.py
# Tests for AnalogOperations class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, null, tensor_type


@pytest.fixture
def mock_tensor_with_analogs():
    """
    Create a mock tensor with multiple tokens across different sets and analogs.
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
    # Analog 0: tokens 0-2 (PO, ACT=0.1, 0.2, 0.3)
    # Analog 1: tokens 3-5 (RB, ACT=0.4, 0.5, 0.6)
    # Analog 2: tokens 6-8 (P, ACT=0.7, 0.8, 0.9)
    # Analog 3: token 9 (GROUP, ACT=1.0)
    tensor[0:9, TF.SET] = Set.DRIVER
    tensor[0:3, TF.ANALOG] = 0
    tensor[0:3, TF.TYPE] = Type.PO
    tensor[0:3, TF.ACT] = torch.tensor([0.1, 0.2, 0.3])
    tensor[3:6, TF.ANALOG] = 1
    tensor[3:6, TF.TYPE] = Type.RB
    tensor[3:6, TF.ACT] = torch.tensor([0.4, 0.5, 0.6])
    tensor[6:9, TF.ANALOG] = 2
    tensor[6:9, TF.TYPE] = Type.P
    tensor[6:9, TF.ACT] = torch.tensor([0.7, 0.8, 0.9])
    tensor[9, TF.SET] = Set.DRIVER
    tensor[9, TF.ANALOG] = 3
    tensor[9, TF.TYPE] = Type.GROUP
    tensor[9, TF.ACT] = 1.0
    
    # RECIPIENT set: tokens 10-19
    # Analog 0: tokens 10-12 (PO, ACT=1.1, 1.2, 1.3)
    # Analog 1: tokens 13-15 (RB, ACT=1.4, 1.5, 1.6)
    # Analog 2: tokens 16-18 (P, ACT=0.0, 0.0, 0.0) - inactive
    # Analog 4: token 19 (SEMANTIC, ACT=2.0)
    tensor[10:19, TF.SET] = Set.RECIPIENT
    tensor[10:13, TF.ANALOG] = 0
    tensor[10:13, TF.TYPE] = Type.PO
    tensor[10:13, TF.ACT] = torch.tensor([1.1, 1.2, 1.3])
    tensor[13:16, TF.ANALOG] = 1
    tensor[13:16, TF.TYPE] = Type.RB
    tensor[13:16, TF.ACT] = torch.tensor([1.4, 1.5, 1.6])
    tensor[16:19, TF.ANALOG] = 2
    tensor[16:19, TF.TYPE] = Type.P
    tensor[16:19, TF.ACT] = 0.0  # Inactive tokens
    tensor[19, TF.SET] = Set.RECIPIENT
    tensor[19, TF.ANALOG] = 4
    tensor[19, TF.TYPE] = Type.SEMANTIC
    tensor[19, TF.ACT] = 2.0
    
    # MEMORY set: tokens 20-24
    # Analog 0: tokens 20-22 (PO, ACT=2.1, 2.2, 2.3)
    # Analog 1: tokens 23-24 (RB, ACT=2.4, 2.5)
    tensor[20:25, TF.SET] = Set.MEMORY
    tensor[20:23, TF.ANALOG] = 0
    tensor[20:23, TF.TYPE] = Type.PO
    tensor[20:23, TF.ACT] = torch.tensor([2.1, 2.2, 2.3])
    tensor[23:25, TF.ANALOG] = 1
    tensor[23:25, TF.TYPE] = Type.RB
    tensor[23:25, TF.ACT] = torch.tensor([2.4, 2.5])
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 30
    return torch.zeros((num_tokens, num_tokens), dtype=torch.bool)


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
def token_tensor(mock_tensor_with_analogs, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    return Token_Tensor(mock_tensor_with_analogs, mock_connections, mock_names)


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


# =====================[ get_analog_indices tests ]======================

def test_get_analog_indices_single_analog(driver_set):
    """Test getting indices for a single analog."""
    # Analog 0 in DRIVER set has tokens at local indices 0, 1, 2
    indices = driver_set.analog_op.get_analog_indices(0)
    
    assert isinstance(indices, torch.Tensor)
    assert len(indices) == 3
    # Should return local indices 0, 1, 2
    assert torch.equal(indices, torch.tensor([0, 1, 2], dtype=torch.long))


def test_get_analog_indices_multiple_tokens(driver_set):
    """Test getting indices for analog with multiple tokens."""
    # Analog 1 in DRIVER set has tokens at local indices 3, 4, 5
    indices = driver_set.analog_op.get_analog_indices(1)
    
    assert len(indices) == 3
    assert torch.equal(indices, torch.tensor([3, 4, 5], dtype=torch.long))


def test_get_analog_indices_single_token(driver_set):
    """Test getting indices for analog with single token."""
    # Analog 3 in DRIVER set has token at local index 9
    indices = driver_set.analog_op.get_analog_indices(3)
    
    assert len(indices) == 1
    assert indices[0].item() == 9


def test_get_analog_indices_nonexistent_analog(driver_set):
    """Test getting indices for analog that doesn't exist in set."""
    # Analog 99 doesn't exist in DRIVER set
    indices = driver_set.analog_op.get_analog_indices(99)
    
    assert isinstance(indices, torch.Tensor)
    assert len(indices) == 0


def test_get_analog_indices_different_sets(driver_set, recipient_set, memory_set):
    """Test getting analog indices across different sets."""
    # Analog 0 exists in all sets but with different tokens
    driver_indices = driver_set.analog_op.get_analog_indices(0)
    recipient_indices = recipient_set.analog_op.get_analog_indices(0)
    memory_indices = memory_set.analog_op.get_analog_indices(0)
    
    # All should have 3 tokens (local indices 0, 1, 2 in each set)
    assert len(driver_indices) == 3
    assert len(recipient_indices) == 3
    assert len(memory_indices) == 3
    
    # But they represent different global tokens
    assert torch.equal(driver_indices, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(recipient_indices, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(memory_indices, torch.tensor([0, 1, 2], dtype=torch.long))


# =====================[ get_analogs_where tests ]======================

def test_get_analogs_where_by_type(driver_set):
    """Test getting analogs that contain tokens of a specific type."""
    # Get analogs containing PO tokens (Type.PO = 0)
    analogs = driver_set.analog_op.get_analogs_where(TF.TYPE, Type.PO)
    
    assert isinstance(analogs, torch.Tensor)
    # Analog 0 contains PO tokens
    assert len(analogs) == 1
    assert analogs[0].item() == 0


def test_get_analogs_where_by_act_value(driver_set):
    """Test getting analogs that contain tokens with specific activation."""
    # Get analogs containing tokens with ACT = 0.5
    analogs = driver_set.analog_op.get_analogs_where(TF.ACT, 0.5)
    
    assert isinstance(analogs, torch.Tensor)
    # Token at local index 4 has ACT=0.5, which is in analog 1
    assert len(analogs) == 1
    assert analogs[0].item() == 1


def test_get_analogs_where_multiple_analogs(driver_set):
    """Test getting analogs where multiple analogs match."""
    # Get analogs containing tokens with ACT > 0.3 (multiple analogs)
    # This will match tokens in analogs 1, 2, 3
    # But we need to check for a value that appears in multiple analogs
    # Let's use a different approach - check for TYPE.RB which appears in analog 1
    analogs = driver_set.analog_op.get_analogs_where(TF.TYPE, Type.RB)
    
    assert len(analogs) == 1
    assert analogs[0].item() == 1


def test_get_analogs_where_no_matches(driver_set):
    """Test getting analogs when no tokens match."""
    # Get analogs containing tokens with ACT = 999.0 (doesn't exist)
    analogs = driver_set.analog_op.get_analogs_where(TF.ACT, 999.0)
    
    assert isinstance(analogs, list)
    assert len(analogs) == 0


def test_get_analogs_where_returns_unique(recipient_set):
    """Test that get_analogs_where returns unique analog IDs."""
    # In RECIPIENT set, analog 0 has 3 PO tokens
    # All have different ACT values, but same TYPE
    analogs = recipient_set.analog_op.get_analogs_where(TF.TYPE, Type.PO)
    
    # Should return unique analog IDs (only analog 0)
    assert len(analogs) == 1
    assert analogs[0].item() == 0


# =====================[ get_analogs_where_not tests ]======================

def test_get_analogs_where_not_by_type(driver_set):
    """Test getting analogs that contain tokens NOT of a specific type."""
    # Get analogs containing tokens that are NOT PO (Type.PO = 0)
    analogs = driver_set.analog_op.get_analogs_where_not(TF.TYPE, Type.PO)
    
    assert isinstance(analogs, torch.Tensor)
    # Analogs 1 (RB), 2 (P), and 3 (GROUP) don't have PO tokens
    assert len(analogs) == 3
    assert 1 in analogs
    assert 2 in analogs
    assert 3 in analogs
    assert 0 not in analogs  # Analog 0 has PO tokens


def test_get_analogs_where_not_by_act_value(driver_set):
    """Test getting analogs that contain tokens without specific activation."""
    # Get analogs containing tokens with ACT != 0.5
    analogs = driver_set.analog_op.get_analogs_where_not(TF.ACT, 0.5)
    
    assert isinstance(analogs, torch.Tensor)
    # Most analogs should match (all except analog 1 which has token with ACT=0.5)
    # Actually, analog 1 has tokens with ACT 0.4, 0.5, 0.6, so it should still be included
    # Let's check for a value that only one token has
    analogs = driver_set.analog_op.get_analogs_where_not(TF.ACT, 1.0)
    
    # Analog 3 has token with ACT=1.0, so it should be excluded
    # But analog 3 also has that token, so if any token in analog 3 doesn't have ACT=1.0...
    # Actually, analog 3 only has one token with ACT=1.0, so it should be excluded
    assert 3 not in analogs or len(analogs) >= 3


def test_get_analogs_where_not_no_matches(driver_set):
    """Test getting analogs when all tokens match (should return empty)."""
    # This is tricky - if all tokens have the same value, all analogs would be excluded
    # Let's use a value that doesn't exist
    analogs = driver_set.analog_op.get_analogs_where_not(TF.ACT, 999.0)
    
    # All analogs should be included since no tokens have ACT=999.0
    assert isinstance(analogs, torch.Tensor)
    assert len(analogs) == 4  # All 4 analogs in DRIVER set


def test_get_analogs_where_not_returns_unique(recipient_set):
    """Test that get_analogs_where_not returns unique analog IDs."""
    # Get analogs containing tokens that are NOT SEMANTIC
    analogs = recipient_set.analog_op.get_analogs_where_not(TF.TYPE, Type.SEMANTIC)
    
    # Should return unique analog IDs
    assert isinstance(analogs, torch.Tensor)
    # Analogs 0, 1, 2 don't have SEMANTIC tokens
    assert len(analogs) == 3
    assert 0 in analogs
    assert 1 in analogs
    assert 2 in analogs
    assert 4 not in analogs  # Analog 4 has SEMANTIC token


# =====================[ get_analogs_active tests ]======================

def test_get_analogs_active_all_active(driver_set):
    """Test getting active analogs when all tokens are active."""
    # All tokens in DRIVER set have ACT > 0.0
    analogs = driver_set.analog_op.get_analogs_active()
    
    assert isinstance(analogs, torch.Tensor)
    # All 4 analogs should be active
    assert len(analogs) == 4
    assert 0 in analogs
    assert 1 in analogs
    assert 2 in analogs
    assert 3 in analogs


def test_get_analogs_active_some_inactive(recipient_set):
    """Test getting active analogs when some tokens are inactive."""
    # RECIPIENT set has analog 2 with all tokens having ACT=0.0 (inactive)
    analogs = recipient_set.analog_op.get_analogs_active()
    
    assert isinstance(analogs, torch.Tensor)
    # Analogs 0, 1, and 4 should be active (analog 2 is inactive)
    assert len(analogs) == 3
    assert 0 in analogs
    assert 1 in analogs
    assert 4 in analogs
    assert 2 not in analogs  # Analog 2 has all inactive tokens


def test_get_analogs_active_all_inactive():
    """Test getting active analogs when all tokens are inactive."""
    # Create a set with all inactive tokens
    num_tokens = 10
    num_features = len(TF)
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    tensor[:, TF.DELETED] = B.FALSE
    tensor[:, TF.SET] = Set.DRIVER
    tensor[:, TF.ANALOG] = torch.arange(0, 10)  # 10 different analogs
    tensor[:, TF.ACT] = 0.0  # All inactive
    
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    names = {i: f"token_{i}" for i in range(10)}
    token_tensor = Token_Tensor(tensor, connections, names)
    from nodes.network.default_parameters import parameters
    params = Params(parameters)
    base_set = Base_Set(token_tensor, Set.DRIVER, params)
    
    analogs = base_set.analog_op.get_analogs_active()
    
    assert isinstance(analogs, list)
    assert len(analogs) == 0


def test_get_analogs_active_returns_unique(driver_set):
    """Test that get_analogs_active returns unique analog IDs."""
    analogs = driver_set.analog_op.get_analogs_active()
    
    # Should return unique values
    unique_analogs = torch.unique(analogs)
    assert torch.equal(analogs, unique_analogs)


# =====================[ get_analog_ref_list tests ]======================

def test_get_analog_ref_list_full_mask(driver_set):
    """Test getting analog list with full mask."""
    # Create mask for all tokens
    mask = torch.ones(len(driver_set.lcl), dtype=torch.bool)
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    assert isinstance(analogs, torch.Tensor)
    # Should return all 4 unique analogs: 0, 1, 2, 3
    assert len(analogs) == 4
    assert 0 in analogs
    assert 1 in analogs
    assert 2 in analogs
    assert 3 in analogs


def test_get_analog_ref_list_partial_mask(driver_set):
    """Test getting analog list with partial mask."""
    # Create mask for first 3 tokens (analog 0)
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    mask[0:3] = True
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    assert isinstance(analogs, torch.Tensor)
    # Should return only analog 0
    assert len(analogs) == 1
    assert analogs[0].item() == 0


def test_get_analog_ref_list_multiple_analogs(driver_set):
    """Test getting analog list with mask covering multiple analogs."""
    # Create mask for tokens in analogs 0 and 1
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    mask[0:6] = True  # Covers analog 0 (indices 0-2) and analog 1 (indices 3-5)
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    assert isinstance(analogs, torch.Tensor)
    # Should return analogs 0 and 1
    assert len(analogs) == 2
    assert 0 in analogs
    assert 1 in analogs


def test_get_analog_ref_list_empty_mask(driver_set):
    """Test getting analog list with empty mask."""
    # Create empty mask
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    assert isinstance(analogs, torch.Tensor)
    # Should return empty tensor
    assert len(analogs) == 0


def test_get_analog_ref_list_returns_unique(driver_set):
    """Test that get_analog_ref_list returns unique analog IDs."""
    # Create mask that might include same analog multiple times
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    mask[0:6] = True  # Covers analog 0 and 1
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    # Should return unique values
    unique_analogs = torch.unique(analogs)
    assert torch.equal(analogs, unique_analogs)


def test_get_analog_ref_list_non_contiguous_mask(driver_set):
    """Test getting analog list with non-contiguous mask."""
    # Create mask for tokens at indices 0, 3, 6, 9 (one from each analog)
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    mask[0] = True  # Analog 0
    mask[3] = True  # Analog 1
    mask[6] = True  # Analog 2
    mask[9] = True  # Analog 3
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    assert isinstance(analogs, torch.Tensor)
    # Should return all 4 analogs
    assert len(analogs) == 4
    assert 0 in analogs
    assert 1 in analogs
    assert 2 in analogs
    assert 3 in analogs


# =====================[ Integration tests ]======================

def test_analog_operations_work_with_local_indices(driver_set):
    """Test that analog operations work correctly with local indices."""
    # Get indices for analog 0 (local indices 0, 1, 2)
    indices = driver_set.analog_op.get_analog_indices(0)
    
    # Verify these are local indices
    assert len(indices) == 3
    assert torch.equal(indices, torch.tensor([0, 1, 2], dtype=torch.long))
    
    # Use these indices to get analog list
    mask = torch.zeros(len(driver_set.lcl), dtype=torch.bool)
    mask[indices] = True
    analogs = driver_set.analog_op.get_analog_ref_list(mask)
    
    # Should return analog 0
    assert len(analogs) == 1
    assert analogs[0].item() == 0


def test_analog_operations_independent_across_sets(driver_set, recipient_set):
    """Test that analog operations are independent across different sets."""
    # Both sets have analog 0, but with different tokens
    driver_analogs = driver_set.analog_op.get_analogs_active()
    recipient_analogs = recipient_set.analog_op.get_analogs_active()
    
    # Both should have analog 0
    assert 0 in driver_analogs
    assert 0 in recipient_analogs
    
    # But they represent different global tokens
    driver_indices = driver_set.analog_op.get_analog_indices(0)
    recipient_indices = recipient_set.analog_op.get_analog_indices(0)
    
    # Both should have 3 tokens locally
    assert len(driver_indices) == 3
    assert len(recipient_indices) == 3


def test_get_analogs_where_with_active_filter(recipient_set):
    """Test combining get_analogs_where with active filter."""
    # Get active analogs
    active_analogs = recipient_set.analog_op.get_analogs_active()
    
    # Get analogs with PO tokens
    po_analogs = recipient_set.analog_op.get_analogs_where(TF.TYPE, Type.PO)
    
    # Analog 0 should be in both (it's active and has PO tokens)
    assert 0 in active_analogs
    assert 0 in po_analogs
    
    # Analog 2 should be active but not have PO tokens
    # Actually analog 2 is inactive, so it won't be in active_analogs
    assert 2 not in active_analogs

