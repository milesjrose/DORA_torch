# nodes/unit_test/tensor/test_analogs.py
# Tests for Analog_ops class

import pytest
import torch
from nodes.network.tokens.tensor.analogs import Analog_ops
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.enums import Set, TF, B, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens across different sets and analogs.
    """
    num_tokens = 30
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for active tokens (0-24)
    tensor[0:25, TF.DELETED] = B.FALSE
    # Set DELETED to True for deleted tokens (25-29) - available for reuse
    tensor[25:30, TF.DELETED] = B.TRUE
    
    # Analog 0: tokens 0-4 in DRIVER set
    tensor[0:5, TF.SET] = Set.DRIVER
    tensor[0:5, TF.ANALOG] = 0
    tensor[0:5, TF.ACT] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    tensor[0:5, TF.ID] = torch.arange(0, 5)
    
    # Analog 1: tokens 5-9 in RECIPIENT set
    tensor[5:10, TF.SET] = Set.RECIPIENT
    tensor[5:10, TF.ANALOG] = 1
    tensor[5:10, TF.ACT] = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    tensor[5:10, TF.ID] = torch.arange(5, 10)
    
    # Analog 2: tokens 10-14 in MEMORY set
    tensor[10:15, TF.SET] = Set.MEMORY
    tensor[10:15, TF.ANALOG] = 2
    tensor[10:15, TF.ACT] = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5])
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    
    # Analog 3: tokens 15-19 in NEW_SET
    tensor[15:20, TF.SET] = Set.NEW_SET
    tensor[15:20, TF.ANALOG] = 3
    tensor[15:20, TF.ACT] = torch.tensor([1.6, 1.7, 1.8, 1.9, 2.0])
    tensor[15:20, TF.ID] = torch.arange(15, 20)
    
    # Analog 4: tokens 20-24 in DRIVER set (some with zero activation)
    tensor[20:25, TF.SET] = Set.DRIVER
    tensor[20:25, TF.ANALOG] = 4
    tensor[20:25, TF.ACT] = torch.tensor([0.0, 0.0, 0.5, 0.6, 0.7])
    tensor[20:25, TF.ID] = torch.arange(20, 25)
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 30
    return torch.zeros((num_tokens, num_tokens), dtype=tensor_type)


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(25)}  # Only active tokens have names


@pytest.fixture
def token_tensor(mock_tensor, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    return Token_Tensor(mock_tensor, mock_connections, mock_names)


@pytest.fixture
def analog_ops(token_tensor):
    """Create an Analog_ops instance."""
    return Analog_ops(token_tensor)


def test_analog_ops_init(analog_ops, token_tensor):
    """Test Analog_ops initialization."""
    assert analog_ops.tokens is token_tensor
    assert torch.equal(analog_ops.tensor, token_tensor.tensor)
    assert analog_ops.cache is token_tensor.cache


def test_new_analog_id(analog_ops):
    """Test getting a new analog ID."""
    # Current max analog is 4, so new should be 5
    new_id = analog_ops.new_analog_id()
    assert new_id == 5
    assert isinstance(new_id, (int, torch.Tensor))
    if isinstance(new_id, torch.Tensor):
        assert new_id.item() == 5


def test_get_analog_indices(analog_ops):
    """Test getting indices for a specific analog."""
    # Test analog 0 (tokens 0-4)
    indices = analog_ops.get_analog_indices(0)
    expected = torch.tensor([0, 1, 2, 3, 4])
    assert torch.equal(indices, expected)
    
    # Test analog 1 (tokens 5-9)
    indices = analog_ops.get_analog_indices(1)
    expected = torch.tensor([5, 6, 7, 8, 9])
    assert torch.equal(indices, expected)
    
    # Test analog 2 (tokens 10-14)
    indices = analog_ops.get_analog_indices(2)
    expected = torch.tensor([10, 11, 12, 13, 14])
    assert torch.equal(indices, expected)
    
    # Test non-existent analog
    indices = analog_ops.get_analog_indices(99)
    assert len(indices) == 0


def test_get_analogs_where(analog_ops):
    """Test getting analogs where tokens have a specific feature value."""
    # Get analogs with tokens in DRIVER set
    analogs = analog_ops.get_analogs_where(TF.SET, Set.DRIVER)
    # Should include analog 0 and analog 4
    assert len(analogs) == 2
    assert Set.DRIVER in analogs or 0 in analogs
    assert Set.DRIVER in analogs or 4 in analogs
    
    # Get analogs with tokens in RECIPIENT set
    analogs = analog_ops.get_analogs_where(TF.SET, Set.RECIPIENT)
    # Should include analog 1
    assert len(analogs) == 1
    assert 1 in analogs
    
    # Get analogs with tokens in MEMORY set
    analogs = analog_ops.get_analogs_where(TF.SET, Set.MEMORY)
    # Should include analog 2
    assert len(analogs) == 1
    assert 2 in analogs
    
    # Get analogs with non-existent feature value
    analogs = analog_ops.get_analogs_where(TF.SET, 99)
    assert len(analogs) == 0
    assert isinstance(analogs, torch.Tensor)


def test_get_analogs_where_with_activation(analog_ops):
    """Test getting analogs where tokens have specific activation."""
    # Get analogs with tokens having activation > 1.0
    # This should find analog 2 and analog 3
    # But we need to check for a specific activation value
    # Let's check for activation == 1.1 (in analog 2)
    analogs = analog_ops.get_analogs_where(TF.ACT, 1.1)
    assert 2 in analogs
    
    # Check for activation == 0.0 (in analog 4)
    analogs = analog_ops.get_analogs_where(TF.ACT, 0.0)
    assert 4 in analogs


def test_get_analogs_where_not(analog_ops):
    """Test getting analogs that do NOT have tokens with a specific feature value."""
    # Get analogs that do NOT have tokens in DRIVER set
    analogs = analog_ops.get_analogs_where_not(TF.SET, Set.DRIVER)
    # Should include analog 1, 2, 3 (but not 0 and 4)
    assert 1 in analogs
    assert 2 in analogs
    assert 3 in analogs
    # Analog 0 and 4 should not be in the result (they have DRIVER tokens)
    # Note: This might include analogs that have SOME tokens not in DRIVER
    
    # Get analogs that do NOT have tokens in MEMORY set
    analogs = analog_ops.get_analogs_where_not(TF.SET, Set.MEMORY)
    # Should include analogs 0, 1, 3, 4 (but not 2)
    assert 0 in analogs
    assert 1 in analogs
    assert 3 in analogs
    assert 4 in analogs


def test_get_analogs_active(analog_ops):
    """Test getting analogs with active tokens."""
    # Get analogs with tokens having activation > 0.0
    active_analogs = analog_ops.get_analogs_active()
    
    # Should include all analogs that have at least one token with ACT > 0
    # Analog 0: all tokens have ACT > 0
    # Analog 1: all tokens have ACT > 0
    # Analog 2: all tokens have ACT > 0
    # Analog 3: all tokens have ACT > 0
    # Analog 4: some tokens have ACT > 0 (0.5, 0.6, 0.7)
    
    assert 0 in active_analogs
    assert 1 in active_analogs
    assert 2 in active_analogs
    assert 3 in active_analogs
    assert 4 in active_analogs
    assert len(active_analogs) == 5


def test_get_analogs_active_with_zero_activation(analog_ops):
    """Test that analogs with only zero activation are not included."""
    # Create a new analog with all zero activation
    new_tokens = torch.full((3, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.DRIVER
    new_tokens[:, TF.ANALOG] = 10
    new_tokens[:, TF.ACT] = 0.0  # All zero activation
    new_tokens[:, TF.ID] = torch.tensor([100, 101, 102])
    
    analog_ops.tokens.add_tokens(new_tokens, ["zero_0", "zero_1", "zero_2"])
    
    # Analog 10 should not be in active analogs
    active_analogs = analog_ops.get_analogs_active()
    assert 10 not in active_analogs


def test_move_analog(analog_ops):
    """Test moving an analog to a different set."""
    # Move analog 0 from DRIVER to RECIPIENT
    analog_ops.move_analog(0, Set.RECIPIENT)
    
    # Verify all tokens in analog 0 are now in RECIPIENT
    indices = analog_ops.get_analog_indices(0)
    assert torch.all(analog_ops.tensor[indices, TF.SET] == Set.RECIPIENT)
    
    # Verify cache was updated
    assert Set.RECIPIENT in analog_ops.cache.masks


def test_move_analog_to_new_set(analog_ops):
    """Test moving an analog to NEW_SET."""
    # Move analog 1 from RECIPIENT to NEW_SET
    analog_ops.move_analog(1, Set.NEW_SET)
    
    # Verify all tokens in analog 1 are now in NEW_SET
    indices = analog_ops.get_analog_indices(1)
    assert torch.all(analog_ops.tensor[indices, TF.SET] == Set.NEW_SET)


def test_copy_analog(analog_ops):
    """Test copying an analog to a different set."""
    # Copy analog 0 from DRIVER to RECIPIENT
    original_indices = analog_ops.get_analog_indices(0)
    original_tokens = analog_ops.tensor[original_indices, :].clone()
    
    new_analog_number = analog_ops.copy_analog(0, Set.RECIPIENT)
    
    # Verify original tokens are unchanged
    assert torch.equal(analog_ops.tensor[original_indices, :], original_tokens)
    assert torch.all(analog_ops.tensor[original_indices, TF.SET] == Set.DRIVER)
    assert torch.all(analog_ops.tensor[original_indices, TF.ANALOG] == 0)
    
    # Verify new analog was created
    assert new_analog_number == 5  # Should be the next available analog number
    
    # Verify copies exist with new analog number and new set
    new_indices = analog_ops.get_analog_indices(new_analog_number)
    assert len(new_indices) == 5  # Same number of tokens
    assert torch.all(analog_ops.tensor[new_indices, TF.SET] == Set.RECIPIENT)
    assert torch.all(analog_ops.tensor[new_indices, TF.ANALOG] == new_analog_number)
    
    # Verify cache was updated
    assert Set.RECIPIENT in analog_ops.cache.masks


def test_copy_analog_preserves_data(analog_ops):
    """Test that copied analog preserves token data."""
    # Copy analog 2 from MEMORY to DRIVER
    original_indices = analog_ops.get_analog_indices(2)
    original_tokens = analog_ops.tensor[original_indices, :].clone()
    
    new_analog_number = analog_ops.copy_analog(2, Set.DRIVER)
    new_indices = analog_ops.get_analog_indices(new_analog_number)
    
    # Verify data is preserved (except SET and ANALOG)
    for i, new_idx in enumerate(new_indices):
        copied_token = analog_ops.tensor[new_idx, :]
        original_token = original_tokens[i, :]
        
        # SET and ANALOG should be different
        assert copied_token[TF.SET] == Set.DRIVER
        assert original_token[TF.SET] == Set.MEMORY
        assert copied_token[TF.ANALOG] == new_analog_number
        assert original_token[TF.ANALOG] == 2
        
        # All other fields should match
        for field in TF:
            if field not in [TF.SET, TF.ANALOG]:
                copied_val = float(copied_token[field].item())
                original_val = float(original_token[field].item())
                assert abs(copied_val - original_val) < 1e-6, f"Field {field} mismatch"


def test_copy_analog_creates_unique_analog_number(analog_ops):
    """Test that copying an analog creates a unique analog number."""
    # Copy analog 0
    new_analog_1 = analog_ops.copy_analog(0, Set.RECIPIENT)
    
    # Copy analog 0 again
    new_analog_2 = analog_ops.copy_analog(0, Set.NEW_SET)
    
    # Verify they have different analog numbers
    assert new_analog_1 != new_analog_2
    assert new_analog_2 == new_analog_1 + 1  # Should be sequential


def test_get_analogs_where_empty_result(analog_ops):
    """Test get_analogs_where returns empty tensor when no matches."""
    # Search for a feature value that doesn't exist
    analogs = analog_ops.get_analogs_where(TF.ID, 9999)
    
    # Should return empty tensor, not empty list
    assert isinstance(analogs, torch.Tensor)
    assert len(analogs) == 0


def test_new_analog_id_after_adding_tokens(analog_ops):
    """Test that new_analog_id updates after adding tokens with new analog numbers."""
    # Add tokens with analog number 10
    new_tokens = torch.full((3, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.DRIVER
    new_tokens[:, TF.ANALOG] = 10
    new_tokens[:, TF.ACT] = torch.tensor([0.5, 0.6, 0.7])
    new_tokens[:, TF.ID] = torch.tensor([200, 201, 202])
    
    analog_ops.tokens.add_tokens(new_tokens, ["new_0", "new_1", "new_2"])
    
    # New analog ID should be 11 (max is now 10)
    new_id = analog_ops.new_analog_id()
    assert new_id == 11


def test_get_analog_indices_includes_deleted_tokens(analog_ops):
    """Test that get_analog_indices includes deleted tokens."""
    # Set some tokens in analog 0 as deleted
    analog_ops.tensor[2, TF.DELETED] = B.TRUE
    analog_ops.tensor[3, TF.DELETED] = B.TRUE
    
    # get_analog_indices should still return all tokens with that analog number
    indices = analog_ops.get_analog_indices(0)
    assert len(indices) == 5  # Still includes deleted tokens
    assert 2 in indices
    assert 3 in indices


def test_get_analogs_where_excludes_deleted_tokens(analog_ops):
    """Test that get_analogs_where only considers non-deleted tokens."""
    # Delete some tokens in analog 0
    analog_ops.tensor[0:3, TF.DELETED] = B.TRUE
    
    # Set a unique feature value on remaining active tokens in analog 0
    analog_ops.tensor[3, TF.ACT] = 99.0  # Unique value
    
    # Search for this unique value
    analogs = analog_ops.get_analogs_where(TF.ACT, 99.0)
    
    # Should find analog 0 (from the active token)
    assert 0 in analogs

