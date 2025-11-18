# nodes/unit_test/tensor/test_token_tensor.py
# Tests for Token_Tensor class

import pytest
import torch
from nodes.network.tensor.token_tensor import Token_Tensor
from nodes.enums import Set, TF, B, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens across different sets.
    Some tokens are active, some are deleted (available for reuse).
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for active tokens (0-14)
    tensor[0:15, TF.DELETED] = B.FALSE
    # Set DELETED to True for deleted tokens (15-19) - available for reuse
    tensor[15:20, TF.DELETED] = B.TRUE
    
    # Create tokens in different sets
    # Set 0 (DRIVER): tokens 0-4
    tensor[0:5, TF.SET] = Set.DRIVER
    tensor[0:5, TF.ANALOG] = 0
    tensor[0:5, TF.ACT] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    tensor[0:5, TF.ID] = torch.arange(0, 5)
    
    # Set 1 (RECIPIENT): tokens 5-9
    tensor[5:10, TF.SET] = Set.RECIPIENT
    tensor[5:10, TF.ANALOG] = 1
    tensor[5:10, TF.ACT] = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    tensor[5:10, TF.ID] = torch.arange(5, 10)
    
    # Set 2 (MEMORY): tokens 10-14
    tensor[10:15, TF.SET] = Set.MEMORY
    tensor[10:15, TF.ANALOG] = 2
    tensor[10:15, TF.ACT] = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5])
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    
    # Tokens 15-19 are deleted (available for reuse)
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 20
    return torch.zeros((num_tokens, num_tokens), dtype=tensor_type)


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(15)}  # Only active tokens have names


@pytest.fixture
def token_tensor(mock_tensor, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    return Token_Tensor(mock_tensor, mock_connections, mock_names)


def test_token_tensor_init(token_tensor, mock_tensor, mock_connections, mock_names):
    """Test Token_Tensor initialization."""
    assert torch.equal(token_tensor.tensor, mock_tensor)
    assert torch.equal(token_tensor.connections, mock_connections)
    assert token_tensor.names == mock_names
    assert token_tensor.expansion_factor == 1.1
    assert token_tensor.cache is not None
    assert torch.equal(token_tensor.cache.tensor, mock_tensor)


def test_add_tokens_to_deleted_slots(token_tensor):
    """Test adding tokens to deleted slots."""
    # Create new tokens to add
    num_new_tokens = 3
    new_tokens = torch.full((num_new_tokens, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.NEW_SET
    new_tokens[:, TF.ANALOG] = 3
    new_tokens[:, TF.ACT] = torch.tensor([2.0, 2.1, 2.2])
    new_tokens[:, TF.ID] = torch.tensor([100, 101, 102])
    
    new_names = ["new_token_0", "new_token_1", "new_token_2"]
    
    # Add tokens
    replace_idxs = token_tensor.add_tokens(new_tokens, new_names)
    
    # Check that tokens were added to deleted slots (indices 15-17)
    assert len(replace_idxs) == 3
    assert torch.all(replace_idxs >= 15)  # Should use deleted slots
    assert torch.all(replace_idxs < 20)
    
    # Check that tokens were actually added
    for idx in replace_idxs:
        assert token_tensor.tensor[idx, TF.DELETED] == B.FALSE
        assert token_tensor.tensor[idx, TF.SET] == Set.NEW_SET
        assert token_tensor.names[idx.item()] in new_names
    
    # Check that cache was updated
    assert Set.NEW_SET in token_tensor.cache.masks


def test_add_tokens_expands_tensor(token_tensor):
    """Test that adding more tokens than deleted slots expands the tensor."""
    # Create many new tokens (more than available deleted slots)
    num_new_tokens = 10  # More than the 5 deleted slots available
    new_tokens = torch.full((num_new_tokens, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.NEW_SET
    new_tokens[:, TF.ANALOG] = 4
    new_tokens[:, TF.ACT] = torch.tensor([3.0 + i * 0.1 for i in range(num_new_tokens)])
    new_tokens[:, TF.ID] = torch.arange(200, 200 + num_new_tokens)
    
    new_names = [f"expanded_token_{i}" for i in range(num_new_tokens)]
    
    original_size = token_tensor.tensor.size(0)
    
    # Add tokens
    replace_idxs = token_tensor.add_tokens(new_tokens, new_names)
    
    # Check that tensor was expanded
    assert token_tensor.tensor.size(0) > original_size
    assert len(replace_idxs) == num_new_tokens
    
    # Check that all tokens were added
    for idx in replace_idxs:
        assert token_tensor.tensor[idx, TF.DELETED] == B.FALSE
        assert token_tensor.tensor[idx, TF.SET] == Set.NEW_SET


def test_expand_tensor(token_tensor):
    """Test tensor expansion."""
    original_size = token_tensor.tensor.size(0)
    original_tensor = token_tensor.tensor.clone()
    
    # Expand tensor
    token_tensor.expand_tensor(min_expansion=10)
    
    # Check that tensor was expanded
    new_size = token_tensor.tensor.size(0)
    # Should be at least original_size + min_expansion, or expanded by factor
    expanded_size = int(original_size * token_tensor.expansion_factor)
    expected_size = max(original_size + 10, expanded_size)
    assert new_size == expected_size
    
    # Check that original data was preserved
    assert torch.equal(token_tensor.tensor[:original_size, :], original_tensor)
    
    # Check that new slots are marked as deleted
    assert torch.all(token_tensor.tensor[original_size:, TF.DELETED] == B.TRUE)


def test_expand_tensor_with_min_expansion(token_tensor):
    """Test tensor expansion with minimum expansion parameter."""
    original_size = token_tensor.tensor.size(0)
    
    # Expand with large min_expansion
    token_tensor.expand_tensor(min_expansion=50)
    
    new_size = token_tensor.tensor.size(0)
    # Should be at least original_size + min_expansion, or expanded by factor
    expanded_size = int(original_size * token_tensor.expansion_factor)
    expected_size = max(original_size + 50, expanded_size)
    assert new_size == expected_size


def test_move_tokens(token_tensor):
    """Test moving tokens to a different set."""
    # Move tokens 0-4 from DRIVER to RECIPIENT
    indices_to_move = torch.tensor([0, 1, 2, 3, 4])
    
    # Verify initial state
    assert torch.all(token_tensor.tensor[indices_to_move, TF.SET] == Set.DRIVER)
    
    # Move tokens
    token_tensor.move_tokens(indices_to_move, Set.RECIPIENT)
    
    # Verify tokens were moved
    assert torch.all(token_tensor.tensor[indices_to_move, TF.SET] == Set.RECIPIENT)
    
    # Verify cache was updated
    assert Set.RECIPIENT in token_tensor.cache.masks


def test_move_tokens_to_new_set(token_tensor):
    """Test moving tokens to NEW_SET."""
    # Move tokens 5-9 from RECIPIENT to NEW_SET
    indices_to_move = torch.tensor([5, 6, 7, 8, 9])
    
    token_tensor.move_tokens(indices_to_move, Set.NEW_SET)
    
    # Verify tokens were moved
    assert torch.all(token_tensor.tensor[indices_to_move, TF.SET] == Set.NEW_SET)
    
    # Verify cache was updated
    assert Set.NEW_SET in token_tensor.cache.masks


def test_copy_tokens(token_tensor):
    """Test copying tokens to a different set."""
    # Copy tokens 0-2 from DRIVER to RECIPIENT
    indices_to_copy = torch.tensor([0, 1, 2])
    original_tokens = token_tensor.tensor[indices_to_copy, :].clone()
    
    # Get original names
    original_names = [token_tensor.names[i.item()] for i in indices_to_copy]
    
    # Copy tokens
    replace_idxs = token_tensor.copy_tokens(indices_to_copy, Set.RECIPIENT)
    
    # Verify original tokens are unchanged
    assert torch.equal(token_tensor.tensor[indices_to_copy, :], original_tokens)
    assert torch.all(token_tensor.tensor[indices_to_copy, TF.SET] == Set.DRIVER)
    
    # Verify copies were created in RECIPIENT set
    assert len(replace_idxs) == 3
    for i, idx in enumerate(replace_idxs):
        copied_token = token_tensor.tensor[idx, :]
        original_token = original_tokens[i, :]
        
        # SET should be different (RECIPIENT vs DRIVER)
        assert copied_token[TF.SET] == Set.RECIPIENT
        assert original_token[TF.SET] == Set.DRIVER
        
        # All other fields should match
        for field in TF:
            if field != TF.SET:
                assert torch.allclose(copied_token[field], original_token[field], atol=1e-6)
        
        # Name should match
        assert token_tensor.names[idx.item()] == original_names[i]
    
    # Verify cache was updated
    assert Set.RECIPIENT in token_tensor.cache.masks


def test_copy_tokens_preserves_data(token_tensor):
    """Test that copied tokens preserve all data except set."""
    # Copy a single token
    idx_to_copy = torch.tensor([10])
    original_token = token_tensor.tensor[10, :].clone()  # Use integer index to get 1D tensor
    original_name = token_tensor.names[10]
    
    # Copy to NEW_SET
    replace_idxs = token_tensor.copy_tokens(idx_to_copy, Set.NEW_SET)
    
    # Verify copy
    assert len(replace_idxs) == 1
    copied_token = token_tensor.tensor[replace_idxs[0].item(), :]
    
    # All fields should match except SET
    for field in TF:
        if field != TF.SET:
            # Convert to float for comparison (handles both scalars and tensors)
            copied_val = float(copied_token[field].item())
            original_val = float(original_token[field].item())
            assert abs(copied_val - original_val) < 1e-6, f"Field {field} mismatch: {copied_val} != {original_val}"
    
    # SET should be different
    assert copied_token[TF.SET] == Set.NEW_SET
    assert original_token[TF.SET] == Set.MEMORY
    
    # Name should match
    assert token_tensor.names[replace_idxs[0].item()] == original_name


def test_add_tokens_updates_cache(token_tensor):
    """Test that adding tokens updates the cache."""
    # Clear cache first
    token_tensor.cache.masks = {}
    
    # Add tokens to MEMORY set
    new_tokens = torch.full((2, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.MEMORY
    new_tokens[:, TF.ANALOG] = 5
    new_tokens[:, TF.ACT] = torch.tensor([4.0, 4.1])
    new_tokens[:, TF.ID] = torch.tensor([300, 301])
    
    token_tensor.add_tokens(new_tokens, ["cache_test_0", "cache_test_1"])
    
    # Verify cache was updated
    assert Set.MEMORY in token_tensor.cache.masks
    # Count should include the new tokens
    assert token_tensor.cache.get_set_count(Set.MEMORY) >= 5  # Original 5 + new 2


def test_add_tokens_multiple_sets(token_tensor):
    """Test adding tokens to multiple sets at once."""
    # Create tokens in different sets
    new_tokens = torch.full((4, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[0:2, TF.SET] = Set.DRIVER
    new_tokens[2:4, TF.SET] = Set.RECIPIENT
    new_tokens[:, TF.ANALOG] = 6
    new_tokens[:, TF.ACT] = torch.tensor([5.0, 5.1, 5.2, 5.3])
    new_tokens[:, TF.ID] = torch.tensor([400, 401, 402, 403])
    
    token_tensor.add_tokens(new_tokens, ["multi_0", "multi_1", "multi_2", "multi_3"])
    
    # Verify both sets are in cache
    assert Set.DRIVER in token_tensor.cache.masks
    assert Set.RECIPIENT in token_tensor.cache.masks


def test_move_tokens_single_token(token_tensor):
    """Test moving a single token."""
    idx_to_move = torch.tensor([0])
    
    token_tensor.move_tokens(idx_to_move, Set.MEMORY)
    
    assert token_tensor.tensor[0, TF.SET] == Set.MEMORY


def test_copy_tokens_to_deleted_slots(token_tensor):
    """Test that copying tokens uses deleted slots when available."""
    # Copy tokens - should use deleted slots (15-19)
    indices_to_copy = torch.tensor([0, 1])
    
    replace_idxs = token_tensor.copy_tokens(indices_to_copy, Set.NEW_SET)
    
    # Should use deleted slots
    assert len(replace_idxs) == 2
    assert torch.all(replace_idxs >= 15)
    assert torch.all(replace_idxs < 20)


def test_expand_tensor_preserves_connections(token_tensor):
    """Test that expanding tensor doesn't affect connections structure."""
    # Note: The current implementation doesn't expand connections,
    # but we should verify the connections tensor still exists
    original_connections = token_tensor.connections.clone()
    
    token_tensor.expand_tensor(min_expansion=10)
    
    # Connections should still exist (though not expanded in current implementation)
    assert token_tensor.connections is not None
    # Original connections should be preserved
    assert torch.equal(token_tensor.connections[:original_connections.size(0), 
                                                 :original_connections.size(1)], 
                      original_connections)


def test_add_tokens_with_empty_tensor():
    """Test adding tokens to an initially empty tensor."""
    # Create empty tensor (all deleted)
    empty_tensor = torch.full((5, len(TF)), null, dtype=tensor_type)
    empty_tensor[:, TF.DELETED] = B.TRUE
    empty_connections = torch.zeros((5, 5), dtype=tensor_type)
    empty_names = {}
    
    token_tensor = Token_Tensor(empty_tensor, empty_connections, empty_names)
    
    # Add tokens
    new_tokens = torch.full((3, len(TF)), null, dtype=tensor_type)
    new_tokens[:, TF.DELETED] = B.FALSE
    new_tokens[:, TF.SET] = Set.DRIVER
    new_tokens[:, TF.ANALOG] = 0
    new_tokens[:, TF.ACT] = torch.tensor([1.0, 2.0, 3.0])
    new_tokens[:, TF.ID] = torch.tensor([0, 1, 2])
    
    replace_idxs = token_tensor.add_tokens(new_tokens, ["empty_0", "empty_1", "empty_2"])
    
    assert len(replace_idxs) == 3
    assert token_tensor.cache.get_set_count(Set.DRIVER) == 3


def test_delete_tokens(token_tensor):
    """Test deleting tokens."""
    # Get initial state
    initial_active = torch.where(token_tensor.tensor[:, TF.DELETED] == B.FALSE)[0]
    initial_active_count = len(initial_active)
    
    # Store original values for tokens we'll delete
    indices_to_delete = torch.tensor([0, 2, 5])
    original_act_values = token_tensor.tensor[indices_to_delete, TF.ACT].clone()
    original_set_values = token_tensor.tensor[indices_to_delete, TF.SET].clone()
    
    # Delete tokens
    token_tensor.delete_tokens(indices_to_delete)
    
    # Verify tokens are marked as deleted
    assert torch.all(token_tensor.tensor[indices_to_delete, TF.DELETED] == B.TRUE)
    
    # Verify all values are set to null (except DELETED which should be TRUE)
    for idx in indices_to_delete:
        # Check that DELETED is TRUE
        assert token_tensor.tensor[idx, TF.DELETED] == B.TRUE, f"Token {idx} DELETED flag should be TRUE"
        # Check that all other values are null
        token_row = token_tensor.tensor[idx, :].clone()
        token_row[TF.DELETED] = null  # Temporarily set DELETED to null for comparison
        assert torch.all(token_row == null), f"Token {idx} should have all null values except DELETED"
    
    # Verify active count decreased
    final_active = torch.where(token_tensor.tensor[:, TF.DELETED] == B.FALSE)[0]
    final_active_count = len(final_active)
    assert final_active_count == initial_active_count - len(indices_to_delete)
    
    # Verify non-deleted tokens are unchanged
    non_deleted_indices = torch.tensor([1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    for idx in non_deleted_indices:
        assert token_tensor.tensor[idx, TF.DELETED] == B.FALSE, f"Token {idx} should still be active"


def test_delete_tokens_single_token(token_tensor):
    """Test deleting a single token."""
    idx_to_delete = torch.tensor([10])
    original_active_count = len(torch.where(token_tensor.tensor[:, TF.DELETED] == B.FALSE)[0])
    
    # Delete token
    token_tensor.delete_tokens(idx_to_delete)
    
    # Verify token is deleted
    assert token_tensor.tensor[10, TF.DELETED] == B.TRUE
    # Check that all values except DELETED are null
    token_row = token_tensor.tensor[10, :].clone()
    token_row[TF.DELETED] = null  # Temporarily set DELETED to null for comparison
    assert torch.all(token_row == null), "All values except DELETED should be null"
    assert token_tensor.tensor[10, TF.DELETED] == B.TRUE  # DELETED flag should be TRUE
    
    # Verify count decreased
    final_active_count = len(torch.where(token_tensor.tensor[:, TF.DELETED] == B.FALSE)[0])
    assert final_active_count == original_active_count - 1


def test_delete_tokens_multiple_sets(token_tensor):
    """Test deleting tokens from multiple sets."""
    # Delete tokens from different sets
    # Token 0-4 are in DRIVER, token 5-9 are in RECIPIENT
    indices_to_delete = torch.tensor([0, 1, 5, 6])  # 2 from DRIVER, 2 from RECIPIENT
    
    token_tensor.delete_tokens(indices_to_delete)
    
    # Verify all are deleted
    assert torch.all(token_tensor.tensor[indices_to_delete, TF.DELETED] == B.TRUE)
    
    # Verify all values are null (except DELETED which should be TRUE)
    for idx in indices_to_delete:
        assert token_tensor.tensor[idx, TF.DELETED] == B.TRUE
        token_row = token_tensor.tensor[idx, :].clone()
        token_row[TF.DELETED] = null  # Temporarily set DELETED to null for comparison
        assert torch.all(token_row == null), f"Token {idx} should have all null values except DELETED"


def test_delete_tokens_already_deleted(token_tensor):
    """Test deleting tokens that are already deleted."""
    # Token 15 is already deleted
    idx_already_deleted = torch.tensor([15])
    
    # Delete it again (should be idempotent)
    token_tensor.delete_tokens(idx_already_deleted)
    
    # Should still be deleted
    assert token_tensor.tensor[15, TF.DELETED] == B.TRUE
    token_row = token_tensor.tensor[15, :].clone()
    token_row[TF.DELETED] = null  # Temporarily set DELETED to null for comparison
    assert torch.all(token_row == null), "All values except DELETED should be null"


def test_delete_tokens_preserves_other_tokens(token_tensor):
    """Test that deleting tokens doesn't affect other tokens."""
    # Store original values for tokens we won't delete
    indices_to_preserve = torch.tensor([1, 3, 7, 11, 13])
    original_values = token_tensor.tensor[indices_to_preserve, :].clone()
    
    # Delete different tokens
    indices_to_delete = torch.tensor([0, 5, 10])
    token_tensor.delete_tokens(indices_to_delete)
    
    # Verify preserved tokens are unchanged
    for i, idx in enumerate(indices_to_preserve):
        assert torch.allclose(token_tensor.tensor[idx, :], original_values[i, :], atol=1e-6), \
            f"Token {idx} should be unchanged"

