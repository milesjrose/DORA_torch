# nodes/unit_test/sets/base_set_ops/test_set_tensor.py
# Tests for TensorOperations class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens of different types across different sets.
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
    # - tokens 0-2: Type.PO
    # - tokens 3-5: Type.RB
    # - tokens 6-8: Type.P
    # - token 9: Type.GROUP
    tensor[0:9, TF.SET] = Set.DRIVER
    tensor[0:3, TF.TYPE] = Type.PO
    tensor[3:6, TF.TYPE] = Type.RB
    tensor[6:9, TF.TYPE] = Type.P
    tensor[9, TF.SET] = Set.DRIVER
    tensor[9, TF.TYPE] = Type.GROUP
    
    # RECIPIENT set: tokens 10-19
    # - tokens 10-12: Type.PO
    # - tokens 13-15: Type.RB
    # - tokens 16-18: Type.P
    # - token 19: Type.SEMANTIC
    tensor[10:19, TF.SET] = Set.RECIPIENT
    tensor[10:13, TF.TYPE] = Type.PO
    tensor[13:16, TF.TYPE] = Type.RB
    tensor[16:19, TF.TYPE] = Type.P
    tensor[19, TF.SET] = Set.RECIPIENT
    tensor[19, TF.TYPE] = Type.SEMANTIC
    
    # MEMORY set: tokens 20-24
    # - tokens 20-22: Type.PO
    # - tokens 23-24: Type.RB
    tensor[20:25, TF.SET] = Set.MEMORY
    tensor[20:23, TF.TYPE] = Type.PO
    tensor[23:25, TF.TYPE] = Type.RB
    
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
def token_tensor(mock_tensor, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    return Token_Tensor(mock_tensor, mock_connections, mock_names)


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


# =====================[ get_mask tests ]======================

def test_get_mask_po(driver_set):
    """Test getting mask for PO tokens in DRIVER set."""
    mask = driver_set.tensor_op.get_mask(Type.PO)
    
    # Should be a boolean tensor
    assert mask.dtype == torch.bool
    # Should have length equal to number of tokens in DRIVER set (10)
    assert len(mask) == 10
    # Should have 3 True values (tokens 0-2 in local view)
    assert mask.sum().item() == 3
    # First 3 should be True
    assert torch.all(mask[0:3] == True)
    # Rest should be False
    assert torch.all(mask[3:] == False)


def test_get_mask_rb(driver_set):
    """Test getting mask for RB tokens in DRIVER set."""
    mask = driver_set.tensor_op.get_mask(Type.RB)
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 3 True values (tokens 3-5 in local view)
    assert mask.sum().item() == 3
    # Indices 3-5 should be True
    assert torch.all(mask[3:6] == True)
    # Rest should be False
    assert torch.all(mask[0:3] == False)
    assert torch.all(mask[6:] == False)


def test_get_mask_p(driver_set):
    """Test getting mask for P tokens in DRIVER set."""
    mask = driver_set.tensor_op.get_mask(Type.P)
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 3 True values (tokens 6-8 in local view)
    assert mask.sum().item() == 3
    # Indices 6-8 should be True
    assert torch.all(mask[6:9] == True)


def test_get_mask_group(driver_set):
    """Test getting mask for GROUP tokens in DRIVER set."""
    mask = driver_set.tensor_op.get_mask(Type.GROUP)
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 1 True value (token 9 in local view)
    assert mask.sum().item() == 1
    # Index 9 should be True
    assert mask[9].item() == True


def test_get_mask_semantic(recipient_set):
    """Test getting mask for SEMANTIC tokens in RECIPIENT set."""
    mask = recipient_set.tensor_op.get_mask(Type.SEMANTIC)
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 1 True value (token 19 in global, which maps to local index 9)
    assert mask.sum().item() == 1
    # Last index should be True
    assert mask[9].item() == True


def test_get_mask_nonexistent_type(driver_set):
    """Test getting mask for type that doesn't exist in set."""
    mask = driver_set.tensor_op.get_mask(Type.SEMANTIC)
    
    # Should return all False
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    assert mask.sum().item() == 0
    assert torch.all(mask == False)


# =====================[ get_combined_mask tests ]======================

def test_get_combined_mask_two_types(driver_set):
    """Test getting combined mask for two types."""
    mask = driver_set.tensor_op.get_combined_mask([Type.PO, Type.RB])
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 6 True values (3 PO + 3 RB)
    assert mask.sum().item() == 6
    # First 6 should be True
    assert torch.all(mask[0:6] == True)
    # Rest should be False
    assert torch.all(mask[6:] == False)


def test_get_combined_mask_three_types(driver_set):
    """Test getting combined mask for three types."""
    mask = driver_set.tensor_op.get_combined_mask([Type.PO, Type.RB, Type.P])
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have 9 True values (3 PO + 3 RB + 3 P)
    assert mask.sum().item() == 9
    # First 9 should be True
    assert torch.all(mask[0:9] == True)
    # Last should be False (GROUP)
    assert mask[9].item() == False


def test_get_combined_mask_all_types(driver_set):
    """Test getting combined mask for all types in set."""
    mask = driver_set.tensor_op.get_combined_mask([Type.PO, Type.RB, Type.P, Type.GROUP])
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should have all 10 True
    assert mask.sum().item() == 10
    assert torch.all(mask == True)


def test_get_combined_mask_single_type(driver_set):
    """Test getting combined mask for single type (should work same as get_mask)."""
    mask1 = driver_set.tensor_op.get_combined_mask([Type.PO])
    mask2 = driver_set.tensor_op.get_mask(Type.PO)
    
    assert torch.equal(mask1, mask2)


def test_get_combined_mask_empty_list(driver_set):
    """Test getting combined mask for empty list."""
    mask = driver_set.tensor_op.get_combined_mask([])
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should be all False
    assert mask.sum().item() == 0
    assert torch.all(mask == False)


def test_get_combined_mask_nonexistent_types(driver_set):
    """Test getting combined mask for types that don't exist in set."""
    mask = driver_set.tensor_op.get_combined_mask([Type.SEMANTIC])
    
    assert mask.dtype == torch.bool
    assert len(mask) == 10
    # Should be all False
    assert mask.sum().item() == 0


# =====================[ get_count tests ]======================

def test_get_count_all_tokens(driver_set):
    """Test getting count of all tokens in set."""
    count = driver_set.tensor_op.get_count()
    
    assert count == 10  # All tokens in DRIVER set
    assert isinstance(count, int)


def test_get_count_single_type(driver_set):
    """Test getting count of a single type."""
    count = driver_set.tensor_op.get_count(Type.PO)
    
    assert count == 3  # 3 PO tokens in DRIVER set
    assert isinstance(count, int)


def test_get_count_multiple_types(driver_set):
    """Test getting count of different types."""
    po_count = driver_set.tensor_op.get_count(Type.PO)
    rb_count = driver_set.tensor_op.get_count(Type.RB)
    p_count = driver_set.tensor_op.get_count(Type.P)
    group_count = driver_set.tensor_op.get_count(Type.GROUP)
    
    assert po_count == 3
    assert rb_count == 3
    assert p_count == 3
    assert group_count == 1


def test_get_count_with_mask(driver_set):
    """Test getting count with a custom mask."""
    # Create a mask for first 5 tokens
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0:5] = True
    
    # Count all tokens in mask
    count_all = driver_set.tensor_op.get_count(mask=mask)
    assert count_all == 5
    
    # Count PO tokens in mask
    count_po = driver_set.tensor_op.get_count(Type.PO, mask=mask)
    assert count_po == 3  # First 3 are PO


def test_get_count_with_mask_partial(driver_set):
    """Test getting count with mask that partially covers tokens."""
    # Create a mask for indices 3-6 (covers some RB and some P)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[3:7] = True
    
    # Count RB tokens in mask
    count_rb = driver_set.tensor_op.get_count(Type.RB, mask=mask)
    assert count_rb == 3  # Indices 3-5 are RB (3 tokens)
    
    # Count P tokens in mask
    count_p = driver_set.tensor_op.get_count(Type.P, mask=mask)
    assert count_p == 1  # Index 6 is P (1 token)


def test_get_count_nonexistent_type(driver_set):
    """Test getting count of type that doesn't exist."""
    count = driver_set.tensor_op.get_count(Type.SEMANTIC)
    
    assert count == 0


def test_get_count_empty_mask(driver_set):
    """Test getting count with empty mask."""
    mask = torch.zeros(10, dtype=torch.bool)
    
    count = driver_set.tensor_op.get_count(mask=mask)
    assert count == 0
    
    count_po = driver_set.tensor_op.get_count(Type.PO, mask=mask)
    assert count_po == 0


def test_get_count_different_sets(driver_set, recipient_set, memory_set):
    """Test getting counts across different sets."""
    # DRIVER set has 10 tokens
    assert driver_set.tensor_op.get_count() == 10
    
    # RECIPIENT set has 10 tokens
    assert recipient_set.tensor_op.get_count() == 10
    
    # MEMORY set has 5 tokens
    assert memory_set.tensor_op.get_count() == 5


# =====================[ print tests ]======================

def test_print_not_implemented(driver_set, caplog):
    """Test that print method logs not implemented message."""
    driver_set.tensor_op.print()
    
    # Check that the log message was recorded
    assert "Printing set not implemmented yet" in caplog.text or "Printing set not implemented yet" in caplog.text


def test_print_tokens_not_implemented(driver_set, caplog):
    """Test that print_tokens method logs not implemented message."""
    driver_set.tensor_op.print_tokens()
    
    # Check that the log message was recorded
    assert "Printing tokens not implemmented yet" in caplog.text or "Printing tokens not implemented yet" in caplog.text


# =====================[ Integration tests ]======================

def test_tensor_operations_work_with_local_indices(driver_set):
    """Test that tensor operations work correctly with local indices."""
    # Get mask for PO tokens (local indices 0-2)
    po_mask = driver_set.tensor_op.get_mask(Type.PO)
    
    # Verify the mask corresponds to local indices
    assert po_mask[0].item() == True
    assert po_mask[1].item() == True
    assert po_mask[2].item() == True
    
    # Get count using the mask
    count = driver_set.tensor_op.get_count(mask=po_mask)
    assert count == 3


def test_tensor_operations_independent_across_sets(driver_set, recipient_set):
    """Test that tensor operations are independent across different sets."""
    # Both sets should have PO tokens
    driver_po_count = driver_set.tensor_op.get_count(Type.PO)
    recipient_po_count = recipient_set.tensor_op.get_count(Type.PO)
    
    # Both should have 3 PO tokens
    assert driver_po_count == 3
    assert recipient_po_count == 3
    
    # But they should be different tokens (different global indices)
    driver_po_mask = driver_set.tensor_op.get_mask(Type.PO)
    recipient_po_mask = recipient_set.tensor_op.get_mask(Type.PO)
    
    # Masks should be same length (both sets have 10 tokens)
    assert len(driver_po_mask) == len(recipient_po_mask)
    # But they represent different tokens in global space


def test_combined_mask_with_count(driver_set):
    """Test using combined mask with count operation."""
    # Get combined mask for PO and RB
    combined_mask = driver_set.tensor_op.get_combined_mask([Type.PO, Type.RB])
    
    # Count tokens matching the combined mask
    count = driver_set.tensor_op.get_count(mask=combined_mask)
    assert count == 6  # 3 PO + 3 RB
    
    # Count specific type within combined mask
    po_in_combined = driver_set.tensor_op.get_count(Type.PO, mask=combined_mask)
    assert po_in_combined == 3

