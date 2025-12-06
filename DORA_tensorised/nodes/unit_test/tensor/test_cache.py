# nodes/unit_test/tensor/test_cache.py
# Tests for Cache class

import pytest
import torch
from nodes.network.tokens.tensor.cache import Cache, Analog_Cache
from nodes.enums import Set, TF, B, Type, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens across different sets.
    Tensor structure: [ID, TYPE, SET, ANALOG, ... (other fields), ACT, ...]
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all tokens (they're active)
    tensor[:, TF.DELETED] = B.FALSE
    
    # Create tokens in different sets
    # Set 0 (DRIVER): tokens 0-4
    tensor[0:5, TF.SET] = Set.DRIVER
    tensor[0:5, TF.ANALOG] = 0
    tensor[0:5, TF.ACT] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Set 1 (RECIPIENT): tokens 5-9
    tensor[5:10, TF.SET] = Set.RECIPIENT
    tensor[5:10, TF.ANALOG] = 1
    tensor[5:10, TF.ACT] = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Set 2 (MEMORY): tokens 10-14
    tensor[10:15, TF.SET] = Set.MEMORY
    tensor[10:15, TF.ANALOG] = 2
    tensor[10:15, TF.ACT] = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5])
    
    # Set 3 (NEW_SET): tokens 15-19
    tensor[15:20, TF.SET] = Set.NEW_SET
    tensor[15:20, TF.ANALOG] = 3
    tensor[15:20, TF.ACT] = torch.tensor([1.6, 1.7, 1.8, 1.9, 2.0])
    
    return tensor


@pytest.fixture
def cache(mock_tensor):
    """Create a Cache instance with mock tensor."""
    return Cache(mock_tensor)


def test_cache_init(cache, mock_tensor):
    """Test Cache initialization."""
    assert torch.equal(cache.tensor, mock_tensor)
    assert cache.masks == {}
    assert cache.analogs.shape == (0,)
    assert cache.analogs.dtype == tensor_type


def test_get_set_mask(cache):
    """Test getting set masks."""
    # Test DRIVER set
    driver_mask = cache.get_set_mask(Set.DRIVER)
    assert driver_mask.dtype == torch.bool
    assert driver_mask.sum() == 5
    assert torch.all(driver_mask[0:5])
    assert not torch.any(driver_mask[5:20])
    
    # Test RECIPIENT set
    recipient_mask = cache.get_set_mask(Set.RECIPIENT)
    assert recipient_mask.sum() == 5
    assert torch.all(recipient_mask[5:10])
    assert not torch.any(recipient_mask[0:5])
    assert not torch.any(recipient_mask[10:20])
    
    # Test MEMORY set
    memory_mask = cache.get_set_mask(Set.MEMORY)
    assert memory_mask.sum() == 5
    assert torch.all(memory_mask[10:15])
    
    # Test NEW_SET
    new_set_mask = cache.get_set_mask(Set.NEW_SET)
    assert new_set_mask.sum() == 5
    assert torch.all(new_set_mask[15:20])
    
    # Test that mask is cached
    assert Set.DRIVER in cache.masks
    assert Set.RECIPIENT in cache.masks
    assert Set.MEMORY in cache.masks
    assert Set.NEW_SET in cache.masks


def test_get_set_indices(cache):
    """Test getting set indices."""
    # Test DRIVER set indices
    driver_indices = cache.get_set_indices(Set.DRIVER)
    expected_driver = torch.tensor([0, 1, 2, 3, 4])
    assert torch.equal(driver_indices, expected_driver)
    
    # Test RECIPIENT set indices
    recipient_indices = cache.get_set_indices(Set.RECIPIENT)
    expected_recipient = torch.tensor([5, 6, 7, 8, 9])
    assert torch.equal(recipient_indices, expected_recipient)
    
    # Test MEMORY set indices
    memory_indices = cache.get_set_indices(Set.MEMORY)
    expected_memory = torch.tensor([10, 11, 12, 13, 14])
    assert torch.equal(memory_indices, expected_memory)
    
    # Test NEW_SET indices
    new_set_indices = cache.get_set_indices(Set.NEW_SET)
    expected_new_set = torch.tensor([15, 16, 17, 18, 19])
    assert torch.equal(new_set_indices, expected_new_set)


def test_get_set_count(cache):
    """Test getting set counts."""
    assert cache.get_set_count(Set.DRIVER) == 5
    assert cache.get_set_count(Set.RECIPIENT) == 5
    assert cache.get_set_count(Set.MEMORY) == 5
    assert cache.get_set_count(Set.NEW_SET) == 5


def test_cache_sets(cache):
    """Test caching multiple sets."""
    # Initially masks should be empty
    assert len(cache.masks) == 0
    
    # Cache multiple sets
    sets_to_cache = [Set.DRIVER, Set.RECIPIENT, Set.MEMORY]
    cache.cache_sets(sets_to_cache)
    
    # Check that masks are now cached
    assert Set.DRIVER in cache.masks
    assert Set.RECIPIENT in cache.masks
    assert Set.MEMORY in cache.masks
    assert Set.NEW_SET not in cache.masks  # Not cached yet
    
    # Verify the masks are correct
    assert cache.masks[Set.DRIVER].sum() == 5
    assert cache.masks[Set.RECIPIENT].sum() == 5
    assert cache.masks[Set.MEMORY].sum() == 5


def test_cache_with_empty_set(cache):
    """Test cache behavior with a set that has no tokens."""
    # Create a tensor with no tokens in NEW_SET
    empty_tensor = torch.full((10, len(TF)), null, dtype=tensor_type)
    empty_tensor[:, TF.DELETED] = B.FALSE
    empty_tensor[:, TF.SET] = Set.DRIVER  # All tokens in DRIVER
    
    empty_cache = Cache(empty_tensor)
    
    # NEW_SET should have count 0
    assert empty_cache.get_set_count(Set.NEW_SET) == 0
    
    # NEW_SET mask should be all False
    new_set_mask = empty_cache.get_set_mask(Set.NEW_SET)
    assert new_set_mask.sum() == 0
    assert not torch.any(new_set_mask)
    
    # NEW_SET indices should be empty
    new_set_indices = empty_cache.get_set_indices(Set.NEW_SET)
    assert len(new_set_indices) == 0


def test_cache_mask_caching(cache):
    """Test that masks are properly cached and reused."""
    # First call should create the mask
    mask1 = cache.get_set_mask(Set.DRIVER)
    
    # Second call should return the same cached mask
    mask2 = cache.get_set_mask(Set.DRIVER)
    
    # They should be the same object (cached)
    assert mask1 is mask2
    
    # Verify it's in the masks dictionary
    assert Set.DRIVER in cache.masks
    assert cache.masks[Set.DRIVER] is mask1


def test_cache_with_different_analogs(mock_tensor):
    """Test cache with tokens having different analog numbers."""
    # Modify mock tensor to have multiple analogs in same set
    mock_tensor[0:3, TF.ANALOG] = 0  # First 3 tokens: analog 0
    mock_tensor[3:5, TF.ANALOG] = 1  # Next 2 tokens: analog 1
    mock_tensor[5:8, TF.ANALOG] = 0  # Next 3 tokens: analog 0
    mock_tensor[8:10, TF.ANALOG] = 2  # Next 2 tokens: analog 2
    
    cache = Cache(mock_tensor)
    
    # All tokens 0-9 are in RECIPIENT set (after modification)
    # But we should still be able to get masks by set
    recipient_mask = cache.get_set_mask(Set.RECIPIENT)
    assert recipient_mask.sum() == 5  # Still 5 tokens in RECIPIENT (indices 5-9)


def test_analogs_initialization(cache):
    """Test that analogs tensor is properly initialized."""
    assert cache.analogs.shape == (0,)
    assert cache.analogs.dtype == tensor_type

def test_cache_analogs(cache):
    """Test caching analog information."""
    cache.cache_analogs()
    
    # Check that analogs tensor has the correct shape
    # Should have 4 columns: [analog_number, analog_set, count, activation]
    assert cache.analogs.shape[1] == 4
    
    # Check that we have 4 unique analogs (0, 1, 2, 3)
    assert cache.analogs.shape[0] == 4
    
    # Check analog numbers
    analog_numbers = cache.analogs[:, Analog_Cache.ANALOG_NUMBERS]
    expected_numbers = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.allclose(analog_numbers, expected_numbers)
    
    # Check counts (each analog should have 5 tokens)
    analog_counts = cache.analogs[:, Analog_Cache.ANALOG_COUNTS]
    expected_counts = torch.tensor([5.0, 5.0, 5.0, 5.0])
    assert torch.allclose(analog_counts, expected_counts)


def test_cache_with_single_token():
    """Test cache with a single token."""
    single_tensor = torch.full((1, len(TF)), null, dtype=tensor_type)
    single_tensor[0, TF.DELETED] = B.FALSE
    single_tensor[0, TF.SET] = Set.DRIVER
    single_tensor[0, TF.ANALOG] = 0
    single_tensor[0, TF.ACT] = 0.5
    
    cache = Cache(single_tensor)
    
    assert cache.get_set_count(Set.DRIVER) == 1
    assert cache.get_set_count(Set.RECIPIENT) == 0
    
    driver_indices = cache.get_set_indices(Set.DRIVER)
    assert torch.equal(driver_indices, torch.tensor([0]))


def test_cache_with_all_tokens_same_set():
    """Test cache where all tokens are in the same set."""
    same_set_tensor = torch.full((10, len(TF)), null, dtype=tensor_type)
    same_set_tensor[:, TF.DELETED] = B.FALSE
    same_set_tensor[:, TF.SET] = Set.MEMORY
    same_set_tensor[:, TF.ANALOG] = 0
    same_set_tensor[:, TF.ACT] = 0.5
    
    cache = Cache(same_set_tensor)
    
    assert cache.get_set_count(Set.MEMORY) == 10
    assert cache.get_set_count(Set.DRIVER) == 0
    assert cache.get_set_count(Set.RECIPIENT) == 0
    assert cache.get_set_count(Set.NEW_SET) == 0
    
    memory_indices = cache.get_set_indices(Set.MEMORY)
    assert len(memory_indices) == 10
    assert torch.equal(memory_indices, torch.arange(10))


# =====================[ get_arbitrary_mask tests ]======================

def test_get_arbitrary_mask_single_feature(cache):
    """Test get_arbitrary_mask with a single feature."""
    # Test matching SET feature
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER})
    assert mask.dtype == torch.bool
    assert mask.sum() == 5
    assert torch.all(mask[0:5])
    assert not torch.any(mask[5:20])
    
    # Test matching ANALOG feature
    mask = cache.get_arbitrary_mask({TF.ANALOG: 0})
    assert mask.sum() == 5
    assert torch.all(mask[0:5])
    
    # Test matching ACT feature (should match none since values are different)
    mask = cache.get_arbitrary_mask({TF.ACT: 0.1})
    assert mask.sum() == 1
    assert mask[0] == True
    assert not torch.any(mask[1:20])


def test_get_arbitrary_mask_multiple_features(cache):
    """Test get_arbitrary_mask with multiple features."""
    # Test matching SET and ANALOG together
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER, TF.ANALOG: 0})
    assert mask.sum() == 5
    assert torch.all(mask[0:5])
    assert not torch.any(mask[5:20])
    
    # Test matching SET and ANALOG that don't match together
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER, TF.ANALOG: 1})
    assert mask.sum() == 0  # No tokens match both
    
    # Test matching SET and ACT
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER, TF.ACT: 0.1})
    assert mask.sum() == 1
    assert mask[0] == True


def test_get_arbitrary_mask_excludes_deleted(cache):
    """Test that get_arbitrary_mask excludes deleted tokens."""
    # Mark some tokens as deleted
    cache.tensor[2, TF.DELETED] = B.TRUE
    cache.tensor[7, TF.DELETED] = B.TRUE
    
    # Test that deleted tokens are excluded even if they match
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER})
    assert mask.sum() == 4  # 5 - 1 deleted = 4
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False  # This one is deleted
    assert mask[3] == True
    assert mask[4] == True
    
    mask = cache.get_arbitrary_mask({TF.SET: Set.RECIPIENT})
    assert mask.sum() == 4  # 5 - 1 deleted = 4
    assert mask[7] == False  # This one is deleted


def test_get_arbitrary_mask_empty_result(cache):
    """Test get_arbitrary_mask when no tokens match."""
    # Test with non-existent SET
    mask = cache.get_arbitrary_mask({TF.SET: 999})
    assert mask.sum() == 0
    assert not torch.any(mask)
    
    # Test with non-existent ANALOG
    mask = cache.get_arbitrary_mask({TF.ANALOG: 999})
    assert mask.sum() == 0
    
    # Test with combination that doesn't exist
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER, TF.ANALOG: 999})
    assert mask.sum() == 0


def test_get_arbitrary_mask_three_features(cache):
    """Test get_arbitrary_mask with three features."""
    # Set up some tokens with specific combinations
    cache.tensor[0, TF.SET] = Set.DRIVER
    cache.tensor[0, TF.ANALOG] = 0
    cache.tensor[0, TF.ACT] = 0.1
    
    cache.tensor[1, TF.SET] = Set.DRIVER
    cache.tensor[1, TF.ANALOG] = 0
    cache.tensor[1, TF.ACT] = 0.2  # Different ACT
    
    # Test matching all three features
    mask = cache.get_arbitrary_mask({TF.SET: Set.DRIVER, TF.ANALOG: 0, TF.ACT: 0.1})
    assert mask.sum() == 1
    assert mask[0] == True
    assert mask[1] == False  # Different ACT value


def test_get_arbitrary_mask_float_values(cache):
    """Test get_arbitrary_mask with float feature values."""
    # Test matching ACT values
    mask = cache.get_arbitrary_mask({TF.ACT: 0.1})
    assert mask.sum() == 1
    assert mask[0] == True
    
    # Test matching different ACT value
    mask = cache.get_arbitrary_mask({TF.ACT: 0.5})
    assert mask.sum() == 1
    assert mask[4] == True


def test_get_arbitrary_mask_empty_dict(cache):
    """Test get_arbitrary_mask with empty dict (should return all non-deleted tokens)."""
    # Empty dict should match all non-deleted tokens
    mask = cache.get_arbitrary_mask({})
    assert mask.sum() == 20  # All 20 tokens are non-deleted
    assert torch.all(mask)
    
    # Mark some tokens as deleted
    cache.tensor[5, TF.DELETED] = B.TRUE
    cache.tensor[10, TF.DELETED] = B.TRUE
    
    # Empty dict should now match only non-deleted tokens
    mask = cache.get_arbitrary_mask({})
    assert mask.sum() == 18  # 20 - 2 deleted = 18
    assert mask[5] == False  # Deleted
    assert mask[10] == False  # Deleted


# =====================[ get_set_type_mask tests ]======================

def test_get_set_type_mask_basic(cache):
    """Test basic get_set_type_mask functionality."""
    # Set up tokens with different types
    # DRIVER set: tokens 0-4
    cache.tensor[0:2, TF.TYPE] = Type.PO
    cache.tensor[2:4, TF.TYPE] = Type.RB
    cache.tensor[4, TF.TYPE] = Type.P
    
    # RECIPIENT set: tokens 5-9
    cache.tensor[5:7, TF.TYPE] = Type.PO
    cache.tensor[7:9, TF.TYPE] = Type.RB
    cache.tensor[9, TF.TYPE] = Type.P
    
    # Test DRIVER + PO
    mask = cache.get_set_type_mask(Set.DRIVER, Type.PO)
    assert mask.sum() == 2
    assert mask[0] == True
    assert mask[1] == True
    assert not torch.any(mask[2:20])
    
    # Test DRIVER + RB
    mask = cache.get_set_type_mask(Set.DRIVER, Type.RB)
    assert mask.sum() == 2
    assert mask[2] == True
    assert mask[3] == True
    assert not torch.any(mask[0:2])
    assert not torch.any(mask[4:20])
    
    # Test RECIPIENT + PO
    mask = cache.get_set_type_mask(Set.RECIPIENT, Type.PO)
    assert mask.sum() == 2
    assert mask[5] == True
    assert mask[6] == True
    assert not torch.any(mask[0:5])
    assert not torch.any(mask[7:20])


def test_get_set_type_mask_no_matches(cache):
    """Test get_set_type_mask when no tokens match."""
    # Set all tokens to specific types
    cache.tensor[0:5, TF.TYPE] = Type.PO
    cache.tensor[5:10, TF.TYPE] = Type.RB
    
    # Test for a combination that doesn't exist
    mask = cache.get_set_type_mask(Set.DRIVER, Type.RB)
    assert mask.sum() == 0  # DRIVER has PO, not RB
    
    mask = cache.get_set_type_mask(Set.RECIPIENT, Type.PO)
    assert mask.sum() == 0  # RECIPIENT has RB, not PO
    
    # Test with non-existent set
    mask = cache.get_set_type_mask(999, Type.PO)
    assert mask.sum() == 0


def test_get_set_type_mask_all_types(cache):
    """Test get_set_type_mask with all different types."""
    # Set up tokens with all types in DRIVER set
    cache.tensor[0, TF.TYPE] = Type.PO
    cache.tensor[1, TF.TYPE] = Type.RB
    cache.tensor[2, TF.TYPE] = Type.P
    cache.tensor[3, TF.TYPE] = Type.GROUP
    cache.tensor[4, TF.TYPE] = Type.SEMANTIC
    
    # Test each type
    for i, token_type in enumerate([Type.PO, Type.RB, Type.P, Type.GROUP, Type.SEMANTIC]):
        mask = cache.get_set_type_mask(Set.DRIVER, token_type)
        assert mask.sum() == 1
        assert mask[i] == True
        assert not torch.any(mask[0:i])
        assert not torch.any(mask[i+1:20])


def test_get_set_type_mask_excludes_deleted(cache):
    """Test that get_set_type_mask excludes deleted tokens."""
    # Set up tokens
    cache.tensor[0:3, TF.TYPE] = Type.PO
    cache.tensor[3:5, TF.TYPE] = Type.RB
    
    # Mark one as deleted
    cache.tensor[1, TF.DELETED] = B.TRUE
    
    # Test that deleted token is excluded
    mask = cache.get_set_type_mask(Set.DRIVER, Type.PO)
    assert mask.sum() == 2  # 3 - 1 deleted = 2
    assert mask[0] == True
    assert mask[1] == False  # Deleted
    assert mask[2] == True


def test_get_set_type_mask_multiple_sets(cache):
    """Test get_set_type_mask across multiple sets."""
    # Set up same type in different sets
    cache.tensor[0:2, TF.TYPE] = Type.PO  # DRIVER
    cache.tensor[5:7, TF.TYPE] = Type.PO  # RECIPIENT
    cache.tensor[10:12, TF.TYPE] = Type.PO  # MEMORY
    
    # Test each set separately
    mask = cache.get_set_type_mask(Set.DRIVER, Type.PO)
    assert mask.sum() == 2
    assert torch.all(mask[0:2])
    
    mask = cache.get_set_type_mask(Set.RECIPIENT, Type.PO)
    assert mask.sum() == 2
    assert torch.all(mask[5:7])
    
    mask = cache.get_set_type_mask(Set.MEMORY, Type.PO)
    assert mask.sum() == 2
    assert torch.all(mask[10:12])


def test_get_set_type_mask_uses_cached_masks(cache):
    """Test that get_set_type_mask uses cached set masks."""
    # First call should cache the set mask
    mask1 = cache.get_set_type_mask(Set.DRIVER, Type.PO)
    
    # Verify the set mask is cached
    assert Set.DRIVER in cache.masks
    
    # Second call should use the cached mask
    mask2 = cache.get_set_type_mask(Set.DRIVER, Type.PO)
    
    # Results should be the same
    assert torch.equal(mask1, mask2)

