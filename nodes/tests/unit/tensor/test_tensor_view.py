# nodes/unit_test/tensor/test_tensor_view.py
# Tests for TensorView class

import pytest
import torch
from nodes.network.tokens.tensor_view import TensorView


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(10, 5)


@pytest.fixture
def tensor_view_contiguous(sample_tensor):
    """Create a TensorView with contiguous indices."""
    indices = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
    return TensorView(sample_tensor, indices)


@pytest.fixture
def tensor_view_non_contiguous(sample_tensor):
    """Create a TensorView with non-contiguous indices."""
    indices = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    return TensorView(sample_tensor, indices)


@pytest.fixture
def tensor_view_with_duplicates(sample_tensor):
    """Create a TensorView with duplicate indices."""
    indices = torch.tensor([2, 2, 4, 4, 6], dtype=torch.long)
    return TensorView(sample_tensor, indices)


class TestToGlobal:
    """Tests for to_global method."""
    
    def test_to_global_single_index(self, tensor_view_contiguous):
        """Test converting a single local index to global."""
        local_idx = torch.tensor([0], dtype=torch.long)
        global_idx = tensor_view_contiguous.to_global(local_idx)
        assert global_idx.item() == 2  # First index in view is 2
    
    def test_to_global_multiple_indices(self, tensor_view_contiguous):
        """Test converting multiple local indices to global."""
        local_indices = torch.tensor([0, 2, 4], dtype=torch.long)
        global_indices = tensor_view_contiguous.to_global(local_indices)
        expected = torch.tensor([2, 4, 6], dtype=torch.long)
        assert torch.equal(global_indices, expected)
    
    def test_to_global_non_contiguous_view(self, tensor_view_non_contiguous):
        """Test to_global with non-contiguous view indices."""
        local_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        global_indices = tensor_view_non_contiguous.to_global(local_indices)
        expected = torch.tensor([1, 3, 5], dtype=torch.long)
        assert torch.equal(global_indices, expected)
    
    def test_to_global_all_indices(self, tensor_view_contiguous):
        """Test converting all local indices to global."""
        local_indices = torch.arange(len(tensor_view_contiguous), dtype=torch.long)
        global_indices = tensor_view_contiguous.to_global(local_indices)
        expected = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
        assert torch.equal(global_indices, expected)
    
    def test_to_global_list_input(self, tensor_view_contiguous):
        """Test to_global with list input."""
        local_indices = [0, 2, 4]
        global_indices = tensor_view_contiguous.to_global(local_indices)
        expected = torch.tensor([2, 4, 6], dtype=torch.long)
        assert torch.equal(global_indices, expected)
    
    def test_to_global_out_of_bounds_negative(self, tensor_view_contiguous):
        """Test to_global raises IndexError for negative indices."""
        local_indices = torch.tensor([-1], dtype=torch.long)
        with pytest.raises(IndexError, match="Invalid local indices"):
            tensor_view_contiguous.to_global(local_indices)
    
    def test_to_global_out_of_bounds_too_large(self, tensor_view_contiguous):
        """Test to_global raises IndexError for indices >= view size."""
        view_size = len(tensor_view_contiguous)
        local_indices = torch.tensor([view_size], dtype=torch.long)
        with pytest.raises(IndexError, match="Invalid local indices"):
            tensor_view_contiguous.to_global(local_indices)
    
    def test_to_global_mixed_valid_invalid(self, tensor_view_contiguous):
        """Test to_global raises error when some indices are invalid."""
        local_indices = torch.tensor([0, 10, 2], dtype=torch.long)
        with pytest.raises(IndexError, match="Invalid local indices"):
            tensor_view_contiguous.to_global(local_indices)
    
    def test_to_global_empty_tensor(self, sample_tensor):
        """Test to_global with empty view."""
        empty_indices = torch.tensor([], dtype=torch.long)
        empty_view = TensorView(sample_tensor, empty_indices)
        local_indices = torch.tensor([], dtype=torch.long)
        global_indices = empty_view.to_global(local_indices)
        assert len(global_indices) == 0


class TestToLocal:
    """Tests for to_local method."""
    
    def test_to_local_single_index(self, tensor_view_contiguous):
        """Test converting a single global index to local."""
        global_idx = torch.tensor([2], dtype=torch.long)
        local_idx = tensor_view_contiguous.to_local(global_idx)
        assert local_idx.item() == 0  # Global index 2 is at local position 0
    
    def test_to_local_multiple_indices(self, tensor_view_contiguous):
        """Test converting multiple global indices to local."""
        global_indices = torch.tensor([2, 4, 6], dtype=torch.long)
        local_indices = tensor_view_contiguous.to_local(global_indices)
        expected = torch.tensor([0, 2, 4], dtype=torch.long)
        assert torch.equal(local_indices, expected)
    
    def test_to_local_non_contiguous_view(self, tensor_view_non_contiguous):
        """Test to_local with non-contiguous view indices."""
        global_indices = torch.tensor([1, 3, 5], dtype=torch.long)
        local_indices = tensor_view_non_contiguous.to_local(global_indices)
        expected = torch.tensor([0, 1, 2], dtype=torch.long)
        assert torch.equal(local_indices, expected)
    
    def test_to_local_all_indices(self, tensor_view_contiguous):
        """Test converting all global indices in view to local."""
        global_indices = torch.tensor([2, 3, 4, 5, 6], dtype=torch.long)
        local_indices = tensor_view_contiguous.to_local(global_indices)
        expected = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        assert torch.equal(local_indices, expected)
    
    def test_to_local_list_input(self, tensor_view_contiguous):
        """Test to_local with list input."""
        global_indices = [2, 4, 6]
        local_indices = tensor_view_contiguous.to_local(global_indices)
        expected = torch.tensor([0, 2, 4], dtype=torch.long)
        assert torch.equal(local_indices, expected)
    
    def test_to_local_with_duplicates_first_occurrence(self, tensor_view_with_duplicates):
        """Test to_local returns first occurrence when duplicates exist in view."""
        # View has indices [2, 2, 4, 4, 6]
        global_indices = torch.tensor([2, 4], dtype=torch.long)
        local_indices = tensor_view_with_duplicates.to_local(global_indices)
        # Should return first occurrence: 2 at position 0, 4 at position 2
        expected = torch.tensor([0, 2], dtype=torch.long)
        assert torch.equal(local_indices, expected)
    
    def test_to_local_invalid_index_not_in_view(self, tensor_view_contiguous):
        """Test to_local raises ValueError for global index not in view."""
        global_indices = torch.tensor([0], dtype=torch.long)  # 0 is not in view [2,3,4,5,6]
        with pytest.raises(ValueError, match="Invalid global indices"):
            tensor_view_contiguous.to_local(global_indices)
    
    def test_to_local_invalid_index_out_of_range(self, tensor_view_contiguous):
        """Test to_local raises ValueError for global index out of tensor range."""
        global_indices = torch.tensor([20], dtype=torch.long)  # Out of range for tensor size 10
        with pytest.raises(ValueError, match="Invalid global indices"):
            tensor_view_contiguous.to_local(global_indices)
    
    def test_to_local_mixed_valid_invalid(self, tensor_view_contiguous):
        """Test to_local raises error when some indices are invalid."""
        global_indices = torch.tensor([2, 0, 4], dtype=torch.long)  # 0 is not in view
        with pytest.raises(ValueError, match="Invalid global indices"):
            tensor_view_contiguous.to_local(global_indices)
    
    def test_to_local_empty_tensor(self, sample_tensor):
        """Test to_local with empty view."""
        empty_indices = torch.tensor([], dtype=torch.long)
        empty_view = TensorView(sample_tensor, empty_indices)
        global_indices = torch.tensor([], dtype=torch.long)
        local_indices = empty_view.to_local(global_indices)
        assert len(local_indices) == 0
    
    def test_to_local_empty_input(self, tensor_view_contiguous):
        """Test to_local with empty input tensor."""
        global_indices = torch.tensor([], dtype=torch.long)
        local_indices = tensor_view_contiguous.to_local(global_indices)
        assert len(local_indices) == 0


class TestToLocalToGlobalRoundTrip:
    """Tests for round-trip conversions between local and global indices."""
    
    def test_round_trip_single_index(self, tensor_view_contiguous):
        """Test round-trip conversion: local -> global -> local."""
        local_idx = torch.tensor([2], dtype=torch.long)
        global_idx = tensor_view_contiguous.to_global(local_idx)
        local_idx_back = tensor_view_contiguous.to_local(global_idx)
        assert torch.equal(local_idx, local_idx_back)
    
    def test_round_trip_multiple_indices(self, tensor_view_contiguous):
        """Test round-trip conversion with multiple indices."""
        local_indices = torch.tensor([0, 2, 4], dtype=torch.long)
        global_indices = tensor_view_contiguous.to_global(local_indices)
        local_indices_back = tensor_view_contiguous.to_local(global_indices)
        assert torch.equal(local_indices, local_indices_back)
    
    def test_round_trip_all_indices(self, tensor_view_contiguous):
        """Test round-trip conversion with all indices."""
        local_indices = torch.arange(len(tensor_view_contiguous), dtype=torch.long)
        global_indices = tensor_view_contiguous.to_global(local_indices)
        local_indices_back = tensor_view_contiguous.to_local(global_indices)
        assert torch.equal(local_indices, local_indices_back)
    
    def test_round_trip_non_contiguous(self, tensor_view_non_contiguous):
        """Test round-trip conversion with non-contiguous view."""
        local_indices = torch.tensor([0, 2, 4], dtype=torch.long)
        global_indices = tensor_view_non_contiguous.to_global(local_indices)
        local_indices_back = tensor_view_non_contiguous.to_local(global_indices)
        assert torch.equal(local_indices, local_indices_back)

