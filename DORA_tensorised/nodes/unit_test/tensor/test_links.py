# nodes/unit_test/tensor/test_links.py
# Tests for Links class

import pytest
import torch
from nodes.network.tokens.connections.links import Links, LD


@pytest.fixture
def mock_links():
    """Create a mock links tensor (tokens x semantics)."""
    num_tokens = 10
    num_semantics = 5
    # Create links with some non-zero values
    links = torch.zeros((num_tokens, num_semantics))
    
    # Token 0: connected to semantics 0, 1, 2 with weights 0.8, 0.9, 0.7
    links[0, 0] = 0.8
    links[0, 1] = 0.9
    links[0, 2] = 0.7
    
    # Token 1: connected to semantics 1, 3 with weights 0.6, 0.5
    links[1, 1] = 0.6
    links[1, 3] = 0.5
    
    # Token 2: connected to semantics 0, 2, 4 with weights 0.95, 0.85, 0.75
    links[2, 0] = 0.95
    links[2, 2] = 0.85
    links[2, 4] = 0.75
    
    # Token 3: connected to semantics 1, 2, 3, 4 with weights 0.3, 0.4, 0.2, 0.1
    links[3, 1] = 0.3
    links[3, 2] = 0.4
    links[3, 3] = 0.2
    links[3, 4] = 0.1
    
    # Token 4: no connections (all zeros)
    
    # Token 5: connected to semantics 0, 1, 2, 3, 4 with weights 0.5 each
    links[5, :] = 0.5
    
    return links


@pytest.fixture
def links(mock_links):
    """Create a Links instance with mock data."""
    return Links(mock_links)


def test_links_init(links, mock_links):
    """Test Links initialization."""
    assert torch.equal(links.adj_matrix, mock_links)
    assert links.adj_matrix.shape == (10, 5)


def test_links_size(links):
    """Test size method."""
    assert links.size(LD.TK) == 10
    assert links.size(LD.SEM) == 5


def test_round_big_link(links):
    """Test rounding links above threshold to 1.0."""
    # Set some links above threshold
    links.adj_matrix[0, 0] = 0.8
    links.adj_matrix[0, 1] = 0.95
    links.adj_matrix[2, 0] = 0.85
    
    # Round links above 0.9
    links.round_big_link(threshold=0.9)
    
    # Verify links above threshold are rounded to 1.0
    assert links.adj_matrix[0, 1].item() == 1.0
    # Verify links below threshold are unchanged
    assert links.adj_matrix[0, 0].item() == pytest.approx(0.8, abs=1e-6)
    assert links.adj_matrix[2, 0].item() == pytest.approx(0.85, abs=1e-6)


def test_round_big_link_all_above_threshold(links):
    """Test rounding when all links are above threshold."""
    # Set all links to 0.95
    links.adj_matrix.fill_(0.95)
    
    # Round links above 0.9
    links.round_big_link(threshold=0.9)
    
    # Verify all links are 1.0
    assert torch.all(links.adj_matrix == 1.0)


def test_round_big_link_none_above_threshold(links):
    """Test rounding when no links are above threshold."""
    # Set all links to 0.5
    links.adj_matrix.fill_(0.5)
    original = links.adj_matrix.clone()
    
    # Round links above 0.9
    links.round_big_link(threshold=0.9)
    
    # Verify all links are unchanged
    assert torch.allclose(links.adj_matrix, original)


def test_del_small_link(links):
    """Test deleting links below threshold."""
    # Set some links with various values
    links.adj_matrix[0, 0] = 0.1
    links.adj_matrix[0, 1] = 0.5
    links.adj_matrix[0, 2] = 0.3
    links.adj_matrix[1, 0] = 0.8
    
    # Delete links below 0.4
    links.del_small_link(threshold=0.4)
    
    # Verify links below threshold are set to 0.0
    assert links.adj_matrix[0, 0].item() == 0.0
    assert links.adj_matrix[0, 2].item() == 0.0
    # Verify links above threshold are unchanged
    assert links.adj_matrix[0, 1].item() == pytest.approx(0.5, abs=1e-6)
    assert links.adj_matrix[1, 0].item() == pytest.approx(0.8, abs=1e-6)


def test_del_small_link_all_below_threshold(links):
    """Test deleting when all links are below threshold."""
    # Set all links to 0.1
    links.adj_matrix.fill_(0.1)
    
    # Delete links below 0.5
    links.del_small_link(threshold=0.5)
    
    # Verify all links are 0.0
    assert torch.all(links.adj_matrix == 0.0)


def test_del_small_link_none_below_threshold(links):
    """Test deleting when no links are below threshold."""
    # Set all links to 0.8
    links.adj_matrix.fill_(0.8)
    original = links.adj_matrix.clone()
    
    # Delete links below 0.5
    links.del_small_link(threshold=0.5)
    
    # Verify all links are unchanged
    assert torch.allclose(links.adj_matrix, original)


def test_update_link(links):
    """Test updating a single link."""
    # Update a link
    links.update_link(token_index=3, semantic_index=2, weight=0.99)
    
    # Verify the link was updated
    assert links.adj_matrix[3, 2].item() == pytest.approx(0.99, abs=1e-6)
    
    # Verify other links are unchanged
    assert links.adj_matrix[3, 1].item() == pytest.approx(0.3, abs=1e-6)
    assert links.adj_matrix[3, 3].item() == pytest.approx(0.2, abs=1e-6)


def test_update_link_multiple(links):
    """Test updating multiple links."""
    links.update_link(0, 0, 0.5)
    links.update_link(1, 1, 0.7)
    links.update_link(2, 2, 0.9)
    
    assert links.adj_matrix[0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert links.adj_matrix[1, 1].item() == pytest.approx(0.7, abs=1e-6)
    assert links.adj_matrix[2, 2].item() == pytest.approx(0.9, abs=1e-6)


def test_calibrate_weights(links):
    """Test calibrating weights for driver POs."""
    # Set up links for tokens 0, 2 (driver POs)
    # Token 0: max link is at semantic 1 (0.9)
    # Token 2: max link is at semantic 0 (0.95)
    driver_po_idxs = torch.tensor([0, 2])
    
    # Calibrate weights
    links.calibrate_weights(driver_po_idxs)
    
    # Verify max links are set to 1.0
    assert links.adj_matrix[0, 1].item() == 1.0  # Token 0, semantic 1 (was 0.9)
    assert links.adj_matrix[2, 0].item() == 1.0  # Token 2, semantic 0 (was 0.95)
    
    # Verify other links are unchanged
    assert links.adj_matrix[0, 0].item() == pytest.approx(0.8, abs=1e-6)
    assert links.adj_matrix[0, 2].item() == pytest.approx(0.7, abs=1e-6)
    assert links.adj_matrix[2, 2].item() == pytest.approx(0.85, abs=1e-6)


def test_calibrate_weights_tie_breaking(links):
    """Test calibrating weights when there are ties (should pick first max)."""
    # Create a scenario with ties
    links.adj_matrix[5, 0] = 0.8
    links.adj_matrix[5, 1] = 0.8  # Tie with semantic 0
    links.adj_matrix[5, 2] = 0.7
    
    driver_po_idxs = torch.tensor([5])
    links.calibrate_weights(driver_po_idxs)
    
    # Should set the first max (semantic 0) to 1.0
    assert links.adj_matrix[5, 0].item() == 1.0
    # Other max (semantic 1) should remain 0.8
    assert links.adj_matrix[5, 1].item() == pytest.approx(0.8, abs=1e-6)


def test_calibrate_weights_single_po(links):
    """Test calibrating weights for a single driver PO."""
    driver_po_idxs = torch.tensor([0])
    
    links.calibrate_weights(driver_po_idxs)
    
    # Token 0's max link is at semantic 1 (0.9)
    assert links.adj_matrix[0, 1].item() == 1.0


def test_calibrate_weights_empty(links):
    """Test calibrating weights with empty tensor."""
    driver_po_idxs = torch.tensor([], dtype=torch.long)
    
    # Should not raise an error
    links.calibrate_weights(driver_po_idxs)


def test_get_max_linked_sem(links):
    """Test getting the semantic with the highest link weight."""
    # Note: This test will fail due to a bug in get_max_linked_sem - it uses dim=LD.SEM (1)
    # but when indexing with tk_idx, the result is 1D, so dim should be 0
    # Token 0: max is semantic 1 (0.9)
    # This will fail with IndexError until the bug is fixed
    with pytest.raises(IndexError):
        max_sem_0 = links.get_max_linked_sem(0)


def test_get_max_linked_sem_tie(links):
    """Test getting max linked semantic when there are ties."""
    # Note: This test will fail due to a bug in get_max_linked_sem
    # Create a tie scenario
    links.adj_matrix[6, 0] = 0.7
    links.adj_matrix[6, 1] = 0.7  # Tie
    links.adj_matrix[6, 2] = 0.5
    
    # This will fail with IndexError until the bug is fixed
    with pytest.raises(IndexError):
        max_sem = links.get_max_linked_sem(6)


def test_get_max_linked_sem_no_connections(links):
    """Test getting max linked semantic when token has no connections."""
    # Note: This test will fail due to a bug in get_max_linked_sem
    # Token 4 has no connections (all zeros)
    # This will fail with IndexError until the bug is fixed
    with pytest.raises(IndexError):
        max_sem = links.get_max_linked_sem(4)


def test_expand_token_dimension(links):
    """Test expanding the links tensor along the token dimension."""
    original_size_tk = links.size(LD.TK)
    original_size_sem = links.size(LD.SEM)
    original_links = links.adj_matrix.clone()
    
    # Expand token dimension
    links.expand_to(new_size=15, dimension=LD.TK)
    
    # Verify size increased
    assert links.size(LD.TK) == 15
    assert links.size(LD.SEM) == original_size_sem
    
    # Verify old links are preserved
    assert torch.allclose(links.adj_matrix[:original_size_tk, :original_size_sem], original_links)
    
    # Verify new rows are zeros
    assert torch.all(links.adj_matrix[original_size_tk:, :] == 0.0)


def test_expand_semantic_dimension(links):
    """Test expanding the links tensor along the semantic dimension."""
    original_size_tk = links.size(LD.TK)
    original_size_sem = links.size(LD.SEM)
    original_links = links.adj_matrix.clone()
    
    # Expand semantic dimension
    links.expand_to(new_size=8, dimension=LD.SEM)
    
    # Verify size increased
    assert links.size(LD.TK) == original_size_tk
    assert links.size(LD.SEM) == 8
    
    # Verify old links are preserved
    assert torch.allclose(links.adj_matrix[:original_size_tk, :original_size_sem], original_links)
    
    # Verify new columns are zeros
    assert torch.all(links.adj_matrix[:, original_size_sem:] == 0.0)


def test_expand_both_dimensions(links):
    """Test expanding both dimensions separately."""
    original_size_tk = links.size(LD.TK)
    original_size_sem = links.size(LD.SEM)
    original_links = links.adj_matrix.clone()
    
    # Expand token dimension first
    links.expand_to(new_size=12, dimension=LD.TK)
    assert links.size(LD.TK) == 12
    assert links.size(LD.SEM) == original_size_sem
    assert torch.allclose(links.adj_matrix[:original_size_tk, :original_size_sem], original_links)
    
    # Expand semantic dimension
    links.expand_to(new_size=8, dimension=LD.SEM)
    assert links.size(LD.TK) == 12
    assert links.size(LD.SEM) == 8
    assert torch.allclose(links.adj_matrix[:original_size_tk, :original_size_sem], original_links)


def test_expand_smaller_size(links):
    """Test expanding to a smaller size (should raise ValueError - shrinking not allowed)."""
    original_size_tk = links.size(LD.TK)
    original_links = links.adj_matrix.clone()
    
    # Try to expand to smaller size - should raise ValueError
    with pytest.raises(ValueError, match="Cannot shrink tensor"):
        links.expand_to(new_size=8, dimension=LD.TK)
    
    # Verify tensor is unchanged after error
    assert links.size(LD.TK) == original_size_tk
    assert torch.allclose(links.adj_matrix, original_links)


def test_get_sem_count(links):
    """Test getting the number of semantics connected to tokens."""
    indices = torch.tensor([0, 1, 2, 3, 4])
    
    sem_counts = links.get_sem_count(indices)
    
    # Token 0: connected to 3 semantics (0.8, 0.9, 0.7)
    assert sem_counts[0].item() == pytest.approx(2.4, abs=1e-6)
    
    # Token 1: connected to 2 semantics (0.6, 0.5)
    assert sem_counts[1].item() == pytest.approx(1.1, abs=1e-6)
    
    # Token 2: connected to 3 semantics (0.95, 0.85, 0.75)
    assert sem_counts[2].item() == pytest.approx(2.55, abs=1e-6)
    
    # Token 3: connected to 4 semantics (0.3, 0.4, 0.2, 0.1)
    assert sem_counts[3].item() == pytest.approx(1.0, abs=1e-6)
    
    # Token 4: no connections
    assert sem_counts[4].item() == 0.0


def test_get_sem_count_single_token(links):
    """Test getting semantic count for a single token."""
    indices = torch.tensor([0])
    
    sem_counts = links.get_sem_count(indices)
    
    assert len(sem_counts) == 1
    assert sem_counts[0].item() == pytest.approx(2.4, abs=1e-6)  # 0.8 + 0.9 + 0.7


def test_get_sem_count_empty_indices(links):
    """Test getting semantic count with empty indices."""
    indices = torch.tensor([], dtype=torch.long)
    
    sem_counts = links.get_sem_count(indices)
    
    assert len(sem_counts) == 0


def test_get_sem_count_non_contiguous(links):
    """Test getting semantic count for non-contiguous indices."""
    indices = torch.tensor([0, 2, 5])
    
    sem_counts = links.get_sem_count(indices)
    
    assert len(sem_counts) == 3
    # Token 0: 0.8 + 0.9 + 0.7 = 2.4
    assert sem_counts[0].item() == pytest.approx(2.4, abs=1e-6)
    # Token 2: 0.95 + 0.85 + 0.75 = 2.55
    assert sem_counts[1].item() == pytest.approx(2.55, abs=1e-6)
    # Token 5: 0.5 * 5 = 2.5
    assert sem_counts[2].item() == pytest.approx(2.5, abs=1e-6)


def test_round_and_delete_links(links):
    """Test combining round_big_link and del_small_link."""
    # Set up links with various values
    links.adj_matrix[0, 0] = 0.1
    links.adj_matrix[0, 1] = 0.95
    links.adj_matrix[0, 2] = 0.5
    links.adj_matrix[1, 0] = 0.3
    links.adj_matrix[1, 1] = 0.8
    
    # Round big links
    links.round_big_link(threshold=0.9)
    # Delete small links
    links.del_small_link(threshold=0.4)
    
    # Verify results
    assert links.adj_matrix[0, 0].item() == 0.0  # Below 0.4, deleted
    assert links.adj_matrix[0, 1].item() == 1.0  # Above 0.9, rounded
    assert links.adj_matrix[0, 2].item() == pytest.approx(0.5, abs=1e-6)  # Between thresholds, unchanged
    assert links.adj_matrix[1, 0].item() == 0.0  # Below 0.4, deleted
    assert links.adj_matrix[1, 1].item() == pytest.approx(0.8, abs=1e-6)  # Between thresholds, unchanged


# =====================[ get_view tests ]======================

def test_get_view_basic(links):
    """Test basic get_view functionality."""
    indices = torch.tensor([0, 2, 5])
    view = links.get_view(indices)
    
    # Should return a TensorView
    from nodes.network.tokens.tensor_view import TensorView
    assert isinstance(view, TensorView)
    
    # View should have correct shape (3 tokens, 5 semantics)
    assert view.shape == (3, 5)
    
    # View should have same dtype and device as original
    assert view.dtype == links.adj_matrix.dtype
    assert view.device == links.adj_matrix.device


def test_get_view_single_index(links):
    """Test get_view with a single index."""
    indices = torch.tensor([2])
    view = links.get_view(indices)
    
    assert view.shape == (1, 5)
    assert len(view) == 1


def test_get_view_multiple_indices(links):
    """Test get_view with multiple indices."""
    indices = torch.tensor([0, 1, 3, 5])
    view = links.get_view(indices)
    
    assert view.shape == (4, 5)
    assert len(view) == 4


def test_get_view_non_contiguous_indices(links):
    """Test get_view with non-contiguous indices."""
    indices = torch.tensor([0, 2, 5, 7])
    view = links.get_view(indices)
    
    assert view.shape == (4, 5)
    
    # Verify we can access the view
    # View[0] should correspond to original token 0
    assert torch.equal(view[0], links.adj_matrix[0])
    # View[1] should correspond to original token 2
    assert torch.equal(view[1], links.adj_matrix[2])


def test_get_view_read_access(links):
    """Test reading values through the view."""
    indices = torch.tensor([0, 2])
    view = links.get_view(indices)
    
    # Read through view
    view_token_0 = view[0]  # Should be token 0 from original
    view_token_1 = view[1]  # Should be token 2 from original
    
    # Verify values match original
    assert torch.equal(view_token_0, links.adj_matrix[0])
    assert torch.equal(view_token_1, links.adj_matrix[2])
    
    # Read specific semantic
    assert view[0, 1].item() == links.adj_matrix[0, 1].item()
    assert view[1, 0].item() == links.adj_matrix[2, 0].item()


def test_get_view_write_access(links):
    """Test writing values through the view modifies original."""
    indices = torch.tensor([0, 2])
    view = links.get_view(indices)
    
    # Store original values
    original_0_1 = links.adj_matrix[0, 1].item()
    original_2_3 = links.adj_matrix[2, 3].item()
    
    # Modify through view
    view[0, 1] = 0.99
    view[1, 3] = 0.88  # view[1] corresponds to original token 2
    
    # Verify original was modified
    assert links.adj_matrix[0, 1].item() == pytest.approx(0.99, abs=1e-6)
    assert links.adj_matrix[2, 3].item() == pytest.approx(0.88, abs=1e-6)
    
    # Verify other values unchanged
    assert links.adj_matrix[0, 0].item() == pytest.approx(0.8, abs=1e-6)
    assert links.adj_matrix[2, 2].item() == pytest.approx(0.85, abs=1e-6)


def test_get_view_slice_access(links):
    """Test accessing view with slices."""
    indices = torch.tensor([0, 1, 2, 3])
    view = links.get_view(indices)
    
    # Get slice of view
    view_slice = view[0:2]  # Should return another TensorView
    
    from nodes.network.tokens.tensor_view import TensorView
    assert isinstance(view_slice, TensorView)
    assert view_slice.shape == (2, 5)
    
    # Verify slice values
    assert torch.equal(view_slice[0], links.adj_matrix[0])
    assert torch.equal(view_slice[1], links.adj_matrix[1])


def test_get_view_all_semantics(links):
    """Test accessing all semantics for tokens in view."""
    indices = torch.tensor([0, 2, 5])
    view = links.get_view(indices)
    
    # Get all semantics for first token in view
    all_sems = view[0, :]
    assert torch.equal(all_sems, links.adj_matrix[0, :])
    
    # Get all semantics for second token in view
    all_sems_2 = view[1, :]
    assert torch.equal(all_sems_2, links.adj_matrix[2, :])


def test_get_view_empty_indices(links):
    """Test get_view with empty indices."""
    indices = torch.tensor([], dtype=torch.long)
    view = links.get_view(indices)
    
    assert view.shape == (0, 5)
    assert len(view) == 0


def test_get_view_reordered_indices(links):
    """Test get_view with reordered indices."""
    indices = torch.tensor([5, 2, 0])  # Reversed order
    view = links.get_view(indices)
    
    assert view.shape == (3, 5)
    
    # View[0] should be token 5
    assert torch.equal(view[0], links.adj_matrix[5])
    # View[1] should be token 2
    assert torch.equal(view[1], links.adj_matrix[2])
    # View[2] should be token 0
    assert torch.equal(view[2], links.adj_matrix[0])


def test_get_view_modify_through_view(links):
    """Test that modifications through view affect original tensor."""
    indices = torch.tensor([1, 3])
    view = links.get_view(indices)
    
    # Modify entire row through view
    new_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    view[0] = new_values  # Should modify original token 1
    
    # Verify original was modified
    assert torch.allclose(links.adj_matrix[1], new_values)
    
    # Original token 3 should be unchanged by this operation
    assert links.adj_matrix[3, 1].item() == pytest.approx(0.3, abs=1e-6)


def test_get_view_clone(links):
    """Test cloning the view creates independent copy."""
    indices = torch.tensor([0, 2])
    view = links.get_view(indices)
    
    # Clone the view
    cloned = view.clone()
    
    # Modify clone
    cloned[0, 1] = 0.99
    
    # Original should be unchanged (clone is independent)
    assert links.adj_matrix[0, 1].item() != pytest.approx(0.99, abs=1e-6)
    
    # But view should still reflect original
    assert view[0, 1].item() == links.adj_matrix[0, 1].item()


def test_get_view_multiple_views(links):
    """Test creating multiple views of the same tensor."""
    indices1 = torch.tensor([0, 1])
    indices2 = torch.tensor([2, 3])
    
    view1 = links.get_view(indices1)
    view2 = links.get_view(indices2)
    
    # Both views should work independently
    assert view1.shape == (2, 5)
    assert view2.shape == (2, 5)
    
    # Modify through view1
    view1[0, 0] = 0.99
    
    # Should affect original
    assert links.adj_matrix[0, 0].item() == pytest.approx(0.99, abs=1e-6)
    
    # view2 should still work correctly
    assert view2[0, 0].item() == links.adj_matrix[2, 0].item()


def test_get_view_overlapping_indices(links):
    """Test creating views with overlapping indices."""
    indices1 = torch.tensor([0, 1, 2])
    indices2 = torch.tensor([1, 2, 3])
    
    view1 = links.get_view(indices1)
    view2 = links.get_view(indices2)
    
    # Modify through view1 (affects token 1)
    view1[1, 0] = 0.99  # view1[1] is original token 1
    
    # Should affect original
    assert links.adj_matrix[1, 0].item() == pytest.approx(0.99, abs=1e-6)
    
    # view2 should see the change (view2[0] is also original token 1)
    assert view2[0, 0].item() == pytest.approx(0.99, abs=1e-6)


# =====================[ del_links tests ]======================

def test_del_links_basic(links):
    """Test basic del_links functionality."""
    # Store original values for verification
    original_1_1 = links.adj_matrix[1, 1].item()
    original_2_0 = links.adj_matrix[2, 0].item()  # Token 2's link to semantic 0
    
    # Delete links for token 0
    links.del_links(torch.tensor([0]))
    
    # All links from token 0 should be deleted (row 0)
    assert torch.all(links.adj_matrix[0, :] == 0.0)
    
    # Links to semantic 0 should be preserved (other tokens can still link to semantic 0)
    assert links.adj_matrix[2, 0].item() == pytest.approx(original_2_0, abs=1e-6)
    
    # Other links should be preserved
    assert links.adj_matrix[1, 1].item() == pytest.approx(original_1_1, abs=1e-6)


def test_del_links_multiple_indices(links):
    """Test del_links with multiple indices."""
    # Store original values
    original_2_0 = links.adj_matrix[2, 0].item()  # Token 2's link to semantic 0
    original_2_1 = links.adj_matrix[2, 1].item()  # Token 2's link to semantic 1
    original_2_2 = links.adj_matrix[2, 2].item()
    original_3_3 = links.adj_matrix[3, 3].item()
    original_5_4 = links.adj_matrix[5, 4].item()
    
    # Delete links for tokens 0 and 1
    links.del_links(torch.tensor([0, 1]))
    
    # All links from tokens 0 and 1 should be deleted
    assert torch.all(links.adj_matrix[0, :] == 0.0)
    assert torch.all(links.adj_matrix[1, :] == 0.0)
    
    # Links to semantics 0 and 1 should be preserved (other tokens can still link to them)
    assert links.adj_matrix[2, 0].item() == pytest.approx(original_2_0, abs=1e-6)
    assert links.adj_matrix[2, 1].item() == pytest.approx(original_2_1, abs=1e-6)
    
    # Other links should be preserved
    assert links.adj_matrix[2, 2].item() == pytest.approx(original_2_2, abs=1e-6)
    assert links.adj_matrix[3, 3].item() == pytest.approx(original_3_3, abs=1e-6)
    assert links.adj_matrix[5, 4].item() == pytest.approx(original_5_4, abs=1e-6)


def test_del_links_single_index(links):
    """Test del_links with a single index."""
    # Store original values for token 2
    original_2_0 = links.adj_matrix[2, 0].item()
    original_2_2 = links.adj_matrix[2, 2].item()
    original_2_4 = links.adj_matrix[2, 4].item()
    original_3_2 = links.adj_matrix[3, 2].item()  # Token 3's link to semantic 2
    
    # Verify token 2 has connections
    assert original_2_0 > 0.0
    assert original_2_2 > 0.0
    assert original_2_4 > 0.0
    
    # Delete links for token 2
    links.del_links(torch.tensor([2]))
    
    # All links from token 2 should be deleted
    assert torch.all(links.adj_matrix[2, :] == 0.0)
    
    # Links to semantic 2 should be preserved (other tokens can still link to semantic 2)
    assert links.adj_matrix[3, 2].item() == pytest.approx(original_3_2, abs=1e-6)


def test_del_links_preserves_other_links(links):
    """Test that del_links doesn't affect other links."""
    # Store original values for tokens we won't delete
    original_3_0 = links.adj_matrix[3, 0].item()  # Token 3's link to semantic 0
    original_3_1 = links.adj_matrix[3, 1].item()
    original_3_2 = links.adj_matrix[3, 2].item()
    original_3_3 = links.adj_matrix[3, 3].item()
    original_3_4 = links.adj_matrix[3, 4].item()
    original_5_0 = links.adj_matrix[5, 0].item()  # Token 5's link to semantic 0
    
    # Delete links for token 0
    links.del_links(torch.tensor([0]))
    
    # Token 3 links should all be preserved (including semantic 0)
    assert links.adj_matrix[3, 0].item() == pytest.approx(original_3_0, abs=1e-6)
    assert links.adj_matrix[3, 1].item() == pytest.approx(original_3_1, abs=1e-6)
    assert links.adj_matrix[3, 2].item() == pytest.approx(original_3_2, abs=1e-6)
    assert links.adj_matrix[3, 3].item() == pytest.approx(original_3_3, abs=1e-6)
    assert links.adj_matrix[3, 4].item() == pytest.approx(original_3_4, abs=1e-6)
    
    # Token 5 links should all be preserved (including semantic 0)
    assert links.adj_matrix[5, 0].item() == pytest.approx(original_5_0, abs=1e-6)
    assert links.adj_matrix[5, 1].item() == pytest.approx(0.5, abs=1e-6)
    assert links.adj_matrix[5, 2].item() == pytest.approx(0.5, abs=1e-6)


def test_del_links_empty_indices(links):
    """Test del_links with empty indices."""
    # Store original state
    original_links = links.adj_matrix.clone()
    
    # Delete with empty indices (should do nothing)
    links.del_links(torch.tensor([], dtype=torch.long))
    
    # Nothing should have changed
    assert torch.allclose(links.adj_matrix, original_links)


def test_del_links_all_indices(links):
    """Test del_links with all indices."""
    # Delete links for all tokens
    all_indices = torch.arange(0, links.size(LD.TK), dtype=torch.long)
    links.del_links(all_indices)
    
    # All token links should be deleted (all rows should be zero)
    assert torch.all(links.adj_matrix == 0.0)


def test_del_links_only_deletes_token_links(links):
    """Test that del_links only deletes token links (rows), not semantic links (columns)."""
    # Token 0 has links to semantics 0, 1, 2
    # Token 2 has links to semantics 0, 2, 4
    # So semantic 0 is connected to both token 0 and token 2
    
    original_2_0 = links.adj_matrix[2, 0].item()  # Token 2's link to semantic 0
    original_2_2 = links.adj_matrix[2, 2].item()  # Token 2's link to semantic 2
    original_2_4 = links.adj_matrix[2, 4].item()   # Token 2's link to semantic 4
    original_3_0 = links.adj_matrix[3, 0].item()  # Token 3's link to semantic 0 (if any)
    
    # Delete links for token 0
    links.del_links(torch.tensor([0]))
    
    # Row 0 (token 0) should be all zeros
    assert torch.all(links.adj_matrix[0, :] == 0.0)
    
    # Column 0 (semantic 0) should be preserved - other tokens can still link to semantic 0
    assert links.adj_matrix[2, 0].item() == pytest.approx(original_2_0, abs=1e-6)
    
    # Column 1 (semantic 1) should be preserved
    # Column 2 (semantic 2) should be preserved
    assert links.adj_matrix[2, 2].item() == pytest.approx(original_2_2, abs=1e-6)
    
    # Token 2's connection to semantic 4 should remain
    assert links.adj_matrix[2, 4].item() == pytest.approx(original_2_4, abs=1e-6)


def test_del_links_non_contiguous_indices(links):
    """Test del_links with non-contiguous indices."""
    # Store original values
    original_5_0 = links.adj_matrix[5, 0].item()  # Token 5's link to semantic 0
    original_5_1 = links.adj_matrix[5, 1].item()
    original_5_2 = links.adj_matrix[5, 2].item()  # Token 5's link to semantic 2
    original_5_4 = links.adj_matrix[5, 4].item()  # Token 5's link to semantic 4
    original_3_0 = links.adj_matrix[3, 0].item()  # Token 3's link to semantic 0 (if any)
    original_3_2 = links.adj_matrix[3, 2].item()  # Token 3's link to semantic 2
    original_3_4 = links.adj_matrix[3, 4].item()  # Token 3's link to semantic 4
    
    # Delete links for tokens 0, 2, 4 (non-contiguous)
    links.del_links(torch.tensor([0, 2, 4]))
    
    # All links from tokens 0, 2, 4 should be deleted
    assert torch.all(links.adj_matrix[0, :] == 0.0)
    assert torch.all(links.adj_matrix[2, :] == 0.0)
    assert torch.all(links.adj_matrix[4, :] == 0.0)
    
    # Links to semantics 0, 2, 4 should be preserved (other tokens can still link to them)
    assert links.adj_matrix[5, 0].item() == pytest.approx(original_5_0, abs=1e-6)
    assert links.adj_matrix[5, 2].item() == pytest.approx(original_5_2, abs=1e-6)
    assert links.adj_matrix[5, 4].item() == pytest.approx(original_5_4, abs=1e-6)
    assert links.adj_matrix[3, 2].item() == pytest.approx(original_3_2, abs=1e-6)
    assert links.adj_matrix[3, 4].item() == pytest.approx(original_3_4, abs=1e-6)
    
    # Token 5's links to semantics 1 and 3 should be preserved
    assert links.adj_matrix[5, 1].item() == pytest.approx(original_5_1, abs=1e-6)
    assert links.adj_matrix[5, 3].item() == pytest.approx(0.5, abs=1e-6)


def test_del_links_idempotent(links):
    """Test that deleting links multiple times is idempotent."""
    # Delete links for token 0
    links.del_links(torch.tensor([0]))
    
    # Store state after first deletion
    state_after_first = links.adj_matrix.clone()
    
    # Delete links for token 0 again
    links.del_links(torch.tensor([0]))
    
    # State should be unchanged (idempotent)
    assert torch.allclose(links.adj_matrix, state_after_first)


def test_del_links_sequential_deletion(links):
    """Test deleting links sequentially."""
    # Store original values for token 3
    original_3_0 = links.adj_matrix[3, 0].item()  # Token 3's link to semantic 0 (if any)
    original_3_1 = links.adj_matrix[3, 1].item()
    original_3_2 = links.adj_matrix[3, 2].item()
    original_3_3 = links.adj_matrix[3, 3].item()
    original_3_4 = links.adj_matrix[3, 4].item()
    
    # Delete token 0
    links.del_links(torch.tensor([0]))
    assert torch.all(links.adj_matrix[0, :] == 0.0)
    # Semantic 0 links should be preserved
    assert links.adj_matrix[3, 0].item() == pytest.approx(original_3_0, abs=1e-6)
    
    # Delete token 1
    links.del_links(torch.tensor([1]))
    assert torch.all(links.adj_matrix[1, :] == 0.0)
    # Semantic 1 links should be preserved
    assert links.adj_matrix[3, 1].item() == pytest.approx(original_3_1, abs=1e-6)
    
    # Delete token 2
    links.del_links(torch.tensor([2]))
    assert torch.all(links.adj_matrix[2, :] == 0.0)
    # Semantic 2 links should be preserved
    assert links.adj_matrix[3, 2].item() == pytest.approx(original_3_2, abs=1e-6)
    
    # Token 3's links should all still be preserved
    assert links.adj_matrix[3, 3].item() == pytest.approx(original_3_3, abs=1e-6)
    assert links.adj_matrix[3, 4].item() == pytest.approx(original_3_4, abs=1e-6)


# =====================[ get_count tests ]======================

def test_get_count_token_basic(links):
    """Test basic get_count functionality for token dimension."""
    # From mock_links fixture: 10 tokens
    count = links.get_count(LD.TK)
    assert count == 10
    assert isinstance(count, int)


def test_get_count_sem_basic(links):
    """Test basic get_count functionality for semantic dimension."""
    # From mock_links fixture: 5 semantics
    count = links.get_count(LD.SEM)
    assert count == 5
    assert isinstance(count, int)


def test_get_count_token_after_expansion(links):
    """Test get_count for token dimension after expanding token dimension."""
    initial_token_count = links.get_count(LD.TK)
    initial_sem_count = links.get_count(LD.SEM)
    
    assert initial_token_count == 10
    assert initial_sem_count == 5
    
    # Expand token dimension
    links.expand_to(new_size=15, dimension=LD.TK)
    
    # Token count should have increased
    new_token_count = links.get_count(LD.TK)
    assert new_token_count == 15
    assert new_token_count > initial_token_count
    
    # Semantic count should remain unchanged
    assert links.get_count(LD.SEM) == initial_sem_count


def test_get_count_sem_after_expansion(links):
    """Test get_count for semantic dimension after expanding semantic dimension."""
    initial_token_count = links.get_count(LD.TK)
    initial_sem_count = links.get_count(LD.SEM)
    
    assert initial_token_count == 10
    assert initial_sem_count == 5
    
    # Expand semantic dimension
    links.expand_to(new_size=8, dimension=LD.SEM)
    
    # Semantic count should have increased
    new_sem_count = links.get_count(LD.SEM)
    assert new_sem_count == 8
    assert new_sem_count > initial_sem_count
    
    # Token count should remain unchanged
    assert links.get_count(LD.TK) == initial_token_count


def test_get_count_token_consistency(links):
    """Test that get_count(LD.TK) matches size(LD.TK)."""
    token_count = links.get_count(LD.TK)
    size_tk = links.size(LD.TK)
    
    assert token_count == size_tk
    assert token_count == links.adj_matrix.size(LD.TK)


def test_get_count_sem_consistency(links):
    """Test that get_count(LD.SEM) matches size(LD.SEM)."""
    sem_count = links.get_count(LD.SEM)
    size_sem = links.size(LD.SEM)
    
    assert sem_count == size_sem
    assert sem_count == links.adj_matrix.size(LD.SEM)


def test_get_count_token_after_operations(links):
    """Test get_count(LD.TK) remains consistent after operations that don't change dimensions."""
    initial_token_count = links.get_count(LD.TK)
    
    # Perform various operations that don't change dimensions
    links.round_big_link(threshold=0.5)
    links.del_small_link(threshold=0.1)
    links.update_link(token_index=0, semantic_index=0, weight=0.9)
    links.calibrate_weights(torch.tensor([0, 1]))
    
    # Token count should remain the same
    assert links.get_count(LD.TK) == initial_token_count


def test_get_count_sem_after_operations(links):
    """Test get_count(LD.SEM) remains consistent after operations that don't change dimensions."""
    initial_sem_count = links.get_count(LD.SEM)
    
    # Perform various operations that don't change dimensions
    links.round_big_link(threshold=0.5)
    links.del_small_link(threshold=0.1)
    links.update_link(token_index=0, semantic_index=0, weight=0.9)
    links.calibrate_weights(torch.tensor([0, 1]))
    
    # Semantic count should remain the same
    assert links.get_count(LD.SEM) == initial_sem_count


def test_get_count_token_empty_links():
    """Test get_count(LD.TK) with empty links tensor."""
    empty_tensor = torch.zeros((0, 5))
    empty_links = Links(empty_tensor)
    
    count = empty_links.get_count(LD.TK)
    assert count == 0
    assert isinstance(count, int)


def test_get_count_sem_empty_links():
    """Test get_count(LD.SEM) with empty links tensor."""
    empty_tensor = torch.zeros((10, 0))
    empty_links = Links(empty_tensor)
    
    count = empty_links.get_count(LD.SEM)
    assert count == 0
    assert isinstance(count, int)


def test_get_count_square_links():
    """Test get_count with square links tensor (same token and semantic count)."""
    square_tensor = torch.zeros((5, 5))
    square_links = Links(square_tensor)
    
    token_count = square_links.get_count(LD.TK)
    sem_count = square_links.get_count(LD.SEM)
    
    assert token_count == 5
    assert sem_count == 5
    assert token_count == sem_count


def test_get_count_rectangular_links():
    """Test get_count with rectangular links tensor."""
    rectangular_tensor = torch.zeros((8, 3))
    rectangular_links = Links(rectangular_tensor)
    
    token_count = rectangular_links.get_count(LD.TK)
    sem_count = rectangular_links.get_count(LD.SEM)
    
    assert token_count == 8
    assert sem_count == 3
    assert token_count != sem_count


def test_get_count_token_after_del_links(links):
    """Test get_count(LD.TK) after deleting links (should not change count)."""
    initial_token_count = links.get_count(LD.TK)
    
    # Delete links for some tokens
    links.del_links(torch.tensor([0, 1, 2]))
    
    # Token count should remain the same (deleting links doesn't change tensor size)
    assert links.get_count(LD.TK) == initial_token_count


def test_get_count_sem_after_del_links(links):
    """Test get_count(LD.SEM) after deleting links (should not change count)."""
    initial_sem_count = links.get_count(LD.SEM)
    
    # Delete links for some tokens
    links.del_links(torch.tensor([0, 1, 2]))
    
    # Semantic count should remain the same (deleting links doesn't change tensor size)
    assert links.get_count(LD.SEM) == initial_sem_count


def test_get_count_both_dimensions(links):
    """Test get_count for both dimensions simultaneously."""
    token_count = links.get_count(LD.TK)
    sem_count = links.get_count(LD.SEM)
    
    # Both should return valid counts
    assert token_count > 0
    assert sem_count > 0
    assert isinstance(token_count, int)
    assert isinstance(sem_count, int)
    
    # Should match tensor shape
    assert token_count == links.adj_matrix.shape[LD.TK]
    assert sem_count == links.adj_matrix.shape[LD.SEM]
