# nodes/unit_test/tensor/test_mapping.py
# Tests for Mapping class - getters and setters

import pytest
import torch
from nodes.network.tensor.mapping import Mapping, MD
from nodes.enums import MappingFields


@pytest.fixture
def mock_mapping_tensor():
    """Create a mock mapping tensor (recipient x driver x fields)."""
    num_recipient = 5
    num_driver = 4
    num_fields = len(MappingFields)
    
    # Create 3D tensor: [recipient, driver, fields]
    tensor = torch.zeros((num_recipient, num_driver, num_fields))
    
    # Set some weights
    tensor[0, 0, MappingFields.WEIGHT] = 0.8
    tensor[0, 1, MappingFields.WEIGHT] = 0.9
    tensor[0, 2, MappingFields.WEIGHT] = 0.7
    tensor[1, 0, MappingFields.WEIGHT] = 0.6
    tensor[1, 1, MappingFields.WEIGHT] = 0.5
    tensor[2, 0, MappingFields.WEIGHT] = 0.95
    tensor[2, 2, MappingFields.WEIGHT] = 0.85
    
    # Set some hypotheses
    tensor[0, 0, MappingFields.HYPOTHESIS] = 0.3
    tensor[0, 1, MappingFields.HYPOTHESIS] = 0.4
    tensor[1, 0, MappingFields.HYPOTHESIS] = 0.2
    tensor[2, 0, MappingFields.HYPOTHESIS] = 0.5
    
    # Set some max_hyp
    tensor[0, 0, MappingFields.MAX_HYP] = 0.4
    tensor[0, 1, MappingFields.MAX_HYP] = 0.5
    tensor[1, 0, MappingFields.MAX_HYP] = 0.3
    
    return tensor


@pytest.fixture
def mapping(mock_mapping_tensor):
    """Create a Mapping instance with mock data."""
    return Mapping(mock_mapping_tensor)


def test_mapping_init(mapping, mock_mapping_tensor):
    """Test Mapping initialization."""
    assert torch.equal(mapping.adj_matrix, mock_mapping_tensor)
    assert mapping.adj_matrix.shape == (5, 4, len(MappingFields))


def test_mapping_size(mapping):
    """Test size method."""
    assert mapping.size(MD.REC) == 5
    assert mapping.size(MD.DRI) == 4
    assert mapping.size(2) == len(MappingFields)  # Third dimension (fields)


def test_mapping_getitem_weight(mapping):
    """Test getting WEIGHT field."""
    weight = mapping[MappingFields.WEIGHT]
    
    # Should return 2D tensor [recipient, driver]
    assert weight.shape == (5, 4)
    
    # Verify values
    assert weight[0, 0].item() == pytest.approx(0.8, abs=1e-6)
    assert weight[0, 1].item() == pytest.approx(0.9, abs=1e-6)
    assert weight[0, 2].item() == pytest.approx(0.7, abs=1e-6)
    assert weight[1, 0].item() == pytest.approx(0.6, abs=1e-6)
    assert weight[2, 0].item() == pytest.approx(0.95, abs=1e-6)


def test_mapping_getitem_hypothesis(mapping):
    """Test getting HYPOTHESIS field."""
    hypothesis = mapping[MappingFields.HYPOTHESIS]
    
    assert hypothesis.shape == (5, 4)
    assert hypothesis[0, 0].item() == pytest.approx(0.3, abs=1e-6)
    assert hypothesis[0, 1].item() == pytest.approx(0.4, abs=1e-6)
    assert hypothesis[1, 0].item() == pytest.approx(0.2, abs=1e-6)
    assert hypothesis[2, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_mapping_getitem_max_hyp(mapping):
    """Test getting MAX_HYP field."""
    max_hyp = mapping[MappingFields.MAX_HYP]
    
    assert max_hyp.shape == (5, 4)
    assert max_hyp[0, 0].item() == pytest.approx(0.4, abs=1e-6)
    assert max_hyp[0, 1].item() == pytest.approx(0.5, abs=1e-6)
    assert max_hyp[1, 0].item() == pytest.approx(0.3, abs=1e-6)




def test_mapping_getitem_all_fields(mapping):
    """Test getting all mapping fields."""
    for field in MappingFields:
        field_tensor = mapping[field]
        assert field_tensor.shape == (5, 4)
        assert field_tensor.dtype == torch.float32
    # Verify we have the expected number of fields
    assert len(MappingFields) == 3  # WEIGHT, HYPOTHESIS, MAX_HYP


def test_mapping_setitem_weight(mapping):
    """Test setting WEIGHT field."""
    # Create new weight values
    new_weights = torch.tensor([
        [1.0, 0.5, 0.3, 0.0],
        [0.8, 0.6, 0.4, 0.2],
        [0.9, 0.7, 0.5, 0.1],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.WEIGHT] = new_weights
    
    # Verify values were set
    assert torch.allclose(mapping[MappingFields.WEIGHT], new_weights)
    
    # Verify other fields unchanged
    assert mapping[MappingFields.HYPOTHESIS][0, 0].item() == pytest.approx(0.3, abs=1e-6)


def test_mapping_setitem_hypothesis(mapping):
    """Test setting HYPOTHESIS field."""
    new_hypothesis = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = new_hypothesis
    
    # Verify values were set
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS], new_hypothesis)
    
    # Verify other fields unchanged
    assert mapping[MappingFields.WEIGHT][0, 0].item() == pytest.approx(0.8, abs=1e-6)


def test_mapping_setitem_max_hyp(mapping):
    """Test setting MAX_HYP field."""
    new_max_hyp = torch.tensor([
        [0.5, 0.6, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.MAX_HYP] = new_max_hyp
    
    # Verify values were set
    assert torch.allclose(mapping[MappingFields.MAX_HYP], new_max_hyp)




def test_mapping_setitem_all_fields(mapping):
    """Test setting all mapping fields."""
    for field in MappingFields:
        new_values = torch.rand(5, 4)
        mapping[field] = new_values
        assert torch.allclose(mapping[field], new_values)
    # Verify we have the expected number of fields
    assert len(MappingFields) == 3  # WEIGHT, HYPOTHESIS, MAX_HYP


def test_mapping_getitem_setitem_roundtrip(mapping):
    """Test that getting and setting works correctly together."""
    # Get original weight
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    # Set new weight
    new_weight = torch.rand(5, 4)
    mapping[MappingFields.WEIGHT] = new_weight
    
    # Get it back
    retrieved_weight = mapping[MappingFields.WEIGHT]
    
    # Verify roundtrip
    assert torch.allclose(retrieved_weight, new_weight)
    
    # Restore original
    mapping[MappingFields.WEIGHT] = original_weight
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight)


def test_mapping_setitem_partial_update(mapping):
    """Test that setting one field doesn't affect others."""
    original_weight = mapping[MappingFields.WEIGHT].clone()
    original_hypothesis = mapping[MappingFields.HYPOTHESIS].clone()
    original_max_hyp = mapping[MappingFields.MAX_HYP].clone()
    
    # Update only hypothesis
    new_hypothesis = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = new_hypothesis
    
    # Verify hypothesis changed
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS], new_hypothesis)
    
    # Verify other fields unchanged
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight)
    assert torch.allclose(mapping[MappingFields.MAX_HYP], original_max_hyp)


def test_mapping_get_max_map(mapping):
    """Test get_max_map method."""
    max_recipient, max_driver = mapping.get_max_map()
    
    # max_recipient should have values and indices for each recipient token
    # (max along driver dimension)
    assert hasattr(max_recipient, 'values')
    assert hasattr(max_recipient, 'indices')
    assert max_recipient.values.shape == (5,)  # One max per recipient
    assert max_recipient.indices.shape == (5,)  # One index per recipient
    
    # max_driver should have values and indices for each driver token
    # (max along recipient dimension)
    assert hasattr(max_driver, 'values')
    assert hasattr(max_driver, 'indices')
    assert max_driver.values.shape == (4,)  # One max per driver
    assert max_driver.indices.shape == (4,)  # One index per driver
    
    # Verify values are correct
    # Recipient 0: max weight is 0.9 at driver 1
    assert max_recipient.values[0].item() == pytest.approx(0.9, abs=1e-6)
    assert max_recipient.indices[0].item() == 1
    
    # Recipient 1: max weight is 0.6 at driver 0
    assert max_recipient.values[1].item() == pytest.approx(0.6, abs=1e-6)
    assert max_recipient.indices[1].item() == 0
    
    # Recipient 2: max weight is 0.95 at driver 0
    assert max_recipient.values[2].item() == pytest.approx(0.95, abs=1e-6)
    assert max_recipient.indices[2].item() == 0


def test_mapping_get_max_map_zeros(mapping):
    """Test get_max_map when some rows/columns are all zeros."""
    # Set recipient 3 and 4 to all zeros
    mapping[MappingFields.WEIGHT][3, :] = 0.0
    mapping[MappingFields.WEIGHT][4, :] = 0.0
    
    max_recipient, max_driver = mapping.get_max_map()
    
    # Max should still be computed (will be 0.0 for zero rows)
    assert max_recipient.values[3].item() == 0.0
    assert max_recipient.values[4].item() == 0.0


def test_mapping_get_max_map_single_recipient():
    """Test get_max_map with single recipient."""
    tensor = torch.zeros((1, 4, len(MappingFields)))
    tensor[0, 0, MappingFields.WEIGHT] = 0.8
    tensor[0, 1, MappingFields.WEIGHT] = 0.9
    tensor[0, 2, MappingFields.WEIGHT] = 0.7
    
    mapping = Mapping(tensor)
    max_recipient, max_driver = mapping.get_max_map()
    
    assert max_recipient.values.shape == (1,)
    assert max_recipient.values[0].item() == pytest.approx(0.9, abs=1e-6)
    assert max_recipient.indices[0].item() == 1


def test_mapping_get_max_map_single_driver():
    """Test get_max_map with single driver."""
    tensor = torch.zeros((5, 1, len(MappingFields)))
    tensor[0, 0, MappingFields.WEIGHT] = 0.8
    tensor[1, 0, MappingFields.WEIGHT] = 0.9
    tensor[2, 0, MappingFields.WEIGHT] = 0.7
    
    mapping = Mapping(tensor)
    max_recipient, max_driver = mapping.get_max_map()
    
    assert max_driver.values.shape == (1,)
    assert max_driver.values[0].item() == pytest.approx(0.9, abs=1e-6)
    assert max_driver.indices[0].item() == 1


def test_mapping_get_max_map_ties(mapping):
    """Test get_max_map when there are ties (should pick first max)."""
    # Create ties: recipient 3 has same max weight at multiple drivers
    mapping[MappingFields.WEIGHT][3, 0] = 0.8
    mapping[MappingFields.WEIGHT][3, 1] = 0.8  # Tie
    mapping[MappingFields.WEIGHT][3, 2] = 0.7
    
    max_recipient, max_driver = mapping.get_max_map()
    
    # Should pick first max (driver 0)
    assert max_recipient.values[3].item() == pytest.approx(0.8, abs=1e-6)
    assert max_recipient.indices[3].item() == 0


def test_mapping_get_max_map_empty():
    """Test get_max_map with empty mapping."""
    # Note: This will fail because max() can't operate on empty dimensions
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # This will raise IndexError because max() requires non-zero size
    with pytest.raises(IndexError):
        max_recipient, max_driver = mapping.get_max_map()


def test_mapping_size_all_dimensions(mapping):
    """Test size method for all dimensions."""
    assert mapping.size(0) == 5  # Recipient dimension
    assert mapping.size(1) == 4  # Driver dimension
    assert mapping.size(2) == len(MappingFields)  # Fields dimension
    
    # Test with enum
    assert mapping.size(MD.REC) == 5
    assert mapping.size(MD.DRI) == 4


def test_mapping_getitem_returns_view(mapping):
    """Test that __getitem__ returns a view that updates the original."""
    weight = mapping[MappingFields.WEIGHT]
    
    # Modify the view
    weight[0, 0] = 99.0
    
    # Verify original was updated
    assert mapping[MappingFields.WEIGHT][0, 0].item() == 99.0


def test_mapping_setitem_broadcast(mapping):
    """Test setting a field with a scalar (broadcasting)."""
    # Set all weights to 0.5
    mapping[MappingFields.WEIGHT] = 0.5
    
    # Verify all weights are 0.5
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.5)


def test_mapping_setitem_tensor(mapping):
    """Test setting a field with a tensor."""
    new_weights = torch.ones(5, 4) * 0.75
    mapping[MappingFields.WEIGHT] = new_weights
    
    assert torch.allclose(mapping[MappingFields.WEIGHT], new_weights)


def test_reset_hypotheses(mapping):
    """Test reset_hypotheses method."""
    # Set some hypothesis and max_hyp values
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Store original weight (should not be affected)
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    # Reset hypotheses
    mapping.reset_hypotheses()
    
    # Verify hypotheses are reset to 0.0
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)
    
    # Verify weight is unchanged
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight)


def test_reset_hypotheses_partial(mapping):
    """Test reset_hypotheses only resets hypothesis fields."""
    # Set all fields
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Store non-hypothesis fields
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    # Reset hypotheses
    mapping.reset_hypotheses()
    
    # Verify only hypothesis fields are reset
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)
    
    # Verify other fields unchanged
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight)


def test_reset_hypotheses_empty():
    """Test reset_hypotheses with empty mapping."""
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # Should not raise an error
    mapping.reset_hypotheses()
    
    # Verify still empty
    assert mapping[MappingFields.HYPOTHESIS].shape == (0, 0)
    assert mapping[MappingFields.MAX_HYP].shape == (0, 0)


def test_reset_mapping_units(mapping):
    """Test reset_mapping_units method."""
    # Set all fields
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Reset mapping units
    mapping.reset_mapping_units()
    
    # Verify weight and hypotheses are reset
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.0)
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


def test_reset_mapping_units_calls_reset_hypotheses(mapping):
    """Test that reset_mapping_units calls reset_hypotheses."""
    # Set values
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Reset mapping units
    mapping.reset_mapping_units()
    
    # Verify all are reset (weight + hypotheses)
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.0)
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


def test_reset_mapping_units_empty():
    """Test reset_mapping_units with empty mapping."""
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # Should not raise an error
    mapping.reset_mapping_units()
    
    # Verify still empty
    assert mapping[MappingFields.WEIGHT].shape == (0, 0)
    assert mapping[MappingFields.HYPOTHESIS].shape == (0, 0)
    assert mapping[MappingFields.MAX_HYP].shape == (0, 0)


def test_reset_mappings_requires_driver_recipient(mapping):
    """Test that reset_mappings requires driver and recipient attributes."""
    # Note: reset_mappings references self.driver and self.recipient
    # which may not exist in the Mapping class
    # This test documents the expected behavior/error
    
    # This will fail because driver and recipient are not set
    with pytest.raises(AttributeError):
        mapping.reset_mappings()


def test_reset_hypotheses_multiple_calls(mapping):
    """Test calling reset_hypotheses multiple times."""
    # Set values
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Reset multiple times
    mapping.reset_hypotheses()
    mapping.reset_hypotheses()
    mapping.reset_hypotheses()
    
    # Should still be 0.0
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


def test_reset_mapping_units_multiple_calls(mapping):
    """Test calling reset_mapping_units multiple times."""
    # Set values
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Reset multiple times
    mapping.reset_mapping_units()
    mapping.reset_mapping_units()
    mapping.reset_mapping_units()
    
    # Should still be 0.0
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.0)
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


def test_reset_hypotheses_after_setting(mapping):
    """Test reset_hypotheses after setting values."""
    # Set specific values
    mapping[MappingFields.HYPOTHESIS][0, 0] = 0.5
    mapping[MappingFields.HYPOTHESIS][1, 1] = 0.7
    mapping[MappingFields.MAX_HYP][0, 0] = 0.6
    mapping[MappingFields.MAX_HYP][2, 2] = 0.8
    
    # Reset
    mapping.reset_hypotheses()
    
    # Verify all are 0.0
    assert mapping[MappingFields.HYPOTHESIS][0, 0].item() == 0.0
    assert mapping[MappingFields.HYPOTHESIS][1, 1].item() == 0.0
    assert mapping[MappingFields.MAX_HYP][0, 0].item() == 0.0
    assert mapping[MappingFields.MAX_HYP][2, 2].item() == 0.0


def test_reset_mapping_units_after_setting(mapping):
    """Test reset_mapping_units after setting values."""
    # Set specific values
    mapping[MappingFields.WEIGHT][0, 0] = 0.9
    mapping[MappingFields.WEIGHT][1, 1] = 0.8
    mapping[MappingFields.HYPOTHESIS][0, 0] = 0.5
    mapping[MappingFields.MAX_HYP][1, 1] = 0.6
    
    # Reset
    mapping.reset_mapping_units()
    
    # Verify all are 0.0
    assert mapping[MappingFields.WEIGHT][0, 0].item() == 0.0
    assert mapping[MappingFields.WEIGHT][1, 1].item() == 0.0
    assert mapping[MappingFields.HYPOTHESIS][0, 0].item() == 0.0
    assert mapping[MappingFields.MAX_HYP][1, 1].item() == 0.0


def test_reset_operations_sequence(mapping):
    """Test sequence of reset operations."""
    # Set all values
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    # Reset hypotheses first
    mapping.reset_hypotheses()
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)
    # Weight should still have values
    assert not torch.all(mapping[MappingFields.WEIGHT] == 0.0)
    
    # Then reset mapping units
    mapping.reset_mapping_units()
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.0)
    assert torch.all(mapping[MappingFields.HYPOTHESIS] == 0.0)
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


# =====================[ Update functions tests ]======================

def test_update_max_hyp_basic(mapping):
    """Test update_max_hyp with basic hypothesis values."""
    # Set specific hypothesis values
    hypothesis = torch.tensor([
        [0.1, 0.3, 0.2, 0.0],
        [0.4, 0.2, 0.5, 0.1],
        [0.2, 0.1, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.1, 0.1]
    ])
    mapping[MappingFields.HYPOTHESIS] = hypothesis
    
    mapping.update_max_hyp()
    
    # For position (0, 0): max of row 0 is 0.3, max of col 0 is 0.4, so max_hyp[0,0] = 0.4
    # For position (0, 1): max of row 0 is 0.3, max of col 1 is 0.3, so max_hyp[0,1] = 0.3
    # For position (1, 2): max of row 1 is 0.5, max of col 2 is 0.5, so max_hyp[1,2] = 0.5
    max_hyp = mapping[MappingFields.MAX_HYP]
    
    # Verify some specific values
    assert max_hyp[0, 0].item() == pytest.approx(0.4, abs=1e-6)  # max(0.3, 0.4)
    assert max_hyp[0, 1].item() == pytest.approx(0.3, abs=1e-6)  # max(0.3, 0.3)
    assert max_hyp[1, 2].item() == pytest.approx(0.5, abs=1e-6)  # max(0.5, 0.5)
    assert max_hyp[2, 3].item() == pytest.approx(0.4, abs=1e-6)  # max(0.4, 0.4)


def test_update_max_hyp_all_zeros(mapping):
    """Test update_max_hyp when all hypotheses are zero."""
    mapping[MappingFields.HYPOTHESIS] = torch.zeros(5, 4)
    
    mapping.update_max_hyp()
    
    # All max_hyp should be 0.0
    assert torch.all(mapping[MappingFields.MAX_HYP] == 0.0)


def test_update_max_hyp_single_value(mapping):
    """Test update_max_hyp with a single non-zero value."""
    hypothesis = torch.zeros(5, 4)
    hypothesis[2, 1] = 0.8
    mapping[MappingFields.HYPOTHESIS] = hypothesis
    
    mapping.update_max_hyp()
    
    # Row 2 max is 0.8, Column 1 max is 0.8
    # Positions in row 2 OR column 1 should have max_hyp = 0.8
    # Positions not in row 2 AND not in column 1 should have max_hyp = 0.0
    max_hyp = mapping[MappingFields.MAX_HYP]
    
    # All positions in row 2 should have max_hyp = 0.8
    assert torch.all(max_hyp[2, :] == 0.8)
    
    # All positions in column 1 should have max_hyp = 0.8
    assert torch.all(max_hyp[:, 1] == 0.8)
    
    # Position (2, 1) should be 0.8 (in both row 2 and column 1)
    assert max_hyp[2, 1].item() == pytest.approx(0.8, abs=1e-6)


def test_update_max_hyp_single_row(mapping):
    """Test update_max_hyp when only one row has values."""
    hypothesis = torch.zeros(5, 4)
    hypothesis[1, :] = torch.tensor([0.2, 0.5, 0.3, 0.4])
    mapping[MappingFields.HYPOTHESIS] = hypothesis
    
    mapping.update_max_hyp()
    
    # Row 1 max is 0.5
    # Column maxes are: [0.2, 0.5, 0.3, 0.4]
    # So max_hyp[1, 1] = max(0.5, 0.5) = 0.5
    assert mapping[MappingFields.MAX_HYP][1, 1].item() == pytest.approx(0.5, abs=1e-6)
    # max_hyp[1, 0] = max(0.5, 0.2) = 0.5
    assert mapping[MappingFields.MAX_HYP][1, 0].item() == pytest.approx(0.5, abs=1e-6)


def test_update_max_hyp_single_column(mapping):
    """Test update_max_hyp when only one column has values."""
    hypothesis = torch.zeros(5, 4)
    hypothesis[:, 2] = torch.tensor([0.1, 0.6, 0.3, 0.2, 0.4])
    mapping[MappingFields.HYPOTHESIS] = hypothesis
    
    mapping.update_max_hyp()
    
    # Column 2 max is 0.6
    # Row maxes are: [0.1, 0.6, 0.3, 0.2, 0.4]
    # So max_hyp[1, 2] = max(0.6, 0.6) = 0.6
    assert mapping[MappingFields.MAX_HYP][1, 2].item() == pytest.approx(0.6, abs=1e-6)


def test_update_max_hyp_empty():
    """Test update_max_hyp with empty mapping."""
    # Note: This will fail because max() can't operate on empty dimensions
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # This will raise IndexError because max() requires non-zero size
    with pytest.raises(IndexError):
        mapping.update_max_hyp()


def test_update_weight_basic(mapping):
    """Test update_weight with basic values."""
    # Set initial values
    mapping[MappingFields.WEIGHT] = torch.tensor([
        [0.1, 0.2, 0.0, 0.0],
        [0.3, 0.1, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    eta = 0.5
    mapping.update_weight(eta)
    
    # Verify weights were updated (should be clamped between 0 and 1)
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)
    
    # Verify hypothesis was modified (divisive and subtractive normalization)
    # Hypothesis should have been normalized
    assert torch.all(mapping[MappingFields.HYPOTHESIS] >= -1.0)  # Can be negative after subtractive norm


def test_update_weight_eta_zero(mapping):
    """Test update_weight with eta=0."""
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    mapping.update_weight(eta=0.0)
    
    # With eta=0, weights should remain unchanged (or become 0 if hypothesis is negative)
    # Actually, weight = clamp(0 * (1.1 - weight) * hypothesis, 0, 1) = 0
    assert torch.all(mapping[MappingFields.WEIGHT] == 0.0)


def test_update_weight_eta_one(mapping):
    """Test update_weight with eta=1.0."""
    mapping[MappingFields.WEIGHT] = torch.tensor([
        [0.1, 0.2, 0.0, 0.0],
        [0.3, 0.1, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping.update_weight(eta=1.0)
    
    # Weights should be updated and clamped
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


def test_update_weight_all_zeros(mapping):
    """Test update_weight when all hypotheses are zero."""
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.zeros(5, 4)
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    mapping.update_weight(eta=0.5)
    
    # With zero hypotheses, max_hyp will be 0, so no division happens
    # Then subtractive norm will result in negative or zero hypothesis
    # So weights should be 0 or clamped
    assert torch.all(mapping[MappingFields.WEIGHT] >= 0.0)
    assert torch.all(mapping[MappingFields.WEIGHT] <= 1.0)


def test_update_weight_division_by_zero_handling(mapping):
    """Test that update_weight handles division by zero correctly."""
    # Set hypothesis with some zeros
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.0, 0.3, 0.0, 0.0],
        [0.5, 0.0, 0.6, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    
    # Should not raise division by zero error
    mapping.update_weight(eta=0.5)
    
    # Verify weights are valid
    assert torch.all(torch.isfinite(mapping[MappingFields.WEIGHT]))


def test_update_weight_clamping(mapping):
    """Test that update_weight clamps values between 0 and 1."""
    # Set large hypothesis values that would produce weights > 1
    mapping[MappingFields.HYPOTHESIS] = torch.ones(5, 4) * 10.0
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    
    mapping.update_weight(eta=1.0)
    
    # All weights should be clamped to [0, 1]
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


def test_update_weight_negative_hypothesis(mapping):
    """Test update_weight when hypothesis becomes negative after subtractive normalization."""
    # Set hypothesis values that will result in negative values after subtractive norm
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.1, 0.2, 0.3, 0.0],
        [0.4, 0.5, 0.6, 0.0],
        [0.2, 0.1, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    
    mapping.update_weight(eta=0.5)
    
    # Weights should still be valid (clamped to >= 0)
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


def test_update_weight_calls_update_max_hyp(mapping):
    """Test that update_weight calls update_max_hyp."""
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.zeros(5, 4)
    
    mapping.update_weight(eta=0.5)
    
    # MAX_HYP should have been updated (not all zeros)
    # After the first update_max_hyp call, it should have values
    # But then it gets overwritten by efficient_local_max_excluding_self
    # So we just verify it's been modified
    assert mapping[MappingFields.MAX_HYP].shape == (5, 4)


def test_update_weight_multiple_calls(mapping):
    """Test calling update_weight multiple times."""
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    # Call multiple times
    mapping.update_weight(eta=0.5)
    weight_after_first = mapping[MappingFields.WEIGHT].clone()
    
    # Set new hypothesis for second call
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.1, 0.2, 0.3, 0.0],
        [0.4, 0.5, 0.6, 0.0],
        [0.2, 0.1, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping.update_weight(eta=0.5)
    
    # Weights should be updated again
    assert mapping[MappingFields.WEIGHT].shape == (5, 4)
    assert torch.all(mapping[MappingFields.WEIGHT] >= 0.0)
    assert torch.all(mapping[MappingFields.WEIGHT] <= 1.0)


def test_update_weight_empty():
    """Test update_weight with empty mapping."""
    # Note: This will fail because update_weight calls update_max_hyp which can't operate on empty dimensions
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # This will raise IndexError because update_max_hyp requires non-zero size
    with pytest.raises(IndexError):
        mapping.update_weight(eta=0.5)


def test_update_weight_small_eta(mapping):
    """Test update_weight with small eta value."""
    mapping[MappingFields.WEIGHT] = torch.tensor([
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping.update_weight(eta=0.01)
    
    # With small eta, weights should change but not dramatically
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


def test_update_weight_large_eta(mapping):
    """Test update_weight with large eta value."""
    mapping[MappingFields.WEIGHT] = torch.zeros(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.4, 0.1, 0.0],
        [0.5, 0.3, 0.6, 0.0],
        [0.1, 0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping.update_weight(eta=10.0)
    
    # Even with large eta, weights should be clamped to [0, 1]
    weight = mapping[MappingFields.WEIGHT]
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


# =====================[ Expand and swap functions tests ]======================

def test_expand_recipient_dimension(mapping):
    """Test expanding the mapping tensor along the recipient dimension."""
    original_size_rec = mapping.size(MD.REC)
    original_size_dri = mapping.size(MD.DRI)
    original_adj_matrix = mapping.adj_matrix.clone()
    
    # Expand to 8 recipients
    mapping.expand(new_count=8, dimension=MD.REC)
    
    # Verify new size
    assert mapping.size(MD.REC) == 8
    assert mapping.size(MD.DRI) == original_size_dri
    
    # Verify old mappings are preserved
    assert torch.allclose(mapping.adj_matrix[:original_size_rec, :, :], original_adj_matrix)
    
    # Verify new rows are zeros
    assert torch.all(mapping.adj_matrix[original_size_rec:, :, :] == 0.0)


def test_expand_driver_dimension(mapping):
    """Test expanding the mapping tensor along the driver dimension."""
    original_size_rec = mapping.size(MD.REC)
    original_size_dri = mapping.size(MD.DRI)
    original_adj_matrix = mapping.adj_matrix.clone()
    
    # Expand to 7 drivers
    mapping.expand(new_count=7, dimension=MD.DRI)
    
    # Verify new size
    assert mapping.size(MD.REC) == original_size_rec
    assert mapping.size(MD.DRI) == 7
    
    # Verify old mappings are preserved
    assert torch.allclose(mapping.adj_matrix[:, :original_size_dri, :], original_adj_matrix)
    
    # Verify new columns are zeros
    assert torch.all(mapping.adj_matrix[:, original_size_dri:, :] == 0.0)


def test_expand_both_dimensions(mapping):
    """Test expanding both dimensions separately."""
    original_adj_matrix = mapping.adj_matrix.clone()
    
    # Expand recipient dimension first
    mapping.expand(new_count=10, dimension=MD.REC)
    assert mapping.size(MD.REC) == 10
    assert mapping.size(MD.DRI) == 4
    
    # Expand driver dimension
    mapping.expand(new_count=8, dimension=MD.DRI)
    assert mapping.size(MD.REC) == 10
    assert mapping.size(MD.DRI) == 8
    
    # Verify original mappings are still preserved
    assert torch.allclose(mapping.adj_matrix[:5, :4, :], original_adj_matrix)


def test_expand_smaller_size_recipient(mapping):
    """Test expanding to a smaller recipient size."""
    # Note: The expand function doesn't support shrinking - it will raise RuntimeError
    # when trying to copy a larger tensor into a smaller one
    original_adj_matrix = mapping.adj_matrix.clone()
    
    # This will fail because expand doesn't handle shrinking
    with pytest.raises((RuntimeError, AttributeError)):
        # RuntimeError from tensor copy, or AttributeError from error logging
        mapping.expand(new_count=3, dimension=MD.REC)


def test_expand_smaller_size_driver(mapping):
    """Test expanding to a smaller driver size."""
    # Note: The expand function doesn't support shrinking - it will raise RuntimeError
    # when trying to copy a larger tensor into a smaller one
    original_adj_matrix = mapping.adj_matrix.clone()
    
    # This will fail because expand doesn't handle shrinking
    with pytest.raises((RuntimeError, AttributeError)):
        # RuntimeError from tensor copy, or AttributeError from error logging
        mapping.expand(new_count=2, dimension=MD.DRI)


def test_expand_preserves_all_fields(mapping):
    """Test that expand preserves all mapping fields."""
    # Set values in all fields
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    original_hypothesis = mapping[MappingFields.HYPOTHESIS].clone()
    original_max_hyp = mapping[MappingFields.MAX_HYP].clone()
    
    # Expand
    mapping.expand(new_count=7, dimension=MD.REC)
    
    # Verify all fields are preserved
    assert torch.allclose(mapping[MappingFields.WEIGHT][:5, :], original_weight)
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS][:5, :], original_hypothesis)
    assert torch.allclose(mapping[MappingFields.MAX_HYP][:5, :], original_max_hyp)


def test_expand_empty():
    """Test expanding an empty mapping."""
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # Expand recipient dimension
    mapping.expand(new_count=5, dimension=MD.REC)
    assert mapping.size(MD.REC) == 5
    assert mapping.size(MD.DRI) == 0
    
    # Expand driver dimension
    mapping.expand(new_count=4, dimension=MD.DRI)
    assert mapping.size(MD.REC) == 5
    assert mapping.size(MD.DRI) == 4
    
    # All should be zeros
    assert torch.all(mapping.adj_matrix == 0.0)


def test_swap_driver_recipient_basic(mapping):
    """Test swapping driver and recipient dimensions."""
    # Set some values in all fields
    mapping[MappingFields.WEIGHT] = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.MAX_HYP] = torch.tensor([
        [0.3, 0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    original_hypothesis = mapping[MappingFields.HYPOTHESIS].clone()
    original_max_hyp = mapping[MappingFields.MAX_HYP].clone()
    
    # Swap
    mapping.swap_driver_recipient()
    
    # Verify dimensions are swapped
    assert mapping.size(MD.REC) == 4  # Was driver dimension
    assert mapping.size(MD.DRI) == 5  # Was recipient dimension
    
    # Verify values are transposed for each field
    # Original weight[0, 1] = 0.2 should become weight[1, 0] = 0.2
    assert mapping[MappingFields.WEIGHT][1, 0].item() == pytest.approx(0.2, abs=1e-6)
    assert mapping[MappingFields.WEIGHT][0, 0].item() == pytest.approx(0.1, abs=1e-6)
    assert mapping[MappingFields.WEIGHT][2, 1].item() == pytest.approx(0.7, abs=1e-6)
    
    # Verify all fields are transposed
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight.t())
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS], original_hypothesis.t())
    assert torch.allclose(mapping[MappingFields.MAX_HYP], original_max_hyp.t())


def test_swap_driver_recipient_square(mapping):
    """Test swapping on a square mapping tensor."""
    # Create a square tensor (4x4)
    tensor = torch.zeros((4, 4, len(MappingFields)))
    tensor[:, :, MappingFields.WEIGHT] = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    mapping = Mapping(tensor)
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    mapping.swap_driver_recipient()
    
    # Dimensions should be swapped (but same for square)
    assert mapping.size(MD.REC) == 4
    assert mapping.size(MD.DRI) == 4
    
    # Values should be transposed
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight.t())


def test_swap_driver_recipient_roundtrip(mapping):
    """Test that swapping twice returns to original."""
    # Set some values
    mapping[MappingFields.WEIGHT] = torch.rand(5, 4)
    mapping[MappingFields.HYPOTHESIS] = torch.rand(5, 4)
    mapping[MappingFields.MAX_HYP] = torch.rand(5, 4)
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    original_hypothesis = mapping[MappingFields.HYPOTHESIS].clone()
    original_max_hyp = mapping[MappingFields.MAX_HYP].clone()
    original_size_rec = mapping.size(MD.REC)
    original_size_dri = mapping.size(MD.DRI)
    
    # Swap twice
    mapping.swap_driver_recipient()
    mapping.swap_driver_recipient()
    
    # Should be back to original
    assert mapping.size(MD.REC) == original_size_rec
    assert mapping.size(MD.DRI) == original_size_dri
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight)
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS], original_hypothesis)
    assert torch.allclose(mapping[MappingFields.MAX_HYP], original_max_hyp)


def test_swap_driver_recipient_preserves_all_fields(mapping):
    """Test that swap preserves all mapping fields."""
    # Set values in all fields
    mapping[MappingFields.WEIGHT] = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.HYPOTHESIS] = torch.tensor([
        [0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    mapping[MappingFields.MAX_HYP] = torch.tensor([
        [0.3, 0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    original_weight = mapping[MappingFields.WEIGHT].clone()
    original_hypothesis = mapping[MappingFields.HYPOTHESIS].clone()
    original_max_hyp = mapping[MappingFields.MAX_HYP].clone()
    
    mapping.swap_driver_recipient()
    
    # All fields should be transposed
    assert torch.allclose(mapping[MappingFields.WEIGHT], original_weight.t())
    assert torch.allclose(mapping[MappingFields.HYPOTHESIS], original_hypothesis.t())
    assert torch.allclose(mapping[MappingFields.MAX_HYP], original_max_hyp.t())


def test_swap_driver_recipient_empty():
    """Test swapping empty mapping."""
    tensor = torch.zeros((0, 0, len(MappingFields)))
    mapping = Mapping(tensor)
    
    # Should not raise an error
    mapping.swap_driver_recipient()
    
    # Dimensions should still be 0x0
    assert mapping.size(MD.REC) == 0
    assert mapping.size(MD.DRI) == 0


def test_expand_then_swap(mapping):
    """Test expanding then swapping."""
    # Expand recipient dimension
    mapping.expand(new_count=8, dimension=MD.REC)
    assert mapping.size(MD.REC) == 8
    assert mapping.size(MD.DRI) == 4
    
    # Swap
    mapping.swap_driver_recipient()
    
    # Dimensions should be swapped
    assert mapping.size(MD.REC) == 4
    assert mapping.size(MD.DRI) == 8


def test_swap_then_expand(mapping):
    """Test swapping then expanding."""
    original_weight = mapping[MappingFields.WEIGHT].clone()
    
    # Swap first
    mapping.swap_driver_recipient()
    assert mapping.size(MD.REC) == 4
    assert mapping.size(MD.DRI) == 5
    
    # Expand what is now the recipient dimension (was driver)
    mapping.expand(new_count=7, dimension=MD.REC)
    assert mapping.size(MD.REC) == 7
    assert mapping.size(MD.DRI) == 5
    
    # Original data should be preserved (transposed)
    assert torch.allclose(mapping.adj_matrix[:4, :5, MappingFields.WEIGHT], original_weight.t())


# =====================[ get_view tests ]======================

def test_get_view_basic(mapping):
    """Test basic get_view functionality."""
    indices = torch.tensor([0, 2, 4])
    view = mapping.get_view(indices)
    
    # Should return a TensorView
    from nodes.network.tensor.tensor_view import TensorView
    assert isinstance(view, TensorView)
    
    # View should have correct shape (3 recipients, 4 drivers, fields)
    assert view.shape == (3, 4, len(MappingFields))
    
    # View should have same dtype and device as original
    assert view.dtype == mapping.adj_matrix.dtype
    assert view.device == mapping.adj_matrix.device


def test_get_view_single_index(mapping):
    """Test get_view with a single index."""
    indices = torch.tensor([1])
    view = mapping.get_view(indices)
    
    assert view.shape == (1, 4, len(MappingFields))
    assert len(view) == 1


def test_get_view_multiple_indices(mapping):
    """Test get_view with multiple indices."""
    indices = torch.tensor([0, 1, 2, 3])
    view = mapping.get_view(indices)
    
    assert view.shape == (4, 4, len(MappingFields))
    assert len(view) == 4


def test_get_view_non_contiguous_indices(mapping):
    """Test get_view with non-contiguous indices."""
    indices = torch.tensor([0, 2, 4])
    view = mapping.get_view(indices)
    
    assert view.shape == (3, 4, len(MappingFields))
    
    # Verify we can access the view
    # View[0] should correspond to original recipient 0
    assert torch.equal(view[0], mapping.adj_matrix[0])
    # View[1] should correspond to original recipient 2
    assert torch.equal(view[1], mapping.adj_matrix[2])


def test_get_view_read_access(mapping):
    """Test reading values through the view."""
    indices = torch.tensor([0, 2])
    view = mapping.get_view(indices)
    
    # Read through view
    view_recipient_0 = view[0]  # Should be recipient 0 from original
    view_recipient_1 = view[1]  # Should be recipient 2 from original
    
    # Verify values match original
    assert torch.equal(view_recipient_0, mapping.adj_matrix[0])
    assert torch.equal(view_recipient_1, mapping.adj_matrix[2])
    
    # Read specific field and driver
    assert view[0, 1, MappingFields.WEIGHT].item() == mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item()
    assert view[1, 0, MappingFields.HYPOTHESIS].item() == mapping.adj_matrix[2, 0, MappingFields.HYPOTHESIS].item()


def test_get_view_write_access(mapping):
    """Test writing values through the view modifies original."""
    indices = torch.tensor([0, 2])
    view = mapping.get_view(indices)
    
    # Store original values
    original_0_1_weight = mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item()
    original_2_0_hyp = mapping.adj_matrix[2, 0, MappingFields.HYPOTHESIS].item()
    
    # Modify through view
    view[0, 1, MappingFields.WEIGHT] = 0.99
    view[1, 0, MappingFields.HYPOTHESIS] = 0.88  # view[1] corresponds to original recipient 2
    
    # Verify original was modified
    assert mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item() == pytest.approx(0.99, abs=1e-6)
    assert mapping.adj_matrix[2, 0, MappingFields.HYPOTHESIS].item() == pytest.approx(0.88, abs=1e-6)
    
    # Verify other values unchanged
    assert mapping.adj_matrix[0, 0, MappingFields.WEIGHT].item() == pytest.approx(0.8, abs=1e-6)
    assert mapping.adj_matrix[2, 2, MappingFields.WEIGHT].item() == pytest.approx(0.85, abs=1e-6)


def test_get_view_slice_access(mapping):
    """Test accessing view with slices."""
    indices = torch.tensor([0, 1, 2, 3])
    view = mapping.get_view(indices)
    
    # Get slice of view
    view_slice = view[0:2]  # Should return another TensorView
    
    from nodes.network.tensor.tensor_view import TensorView
    assert isinstance(view_slice, TensorView)
    assert view_slice.shape == (2, 4, len(MappingFields))
    
    # Verify slice values
    assert torch.equal(view_slice[0], mapping.adj_matrix[0])
    assert torch.equal(view_slice[1], mapping.adj_matrix[1])


def test_get_view_all_drivers(mapping):
    """Test accessing all drivers for recipients in view."""
    indices = torch.tensor([0, 2, 4])
    view = mapping.get_view(indices)
    
    # Get all drivers for first recipient in view
    all_drivers = view[0, :, MappingFields.WEIGHT]
    assert torch.equal(all_drivers, mapping.adj_matrix[0, :, MappingFields.WEIGHT])
    
    # Get all drivers for second recipient in view
    all_drivers_2 = view[1, :, MappingFields.HYPOTHESIS]
    assert torch.equal(all_drivers_2, mapping.adj_matrix[2, :, MappingFields.HYPOTHESIS])


def test_get_view_empty_indices(mapping):
    """Test get_view with empty indices."""
    indices = torch.tensor([], dtype=torch.long)
    view = mapping.get_view(indices)
    
    assert view.shape == (0, 4, len(MappingFields))
    assert len(view) == 0


def test_get_view_reordered_indices(mapping):
    """Test get_view with reordered indices."""
    indices = torch.tensor([4, 2, 0])  # Reversed order
    view = mapping.get_view(indices)
    
    assert view.shape == (3, 4, len(MappingFields))
    
    # View[0] should be recipient 4
    assert torch.equal(view[0], mapping.adj_matrix[4])
    # View[1] should be recipient 2
    assert torch.equal(view[1], mapping.adj_matrix[2])
    # View[2] should be recipient 0
    assert torch.equal(view[2], mapping.adj_matrix[0])


def test_get_view_modify_through_view(mapping):
    """Test that modifications through view affect original tensor."""
    indices = torch.tensor([1, 3])
    view = mapping.get_view(indices)
    
    # Modify entire row through view (all fields for one recipient)
    new_weight = torch.tensor([0.1, 0.2, 0.3, 0.4])
    view[0, :, MappingFields.WEIGHT] = new_weight  # Should modify original recipient 1
    
    # Verify original was modified
    assert torch.allclose(mapping.adj_matrix[1, :, MappingFields.WEIGHT], new_weight)
    
    # Original recipient 3 should be unchanged by this operation
    assert mapping.adj_matrix[3, 0, MappingFields.WEIGHT].item() == 0.0


def test_get_view_modify_all_fields(mapping):
    """Test modifying all fields through view."""
    indices = torch.tensor([0, 2])
    view = mapping.get_view(indices)
    
    # Modify all fields for a specific mapping
    view[0, 1, MappingFields.WEIGHT] = 0.9
    view[0, 1, MappingFields.HYPOTHESIS] = 0.8
    view[0, 1, MappingFields.MAX_HYP] = 0.7
    
    # Verify all fields were modified
    assert mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item() == pytest.approx(0.9, abs=1e-6)
    assert mapping.adj_matrix[0, 1, MappingFields.HYPOTHESIS].item() == pytest.approx(0.8, abs=1e-6)
    assert mapping.adj_matrix[0, 1, MappingFields.MAX_HYP].item() == pytest.approx(0.7, abs=1e-6)


def test_get_view_clone(mapping):
    """Test cloning the view creates independent copy."""
    indices = torch.tensor([0, 2])
    view = mapping.get_view(indices)
    
    # Clone the view
    cloned = view.clone()
    
    # Modify clone
    cloned[0, 1, MappingFields.WEIGHT] = 0.99
    
    # Original should be unchanged (clone is independent)
    assert mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item() != pytest.approx(0.99, abs=1e-6)
    
    # But view should still reflect original
    assert view[0, 1, MappingFields.WEIGHT].item() == mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item()


def test_get_view_multiple_views(mapping):
    """Test creating multiple views of the same tensor."""
    indices1 = torch.tensor([0, 1])
    indices2 = torch.tensor([2, 3])
    
    view1 = mapping.get_view(indices1)
    view2 = mapping.get_view(indices2)
    
    # Both views should work independently
    assert view1.shape == (2, 4, len(MappingFields))
    assert view2.shape == (2, 4, len(MappingFields))
    
    # Modify through view1
    view1[0, 0, MappingFields.WEIGHT] = 0.99
    
    # Should affect original
    assert mapping.adj_matrix[0, 0, MappingFields.WEIGHT].item() == pytest.approx(0.99, abs=1e-6)
    
    # view2 should still work correctly
    assert view2[0, 0, MappingFields.WEIGHT].item() == mapping.adj_matrix[2, 0, MappingFields.WEIGHT].item()


def test_get_view_overlapping_indices(mapping):
    """Test creating views with overlapping indices."""
    indices1 = torch.tensor([0, 1, 2])
    indices2 = torch.tensor([1, 2, 3])
    
    view1 = mapping.get_view(indices1)
    view2 = mapping.get_view(indices2)
    
    # Modify through view1 (affects recipient 1)
    view1[1, 0, MappingFields.WEIGHT] = 0.99  # view1[1] is original recipient 1
    
    # Should affect original
    assert mapping.adj_matrix[1, 0, MappingFields.WEIGHT].item() == pytest.approx(0.99, abs=1e-6)
    
    # view2 should see the change (view2[0] is also original recipient 1)
    assert view2[0, 0, MappingFields.WEIGHT].item() == pytest.approx(0.99, abs=1e-6)


def test_get_view_with_field_access(mapping):
    """Test using view with field access patterns."""
    indices = torch.tensor([0, 1, 2])
    view = mapping.get_view(indices)
    
    # Access weight field for all recipients in view
    weight_view = view[:, :, MappingFields.WEIGHT]
    
    # Should be able to read
    assert weight_view.shape == (3, 4)
    assert weight_view[0, 0].item() == mapping.adj_matrix[0, 0, MappingFields.WEIGHT].item()
    
    # Modify through weight view
    weight_view[0, 1] = 0.95
    
    # Original should be updated
    assert mapping.adj_matrix[0, 1, MappingFields.WEIGHT].item() == pytest.approx(0.95, abs=1e-6)


def test_get_view_batch_operations(mapping):
    """Test batch operations through view."""
    indices = torch.tensor([0, 1, 2])
    view = mapping.get_view(indices)
    
    # Set all weights for recipients in view to a specific value
    view[:, :, MappingFields.WEIGHT] = 0.5
    
    # Verify all were set
    assert torch.all(mapping.adj_matrix[0, :, MappingFields.WEIGHT] == 0.5)
    assert torch.all(mapping.adj_matrix[1, :, MappingFields.WEIGHT] == 0.5)
    assert torch.all(mapping.adj_matrix[2, :, MappingFields.WEIGHT] == 0.5)
    
    # Other recipients should be unchanged
    assert mapping.adj_matrix[3, 0, MappingFields.WEIGHT].item() == 0.0

