# nodes/unit_test/operations/test_mapping_ops.py
# Tests for MappingOperations class

import sys
from pathlib import Path

# Add DORA_tensorised to Python path so 'nodes' module can be imported
dora_tensorised_dir = Path(__file__).parent.parent.parent.parent
if str(dora_tensorised_dir) not in sys.path:
    sys.path.insert(0, str(dora_tensorised_dir))

import pytest
import torch
from unittest.mock import Mock, patch
from nodes.network.network import Network
from nodes.network.tokens.tokens import Tokens
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.tokens.connections.connections import Connections_Tensor
from nodes.network.tokens.connections.mapping import Mapping
from nodes.network.tokens.connections.links import Links
from nodes.network.sets_new.semantics import Semantics
from nodes.network.network_params import Params, default_params
from nodes.enums import Set, TF, SF, MappingFields


@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """Create minimal Token_Tensor for testing."""
    num_tokens = 10
    num_features = len(TF)
    tokens = torch.zeros((num_tokens, num_features))
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    names = {}
    return Token_Tensor(tokens, Connections_Tensor(connections), names)


@pytest.fixture
def minimal_connections():
    """Create minimal Connections_Tensor for testing."""
    num_tokens = 10
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    return Connections_Tensor(connections)


@pytest.fixture
def minimal_links(minimal_token_tensor):
    """Create minimal Links object for testing."""
    num_tokens = minimal_token_tensor.get_count()
    num_semantics = 5
    links = torch.zeros((num_tokens, num_semantics))
    return Links(links)


@pytest.fixture
def minimal_mapping():
    """Create minimal Mapping object for testing."""
    num_recipient = 5
    num_driver = 4
    num_fields = len(MappingFields)
    adj_matrix = torch.zeros((num_recipient, num_driver, num_fields))
    return Mapping(adj_matrix)


@pytest.fixture
def minimal_semantics():
    """Create minimal Semantics object for testing."""
    num_semantics = 5
    num_features = len(SF)
    nodes = torch.zeros((num_semantics, num_features))
    connections = torch.zeros((num_semantics, num_semantics))
    IDs = {i: i for i in range(num_semantics)}
    names = {}
    return Semantics(nodes, connections, IDs, names)


@pytest.fixture
def minimal_tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping):
    """Create minimal Tokens object for testing."""
    return Tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping)


@pytest.fixture
def network(minimal_tokens, minimal_semantics, minimal_mapping, minimal_links, minimal_params):
    """Create minimal Network object for testing."""
    return Network(minimal_tokens, minimal_semantics, minimal_mapping, minimal_links, minimal_params)


# =====================[ Wrapper Function Tests ]======================
# These functions are just wrappers, so we only test they run without error

def test_reset_mapping_units_runs_without_error(network):
    """Test that reset_mapping_units runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    network.mapping_ops.reset_mapping_units()
    # No assertion needed - just verifying it runs


def test_reset_mappings_runs_without_error(network):
    """Test that reset_mappings runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    # Note: This may fail if mapping.py has bugs (e.g., using token_ops instead of token_op)
    # but we're testing the wrapper, so we expect it to work if the underlying code is correct
    network.mapping_ops.reset_mappings()
    # No assertion needed - just verifying it runs


def test_update_mapping_hyps_runs_without_error(network):
    """Test that update_mapping_hyps runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    # Note: update_hypotheses() requires driver_mask and recipient_mask but raises NotImplementedError
    # So this will fail, but we're testing the wrapper
    network.mapping_ops.update_mapping_hyps()
    # No assertion needed - just verifying it runs


def test_reset_mapping_hyps_runs_without_error(network):
    """Test that reset_mapping_hyps runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    network.mapping_ops.reset_mapping_hyps()
    # No assertion needed - just verifying it runs


def test_update_mapping_connections_runs_without_error(network):
    """Test that update_mapping_connections runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    # Note: The wrapper calls update_connections() but the actual method is update_weight()
    # So this will fail, but we're testing the wrapper
    network.mapping_ops.update_mapping_connections()
    # No assertion needed - just verifying it runs


# =====================[ get_max_maps Tests ]======================
# This function has its own logic, so we need more in-depth tests

def test_get_max_maps_default_sets_both_driver_and_recipient(network):
    """Test that get_max_maps with default parameter sets max_maps for both driver and recipient."""
    # Create mock return values from get_max_map()
    # torch.max returns a named tuple-like object with .values and .indices
    num_recipient = network.mappings.size(0)
    num_driver = network.mappings.size(1)
    
    # Get actual set sizes (may differ from mapping dimensions)
    recipient_set_size = network.sets[Set.RECIPIENT].lcl.shape[0]
    driver_set_size = network.sets[Set.DRIVER].lcl.shape[0]
    
    # Create mock objects that mimic torch.return_types.max structure
    # Use sizes matching the actual set sizes (not mapping dimensions)
    # Note: indices from torch.max() are Long, but need to be converted to Float for tensor storage
    max_recipient = Mock()
    max_recipient.values = torch.rand(recipient_set_size)  # Match actual set size
    max_recipient.indices = torch.randint(0, num_driver, (recipient_set_size,)).float()  # Convert to float
    
    max_driver = Mock()
    max_driver.values = torch.rand(driver_set_size)  # Match actual set size
    max_driver.indices = torch.randint(0, num_recipient, (driver_set_size,)).float()  # Convert to float
    
    # Mock the get_max_map method
    with patch.object(network.mappings, 'get_max_map', return_value=(max_recipient, max_driver)):
        network.mapping_ops.get_max_maps()
    
    # Verify that set_max_maps and set_max_map_units were called for both sets
    # Check recipient - verify values were set correctly
    recipient_max_maps = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP]
    recipient_max_map_units = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT]
    assert torch.allclose(recipient_max_maps, max_recipient.values)
    assert torch.equal(recipient_max_map_units.long(), max_recipient.indices.long())
    
    # Check driver - verify values were set correctly
    driver_max_maps = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP]
    driver_max_map_units = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT]
    assert torch.allclose(driver_max_maps, max_driver.values)
    assert torch.equal(driver_max_map_units.long(), max_driver.indices.long())


def test_get_max_maps_only_recipient(network):
    """Test that get_max_maps with only RECIPIENT set only updates recipient."""
    num_recipient = network.mappings.size(0)
    num_driver = network.mappings.size(1)
    
    # Get actual set sizes
    recipient_set_size = network.sets[Set.RECIPIENT].lcl.shape[0]
    driver_set_size = network.sets[Set.DRIVER].lcl.shape[0]
    
    max_recipient = Mock()
    max_recipient.values = torch.rand(recipient_set_size)
    max_recipient.indices = torch.randint(0, num_driver, (recipient_set_size,)).float()  # Convert to float
    
    max_driver = Mock()
    max_driver.values = torch.rand(driver_set_size)
    max_driver.indices = torch.randint(0, num_recipient, (driver_set_size,)).float()  # Convert to float
    
    # Store original driver values
    original_driver_max_maps = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP].clone()
    original_driver_max_map_units = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT].clone()
    
    with patch.object(network.mappings, 'get_max_map', return_value=(max_recipient, max_driver)):
        network.mapping_ops.get_max_maps(set=[Set.RECIPIENT])
    
    # Verify recipient was updated
    recipient_max_maps = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP]
    recipient_max_map_units = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT]
    assert torch.allclose(recipient_max_maps, max_recipient.values)
    assert torch.equal(recipient_max_map_units.long(), max_recipient.indices.long())
    
    # Verify driver was NOT updated
    assert torch.allclose(network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP], original_driver_max_maps)
    assert torch.equal(network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT].long(), original_driver_max_map_units.long())


def test_get_max_maps_only_driver(network):
    """Test that get_max_maps with only DRIVER set only updates driver."""
    num_recipient = network.mappings.size(0)
    num_driver = network.mappings.size(1)
    
    # Get actual set sizes
    recipient_set_size = network.sets[Set.RECIPIENT].lcl.shape[0]
    driver_set_size = network.sets[Set.DRIVER].lcl.shape[0]
    
    max_recipient = Mock()
    max_recipient.values = torch.rand(recipient_set_size)
    max_recipient.indices = torch.randint(0, num_driver, (recipient_set_size,)).float()  # Convert to float
    
    max_driver = Mock()
    max_driver.values = torch.rand(driver_set_size)
    max_driver.indices = torch.randint(0, num_recipient, (driver_set_size,)).float()  # Convert to float
    
    # Store original recipient values
    original_recipient_max_maps = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP].clone()
    original_recipient_max_map_units = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT].clone()
    
    with patch.object(network.mappings, 'get_max_map', return_value=(max_recipient, max_driver)):
        network.mapping_ops.get_max_maps(set=[Set.DRIVER])
    
    # Verify driver was updated
    driver_max_maps = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP]
    driver_max_map_units = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT]
    assert torch.allclose(driver_max_maps, max_driver.values)
    assert torch.equal(driver_max_map_units.long(), max_driver.indices.long())
    
    # Verify recipient was NOT updated
    assert torch.allclose(network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP], original_recipient_max_maps)
    assert torch.equal(network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT].long(), original_recipient_max_map_units.long())


def test_get_max_maps_empty_set_list(network):
    """Test that get_max_maps with empty set list doesn't update anything."""
    num_recipient = network.mappings.size(0)
    num_driver = network.mappings.size(1)
    
    # Get actual set sizes
    recipient_set_size = network.sets[Set.RECIPIENT].lcl.shape[0]
    driver_set_size = network.sets[Set.DRIVER].lcl.shape[0]
    
    max_recipient = Mock()
    max_recipient.values = torch.rand(recipient_set_size)
    max_recipient.indices = torch.randint(0, num_driver, (recipient_set_size,)).float()  # Convert to float
    
    max_driver = Mock()
    max_driver.values = torch.rand(driver_set_size)
    max_driver.indices = torch.randint(0, num_recipient, (driver_set_size,)).float()  # Convert to float
    
    # Store original values
    original_recipient_max_maps = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP].clone()
    original_recipient_max_map_units = network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT].clone()
    original_driver_max_maps = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP].clone()
    original_driver_max_map_units = network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT].clone()
    
    with patch.object(network.mappings, 'get_max_map', return_value=(max_recipient, max_driver)):
        network.mapping_ops.get_max_maps(set=[])
    
    # Verify nothing was updated
    assert torch.allclose(network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP], original_recipient_max_maps)
    assert torch.equal(network.sets[Set.RECIPIENT].lcl[:, TF.MAX_MAP_UNIT].long(), original_recipient_max_map_units.long())
    assert torch.allclose(network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP], original_driver_max_maps)
    assert torch.equal(network.sets[Set.DRIVER].lcl[:, TF.MAX_MAP_UNIT].long(), original_driver_max_map_units.long())


# =====================[ get_max_map_unit Tests ]======================
# This function has its own logic, so we need more in-depth tests

def test_get_max_map_unit_returns_correct_value(network):
    """Test that get_max_map_unit returns the correct integer value."""
    idx = 5
    expected_value = 7
    
    # Mock get_feature to return a tensor with the expected value
    mock_tensor = torch.tensor([expected_value], dtype=torch.float32)
    
    with patch.object(network.token_tensor, 'get_feature', return_value=mock_tensor):
        result = network.mapping_ops.get_max_map_unit(idx)
    
    assert result == expected_value
    assert isinstance(result, int)


def test_get_max_map_unit_calls_get_feature_with_correct_parameters(network):
    """Test that get_max_map_unit calls get_feature with correct index and feature."""
    idx = 3
    mock_tensor = torch.tensor([5.0], dtype=torch.float32)
    
    with patch.object(network.token_tensor, 'get_feature', return_value=mock_tensor) as mock_get_feature:
        network.mapping_ops.get_max_map_unit(idx)
    
    # Verify get_feature was called with correct parameters
    mock_get_feature.assert_called_once()
    call_args = mock_get_feature.call_args
    assert call_args[0][0] == idx  # First positional argument should be idx
    assert call_args[0][1] == TF.MAX_MAP_UNIT  # Second positional argument should be TF.MAX_MAP_UNIT


def test_get_max_map_unit_converts_tensor_to_int(network):
    """Test that get_max_map_unit correctly converts tensor to int."""
    idx = 2
    # Test with different tensor types that should all convert to int
    test_values = [0, 1, 42, 100]
    
    for expected_value in test_values:
        mock_tensor = torch.tensor([expected_value], dtype=torch.float32)
        
        with patch.object(network.token_tensor, 'get_feature', return_value=mock_tensor):
            result = network.mapping_ops.get_max_map_unit(idx)
        
        assert result == expected_value
        assert isinstance(result, int)

