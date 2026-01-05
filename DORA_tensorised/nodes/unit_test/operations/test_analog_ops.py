# nodes/unit_test/operations/test_analog_ops.py
# Tests for AnalogOperations class

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
from nodes.enums import Set, TF, SF, MappingFields, Type


@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """Create minimal Token_Tensor for testing."""
    num_tokens = 20
    num_features = len(TF)
    tokens = torch.zeros((num_tokens, num_features))
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    names = {}
    return Token_Tensor(tokens, Connections_Tensor(connections), names)


@pytest.fixture
def minimal_connections():
    """Create minimal Connections_Tensor for testing."""
    num_tokens = 20
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

def test_copy_runs_without_error(network):
    """Test that copy runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    # We need to set up a valid analog first
    analog_num = 1
    to_set = Set.DRIVER
    # Mock the underlying copy_analog to avoid actual implementation details
    with patch.object(network.tokens.analog_ops, 'copy_analog', return_value=2):
        result = network.analog_ops.copy(analog_num, to_set)
        assert result == 2


def test_delete_runs_without_error(network):
    """Test that delete runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    analog_num = 1
    # Mock the underlying delete_analog to avoid actual implementation details
    with patch.object(network.tokens.analog_ops, 'delete_analog'):
        network.analog_ops.delete(analog_num)
        # Verify it was called
        network.tokens.analog_ops.delete_analog.assert_called_once_with(analog_num)


def test_move_runs_without_error(network):
    """Test that move runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    analog_num = 1
    to_set = Set.RECIPIENT
    # Mock the underlying move_analog to avoid actual implementation details
    with patch.object(network.tokens.analog_ops, 'move_analog'):
        network.analog_ops.move(analog_num, to_set)
        # Verify it was called
        network.tokens.analog_ops.move_analog.assert_called_once_with(analog_num, to_set)


def test_check_for_copy_runs_without_error(network):
    """Test that check_for_copy runs without error."""
    # This is a wrapper function, so we just verify it doesn't raise an exception
    # Mock the underlying get_analogs_where_not
    mock_analogs = torch.tensor([1, 2, 3])
    with patch.object(network.sets[Set.MEMORY].analog_ops, 'get_analogs_where_not', return_value=mock_analogs):
        result = network.analog_ops.check_for_copy()
        assert torch.equal(result, mock_analogs)


def test_clear_set_runs_without_error(network):
    """Test that clear_set runs without error."""
    # This is a wrapper function that calls set_analog_features
    analog_num = 1
    # Mock get_analog_indices and set_features
    mock_indices = torch.tensor([5, 6, 7])
    with patch.object(network.analog_ops, 'get_analog_indices', return_value=mock_indices):
        with patch.object(network.tokens.token_tensor, 'set_features') as mock_set_features:
            network.analog_ops.clear_set(analog_num)
            # Verify set_features was called with correct parameters
            mock_set_features.assert_called_once_with(mock_indices, TF.SET, Set.MEMORY)


def test_get_analog_indices_runs_without_error(network):
    """Test that get_analog_indices runs without error."""
    # This is a wrapper function
    analog_num = 1
    mock_indices = torch.tensor([2, 3, 4])
    with patch.object(network.tokens.analog_ops, 'get_analog_indices', return_value=mock_indices):
        result = network.analog_ops.get_analog_indices(analog_num)
        assert torch.equal(result, mock_indices)


def test_set_analog_features_runs_without_error(network):
    """Test that set_analog_features runs without error."""
    # This function has its own logic (calls get_analog_indices then set_features)
    analog_num = 1
    feature = TF.ACT
    value = 0.5
    mock_indices = torch.tensor([1, 2, 3])
    
    with patch.object(network.analog_ops, 'get_analog_indices', return_value=mock_indices):
        with patch.object(network.tokens.token_tensor, 'set_features') as mock_set_features:
            network.analog_ops.set_analog_features(analog_num, feature, value)
            # Verify get_analog_indices was called
            network.analog_ops.get_analog_indices.assert_called_once_with(analog_num)
            # Verify set_features was called with correct parameters
            mock_set_features.assert_called_once_with(mock_indices, feature, value)


# =====================[ NotImplementedError Tests ]======================

def test_get_analog_raises_not_implemented(network):
    """Test that get_analog raises NotImplementedError."""
    from nodes.network.single_nodes import Ref_Analog
    analog = Ref_Analog(1, Set.DRIVER)
    with pytest.raises(NotImplementedError, match="get_analog is not implemented"):
        network.analog_ops.get_analog(analog)


def test_add_analog_raises_not_implemented(network):
    """Test that add_analog raises NotImplementedError."""
    from nodes.network.single_nodes import Analog
    analog = Analog(1, Set.DRIVER)
    with pytest.raises(NotImplementedError, match="add_analog is not implemented"):
        network.analog_ops.add_analog(analog)


# =====================[ make_AM_copy Tests ]======================
# This function has its own logic, so we need more in-depth tests

def test_make_AM_copy_with_single_analog(network):
    """Test that make_AM_copy correctly copies a single analog."""
    # Set up: Create an analog in memory with some tokens not in memory
    # Create tokens in memory with analog 1
    memory_indices = torch.tensor([0, 1, 2])
    driver_indices = torch.tensor([3, 4])
    
    # Set up tokens
    network.tokens.token_tensor.set_features(memory_indices, TF.SET, Set.MEMORY)
    network.tokens.token_tensor.set_features(driver_indices, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(torch.cat([memory_indices, driver_indices]), TF.ANALOG, 1.0)
    
    # Mock check_for_copy to return analog 1
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([1])):
        # Mock get_analog_indices to return all tokens in analog
        all_analog_indices = torch.cat([memory_indices, driver_indices])
        with patch.object(network.tokens.analog_ops, 'get_analog_indices', return_value=all_analog_indices):
            # Mock copy_tokens to return new indices
            new_indices = torch.tensor([10, 11])
            with patch.object(network.tokens, 'copy_tokens', return_value=new_indices):
                # Mock new_analog_id to return a new analog number
                with patch.object(network.tokens.analog_ops, 'new_analog_id', return_value=2):
                    with patch.object(network, 'cache_sets'):
                        with patch.object(network, 'cache_analogs'):
                            result = network.analog_ops.make_AM_copy()
    
    # Verify result
    assert result == [2]
    # Verify new analog number was set on copied tokens
    network.tokens.token_tensor.set_feature.assert_called_once_with(new_indices, TF.ANALOG, 2)


def test_make_AM_copy_with_multiple_analogs(network):
    """Test that make_AM_copy correctly copies multiple analogs."""
    # Set up multiple analogs
    analog1_tokens = torch.tensor([0, 1])
    analog2_tokens = torch.tensor([2, 3])
    
    network.tokens.token_tensor.set_features(analog1_tokens, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(analog2_tokens, TF.SET, Set.RECIPIENT)
    network.tokens.token_tensor.set_features(analog1_tokens, TF.ANALOG, 1.0)
    network.tokens.token_tensor.set_features(analog2_tokens, TF.ANALOG, 2.0)
    
    # Mock check_for_copy to return both analogs
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([1, 2])):
        # Mock get_analog_indices to return tokens for each analog
        def get_analog_indices_side_effect(analog):
            if analog == 1:
                return analog1_tokens
            elif analog == 2:
                return analog2_tokens
        with patch.object(network.tokens.analog_ops, 'get_analog_indices', side_effect=get_analog_indices_side_effect):
            # Mock copy_tokens to return new indices
            new_indices1 = torch.tensor([10, 11])
            new_indices2 = torch.tensor([12, 13])
            copy_call_count = 0
            def copy_tokens_side_effect(indices, to_set, connect_to_copies):
                nonlocal copy_call_count
                copy_call_count += 1
                if copy_call_count == 1:
                    return new_indices1
                else:
                    return new_indices2
            with patch.object(network.tokens, 'copy_tokens', side_effect=copy_tokens_side_effect):
                # Mock new_analog_id to return sequential analog numbers
                analog_id_call_count = 0
                def new_analog_id_side_effect():
                    nonlocal analog_id_call_count
                    analog_id_call_count += 1
                    return analog_id_call_count + 2  # Return 3, 4
                with patch.object(network.tokens.analog_ops, 'new_analog_id', side_effect=new_analog_id_side_effect):
                    with patch.object(network, 'cache_sets'):
                        with patch.object(network, 'cache_analogs'):
                            result = network.analog_ops.make_AM_copy()
    
    # Verify result contains both new analog numbers
    assert result == [3, 4]


def test_make_AM_copy_skips_analogs_with_no_non_memory_tokens(network):
    """Test that make_AM_copy skips analogs that have no non-memory tokens."""
    # Set up: Create an analog where all tokens are in memory
    memory_indices = torch.tensor([0, 1, 2])
    network.tokens.token_tensor.set_features(memory_indices, TF.SET, Set.MEMORY)
    network.tokens.token_tensor.set_features(memory_indices, TF.ANALOG, 1.0)
    
    # Mock check_for_copy to return analog 1
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([1])):
        # Mock get_analog_indices to return memory tokens
        with patch.object(network.tokens.analog_ops, 'get_analog_indices', return_value=memory_indices):
            # get_tokens_where_not should return empty tensor
            with patch.object(network.tokens.token_tensor, 'get_tokens_where_not', return_value=torch.tensor([], dtype=torch.long)):
                with patch.object(network, 'cache_sets'):
                    with patch.object(network, 'cache_analogs'):
                        result = network.analog_ops.make_AM_copy()
    
    # Verify result is empty (analog was skipped)
    assert result == []


def test_make_AM_copy_includes_children(network):
    """Test that make_AM_copy includes children tokens in the copy."""
    # Set up: Create tokens with children
    parent_indices = torch.tensor([3, 4])
    child_indices = torch.tensor([5, 6])
    
    network.tokens.token_tensor.set_features(parent_indices, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(parent_indices, TF.ANALOG, 1.0)
    
    # Set up connections: parents -> children
    network.tokens.connections.tensor[3, 5] = True
    network.tokens.connections.tensor[4, 6] = True
    
    # Mock check_for_copy
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([1])):
        # Mock get_analog_indices to return parent indices
        with patch.object(network.tokens.analog_ops, 'get_analog_indices', return_value=parent_indices):
            # get_children_recursive should return child indices
            with patch.object(network.tokens.connections, 'get_children_recursive', return_value=child_indices):
                # Mock copy_tokens
                new_indices = torch.tensor([10, 11, 12, 13])  # Both parents and children
                with patch.object(network.tokens, 'copy_tokens', return_value=new_indices) as mock_copy:
                    with patch.object(network.tokens.analog_ops, 'new_analog_id', return_value=2):
                        with patch.object(network, 'cache_sets'):
                            with patch.object(network, 'cache_analogs'):
                                result = network.analog_ops.make_AM_copy()
                
                # Verify copy_tokens was called with combined indices (parents + children)
                call_args = mock_copy.call_args
                combined_indices = call_args[0][0]
                # Should include both parents and children (order may vary due to unique)
                assert len(combined_indices) == 4
                assert set(combined_indices.tolist()) == set(torch.cat([parent_indices, child_indices]).tolist())


# =====================[ make_AM_move Tests ]======================

def test_make_AM_move_updates_children_sets(network):
    """Test that make_AM_move updates children tokens to match parent set."""
    # Set up: Create tokens in DRIVER with children
    parent_indices = torch.tensor([0, 1])
    child_indices = torch.tensor([2, 3])
    
    network.tokens.token_tensor.set_features(parent_indices, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(child_indices, TF.SET, Set.MEMORY)  # Children in different set
    
    # Set up connections
    network.tokens.connections.tensor[0, 2] = True
    network.tokens.connections.tensor[1, 3] = True
    
    # Mock check_for_copy to return empty (no analogs to move)
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([])):
        # get_tokens_where is called with only 2 args - mock it to use cache.get_set_indices
        def get_tokens_where_side_effect(feature, value, *args, **kwargs):
            if len(args) == 0 and 'indices' not in kwargs:
                # Called with 2 args - use cache to get set indices
                if feature == TF.SET:
                    return network.tokens.token_tensor.cache.get_set_indices(Set(int(value)))
                return torch.tensor([], dtype=torch.long)
            # Otherwise use real method
            return network.tokens.token_tensor.get_tokens_where(feature, value, *args, **kwargs)
        
        with patch.object(network.tokens.token_tensor, 'get_tokens_where', side_effect=get_tokens_where_side_effect):
            with patch.object(network, 'cache_sets'):
                with patch.object(network, 'cache_analogs'):
                    network.analog_ops.make_AM_move()
    
    # Verify children were updated to DRIVER set
    child_sets = network.tokens.token_tensor.get_features(child_indices, TF.SET)
    assert torch.all(child_sets == Set.DRIVER)


def test_make_AM_move_handles_empty_sets(network):
    """Test that make_AM_move handles empty sets gracefully."""
    # Mock check_for_copy
    with patch.object(network.analog_ops, 'check_for_copy', return_value=torch.tensor([])):
        # get_tokens_where is called with only 2 args - mock it to use cache.get_set_indices
        def get_tokens_where_side_effect(feature, value, *args, **kwargs):
            if len(args) == 0 and 'indices' not in kwargs:
                # Called with 2 args - use cache to get set indices
                if feature == TF.SET:
                    return network.tokens.token_tensor.cache.get_set_indices(Set(int(value)))
                return torch.tensor([], dtype=torch.long)
            # Otherwise use real method
            return network.tokens.token_tensor.get_tokens_where(feature, value, *args, **kwargs)
        
        with patch.object(network.tokens.token_tensor, 'get_tokens_where', side_effect=get_tokens_where_side_effect):
            with patch.object(network, 'cache_sets'):
                with patch.object(network, 'cache_analogs'):
                    # Should not raise an error
                    network.analog_ops.make_AM_move()


# =====================[ find_mapped_analog Tests ]======================

def test_find_mapped_analog_returns_analog_number(network):
    """Test that find_mapped_analog returns the correct analog number."""
    # Set up: Create a mapped PO in RECIPIENT
    po_index = 5
    analog_num = 2
    
    # Set up token in RECIPIENT with max_map > 0
    network.tokens.token_tensor.set_feature(po_index, TF.SET, Set.RECIPIENT)
    network.tokens.token_tensor.set_feature(po_index, TF.ANALOG, float(analog_num))
    network.tokens.token_tensor.set_feature(po_index, TF.MAX_MAP, 0.5)
    
    # Mock get_max_maps
    with patch.object(network.mapping_ops, 'get_max_maps'):
        # Mock get_mapped_pos to return the PO index
        with patch.object(network.sets[Set.RECIPIENT].token_op, 'get_mapped_pos', return_value=torch.tensor([po_index])):
            result = network.analog_ops.find_mapped_analog(Set.RECIPIENT)
    
    assert result == analog_num


def test_find_mapped_analog_raises_error_when_no_mapped_pos(network):
    """Test that find_mapped_analog raises ValueError when no mapped POs exist."""
    # Mock get_max_maps
    with patch.object(network.mapping_ops, 'get_max_maps'):
        # Mock get_mapped_pos to return empty tensor
        with patch.object(network.sets[Set.RECIPIENT].token_op, 'get_mapped_pos', return_value=torch.tensor([], dtype=torch.long)):
            with pytest.raises(ValueError, match="No mapped POs in set"):
                network.analog_ops.find_mapped_analog(Set.RECIPIENT)


# =====================[ find_mapping_analog Tests ]======================

def test_find_mapping_analog_returns_analogs(network):
    """Test that find_mapping_analog returns unique analog numbers."""
    # Set up: Create tokens in DRIVER with max_map > 0
    token_indices = torch.tensor([0, 1, 2])
    analog_nums = torch.tensor([1.0, 2.0, 1.0])  # Two tokens with analog 1, one with analog 2
    
    network.tokens.token_tensor.set_features(token_indices, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(token_indices, TF.ANALOG, analog_nums)
    network.tokens.token_tensor.set_features(token_indices, TF.MAX_MAP, torch.tensor([0.5, 0.3, 0.7]))
    
    # Update driver local tensor - need to refresh cache first
    network.cache_sets()
    driver_local = network.sets[Set.DRIVER].lcl
    for i, idx in enumerate(token_indices):
        local_idx = network.sets[Set.DRIVER].lcl.to_local(torch.tensor([idx]))[0]
        driver_local[local_idx, TF.MAX_MAP] = 0.5
        driver_local[local_idx, TF.ANALOG] = analog_nums[i]
    
    # Mock get_max_maps
    with patch.object(network.mapping_ops, 'get_max_maps'):
        result = network.analog_ops.find_mapping_analog()
    
    # Should return unique analogs: [1, 2]
    assert result is not None
    assert len(result) == 2
    assert set(result.tolist()) == {1, 2}


def test_find_mapping_analog_returns_none_when_no_mappings(network):
    """Test that find_mapping_analog returns None when no tokens have max_map > 0."""
    # Set up: All tokens have max_map = 0
    token_indices = torch.tensor([0, 1])
    network.tokens.token_tensor.set_features(token_indices, TF.SET, Set.DRIVER)
    network.tokens.token_tensor.set_features(token_indices, TF.MAX_MAP, torch.tensor([0.0, 0.0]))
    
    # Update driver local tensor - need to refresh cache first
    network.cache_sets()
    driver_local = network.sets[Set.DRIVER].lcl
    for idx in token_indices:
        local_idx = network.sets[Set.DRIVER].lcl.to_local(torch.tensor([idx]))[0]
        driver_local[local_idx, TF.MAX_MAP] = 0.0
    
    # Mock get_max_maps
    with patch.object(network.mapping_ops, 'get_max_maps'):
        result = network.analog_ops.find_mapping_analog()
    
    assert result is None


# =====================[ move_mapping_analogs_to_new Tests ]======================

def test_move_mapping_analogs_to_new_creates_new_analog(network):
    """Test that move_mapping_analogs_to_new creates a new analog for mapped analogs."""
    # Set up: Create mapped analogs
    mapped_analogs = torch.tensor([1, 2])
    token_indices = torch.tensor([0, 1, 2, 3])
    
    # Mock find_mapping_analog
    with patch.object(network.analog_ops, 'find_mapping_analog', return_value=mapped_analogs):
        # Mock get_analog_indices_multiple
        with patch.object(network.tokens.analog_ops, 'get_analog_indices_multiple', return_value=token_indices):
            # Mock new_analog_id
            with patch.object(network.tokens.analog_ops, 'new_analog_id', return_value=5):
                with patch.object(network, 'cache_sets'):
                    with patch.object(network, 'cache_analogs'):
                        result = network.analog_ops.move_mapping_analogs_to_new()
    
    # Verify new analog number was returned
    assert result == 5
    # Verify set_feature was called with new analog number
    network.tokens.token_tensor.set_feature.assert_called_once_with(token_indices, TF.ANALOG, 5)


def test_move_mapping_analogs_to_new_returns_none_when_no_mappings(network):
    """Test that move_mapping_analogs_to_new returns None when no mapped analogs exist."""
    # Mock find_mapping_analog to return None
    with patch.object(network.analog_ops, 'find_mapping_analog', return_value=None):
        result = network.analog_ops.move_mapping_analogs_to_new()
    
    assert result is None


# =====================[ new_set_to_analog Tests ]======================

def test_new_set_to_analog_creates_analog_for_new_set(network):
    """Test that new_set_to_analog creates a new analog for all tokens in NEW_SET."""
    # Set up: Create tokens in NEW_SET
    new_set_indices = torch.tensor([0, 1, 2])
    network.tokens.token_tensor.set_features(new_set_indices, TF.SET, Set.NEW_SET)
    
    # Mock new_analog_id
    with patch.object(network.tokens.analog_ops, 'new_analog_id', return_value=3):
        # Mock set_features_all
        with patch.object(network.sets[Set.NEW_SET].token_op, 'set_features_all') as mock_set_features_all:
            result = network.analog_ops.new_set_to_analog()
    
    # Verify new analog number was returned
    assert result == 3
    # Verify set_features_all was called with correct parameters
    mock_set_features_all.assert_called_once_with(TF.ANALOG, 3)

