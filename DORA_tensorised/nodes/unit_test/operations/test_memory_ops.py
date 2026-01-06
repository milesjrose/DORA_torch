# nodes/unit_test/operations/test_memory_ops.py
# Tests for TensorOperations class (memory operations)

import sys
from pathlib import Path

# Add DORA_tensorised to Python path so 'nodes' module can be imported
dora_tensorised_dir = Path(__file__).parent.parent.parent.parent
if str(dora_tensorised_dir) not in sys.path:
    sys.path.insert(0, str(dora_tensorised_dir))

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
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


# =====================[ NotImplementedError Tests ]======================

def test_make_am_raises_not_implemented(network):
    """Test that make_am raises NotImplementedError (moved to analog_ops.py)."""
    with pytest.raises(NotImplementedError, match="Function is moved to analog_ops.py"):
        network.tensor_ops.make_am()


def test_make_am_with_copy_false_raises_not_implemented(network):
    """Test that make_am with copy=False raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Function is moved to analog_ops.py"):
        network.tensor_ops.make_am(copy=False)


def test_check_analog_for_tokens_to_copy_raises_not_implemented(network):
    """Test that check_analog_for_tokens_to_copy raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Function is moved to analog_ops.py"):
        network.tensor_ops.check_analog_for_tokens_to_copy()


def test_del_mem_tokens_raises_not_implemented(network):
    """Test that del_mem_tokens raises NotImplementedError (function not used anymore)."""
    with pytest.raises(NotImplementedError, match="Function is not used anymore"):
        network.tensor_ops.del_mem_tokens()


def test_del_mem_tokens_with_sets_raises_not_implemented(network):
    """Test that del_mem_tokens with custom sets raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Function is not used anymore"):
        network.tensor_ops.del_mem_tokens(sets=[Set.DRIVER])


# =====================[ clear_all_sets Tests ]======================

def test_clear_all_sets_sets_all_tokens_to_memory(network):
    """Test that clear_all_sets sets all tokens to Set.MEMORY."""
    # Set up: Create tokens in different sets
    driver_indices = torch.tensor([0, 1, 2])
    recipient_indices = torch.tensor([3, 4])
    memory_indices = torch.tensor([5, 6, 7])
    
    # Set the initial sets
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.token_tensor.set_feature(memory_indices, TF.SET, float(Set.MEMORY))
    network.recache()
    
    # Verify initial state
    assert torch.all(network.token_tensor.get_feature(driver_indices, TF.SET) == float(Set.DRIVER))
    assert torch.all(network.token_tensor.get_feature(recipient_indices, TF.SET) == float(Set.RECIPIENT))
    
    # Call clear_all_sets
    network.tensor_ops.clear_all_sets()
    
    # All tokens should now be in MEMORY
    all_indices = torch.cat([driver_indices, recipient_indices, memory_indices])
    for idx in all_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.MEMORY), f"Token {idx} should be in MEMORY, got {set_value}"


def test_clear_all_sets_calls_recache(network):
    """Test that clear_all_sets calls recache to update caches."""
    with patch.object(network, 'recache') as mock_recache:
        network.tensor_ops.clear_all_sets()
        mock_recache.assert_called_once()


def test_clear_all_sets_uses_cache_to_get_indices(network):
    """Test that clear_all_sets uses the cache to get all node indices."""
    # Set up some tokens
    network.token_tensor.set_feature(torch.tensor([0, 1]), TF.SET, float(Set.DRIVER))
    network.recache()
    
    mock_indices = torch.tensor([0, 1, 2, 3, 4])
    with patch.object(network.token_tensor.cache, 'get_all_nodes_indices', return_value=mock_indices) as mock_get_indices:
        with patch.object(network.token_tensor, 'set_feature') as mock_set_feature:
            with patch.object(network, 'recache'):
                network.tensor_ops.clear_all_sets()
                
                mock_get_indices.assert_called_once()
                mock_set_feature.assert_called_once()
                # Verify set_feature was called with correct arguments
                call_args = mock_set_feature.call_args
                assert torch.equal(call_args[0][0], mock_indices)
                assert call_args[0][1] == TF.SET
                assert call_args[0][2] == Set.MEMORY


# =====================[ clear_set Tests ]======================

def test_clear_set_clears_driver_set(network):
    """Test that clear_set calls set_features_all with correct parameters."""
    # Set up driver tokens
    driver_indices = torch.tensor([0, 1, 2])
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.recache()
    
    # Mock set_features_all to verify it's called correctly
    with patch.object(network.sets[Set.DRIVER].token_op, 'set_features_all') as mock_set_features:
        with patch.object(network, 'recache') as mock_recache:
            network.tensor_ops.clear_set(Set.DRIVER)
            
            # Verify set_features_all was called with correct parameters
            mock_set_features.assert_called_once_with(TF.SET, Set.MEMORY)
            mock_recache.assert_called_once()


def test_clear_set_clears_recipient_set(network):
    """Test that clear_set calls set_features_all for RECIPIENT set."""
    # Set up recipient tokens
    recipient_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    # Mock set_features_all to verify it's called correctly
    with patch.object(network.sets[Set.RECIPIENT].token_op, 'set_features_all') as mock_set_features:
        with patch.object(network, 'recache') as mock_recache:
            network.tensor_ops.clear_set(Set.RECIPIENT)
            
            # Verify set_features_all was called with correct parameters
            mock_set_features.assert_called_once_with(TF.SET, Set.MEMORY)
            mock_recache.assert_called_once()


def test_clear_set_clears_new_set(network):
    """Test that clear_set calls set_features_all for NEW_SET."""
    # Set up new_set tokens
    new_set_indices = torch.tensor([0, 1, 2, 3])
    network.token_tensor.set_feature(new_set_indices, TF.SET, float(Set.NEW_SET))
    network.recache()
    
    # Mock set_features_all to verify it's called correctly
    with patch.object(network.sets[Set.NEW_SET].token_op, 'set_features_all') as mock_set_features:
        with patch.object(network, 'recache') as mock_recache:
            network.tensor_ops.clear_set(Set.NEW_SET)
            
            # Verify set_features_all was called with correct parameters
            mock_set_features.assert_called_once_with(TF.SET, Set.MEMORY)
            mock_recache.assert_called_once()


def test_clear_set_does_not_affect_other_sets(network):
    """Test that clear_set only calls set_features_all on the specified set."""
    # Set up tokens in multiple sets
    driver_indices = torch.tensor([0, 1])
    recipient_indices = torch.tensor([2, 3])
    
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    # Mock set_features_all for both sets
    with patch.object(network.sets[Set.DRIVER].token_op, 'set_features_all') as mock_driver:
        with patch.object(network.sets[Set.RECIPIENT].token_op, 'set_features_all') as mock_recipient:
            with patch.object(network, 'recache'):
                # Clear only DRIVER
                network.tensor_ops.clear_set(Set.DRIVER)
                
                # Driver's set_features_all should be called
                mock_driver.assert_called_once_with(TF.SET, Set.MEMORY)
                # Recipient's set_features_all should NOT be called
                mock_recipient.assert_not_called()


def test_clear_set_calls_recache(network):
    """Test that clear_set calls recache after clearing."""
    with patch.object(network, 'recache') as mock_recache:
        network.tensor_ops.clear_set(Set.DRIVER)
        mock_recache.assert_called_once()


# =====================[ reset_inferences Tests ]======================

def test_reset_inferences_calls_reset_on_all_sets(network):
    """Test that reset_inferences calls reset_inferences on all sets."""
    # Mock reset_inferences for each set
    mocks = {}
    for set_type in Set:
        mocks[set_type] = patch.object(network.sets[set_type].token_op, 'reset_inferences').start()
    
    try:
        network.tensor_ops.reset_inferences()
        
        # Verify reset_inferences was called on each set
        for set_type in Set:
            mocks[set_type].assert_called_once()
    finally:
        patch.stopall()


def test_reset_inferences_resets_inferred_field(network):
    """Test that reset_inferences properly resets INFERRED field."""
    # First, set all tokens to MEMORY to isolate test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with INFERRED = 1
    test_indices = torch.tensor([0, 1, 2])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(test_indices, TF.INFERRED, 1.0)
    network.recache()
    
    # Call reset_inferences
    network.tensor_ops.reset_inferences()
    
    # Verify INFERRED is now 0 (B.FALSE) for our test tokens
    for idx in test_indices:
        inferred_value = network.token_tensor.get_feature(idx, TF.INFERRED)
        assert inferred_value == 0.0, f"Token {idx} INFERRED should be 0.0"


def test_reset_inferences_resets_maker_unit(network):
    """Test that reset_inferences properly resets MAKER_UNIT field."""
    from nodes.enums import null
    
    # First, set all tokens to MEMORY to isolate test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with MAKER_UNIT set
    test_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.RECIPIENT))
    network.token_tensor.set_feature(test_indices, TF.MAKER_UNIT, 5.0)
    network.recache()
    
    # Call reset_inferences
    network.tensor_ops.reset_inferences()
    
    # Verify MAKER_UNIT is now null for our test tokens
    for idx in test_indices:
        maker_value = network.token_tensor.get_feature(idx, TF.MAKER_UNIT)
        assert maker_value == null, f"Token {idx} MAKER_UNIT should be null"


def test_reset_inferences_resets_made_unit(network):
    """Test that reset_inferences properly resets MADE_UNIT field."""
    from nodes.enums import null
    
    # First, set all tokens to MEMORY to isolate test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with MADE_UNIT set
    test_indices = torch.tensor([0, 1, 2])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(test_indices, TF.MADE_UNIT, 10.0)
    network.recache()
    
    # Call reset_inferences
    network.tensor_ops.reset_inferences()
    
    # Verify MADE_UNIT is now null for our test tokens
    for idx in test_indices:
        made_value = network.token_tensor.get_feature(idx, TF.MADE_UNIT)
        assert made_value == null, f"Token {idx} MADE_UNIT should be null"


# =====================[ reset_maker_made_units Tests ]======================

def test_reset_maker_made_units_calls_reset_on_all_sets(network):
    """Test that reset_maker_made_units calls reset_maker_made_units on all sets."""
    # Mock reset_maker_made_units for each set
    mocks = {}
    for set_type in Set:
        mocks[set_type] = patch.object(network.sets[set_type].token_op, 'reset_maker_made_units').start()
    
    try:
        network.tensor_ops.reset_maker_made_units()
        
        # Verify reset_maker_made_units was called on each set
        for set_type in Set:
            mocks[set_type].assert_called_once()
    finally:
        patch.stopall()


def test_reset_maker_made_units_resets_maker_unit(network):
    """Test that reset_maker_made_units properly resets MAKER_UNIT field."""
    from nodes.enums import null
    
    # First, set all tokens to MEMORY to isolate test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with MAKER_UNIT set
    test_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(test_indices, TF.MAKER_UNIT, 7.0)
    network.recache()
    
    # Call reset_maker_made_units
    network.tensor_ops.reset_maker_made_units()
    
    # Verify MAKER_UNIT is now null for our test tokens
    for idx in test_indices:
        maker_value = network.token_tensor.get_feature(idx, TF.MAKER_UNIT)
        assert maker_value == null, f"Token {idx} MAKER_UNIT should be null"


def test_reset_maker_made_units_resets_made_unit(network):
    """Test that reset_maker_made_units properly resets MADE_UNIT field."""
    from nodes.enums import null
    
    # First, set all tokens to MEMORY to isolate test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with MADE_UNIT set
    test_indices = torch.tensor([0, 1, 2])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.RECIPIENT))
    network.token_tensor.set_feature(test_indices, TF.MADE_UNIT, 3.0)
    network.recache()
    
    # Call reset_maker_made_units
    network.tensor_ops.reset_maker_made_units()
    
    # Verify MADE_UNIT is now null for our test tokens
    for idx in test_indices:
        made_value = network.token_tensor.get_feature(idx, TF.MADE_UNIT)
        assert made_value == null, f"Token {idx} MADE_UNIT should be null"


def test_reset_maker_made_units_does_not_affect_inferred(network):
    """Test that reset_maker_made_units does NOT reset INFERRED field."""
    # First, set all tokens to MEMORY to isolate our test tokens
    all_indices = torch.arange(network.token_tensor.get_count())
    network.token_tensor.set_feature(all_indices, TF.SET, float(Set.MEMORY))
    
    # Set up specific tokens with INFERRED = 1 in DRIVER
    test_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(test_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(test_indices, TF.INFERRED, 1.0)
    network.recache()
    
    # Call reset_maker_made_units
    network.tensor_ops.reset_maker_made_units()
    
    # Verify INFERRED is still 1 for our test tokens
    # Check directly on the global tensor since local view behavior may vary
    for idx in test_indices:
        inferred_value = network.token_tensor.get_feature(idx, TF.INFERRED)
        assert inferred_value == 1.0, f"Token {idx} INFERRED should still be 1.0"


# =====================[ swap_driver_recipient Tests ]======================

def test_swap_driver_recipient_swaps_set_values(network):
    """Test that swap_driver_recipient swaps driver and recipient set values."""
    # Set up tokens in driver and recipient
    driver_indices = torch.tensor([0, 1, 2])
    recipient_indices = torch.tensor([3, 4])
    
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    # Call swap_driver_recipient
    network.tensor_ops.swap_driver_recipient()
    
    # Driver tokens should now be RECIPIENT
    for idx in driver_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.RECIPIENT), f"Token {idx} should be RECIPIENT after swap"
    
    # Recipient tokens should now be DRIVER
    for idx in recipient_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.DRIVER), f"Token {idx} should be DRIVER after swap"


def test_swap_driver_recipient_calls_mappings_swap(network):
    """Test that swap_driver_recipient calls mappings.swap_driver_recipient."""
    # Set up tokens
    network.token_tensor.set_feature(torch.tensor([0, 1]), TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(torch.tensor([2, 3]), TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    with patch.object(network.mappings, 'swap_driver_recipient') as mock_swap:
        with patch.object(network, 'recache'):
            network.tensor_ops.swap_driver_recipient()
            mock_swap.assert_called_once()


def test_swap_driver_recipient_calls_recache(network):
    """Test that swap_driver_recipient calls recache."""
    with patch.object(network, 'recache') as mock_recache:
        network.tensor_ops.swap_driver_recipient()
        mock_recache.assert_called_once()


def test_swap_driver_recipient_does_not_affect_memory(network):
    """Test that swap_driver_recipient does not affect MEMORY set."""
    # Set up tokens in all sets
    driver_indices = torch.tensor([0, 1])
    recipient_indices = torch.tensor([2, 3])
    memory_indices = torch.tensor([4, 5])
    
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.token_tensor.set_feature(memory_indices, TF.SET, float(Set.MEMORY))
    network.recache()
    
    # Call swap_driver_recipient
    network.tensor_ops.swap_driver_recipient()
    
    # Memory tokens should still be in MEMORY
    for idx in memory_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.MEMORY), f"Token {idx} should still be MEMORY"


def test_swap_driver_recipient_does_not_affect_new_set(network):
    """Test that swap_driver_recipient does not affect NEW_SET."""
    # Set up tokens
    driver_indices = torch.tensor([0])
    new_set_indices = torch.tensor([1, 2])
    
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(new_set_indices, TF.SET, float(Set.NEW_SET))
    network.recache()
    
    # Call swap_driver_recipient
    network.tensor_ops.swap_driver_recipient()
    
    # New_set tokens should still be in NEW_SET
    for idx in new_set_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.NEW_SET), f"Token {idx} should still be NEW_SET"


def test_swap_driver_recipient_with_empty_driver(network):
    """Test that swap_driver_recipient handles empty driver set."""
    # Set up only recipient tokens
    recipient_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    # Should not raise error
    network.tensor_ops.swap_driver_recipient()
    
    # Recipient tokens should now be DRIVER
    for idx in recipient_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.DRIVER), f"Token {idx} should be DRIVER after swap"


def test_swap_driver_recipient_with_empty_recipient(network):
    """Test that swap_driver_recipient handles empty recipient set."""
    # Set up only driver tokens
    driver_indices = torch.tensor([0, 1])
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.recache()
    
    # Should not raise error
    network.tensor_ops.swap_driver_recipient()
    
    # Driver tokens should now be RECIPIENT
    for idx in driver_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.RECIPIENT), f"Token {idx} should be RECIPIENT after swap"


def test_swap_driver_recipient_double_swap_returns_original(network):
    """Test that swapping twice returns to original state."""
    # Set up tokens
    driver_indices = torch.tensor([0, 1, 2])
    recipient_indices = torch.tensor([3, 4])
    
    network.token_tensor.set_feature(driver_indices, TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(recipient_indices, TF.SET, float(Set.RECIPIENT))
    network.recache()
    
    # Swap twice
    network.tensor_ops.swap_driver_recipient()
    network.tensor_ops.swap_driver_recipient()
    
    # Should be back to original state
    for idx in driver_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.DRIVER), f"Token {idx} should be DRIVER after double swap"
    
    for idx in recipient_indices:
        set_value = network.token_tensor.get_feature(idx, TF.SET)
        assert set_value == float(Set.RECIPIENT), f"Token {idx} should be RECIPIENT after double swap"


# =====================[ Integration Tests ]======================

def test_tensor_ops_initialization(network):
    """Test that TensorOperations is properly initialized with network reference."""
    assert network.tensor_ops.network is network


def test_tensor_ops_accessible_via_network_tensor_property(network):
    """Test that TensorOperations is accessible via network.tensor property."""
    assert network.tensor is network.tensor_ops


def test_clear_workflow(network):
    """Test a typical clear workflow: reset inferences, reset maker/made, clear sets."""
    # Set up tokens with various states
    indices = torch.tensor([0, 1, 2, 3])
    network.token_tensor.set_feature(indices[:2], TF.SET, float(Set.DRIVER))
    network.token_tensor.set_feature(indices[2:], TF.SET, float(Set.RECIPIENT))
    network.token_tensor.set_feature(indices, TF.INFERRED, 1.0)
    network.token_tensor.set_feature(indices, TF.MAKER_UNIT, 5.0)
    network.token_tensor.set_feature(indices, TF.MADE_UNIT, 10.0)
    network.recache()
    
    # Perform clear workflow
    network.tensor_ops.reset_inferences()
    network.tensor_ops.reset_maker_made_units()
    network.tensor_ops.clear_all_sets()
    
    # All tokens should be in MEMORY with reset fields
    from nodes.enums import null
    for idx in indices:
        assert network.token_tensor.get_feature(idx, TF.SET) == float(Set.MEMORY)

