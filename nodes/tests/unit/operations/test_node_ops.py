# nodes/tests/unit/operations/test_node_ops.py
# Tests for NodeOperations class (using global indexes)

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
from nodes.network.network_params import default_params
from nodes.network.single_nodes import Token, Semantic
from nodes.enums import Set, TF, SF, MappingFields, Type, null


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
    # Set unique IDs for each token
    for i in range(num_tokens):
        tokens[i, TF.ID] = i
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    names = {i: f"token_{i}" for i in range(num_tokens)}
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


# =====================[ Initialization Tests ]======================

def test_node_ops_initialization(network):
    """Test that NodeOperations is properly initialized."""
    assert network.node_ops is not None
    assert network.node_ops.network is network


def test_node_ops_accessible_via_property(network):
    """Test that NodeOperations is accessible via network.node property."""
    assert network.node is network.node_ops


# =====================[ P Mode Tests ]======================

def test_get_pmode_calls_update_op(network):
    """Test that get_pmode calls p_get_mode on DRIVER and RECIPIENT."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'p_get_mode') as mock_driver:
        with patch.object(network.sets[Set.RECIPIENT].update_op, 'p_get_mode') as mock_recipient:
            network.node_ops.get_pmode()
            mock_driver.assert_called_once()
            mock_recipient.assert_called_once()


def test_initialise_p_mode_default(network):
    """Test that initialise_p_mode calls p_initialise_mode on RECIPIENT by default."""
    with patch.object(network.sets[Set.RECIPIENT].update_op, 'p_initialise_mode') as mock_init:
        network.node_ops.initialise_p_mode()
        mock_init.assert_called_once()


def test_initialise_p_mode_specified_set(network):
    """Test that initialise_p_mode calls p_initialise_mode on specified set."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'p_initialise_mode') as mock_init:
        network.node_ops.initialise_p_mode(tk_set=Set.DRIVER)
        mock_init.assert_called_once()


# =====================[ Token Value Tests ]======================

def test_get_tk_value(network):
    """Test getting a token feature value by global index."""
    idx = 5
    test_value = 0.75
    network.token_tensor.tensor[idx, TF.ACT] = test_value
    
    result = network.node_ops.get_tk_value(idx, TF.ACT)
    assert abs(result - test_value) < 0.001


def test_set_tk_value(network):
    """Test setting a token feature value by global index."""
    idx = 6
    test_value = 0.88
    
    network.node_ops.set_tk_value(idx, TF.ACT, test_value)
    
    result = network.token_tensor.tensor[idx, TF.ACT].item()
    assert abs(result - test_value) < 0.001


def test_get_tk_name(network):
    """Test getting a token name by global index."""
    idx = 3
    result = network.node_ops.get_tk_name(idx)
    assert result == f"token_{idx}"


def test_set_tk_name(network):
    """Test setting a token name by global index."""
    idx = 4
    new_name = "custom_token"
    
    network.node_ops.set_tk_name(idx, new_name)
    
    result = network.token_tensor.get_name(idx)
    assert result == new_name


def test_get_tk_set(network):
    """Test getting the set of a token by global index."""
    idx = 7
    network.token_tensor.tensor[idx, TF.SET] = Set.RECIPIENT.value
    network.recache()
    
    result = network.node_ops.get_tk_set(idx)
    assert result == Set.RECIPIENT


# =====================[ Token Add/Delete Tests ]======================

def test_del_token(network):
    """Test deleting a token by global index."""
    idx = 8
    
    with patch.object(network.tokens, 'delete_tokens') as mock_delete:
        with patch.object(network, 'recache') as mock_recache:
            network.node_ops.del_token(idx)
            mock_delete.assert_called_once()
            mock_recache.assert_called_once()


def test_get_token(network):
    """Test getting a token object by global index."""
    idx = 9
    network.token_tensor.tensor[idx, TF.ACT] = 0.5
    network.token_tensor.tensor[idx, TF.TYPE] = Type.RB.value
    
    result = network.node_ops.get_token(idx)
    
    assert isinstance(result, Token)
    assert result.tensor[TF.ACT] == 0.5
    assert result.tensor[TF.TYPE] == Type.RB.value


# =====================[ Semantic Value Tests ]======================

def test_get_sem_value(network):
    """Test getting a semantic feature value."""
    idx = 0
    test_value = 1.5
    network.semantics.nodes[idx, SF.AMOUNT] = test_value
    
    with patch.object(network.semantics, 'get', return_value=test_value) as mock_get:
        result = network.node_ops.get_sem_value(idx, SF.AMOUNT)
        mock_get.assert_called_once_with(idx, SF.AMOUNT)


def test_set_sem_value(network):
    """Test setting a semantic feature value."""
    idx = 1
    test_value = 2.5
    
    with patch.object(network.semantics, 'set') as mock_set:
        network.node_ops.set_sem_value(idx, SF.AMOUNT, test_value)
        mock_set.assert_called_once_with(idx, SF.AMOUNT, test_value)


def test_set_sem_max_input(network):
    """Test setting the maximum input for semantics."""
    mock_max = 0.9
    
    with patch.object(network.semantics, 'get_max_input', return_value=mock_max) as mock_get:
        with patch.object(network.semantics, 'set_max_input') as mock_set:
            network.node_ops.set_sem_max_input()
            mock_get.assert_called_once()
            mock_set.assert_called_once_with(mock_max)


# =====================[ Most Active Token Tests ]======================

def test_get_most_active_token_returns_dict(network):
    """Test that get_most_active_token returns a dictionary."""
    # Set up tokens with activations
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[0, TF.ACT] = 0.9
    network.recache()
    
    result = network.node_ops.get_most_active_token()
    assert isinstance(result, dict)


def test_get_most_active_token_specific_sets(network):
    """Test get_most_active_token with specific sets."""
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[0, TF.ACT] = 0.9
    network.recache()
    
    result = network.node_ops.get_most_active_token(sets=[Set.DRIVER])
    
    assert isinstance(result, dict)
    if Set.DRIVER in result:
        assert isinstance(result[Set.DRIVER], int)


# =====================[ Made/Maker Unit Tests ]======================

def test_get_made_unit_returns_none_for_null(network):
    """Test that get_made_unit returns None when made unit is null."""
    idx = 10
    network.token_tensor.tensor[idx, TF.MADE_UNIT] = null
    
    result = network.node_ops.get_made_unit(idx)
    assert result is None


def test_get_made_unit_returns_tuple(network):
    """Test that get_made_unit returns (local_idx, set) tuple."""
    idx = 11
    made_idx = 5
    made_set = Set.NEW_SET
    network.token_tensor.tensor[idx, TF.MADE_UNIT] = made_idx
    network.token_tensor.tensor[idx, TF.MADE_SET] = made_set.value
    
    result = network.node_ops.get_made_unit(idx)
    
    assert result is not None
    assert result == (made_idx, made_set)


def test_get_maker_unit_returns_none_for_null(network):
    """Test that get_maker_unit returns None when maker unit is null."""
    idx = 12
    network.token_tensor.tensor[idx, TF.MAKER_UNIT] = null
    
    result = network.node_ops.get_maker_unit(idx)
    assert result is None


def test_get_maker_unit_returns_tuple(network):
    """Test that get_maker_unit returns (local_idx, set) tuple."""
    idx = 13
    maker_idx = 3
    maker_set = Set.DRIVER
    network.token_tensor.tensor[idx, TF.MAKER_UNIT] = maker_idx
    network.token_tensor.tensor[idx, TF.MAKER_SET] = maker_set.value
    
    result = network.node_ops.get_maker_unit(idx)
    
    assert result is not None
    assert result == (maker_idx, maker_set)


# =====================[ Index Lookup Tests ]======================

def test_get_index_by_name(network):
    """Test getting a global index by token name."""
    idx = 2
    network.token_tensor.tensor[idx, TF.SET] = Set.DRIVER.value
    network.recache()
    
    result = network.node_ops.get_index_by_name(Set.DRIVER, f"token_{idx}")
    assert result == idx


def test_get_index_by_name_not_found(network):
    """Test that get_index_by_name raises error for non-existent name."""
    with pytest.raises(ValueError, match="not found"):
        network.node_ops.get_index_by_name(Set.DRIVER, "nonexistent_token")


def test_get_index_by_id(network):
    """Test getting a global index by token ID."""
    idx = 3
    token_id = 3  # IDs are set to index values in fixture
    network.token_tensor.tensor[idx, TF.SET] = Set.DRIVER.value
    network.recache()
    
    result = network.node_ops.get_index_by_id(Set.DRIVER, token_id)
    assert result == idx


def test_get_index_by_id_not_found(network):
    """Test that get_index_by_id raises error for non-existent ID."""
    with pytest.raises(ValueError, match="not found"):
        network.node_ops.get_index_by_id(Set.DRIVER, 9999)


def test_local_to_global(network):
    """Test converting local index to global index."""
    # Set up a token in DRIVER
    global_idx = 4
    network.token_tensor.tensor[global_idx, TF.SET] = Set.DRIVER.value
    network.recache()
    
    # The first DRIVER token should have local index 0
    result = network.node_ops.local_to_global(Set.DRIVER, 0)
    
    # Result should be a valid global index
    assert isinstance(result, int)


def test_global_to_local(network):
    """Test converting global index to local index."""
    # All tokens start with SET=0 (DRIVER) from the fixture
    # So we test with a DRIVER token instead
    global_idx = 5
    # Token is already in DRIVER (SET=0) from fixture initialization
    network.recache()
    
    local_idx, tk_set = network.node_ops.global_to_local(global_idx)
    
    assert isinstance(local_idx, int)
    assert tk_set == Set.DRIVER  # All tokens start in DRIVER from fixture


# =====================[ Integration Tests ]======================

def test_set_and_get_tk_value_workflow(network):
    """Test setting and getting a token value."""
    idx = 14
    
    # Set value
    network.node_ops.set_tk_value(idx, TF.ACT, 0.55)
    
    # Get value
    result = network.node_ops.get_tk_value(idx, TF.ACT)
    
    assert abs(result - 0.55) < 0.001


def test_token_set_detection(network):
    """Test that we can correctly detect token sets."""
    # Set up tokens in different sets
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[1, TF.SET] = Set.RECIPIENT.value
    network.token_tensor.tensor[2, TF.SET] = Set.MEMORY.value
    network.recache()
    
    assert network.node_ops.get_tk_set(0) == Set.DRIVER
    assert network.node_ops.get_tk_set(1) == Set.RECIPIENT
    assert network.node_ops.get_tk_set(2) == Set.MEMORY
