# nodes/tests/unit/operations/test_update_ops.py
# Tests for UpdateOperations class

import pytest
import torch
from unittest.mock import Mock, patch, call
from nodes.network.network import Network
from nodes.network.tokens.tokens import Tokens
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.tokens.connections.connections import Connections_Tensor
from nodes.network.tokens.connections.mapping import Mapping
from nodes.network.tokens.connections.links import Links
from nodes.network.sets_new.semantics import Semantics
from nodes.network.network_params import default_params
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

def test_update_ops_initialization(network):
    """Test that UpdateOperations is properly initialized."""
    assert network.update_ops is not None
    assert network.update_ops.network is network


def test_update_ops_accessible_via_property(network):
    """Test that UpdateOperations is accessible via network.update property."""
    assert network.update is network.update_ops


# =====================[ initialise_act Tests ]======================

def test_initialise_act_calls_update_op_init_act_on_all_active_sets(network):
    """Test that initialise_act calls init_act on DRIVER, RECIPIENT, and NEW_SET."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'init_act') as mock_driver:
        with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_act') as mock_recipient:
            with patch.object(network.sets[Set.NEW_SET].update_op, 'init_act') as mock_new_set:
                with patch.object(network.semantics, 'init_sem') as mock_sem:
                    network.update_ops.initialise_act()
                    
                    expected_types = [Type.GROUP, Type.P, Type.RB, Type.PO]
                    mock_driver.assert_called_once_with(expected_types)
                    mock_recipient.assert_called_once_with(expected_types)
                    mock_new_set.assert_called_once_with(expected_types)
                    mock_sem.assert_called_once()


def test_initialise_act_calls_init_sem_on_semantics(network):
    """Test that initialise_act calls init_sem on semantics."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'init_act'):
        with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_act'):
            with patch.object(network.sets[Set.NEW_SET].update_op, 'init_act'):
                with patch.object(network.semantics, 'init_sem') as mock_init_sem:
                    network.update_ops.initialise_act()
                    mock_init_sem.assert_called_once()


# =====================[ initialise_act_memory Tests ]======================

def test_initialise_act_memory_calls_update_op_init_act_on_memory(network):
    """Test that initialise_act_memory calls init_act on MEMORY set."""
    with patch.object(network.sets[Set.MEMORY].update_op, 'init_act') as mock_memory:
        network.update_ops.initialise_act_memory()
        
        expected_types = [Type.GROUP, Type.P, Type.RB, Type.PO]
        mock_memory.assert_called_once_with(expected_types)


def test_initialise_act_memory_does_not_call_other_sets(network):
    """Test that initialise_act_memory does not call init_act on other sets."""
    with patch.object(network.sets[Set.MEMORY].update_op, 'init_act'):
        with patch.object(network.sets[Set.DRIVER].update_op, 'init_act') as mock_driver:
            with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_act') as mock_recipient:
                with patch.object(network.sets[Set.NEW_SET].update_op, 'init_act') as mock_new_set:
                    network.update_ops.initialise_act_memory()
                    
                    mock_driver.assert_not_called()
                    mock_recipient.assert_not_called()
                    mock_new_set.assert_not_called()


# =====================[ acts Tests ]======================

def test_acts_calls_update_op_update_act_on_driver(network):
    """Test that acts calls update_act on DRIVER set."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'update_act') as mock_update_act:
        network.update_ops.acts(Set.DRIVER)
        mock_update_act.assert_called_once()


def test_acts_calls_update_op_update_act_on_recipient(network):
    """Test that acts calls update_act on RECIPIENT set."""
    with patch.object(network.sets[Set.RECIPIENT].update_op, 'update_act') as mock_update_act:
        network.update_ops.acts(Set.RECIPIENT)
        mock_update_act.assert_called_once()


def test_acts_calls_update_op_update_act_on_memory(network):
    """Test that acts calls update_act on MEMORY set."""
    with patch.object(network.sets[Set.MEMORY].update_op, 'update_act') as mock_update_act:
        network.update_ops.acts(Set.MEMORY)
        mock_update_act.assert_called_once()


def test_acts_calls_update_op_update_act_on_new_set(network):
    """Test that acts calls update_act on NEW_SET."""
    with patch.object(network.sets[Set.NEW_SET].update_op, 'update_act') as mock_update_act:
        network.update_ops.acts(Set.NEW_SET)
        mock_update_act.assert_called_once()


# =====================[ acts_sem Tests ]======================

def test_acts_sem_calls_update_act_on_semantics(network):
    """Test that acts_sem calls update_act on semantics."""
    with patch.object(network.semantics, 'update_act') as mock_update_act:
        network.update_ops.acts_sem()
        mock_update_act.assert_called_once()


# =====================[ acts_am Tests ]======================

def test_acts_am_calls_acts_on_all_active_sets(network):
    """Test that acts_am calls acts on DRIVER, RECIPIENT, and NEW_SET."""
    with patch.object(network.update_ops, 'acts') as mock_acts:
        with patch.object(network.update_ops, 'acts_sem') as mock_acts_sem:
            network.update_ops.acts_am()
            
            # Verify acts was called for each active set
            expected_calls = [call(Set.DRIVER), call(Set.RECIPIENT), call(Set.NEW_SET)]
            mock_acts.assert_has_calls(expected_calls, any_order=False)
            assert mock_acts.call_count == 3


def test_acts_am_calls_acts_sem(network):
    """Test that acts_am calls acts_sem."""
    with patch.object(network.update_ops, 'acts'):
        with patch.object(network.update_ops, 'acts_sem') as mock_acts_sem:
            network.update_ops.acts_am()
            mock_acts_sem.assert_called_once()


def test_acts_am_does_not_call_memory(network):
    """Test that acts_am does not call acts on MEMORY set."""
    with patch.object(network.update_ops, 'acts') as mock_acts:
        with patch.object(network.update_ops, 'acts_sem'):
            network.update_ops.acts_am()
            
            # Verify MEMORY was not called
            for call_args in mock_acts.call_args_list:
                assert call_args[0][0] != Set.MEMORY


# =====================[ initialise_input Tests ]======================

def test_initialise_input_calls_update_op_init_input_on_all_active_sets(network):
    """Test that initialise_input calls init_input on DRIVER, RECIPIENT, and NEW_SET."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'init_input') as mock_driver:
        with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_input') as mock_recipient:
            with patch.object(network.sets[Set.NEW_SET].update_op, 'init_input') as mock_new_set:
                with patch.object(network.semantics, 'init_input') as mock_sem:
                    network.update_ops.initialise_input()
                    
                    expected_types = [Type.GROUP, Type.P, Type.RB, Type.PO]
                    mock_driver.assert_called_once_with(expected_types, 0.0)
                    mock_recipient.assert_called_once_with(expected_types, 0.0)
                    mock_new_set.assert_called_once_with(expected_types, 0.0)


def test_initialise_input_calls_init_input_on_semantics(network):
    """Test that initialise_input calls init_input on semantics with 0.0."""
    with patch.object(network.sets[Set.DRIVER].update_op, 'init_input'):
        with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_input'):
            with patch.object(network.sets[Set.NEW_SET].update_op, 'init_input'):
                with patch.object(network.semantics, 'init_input') as mock_init_input:
                    network.update_ops.initialise_input()
                    mock_init_input.assert_called_once_with(0.0)


# =====================[ initialise_input_memory Tests ]======================

def test_initialise_input_memory_calls_update_op_init_input_on_memory(network):
    """Test that initialise_input_memory calls init_input on MEMORY set."""
    with patch.object(network.sets[Set.MEMORY].update_op, 'init_input') as mock_memory:
        network.update_ops.initialise_input_memory()
        
        expected_types = [Type.GROUP, Type.P, Type.RB, Type.PO]
        mock_memory.assert_called_once_with(expected_types, 0.0)


def test_initialise_input_memory_does_not_call_other_sets(network):
    """Test that initialise_input_memory does not call init_input on other sets."""
    with patch.object(network.sets[Set.MEMORY].update_op, 'init_input'):
        with patch.object(network.sets[Set.DRIVER].update_op, 'init_input') as mock_driver:
            with patch.object(network.sets[Set.RECIPIENT].update_op, 'init_input') as mock_recipient:
                with patch.object(network.sets[Set.NEW_SET].update_op, 'init_input') as mock_new_set:
                    network.update_ops.initialise_input_memory()
                    
                    mock_driver.assert_not_called()
                    mock_recipient.assert_not_called()
                    mock_new_set.assert_not_called()


# =====================[ inputs Tests ]======================

def test_inputs_calls_update_input_on_driver_with_no_args(network):
    """Test that inputs calls update_input on DRIVER with no arguments."""
    with patch.object(network.sets[Set.DRIVER], 'update_input') as mock_update_input:
        network.update_ops.inputs(Set.DRIVER)
        mock_update_input.assert_called_once_with()


def test_inputs_calls_update_input_on_new_set_with_no_args(network):
    """Test that inputs calls update_input on NEW_SET with no arguments."""
    with patch.object(network.sets[Set.NEW_SET], 'update_input') as mock_update_input:
        network.update_ops.inputs(Set.NEW_SET)
        mock_update_input.assert_called_once_with()


def test_inputs_calls_update_input_on_recipient_with_semantics_and_links(network):
    """Test that inputs calls update_input on RECIPIENT with semantics and links."""
    with patch.object(network.sets[Set.RECIPIENT], 'update_input') as mock_update_input:
        network.update_ops.inputs(Set.RECIPIENT)
        mock_update_input.assert_called_once_with(network.semantics, network.links)


def test_inputs_calls_update_input_on_memory_with_semantics_and_links(network):
    """Test that inputs calls update_input on MEMORY with semantics and links."""
    with patch.object(network.sets[Set.MEMORY], 'update_input') as mock_update_input:
        network.update_ops.inputs(Set.MEMORY)
        mock_update_input.assert_called_once_with(network.semantics, network.links)


# =====================[ inputs_sem Tests ]======================

def test_inputs_sem_calls_update_input_on_semantics(network):
    """Test that inputs_sem calls update_input on semantics with driver and recipient."""
    with patch.object(network.semantics, 'update_input') as mock_update_input:
        network.update_ops.inputs_sem()
        mock_update_input.assert_called_once_with(
            network.sets[Set.DRIVER], 
            network.sets[Set.RECIPIENT]
        )


def test_inputs_sem_passes_correct_set_objects(network):
    """Test that inputs_sem passes the correct set objects to semantics.update_input."""
    with patch.object(network.semantics, 'update_input') as mock_update_input:
        network.update_ops.inputs_sem()
        
        call_args = mock_update_input.call_args[0]
        assert call_args[0] is network.sets[Set.DRIVER]
        assert call_args[1] is network.sets[Set.RECIPIENT]


# =====================[ inputs_am Tests ]======================

def test_inputs_am_calls_inputs_on_all_active_sets(network):
    """Test that inputs_am calls inputs on DRIVER, RECIPIENT, and NEW_SET."""
    with patch.object(network.update_ops, 'inputs') as mock_inputs:
        with patch.object(network.update_ops, 'inputs_sem') as mock_inputs_sem:
            network.update_ops.inputs_am()
            
            # Verify inputs was called for each active set
            expected_calls = [call(Set.DRIVER), call(Set.RECIPIENT), call(Set.NEW_SET)]
            mock_inputs.assert_has_calls(expected_calls, any_order=False)
            assert mock_inputs.call_count == 3


def test_inputs_am_calls_inputs_sem(network):
    """Test that inputs_am calls inputs_sem."""
    with patch.object(network.update_ops, 'inputs'):
        with patch.object(network.update_ops, 'inputs_sem') as mock_inputs_sem:
            network.update_ops.inputs_am()
            mock_inputs_sem.assert_called_once()


def test_inputs_am_does_not_call_memory(network):
    """Test that inputs_am does not call inputs on MEMORY set."""
    with patch.object(network.update_ops, 'inputs') as mock_inputs:
        with patch.object(network.update_ops, 'inputs_sem'):
            network.update_ops.inputs_am()
            
            # Verify MEMORY was not called
            for call_args in mock_inputs.call_args_list:
                assert call_args[0][0] != Set.MEMORY


# =====================[ get_max_sem_input Tests ]======================

def test_get_max_sem_input_calls_semantics_get_max_input(network):
    """Test that get_max_sem_input calls get_max_input on semantics."""
    with patch.object(network.semantics, 'get_max_input', return_value=0.9) as mock_get_max:
        result = network.update_ops.get_max_sem_input()
        
        mock_get_max.assert_called_once()
        assert result == 0.9


def test_get_max_sem_input_returns_correct_value(network):
    """Test that get_max_sem_input returns the correct max input value."""
    expected_max = 1.5
    with patch.object(network.semantics, 'get_max_input', return_value=expected_max):
        result = network.update_ops.get_max_sem_input()
        assert result == expected_max


# =====================[ del_small_link Tests ]======================

def test_del_small_link_calls_links_del_small_link(network):
    """Test that del_small_link calls del_small_link on links."""
    threshold = 0.1
    with patch.object(network.links, 'del_small_link') as mock_del:
        network.update_ops.del_small_link(threshold)
        mock_del.assert_called_once_with(threshold)


def test_del_small_link_passes_threshold_correctly(network):
    """Test that del_small_link passes the threshold value correctly."""
    threshold = 0.25
    with patch.object(network.links, 'del_small_link') as mock_del:
        network.update_ops.del_small_link(threshold)
        
        call_args = mock_del.call_args[0]
        assert call_args[0] == threshold


# =====================[ round_big_link Tests ]======================

def test_round_big_link_calls_links_round_big_link(network):
    """Test that round_big_link calls round_big_link on links."""
    threshold = 0.9
    with patch.object(network.links, 'round_big_link') as mock_round:
        network.update_ops.round_big_link(threshold)
        mock_round.assert_called_once_with(threshold)


def test_round_big_link_passes_threshold_correctly(network):
    """Test that round_big_link passes the threshold value correctly."""
    threshold = 0.85
    with patch.object(network.links, 'round_big_link') as mock_round:
        network.update_ops.round_big_link(threshold)
        
        call_args = mock_round.call_args[0]
        assert call_args[0] == threshold


# =====================[ Integration Tests ]======================

def test_initialise_act_integration(network):
    """Integration test: initialise_act should reset activations."""
    # Set some non-zero activations first
    network.token_tensor.tensor[0, TF.ACT] = 0.5
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[0, TF.TYPE] = Type.PO.value
    network.recache()
    
    # This might fail if the underlying implementation has issues,
    # but the method should run without exceptions
    try:
        network.update_ops.initialise_act()
    except Exception as e:
        pytest.fail(f"initialise_act raised exception: {e}")


def test_acts_integration(network):
    """Integration test: acts should run without error."""
    # Set up a simple token in DRIVER
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[0, TF.TYPE] = Type.PO.value
    network.token_tensor.tensor[0, TF.ACT] = 0.5
    network.token_tensor.tensor[0, TF.TD_INPUT] = 0.1
    network.recache()
    
    try:
        network.update_ops.acts(Set.DRIVER)
    except Exception as e:
        pytest.fail(f"acts raised exception: {e}")


def test_workflow_initialise_then_update(network):
    """Integration test: typical workflow of initialise then update."""
    # Set up tokens in different sets
    network.token_tensor.tensor[0, TF.SET] = Set.DRIVER.value
    network.token_tensor.tensor[0, TF.TYPE] = Type.PO.value
    network.token_tensor.tensor[1, TF.SET] = Set.RECIPIENT.value
    network.token_tensor.tensor[1, TF.TYPE] = Type.PO.value
    network.recache()
    
    # Run typical workflow
    try:
        network.update_ops.initialise_act()
        network.update_ops.initialise_input()
        network.update_ops.acts_am()
    except Exception as e:
        pytest.fail(f"Workflow raised exception: {e}")


# =====================[ Edge Cases ]======================

def test_inputs_with_empty_driver_set(network):
    """Test that inputs handles empty DRIVER set gracefully."""
    # All tokens default to DRIVER in the fixture, but we can test that
    # the method doesn't crash even if update_input has no tokens to process
    with patch.object(network.sets[Set.DRIVER], 'update_input') as mock_update:
        network.update_ops.inputs(Set.DRIVER)
        mock_update.assert_called_once()


def test_get_max_sem_input_with_zero_semantics(network):
    """Test that get_max_sem_input handles semantics with zero input."""
    # Semantics are initialized with zeros in fixture
    result = network.update_ops.get_max_sem_input()
    assert result == 0.0


def test_del_small_link_with_zero_threshold(network):
    """Test del_small_link with zero threshold."""
    with patch.object(network.links, 'del_small_link') as mock_del:
        network.update_ops.del_small_link(0.0)
        mock_del.assert_called_once_with(0.0)


def test_round_big_link_with_one_threshold(network):
    """Test round_big_link with threshold of 1.0."""
    with patch.object(network.links, 'round_big_link') as mock_round:
        network.update_ops.round_big_link(1.0)
        mock_round.assert_called_once_with(1.0)

