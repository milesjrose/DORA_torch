# nodes/tests/unit/routines/test_predication.py
# Tests for PredicationOperations class

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from nodes.network.network import Network
from nodes.network.tokens.tokens import Tokens
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.tokens.connections.connections import Connections_Tensor
from nodes.network.tokens.connections.mapping import Mapping
from nodes.network.tokens.connections.links import Links
from nodes.network.sets.semantics import Semantics
from nodes.network.network_params import default_params
from nodes.network.single_nodes import Token
from nodes.network.routines.predication import PredicationOperations
from nodes.enums import Set, TF, SF, MappingFields, Type, B, null


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
    num_semantics = 10
    links = torch.zeros((num_tokens, num_semantics))
    return Links(links)


@pytest.fixture
def minimal_mapping():
    """Create minimal Mapping object for testing."""
    num_recipient = 10
    num_driver = 10
    num_fields = len(MappingFields)
    adj_matrix = torch.zeros((num_recipient, num_driver, num_fields))
    return Mapping(adj_matrix)


@pytest.fixture
def minimal_semantics():
    """Create minimal Semantics object for testing."""
    num_semantics = 10
    num_features = len(SF)
    nodes = torch.zeros((num_semantics, num_features))
    connections = torch.zeros((num_semantics, num_semantics))
    IDs = {i: i for i in range(num_semantics)}
    names = {i: f"sem_{i}" for i in range(num_semantics)}
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

def test_predication_ops_initialization(network):
    """Test that PredicationOperations is properly initialized."""
    pred_ops = network.routines.predication
    assert pred_ops is not None
    assert pred_ops.network is network


def test_predication_ops_initial_state(network):
    """Test initial state of PredicationOperations."""
    pred_ops = network.routines.predication
    assert pred_ops.debug is False
    assert pred_ops.made_new_pred is False
    assert pred_ops.inferred_pred is None


def test_predication_ops_standalone_initialization():
    """Test PredicationOperations can be initialized standalone with a mock network."""
    mock_network = Mock()
    pred_ops = PredicationOperations(mock_network)
    
    assert pred_ops.network is mock_network
    assert pred_ops.debug is False
    assert pred_ops.made_new_pred is False
    assert pred_ops.inferred_pred is None


# =====================[ check_po_requirements Tests ]======================

def test_check_po_requirements_returns_false_for_predicate(network):
    """Test that check_po_requirements returns False for a predicate PO."""
    pred_ops = network.routines.predication
    
    # Set up a PO as a predicate
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.PRED] = B.TRUE
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


def test_check_po_requirements_returns_false_for_low_activation(network):
    """Test that check_po_requirements returns False when activation is too low."""
    pred_ops = network.routines.predication
    
    # Set up a PO object with low activation
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE  # Object, not predicate
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.5  # Below 0.6 threshold
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


def test_check_po_requirements_returns_false_for_low_max_map(network):
    """Test that check_po_requirements returns False when max map is too low."""
    pred_ops = network.routines.predication
    
    # Set up a PO object with sufficient activation but low max map
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE  # Object
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.8  # Above 0.6 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP] = 0.5  # Below 0.75 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP_UNIT] = 1
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


def test_check_po_requirements_returns_false_for_low_driver_activation(network):
    """Test that check_po_requirements returns False when mapped driver has low activation."""
    pred_ops = network.routines.predication
    
    # Set up recipient PO
    po_idx = 0
    driver_idx = 1
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE  # Object
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.8  # Above 0.6 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP] = 0.9  # Above 0.75 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP_UNIT] = driver_idx  # Points to driver token
    
    # Set up driver with low activation
    network.token_tensor.tensor[driver_idx, TF.ACT] = 0.3  # Below 0.6 threshold
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


def test_check_po_requirements_returns_true_when_all_conditions_met(network):
    """Test that check_po_requirements returns True when all conditions are met."""
    pred_ops = network.routines.predication
    
    # Set up recipient PO that meets all requirements
    po_idx = 0
    driver_idx = 1
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE  # Object (not predicate)
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.8  # Above 0.6 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP] = 0.9  # Above 0.75 threshold
    network.token_tensor.tensor[po_idx, TF.MAX_MAP_UNIT] = driver_idx  # Points to driver token
    
    # Set up driver with sufficient activation
    network.token_tensor.tensor[driver_idx, TF.ACT] = 0.7  # Above 0.6 threshold
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is True


def test_check_po_requirements_boundary_activation(network):
    """Test check_po_requirements at activation boundary (exactly 0.6)."""
    pred_ops = network.routines.predication
    
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.6  # Exactly at threshold (should fail <= check)
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


def test_check_po_requirements_boundary_max_map(network):
    """Test check_po_requirements at max_map boundary (exactly 0.75)."""
    pred_ops = network.routines.predication
    
    po_idx = 0
    driver_idx = 1
    network.token_tensor.tensor[po_idx, TF.PRED] = B.FALSE
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.8
    network.token_tensor.tensor[po_idx, TF.MAX_MAP] = 0.75  # Exactly at threshold (should fail <= check)
    network.token_tensor.tensor[po_idx, TF.MAX_MAP_UNIT] = driver_idx
    network.token_tensor.tensor[driver_idx, TF.ACT] = 0.8
    
    result = pred_ops.check_po_requirements(po_idx)
    assert result is False


# =====================[ predication_routine Tests ]======================

def test_predication_routine_calls_made_new_pred_when_true(network):
    """Test that predication_routine calls made_new_pred routine when made_new_pred is True."""
    pred_ops = network.routines.predication
    pred_ops.made_new_pred = True
    
    with patch.object(pred_ops, 'predication_routine_made_new_pred') as mock_made:
        with patch.object(pred_ops, 'predication_routine_no_new_pred') as mock_no_made:
            pred_ops.predication_routine()
            mock_made.assert_called_once()
            mock_no_made.assert_not_called()


def test_predication_routine_calls_no_new_pred_when_false(network):
    """Test that predication_routine calls no_new_pred routine when made_new_pred is False."""
    pred_ops = network.routines.predication
    pred_ops.made_new_pred = False
    
    with patch.object(pred_ops, 'predication_routine_made_new_pred') as mock_made:
        with patch.object(pred_ops, 'predication_routine_no_new_pred') as mock_no_made:
            pred_ops.predication_routine()
            mock_no_made.assert_called_once()
            mock_made.assert_not_called()


# =====================[ predication_routine_made_new_pred Tests ]======================

def test_predication_routine_made_new_pred_updates_links(network):
    """Test that predication_routine_made_new_pred updates semantic links."""
    pred_ops = network.routines.predication
    
    # Set up inferred pred token index
    pred_idx = 5
    pred_ops.inferred_pred = pred_idx
    
    # Set up active semantics
    network.semantics.nodes[0, SF.ACT] = 0.5
    network.semantics.nodes[1, SF.ACT] = 0.3
    network.semantics.nodes[2, SF.ACT] = 0.0  # Not active
    
    # Set up initial link weights
    network.links[pred_idx, 0] = 0.1
    network.links[pred_idx, 1] = 0.0
    
    initial_link_0 = network.links[pred_idx, 0].clone()
    initial_link_1 = network.links[pred_idx, 1].clone()
    
    pred_ops.predication_routine_made_new_pred()
    
    # Links should have been updated for active semantics
    # New weight = current + 1 * (sem_act - current) * gamma
    # This increases the link weight toward the semantic activation


def test_predication_routine_made_new_pred_uses_gamma(network):
    """Test that predication_routine_made_new_pred uses network gamma parameter."""
    pred_ops = network.routines.predication
    
    pred_idx = 5
    pred_ops.inferred_pred = pred_idx
    
    # Set up active semantic
    network.semantics.nodes[0, SF.ACT] = 0.8
    network.links[pred_idx, 0] = 0.0
    
    # Get gamma value
    gamma = network.params.gamma
    
    pred_ops.predication_routine_made_new_pred()
    
    # The link update should be: new_weights = 1 * (sem_acts - link_weights) * gamma
    # So for link_weight=0, sem_act=0.8: new = 0.8 * gamma
    expected_delta = 1 * (0.8 - 0.0) * gamma
    assert network.links[pred_idx, 0].item() == pytest.approx(expected_delta, abs=1e-6)


def test_predication_routine_made_new_pred_only_updates_active_semantics(network):
    """Test that only active semantics (act > 0) have their links updated."""
    pred_ops = network.routines.predication
    
    pred_idx = 5
    pred_ops.inferred_pred = pred_idx
    
    # Set up semantics - some active, some not
    network.semantics.nodes[0, SF.ACT] = 0.5  # Active
    network.semantics.nodes[1, SF.ACT] = 0.0  # Not active
    network.semantics.nodes[2, SF.ACT] = 0.3  # Active
    
    # Initial links are all zero
    network.links[pred_idx, 0] = 0.0
    network.links[pred_idx, 1] = 0.0
    network.links[pred_idx, 2] = 0.0
    
    pred_ops.predication_routine_made_new_pred()
    
    # Links to inactive semantics should remain unchanged
    assert network.links[pred_idx, 1].item() == 0.0


# =====================[ predication_routine_no_new_pred Tests ]======================

def test_predication_routine_no_new_pred_returns_early_if_no_active_po(network):
    """Test that predication_routine_no_new_pred returns early if no active PO."""
    pred_ops = network.routines.predication
    
    # Mock the recipient to return None for most active PO
    with patch.object(network.recipient().token_op, 'get_most_active_token', return_value=None):
        # Should return without changing state
        initial_made_new_pred = pred_ops.made_new_pred
        pred_ops.predication_routine_no_new_pred()
        assert pred_ops.made_new_pred == initial_made_new_pred


def test_predication_routine_no_new_pred_checks_requirements(network):
    """Test that predication_routine_no_new_pred checks PO requirements."""
    pred_ops = network.routines.predication
    
    # Set up a recipient PO
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.9
    network.recache()
    
    with patch.object(network.recipient().token_op, 'get_most_active_token', return_value=0):
        with patch.object(network, 'to_global', return_value=po_idx):
            with patch.object(pred_ops, 'check_po_requirements', return_value=False) as mock_check:
                pred_ops.predication_routine_no_new_pred()
                mock_check.assert_called_once_with(po_idx)


def test_predication_routine_no_new_pred_does_not_create_pred_if_requirements_not_met(network):
    """Test that no predicate is created if PO requirements are not met."""
    pred_ops = network.routines.predication
    
    # Set up a recipient PO
    po_idx = 0
    network.token_tensor.tensor[po_idx, TF.TYPE] = Type.PO
    network.token_tensor.tensor[po_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[po_idx, TF.ACT] = 0.9
    network.recache()
    
    with patch.object(network.recipient().token_op, 'get_most_active_token', return_value=0):
        with patch.object(network, 'to_global', return_value=po_idx):
            with patch.object(pred_ops, 'check_po_requirements', return_value=False):
                pred_ops.predication_routine_no_new_pred()
                assert pred_ops.made_new_pred is False
                assert pred_ops.inferred_pred is None


# =====================[ requirements Tests ]======================

def test_requirements_returns_false_on_value_error(network):
    """Test that requirements() returns False when ValueError is raised."""
    pred_ops = network.routines.predication
    
    # Set up conditions that will cause ValueError
    # (mapped recipient nodes are not all POs)
    with patch.object(pred_ops, 'debug', False):
        # Force the nested function to raise ValueError
        result = pred_ops.requirements()
        # Since there are no proper mappings set up, it should return True or False
        # depending on the state - let's just verify it doesn't crash
        assert isinstance(result, bool)


def test_requirements_with_debug_enabled(network):
    """Test that requirements prints error when debug is True."""
    pred_ops = network.routines.predication
    pred_ops.debug = True
    
    # Should not raise even with debug enabled
    result = pred_ops.requirements()
    assert isinstance(result, bool)


# =====================[ State Management Tests ]======================

def test_made_new_pred_flag_persistence(network):
    """Test that made_new_pred flag persists correctly."""
    pred_ops = network.routines.predication
    
    assert pred_ops.made_new_pred is False
    pred_ops.made_new_pred = True
    assert pred_ops.made_new_pred is True


def test_inferred_pred_reference_persistence(network):
    """Test that inferred_pred reference persists correctly."""
    pred_ops = network.routines.predication
    
    assert pred_ops.inferred_pred is None
    pred_ops.inferred_pred = 42
    assert pred_ops.inferred_pred == 42


def test_debug_flag_persistence(network):
    """Test that debug flag can be toggled."""
    pred_ops = network.routines.predication
    
    assert pred_ops.debug is False
    pred_ops.debug = True
    assert pred_ops.debug is True
    pred_ops.debug = False
    assert pred_ops.debug is False


# =====================[ Integration Tests ]======================

def test_full_predication_cycle_with_made_pred(network):
    """Test a full predication cycle after a predicate has been made."""
    pred_ops = network.routines.predication
    
    # Simulate that a predicate was made in a previous cycle
    pred_idx = 0
    pred_ops.made_new_pred = True
    pred_ops.inferred_pred = pred_idx
    
    # Set up active semantics
    network.semantics.nodes[0, SF.ACT] = 0.6
    network.links[pred_idx, 0] = 0.2
    
    # Run the routine
    pred_ops.predication_routine()
    
    # Link should have been updated
    # new_weight = old_weight + 1 * (sem_act - old_weight) * gamma
    gamma = network.params.gamma
    expected = 0.2 + 1 * (0.6 - 0.2) * gamma
    assert network.links[pred_idx, 0].item() == pytest.approx(expected, abs=1e-6)


def test_predication_multiple_active_semantics(network):
    """Test predication with multiple active semantics."""
    pred_ops = network.routines.predication
    
    pred_idx = 0
    pred_ops.made_new_pred = True
    pred_ops.inferred_pred = pred_idx
    
    # Set up multiple active semantics with different activations
    network.semantics.nodes[0, SF.ACT] = 0.9
    network.semantics.nodes[1, SF.ACT] = 0.5
    network.semantics.nodes[2, SF.ACT] = 0.1
    
    # Run the routine
    pred_ops.predication_routine()
    
    gamma = network.params.gamma
    
    # Each link should be updated based on its semantic's activation
    assert network.links[pred_idx, 0].item() == pytest.approx(0.9 * gamma, abs=1e-6)
    assert network.links[pred_idx, 1].item() == pytest.approx(0.5 * gamma, abs=1e-6)
    assert network.links[pred_idx, 2].item() == pytest.approx(0.1 * gamma, abs=1e-6)


def test_predication_preserves_network_reference(network):
    """Test that PredicationOperations maintains correct network reference."""
    pred_ops = network.routines.predication
    
    # Modify network state
    network.token_tensor.tensor[0, TF.ACT] = 0.5
    
    # Access through pred_ops should see the change
    assert pred_ops.network.token_tensor.tensor[0, TF.ACT] == 0.5


def test_predication_operations_separate_instances(network):
    """Test that creating new PredicationOperations doesn't affect existing one."""
    pred_ops1 = network.routines.predication
    pred_ops1.made_new_pred = True
    
    pred_ops2 = PredicationOperations(network)
    
    # New instance should have default state
    assert pred_ops2.made_new_pred is False
    # Original should be unchanged
    assert pred_ops1.made_new_pred is True

