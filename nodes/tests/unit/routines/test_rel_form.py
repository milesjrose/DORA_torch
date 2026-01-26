# nodes/tests/unit/routines/test_rel_form.py
# Tests for RelFormOperations class

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
from nodes.network.routines.rel_form import RelFormOperations
from nodes.enums import Set, TF, SF, MappingFields, Type, B, null
from logging import getLogger
logger = getLogger(__name__)


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

def test_rel_form_ops_initialization(network):
    """Test that RelFormOperations is properly initialized."""
    rel_form_ops = network.routines.rel_form
    assert rel_form_ops is not None
    assert rel_form_ops.network is network


def test_rel_form_ops_initial_state(network):
    """Test initial state of RelFormOperations."""
    rel_form_ops = network.routines.rel_form
    assert rel_form_ops.debug is False
    assert rel_form_ops.inferred_new_p is False
    assert rel_form_ops.inferred_p is None


def test_rel_form_ops_standalone_initialization():
    """Test RelFormOperations can be initialized standalone with a mock network."""
    mock_network = Mock()
    rel_form_ops = RelFormOperations(mock_network)
    
    assert rel_form_ops.network is mock_network
    assert rel_form_ops.debug is False
    assert rel_form_ops.inferred_new_p is False
    assert rel_form_ops.inferred_p is None


# =====================[ requirements Tests ]======================

def test_requirements_returns_false_with_less_than_2_rbs_in_recipient(network):
    """Test that requirements returns False when there are less than 2 RBs in recipient."""
    rel_form_ops = network.routines.rel_form
    
    # Set up only 1 RB in recipient
    network.token_tensor.tensor[0, TF.TYPE] = Type.RB
    network.token_tensor.tensor[0, TF.SET] = Set.RECIPIENT
    network.recache()
    
    result = rel_form_ops.requirements()
    assert result is False


def test_requirements_returns_false_when_rbs_have_parent_p(network):
    """Test that requirements returns False when RBs already have parent P connections."""
    rel_form_ops = network.routines.rel_form
    
    # Set up 3 RBs in recipient
    for i in range(3):
        network.token_tensor.tensor[i, TF.TYPE] = Type.RB
        network.token_tensor.tensor[i, TF.SET] = Set.RECIPIENT
        network.token_tensor.tensor[i, TF.ID] = i
    
    # Set up a P in recipient
    network.token_tensor.tensor[3, TF.TYPE] = Type.P
    network.token_tensor.tensor[3, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[3, TF.ID] = 3
    
    # Connect P to all RBs (P is parent of RBs)
    network.token_tensor.connections.connect(3, 0)
    network.token_tensor.connections.connect(3, 1)
    network.token_tensor.connections.connect(3, 2)
    
    network.recache()
    
    result = rel_form_ops.requirements()
    assert result is False


def test_requirements_returns_bool(network):
    """Test that requirements() always returns a boolean."""
    rel_form_ops = network.routines.rel_form
    result = rel_form_ops.requirements()
    assert isinstance(result, bool)


# =====================[ rel_form_routine Tests - No New P Inferred ]======================

def test_rel_form_routine_creates_new_p_when_not_inferred(network):
    """Test that rel_form_routine creates a new P token when inferred_new_p is False."""
    rel_form_ops = network.routines.rel_form
    rel_form_ops.inferred_new_p = False
    
    initial_count = network.token_tensor.get_count()
    
    rel_form_ops.rel_form_routine()
    
    # Should have inferred a new P
    assert rel_form_ops.inferred_new_p is True
    assert rel_form_ops.inferred_p is not None
    
    # Verify the new P token properties
    new_p_idx = rel_form_ops.inferred_p
    assert network.token_tensor.tensor[new_p_idx, TF.TYPE] == Type.P
    assert network.token_tensor.tensor[new_p_idx, TF.SET] == Set.RECIPIENT
    assert network.token_tensor.tensor[new_p_idx, TF.INFERRED] == B.TRUE


def test_rel_form_routine_sets_inferred_p_flag(network):
    """Test that rel_form_routine sets the inferred_new_p flag to True after creating P."""
    rel_form_ops = network.routines.rel_form
    rel_form_ops.inferred_new_p = False
    
    assert rel_form_ops.inferred_new_p is False
    
    rel_form_ops.rel_form_routine()
    
    assert rel_form_ops.inferred_new_p is True


def test_rel_form_routine_stores_inferred_p_index(network):
    """Test that rel_form_routine stores the index of the newly created P token."""
    rel_form_ops = network.routines.rel_form
    rel_form_ops.inferred_new_p = False
    rel_form_ops.inferred_p = None
    
    rel_form_ops.rel_form_routine()
    
    assert rel_form_ops.inferred_p is not None
    assert isinstance(rel_form_ops.inferred_p, int)


# =====================[ rel_form_routine Tests - New P Already Inferred ]======================

def test_rel_form_routine_connects_p_to_active_rbs_when_inferred(network):
    """Test that rel_form_routine connects existing P to active RBs when inferred_new_p is True."""
    rel_form_ops = network.routines.rel_form
    
    # Set up RBs in recipient with high activation
    for i in range(3):
        network.token_tensor.tensor[i, TF.TYPE] = Type.RB
        network.token_tensor.tensor[i, TF.SET] = Set.RECIPIENT
        network.token_tensor.tensor[i, TF.ID] = i
    
    # First RB has activation above threshold
    network.token_tensor.tensor[0, TF.ACT] = 0.9
    # Second RB below threshold
    network.token_tensor.tensor[1, TF.ACT] = 0.5
    # Third RB above threshold
    network.token_tensor.tensor[2, TF.ACT] = 0.85
    
    network.recache()
    
    # First, create a new P
    rel_form_ops.rel_form_routine()
    
    # Verify P was created
    assert rel_form_ops.inferred_new_p is True
    inferred_p_idx = rel_form_ops.inferred_p
    
    # Now run again to connect RBs to the P
    rel_form_ops.rel_form_routine()
    
    # Check connections - P should be connected to RBs with act >= 0.8
    children = network.token_tensor.connections.get_children(inferred_p_idx)
    
    # Should have connected to RBs at indices 0 and 2 (act >= 0.8)
    assert len(children) >= 1  # At least one RB should be connected


def test_rel_form_routine_raises_when_inferred_p_not_set(network):
    """Test that rel_form_routine raises ValueError when inferred_new_p is True but inferred_p is None."""
    rel_form_ops = network.routines.rel_form
    rel_form_ops.inferred_new_p = True
    rel_form_ops.inferred_p = None
    
    with pytest.raises(ValueError, match="Inferred P is not set"):
        rel_form_ops.rel_form_routine()


def test_rel_form_routine_logs_critical_when_no_rbs_to_connect(network):
    """Test that rel_form_routine logs critical when no RBs meet activation threshold."""
    rel_form_ops = network.routines.rel_form
    
    # Set up an RB with low activation
    network.token_tensor.tensor[0, TF.TYPE] = Type.RB
    network.token_tensor.tensor[0, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[0, TF.ACT] = 0.3  # Below 0.8 threshold
    network.token_tensor.tensor[0, TF.ID] = 0
    
    network.recache()
    
    # First create a P
    rel_form_ops.rel_form_routine()
    
    # Now try to connect - should return without error but log critical
    # (No RBs above threshold to connect)
    rel_form_ops.rel_form_routine()  # Should not raise


# =====================[ name_inferred_p Tests ]======================

def test_name_inferred_p_raises_when_inferred_p_not_set(network):
    """Test that name_inferred_p raises ValueError when inferred_p is None."""
    rel_form_ops = network.routines.rel_form
    rel_form_ops.inferred_p = None
    
    with pytest.raises(ValueError, match="Inferred P is not set"):
        rel_form_ops.name_inferred_p()


def test_name_inferred_p_raises_when_p_has_no_rbs(network):
    """Test that name_inferred_p raises ValueError when P has no RB children."""
    rel_form_ops = network.routines.rel_form
    
    # Create a P token manually
    network.token_tensor.tensor[0, TF.TYPE] = Type.P
    network.token_tensor.tensor[0, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[0, TF.ID] = 0
    network.recache()
    
    rel_form_ops.inferred_p = 0
    
    with pytest.raises(ValueError):
        rel_form_ops.name_inferred_p()


def test_name_inferred_p_creates_name_from_single_rb(network):
    """Test that name_inferred_p creates name from a single connected RB."""
    rel_form_ops = network.routines.rel_form
    
    # Set up P token
    p_idx = 0
    network.token_tensor.tensor[p_idx, TF.TYPE] = Type.P
    network.token_tensor.tensor[p_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[p_idx, TF.ID] = p_idx
    
    # Set up RB token
    rb_idx = 1
    network.token_tensor.tensor[rb_idx, TF.TYPE] = Type.RB
    network.token_tensor.tensor[rb_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[rb_idx, TF.ID] = rb_idx
    network.set_name(rb_idx, "RB_test")
    
    # Connect P to RB
    network.token_tensor.connections.connect(p_idx, rb_idx)
    network.recache()
    
    rel_form_ops.inferred_p = p_idx
    rel_form_ops.name_inferred_p()
    
    # Name should be the RB's name
    assert network.get_name(p_idx) == "RB_test"


def test_name_inferred_p_creates_name_from_multiple_rbs(network):
    """Test that name_inferred_p creates name from multiple connected RBs with + separator."""
    rel_form_ops = network.routines.rel_form
    
    # Set up P token
    p_idx = 0
    network.token_tensor.tensor[p_idx, TF.TYPE] = Type.P
    network.token_tensor.tensor[p_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[p_idx, TF.ID] = p_idx
    
    # Set up first RB token
    rb1_idx = 1
    network.token_tensor.tensor[rb1_idx, TF.TYPE] = Type.RB
    network.token_tensor.tensor[rb1_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[rb1_idx, TF.ID] = rb1_idx
    network.set_name(rb1_idx, "RB_first")
    
    # Set up second RB token
    rb2_idx = 2
    network.token_tensor.tensor[rb2_idx, TF.TYPE] = Type.RB
    network.token_tensor.tensor[rb2_idx, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[rb2_idx, TF.ID] = rb2_idx
    network.set_name(rb2_idx, "RB_second")
    
    # Connect P to both RBs
    network.token_tensor.connections.connect(p_idx, rb1_idx)
    network.token_tensor.connections.connect(p_idx, rb2_idx)
    network.recache()
    
    rel_form_ops.inferred_p = p_idx
    rel_form_ops.name_inferred_p()
    
    # Name should contain both RB names with + separator
    p_name = network.get_name(p_idx)
    assert "RB_first" in p_name
    assert "RB_second" in p_name
    assert "+" in p_name


# =====================[ State Management Tests ]======================

def test_inferred_new_p_flag_persistence(network):
    """Test that inferred_new_p flag persists correctly."""
    rel_form_ops = network.routines.rel_form
    
    assert rel_form_ops.inferred_new_p is False
    rel_form_ops.inferred_new_p = True
    assert rel_form_ops.inferred_new_p is True


def test_inferred_p_reference_persistence(network):
    """Test that inferred_p reference persists correctly."""
    rel_form_ops = network.routines.rel_form
    
    assert rel_form_ops.inferred_p is None
    rel_form_ops.inferred_p = 42
    assert rel_form_ops.inferred_p == 42


def test_debug_flag_persistence(network):
    """Test that debug flag can be toggled."""
    rel_form_ops = network.routines.rel_form
    
    assert rel_form_ops.debug is False
    rel_form_ops.debug = True
    assert rel_form_ops.debug is True
    rel_form_ops.debug = False
    assert rel_form_ops.debug is False


# =====================[ Integration Tests ]======================

def test_full_rel_form_cycle_creates_and_connects_p(network):
    """Test a full relation formation cycle: create P, then connect to RBs."""
    rel_form_ops = network.routines.rel_form
    
    # Set up RBs in recipient with activation
    for i in range(3):
        network.token_tensor.tensor[i, TF.TYPE] = Type.RB
        network.token_tensor.tensor[i, TF.SET] = Set.RECIPIENT
        network.token_tensor.tensor[i, TF.ID] = i
        network.token_tensor.tensor[i, TF.ACT] = 0.9  # Above threshold
        network.set_name(i, f"RB_{i}")
    
    network.recache()
    
    # Phase 1: Create new P
    rel_form_ops.rel_form_routine()
    assert rel_form_ops.inferred_new_p is True
    assert rel_form_ops.inferred_p is not None
    
    p_idx = rel_form_ops.inferred_p
    
    # Verify P was created in recipient with INFERRED flag
    assert network.token_tensor.tensor[p_idx, TF.TYPE] == Type.P
    assert network.token_tensor.tensor[p_idx, TF.SET] == Set.RECIPIENT
    assert network.token_tensor.tensor[p_idx, TF.INFERRED] == B.TRUE
    
    # Phase 2: Connect P to active RBs
    rel_form_ops.rel_form_routine()
    
    # Check P is connected to RBs
    children = network.token_tensor.connections.get_children(p_idx)
    assert len(children) > 0


def test_rel_form_preserves_network_reference(network):
    """Test that RelFormOperations maintains correct network reference."""
    rel_form_ops = network.routines.rel_form
    
    # Modify network state
    network.token_tensor.tensor[0, TF.ACT] = 0.5
    
    # Access through rel_form_ops should see the change
    assert rel_form_ops.network.token_tensor.tensor[0, TF.ACT] == 0.5


def test_rel_form_operations_separate_instances(network):
    """Test that creating new RelFormOperations doesn't affect existing one."""
    rel_form_ops1 = network.routines.rel_form
    rel_form_ops1.inferred_new_p = True
    
    rel_form_ops2 = RelFormOperations(network)
    
    # New instance should have default state
    assert rel_form_ops2.inferred_new_p is False
    # Original should be unchanged
    assert rel_form_ops1.inferred_new_p is True


# =====================[ Edge Cases ]======================

def test_requirements_with_empty_network(network):
    """Test requirements() on a network with no tokens set up."""
    rel_form_ops = network.routines.rel_form
    # With no RBs in recipient, should return False
    result = rel_form_ops.requirements()
    assert result is False


def test_rel_form_routine_threshold_boundary(network):
    """Test rel_form_routine with RB activation exactly at threshold (0.8)."""
    rel_form_ops = network.routines.rel_form
    
    # Set up RB with exactly 0.8 activation
    network.token_tensor.tensor[0, TF.TYPE] = Type.RB
    network.token_tensor.tensor[0, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[0, TF.ACT] = 0.8  # Exactly at threshold
    network.token_tensor.tensor[0, TF.ID] = 0
    network.set_name(0, "RB_boundary")
    
    network.recache()
    
    # Create P first
    rel_form_ops.rel_form_routine()
    
    # Try to connect - should connect since act >= threshold
    rel_form_ops.rel_form_routine()
    
    # Should have connected (0.8 >= 0.8)
    p_idx = rel_form_ops.inferred_p
    children = network.token_tensor.connections.get_children(p_idx)
    assert len(children) > 0


def test_rel_form_routine_below_threshold(network):
    """Test rel_form_routine with RB activation just below threshold."""
    rel_form_ops = network.routines.rel_form
    network: 'Network' = network
    # Set up RB with activation below threshold
    network.token_tensor.tensor[0, TF.TYPE] = Type.RB
    network.token_tensor.tensor[0, TF.SET] = Set.RECIPIENT
    network.token_tensor.tensor[0, TF.ACT] = 0.79  # Just below threshold
    network.token_tensor.tensor[0, TF.ID] = 0
    
    network.recache()
    
    # Create P first
    rel_form_ops.rel_form_routine()
    
    # Try to connect - should log critical and not connect
    rel_form_ops.rel_form_routine()

    network.recache()
    logger.debug(f"tk_con_obj: {network.tokens.connections}, tk_tens_con_obj: {network.token_tensor.connections}")
    logger.debug(f"tk_cons: {network.tokens.connections.tensor.size()}")
    logger.debug(f"connections: {network.token_tensor.connections.tensor.size()}")
    
    # P should have no children (RB below threshold)
    p_idx = rel_form_ops.inferred_p
    children = network.token_tensor.connections.get_children(p_idx)
    assert len(children) == 0

