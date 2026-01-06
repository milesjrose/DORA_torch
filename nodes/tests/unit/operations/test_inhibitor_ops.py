# nodes/unit_test/operations/test_inhibitor_ops.py
# Tests for InhibitorOperations class

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
from nodes.enums import Set, TF, SF, MappingFields, Type, B


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


# =====================[ Initialization Tests ]======================

def test_inhibitor_ops_initialization(network):
    """Test that InhibitorOperations initializes with correct default values."""
    assert network.inhibitor_ops.local == 0.0
    assert network.inhibitor_ops.glbal == 0.0
    assert network.inhibitor_ops.network is network


# =====================[ update() Tests ]======================

def test_update_calls_update_inhibitor_input_on_driver_and_recipient(network):
    """Test that update() calls update_inhibitor_input on driver and recipient."""
    with patch.object(network.driver().update_op, 'update_inhibitor_input') as mock_driver_input:
        with patch.object(network.recipient().update_op, 'update_inhibitor_input') as mock_recipient_input:
            with patch.object(network.driver().update_op, 'update_inhibitor_act'):
                with patch.object(network.recipient().update_op, 'update_inhibitor_act'):
                    network.inhibitor_ops.update()
    
    # Verify both driver and recipient had update_inhibitor_input called
    mock_driver_input.assert_called_once_with([Type.RB, Type.PO])
    mock_recipient_input.assert_called_once_with([Type.RB, Type.PO])


def test_update_in_DORA_mode_updates_driver_rb_po_and_recipient_po(network):
    """Test that update() in DORA mode updates driver RB/PO and recipient PO inhibitor act."""
    network.params.as_DORA = True
    
    with patch.object(network.driver().update_op, 'update_inhibitor_input'):
        with patch.object(network.recipient().update_op, 'update_inhibitor_input'):
            with patch.object(network.driver().update_op, 'update_inhibitor_act') as mock_driver_act:
                with patch.object(network.recipient().update_op, 'update_inhibitor_act') as mock_recipient_act:
                    network.inhibitor_ops.update()
    
    # In DORA mode: driver updates RB and PO, recipient updates PO only
    mock_driver_act.assert_called_once_with([Type.RB, Type.PO])
    mock_recipient_act.assert_called_once_with([Type.PO])


def test_update_not_in_DORA_mode_updates_driver_rb_only(network):
    """Test that update() not in DORA mode only updates driver RB inhibitor act."""
    network.params.as_DORA = False
    
    with patch.object(network.driver().update_op, 'update_inhibitor_input'):
        with patch.object(network.recipient().update_op, 'update_inhibitor_input'):
            with patch.object(network.driver().update_op, 'update_inhibitor_act') as mock_driver_act:
                with patch.object(network.recipient().update_op, 'update_inhibitor_act') as mock_recipient_act:
                    network.inhibitor_ops.update()
    
    # Not in DORA mode: only driver RB is updated
    mock_driver_act.assert_called_once_with([Type.RB])
    mock_recipient_act.assert_not_called()


# =====================[ reset() Tests ]======================

def test_reset_calls_reset_inhibitor_on_all_sets(network):
    """Test that reset() calls reset_inhibitor on all four sets."""
    with patch.object(network.driver().update_op, 'reset_inhibitor') as mock_driver:
        with patch.object(network.recipient().update_op, 'reset_inhibitor') as mock_recipient:
            with patch.object(network.memory().update_op, 'reset_inhibitor') as mock_memory:
                with patch.object(network.new_set().update_op, 'reset_inhibitor') as mock_new_set:
                    network.inhibitor_ops.reset()
    
    # All sets should have reset_inhibitor called with RB and PO types
    mock_driver.assert_called_once_with([Type.RB, Type.PO])
    mock_recipient.assert_called_once_with([Type.RB, Type.PO])
    mock_memory.assert_called_once_with([Type.RB, Type.PO])
    mock_new_set.assert_called_once_with([Type.RB, Type.PO])


# =====================[ check_local() Tests ]======================

def test_check_local_sets_local_to_1_when_driver_has_local_inhibition(network):
    """Test that check_local() sets local to 1.0 when driver reports local inhibition."""
    # Initial state
    assert network.inhibitor_ops.local == 0.0
    
    # Mock driver to return True for local inhibitor check
    with patch.object(network.driver(), 'check_local_inhibitor', return_value=True):
        network.inhibitor_ops.check_local()
    
    assert network.inhibitor_ops.local == 1.0


def test_check_local_keeps_local_at_0_when_no_local_inhibition(network):
    """Test that check_local() keeps local at 0.0 when driver reports no local inhibition."""
    # Initial state
    assert network.inhibitor_ops.local == 0.0
    
    # Mock driver to return False for local inhibitor check
    with patch.object(network.driver(), 'check_local_inhibitor', return_value=False):
        network.inhibitor_ops.check_local()
    
    assert network.inhibitor_ops.local == 0.0


def test_check_local_inhibitor_with_po_inhibitor_act_1(network):
    """Test actual local inhibitor check with PO token having inhibitor_act == 1.0."""
    # Set up a PO token in DRIVER with inhibitor_act = 1.0
    po_index = 5
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.TYPE, float(Type.PO))
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.INHIBITOR_ACT, 1.0)
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Check local inhibitor
    network.inhibitor_ops.check_local()
    
    assert network.inhibitor_ops.local == 1.0


# =====================[ fire_local() Tests ]======================

def test_fire_local_initializes_po_acts_and_semantics(network):
    """Test that fire_local() initializes PO acts for driver/recipient and semantics."""
    with patch.object(network.driver().update_op, 'init_act') as mock_driver_act:
        with patch.object(network.recipient().update_op, 'init_act') as mock_recipient_act:
            with patch.object(network.semantics, 'init_sem') as mock_init_sem:
                network.inhibitor_ops.fire_local()
    
    # Verify PO acts are initialized
    mock_driver_act.assert_called_once_with([Type.PO])
    mock_recipient_act.assert_called_once_with([Type.PO])
    # Verify semantics are initialized
    mock_init_sem.assert_called_once()


# =====================[ check_global() Tests ]======================

def test_check_global_sets_glbal_to_1_when_driver_has_global_inhibition(network):
    """Test that check_global() sets glbal to 1.0 when driver reports global inhibition."""
    # Initial state
    assert network.inhibitor_ops.glbal == 0.0
    
    # Mock driver to return True for global inhibitor check
    with patch.object(network.driver(), 'check_global_inhibitor', return_value=True):
        network.inhibitor_ops.check_global()
    
    assert network.inhibitor_ops.glbal == 1.0


def test_check_global_keeps_glbal_at_0_when_no_global_inhibition(network):
    """Test that check_global() keeps glbal at 0.0 when driver reports no global inhibition."""
    # Initial state
    assert network.inhibitor_ops.glbal == 0.0
    
    # Mock driver to return False for global inhibitor check
    with patch.object(network.driver(), 'check_global_inhibitor', return_value=False):
        network.inhibitor_ops.check_global()
    
    assert network.inhibitor_ops.glbal == 0.0


def test_check_global_inhibitor_with_rb_inhibitor_act_1(network):
    """Test actual global inhibitor check with RB token having inhibitor_act == 1.0."""
    # Set up an RB token in DRIVER with inhibitor_act = 1.0
    rb_index = 3
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.TYPE, float(Type.RB))
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.INHIBITOR_ACT, 1.0)
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Check global inhibitor
    network.inhibitor_ops.check_global()
    
    assert network.inhibitor_ops.glbal == 1.0


# =====================[ fire_global() Tests ]======================

def test_fire_global_initializes_all_token_acts_and_semantics(network):
    """Test that fire_global() initializes PO/RB/P acts for driver/recipient/memory and semantics."""
    with patch.object(network.driver().update_op, 'init_act') as mock_driver_act:
        with patch.object(network.recipient().update_op, 'init_act') as mock_recipient_act:
            with patch.object(network.memory().update_op, 'init_act') as mock_memory_act:
                with patch.object(network.semantics, 'init_sem') as mock_init_sem:
                    network.inhibitor_ops.fire_global()
    
    # Verify all token types are initialized for driver, recipient, memory
    mock_driver_act.assert_called_once_with([Type.PO, Type.RB, Type.P])
    mock_recipient_act.assert_called_once_with([Type.PO, Type.RB, Type.P])
    mock_memory_act.assert_called_once_with([Type.PO, Type.RB, Type.P])
    # Verify semantics are initialized
    mock_init_sem.assert_called_once()


# =====================[ Integration Tests ]======================

def test_inhibitor_update_accumulates_input(network):
    """Test that inhibitor input accumulates activation over multiple update calls."""
    # Set up tokens with some activation
    po_indices = torch.tensor([0, 1])
    rb_indices = torch.tensor([2, 3])
    
    # Set up PO tokens in DRIVER
    network.tokens.token_tensor.set_feature(po_indices, TF.TYPE, float(Type.PO))
    network.tokens.token_tensor.set_feature(po_indices, TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(po_indices, TF.ACT, 0.5)
    network.tokens.token_tensor.set_feature(po_indices, TF.DELETED, float(B.FALSE))
    
    # Set up RB tokens in DRIVER
    network.tokens.token_tensor.set_feature(rb_indices, TF.TYPE, float(Type.RB))
    network.tokens.token_tensor.set_feature(rb_indices, TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(rb_indices, TF.ACT, 0.3)
    network.tokens.token_tensor.set_feature(rb_indices, TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Get initial inhibitor input (should be 0)
    initial_po_inhib = network.tokens.token_tensor.get_feature(po_indices, TF.INHIBITOR_INPUT).clone()
    initial_rb_inhib = network.tokens.token_tensor.get_feature(rb_indices, TF.INHIBITOR_INPUT).clone()
    
    # Call update - this should add ACT to INHIBITOR_INPUT
    network.inhibitor_ops.update()
    
    # Check that inhibitor input increased
    new_po_inhib = network.tokens.token_tensor.get_feature(po_indices, TF.INHIBITOR_INPUT)
    new_rb_inhib = network.tokens.token_tensor.get_feature(rb_indices, TF.INHIBITOR_INPUT)
    
    # Inhibitor input should have increased by the activation amount
    assert torch.all(new_po_inhib > initial_po_inhib)
    assert torch.all(new_rb_inhib > initial_rb_inhib)


def test_reset_clears_inhibitor_state(network):
    """Test that reset clears inhibitor input and act values."""
    # Set up tokens with inhibitor values
    indices = torch.tensor([0, 1, 2])
    
    # Set up RB tokens with inhibitor values
    network.tokens.token_tensor.set_feature(indices, TF.TYPE, float(Type.RB))
    network.tokens.token_tensor.set_feature(indices, TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(indices, TF.INHIBITOR_INPUT, 100.0)
    network.tokens.token_tensor.set_feature(indices, TF.INHIBITOR_ACT, 1.0)
    network.tokens.token_tensor.set_feature(indices, TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Call reset
    network.inhibitor_ops.reset()
    
    # Get the values from the local view (driver)
    driver = network.driver()
    rb_mask = driver.tensor_op.get_mask(Type.RB)
    
    if torch.any(rb_mask):
        inhib_input = driver.lcl[rb_mask, TF.INHIBITOR_INPUT]
        inhib_act = driver.lcl[rb_mask, TF.INHIBITOR_ACT]
        
        # Both should be reset to 0
        assert torch.all(inhib_input == 0.0)
        assert torch.all(inhib_act == 0.0)


def test_full_inhibitor_cycle_local(network):
    """Test a full local inhibitor cycle: update until threshold reached, check, fire."""
    # Set up a PO token in DRIVER with high threshold that will be reached
    po_index = 5
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.TYPE, float(Type.PO))
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.ACT, 0.5)
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.INHIBITOR_THRESHOLD, 1.0)  # Low threshold
    network.tokens.token_tensor.set_feature(torch.tensor([po_index]), TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Reset inhibitors first
    network.inhibitor_ops.reset()
    
    # Update several times to accumulate inhibitor input
    for _ in range(5):  # 5 * 0.5 = 2.5 > threshold of 1.0
        network.inhibitor_ops.update()
    
    # Now check local - should trigger
    network.inhibitor_ops.check_local()
    
    # Local inhibitor should be 1.0
    assert network.inhibitor_ops.local == 1.0


def test_full_inhibitor_cycle_global(network):
    """Test a full global inhibitor cycle: update until threshold reached, check, fire."""
    # Set up an RB token in DRIVER with low threshold
    rb_index = 3
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.TYPE, float(Type.RB))
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.SET, float(Set.DRIVER))
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.ACT, 0.5)
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.INHIBITOR_THRESHOLD, 1.0)  # Low threshold
    network.tokens.token_tensor.set_feature(torch.tensor([rb_index]), TF.DELETED, float(B.FALSE))
    
    # Refresh cache
    network.cache_sets()
    
    # Reset inhibitors first
    network.inhibitor_ops.reset()
    
    # Update several times to accumulate inhibitor input
    for _ in range(5):  # 5 * 0.5 = 2.5 > threshold of 1.0
        network.inhibitor_ops.update()
    
    # Now check global - should trigger
    network.inhibitor_ops.check_global()
    
    # Global inhibitor should be 1.0
    assert network.inhibitor_ops.glbal == 1.0

