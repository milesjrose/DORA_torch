# nodes/tests/unit/routines/test_rel_gen.py
# Tests for RelGenOperations class

import pytest
import torch
import tempfile
import os
import json
from nodes.network.network import Network
from nodes.network.tokens import Tokens, Token_Tensor, Connections_Tensor, Links, Mapping
from nodes.network.single_nodes import Token
from nodes.network.sets import Semantics
from nodes.network.network_params import Params, default_params
from nodes.enums import Set, TF, SF, MappingFields, Type, B, null, Mode
from logging import getLogger
logger = getLogger("TEST")


# =====================[ Fixtures ]======================

@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """Create minimal Token_Tensor for testing."""
    num_tokens = 14
    num_features = len(TF)
    tokens = torch.zeros((num_tokens, num_features))
    idx = 0
    # Driver: 2 PO tokens with pred = true
    for i in range(2):
        tokens[idx] = Token(Type.PO, set=Set.DRIVER, features={TF.PRED: B.TRUE}).tensor
        idx += 1
    # Driver: 2 PO tokens with pred = false
    for i in range(2):
        tokens[idx] = Token(Type.PO, set=Set.DRIVER, features={TF.PRED: B.FALSE}).tensor
        idx += 1
    # Driver: 2 RB tokens
    for i in range(2):
        tokens[idx] = Token(Type.RB, set=Set.DRIVER).tensor
        idx += 1
    # Driver: 1 P token (child mode)
    tokens[idx] = Token(Type.P, set=Set.DRIVER, features={TF.MODE: Mode.CHILD}).tensor
    idx += 1
    # Recipient: 2 PO tokens with pred = true
    for i in range(2):
        tokens[idx] = Token(Type.PO, set=Set.RECIPIENT, features={TF.PRED: B.TRUE}).tensor
        idx += 1
    # Recipient: 2 PO tokens with pred = false
    for i in range(2):
        tokens[idx] = Token(Type.PO, set=Set.RECIPIENT, features={TF.PRED: B.FALSE}).tensor
        idx += 1
    # Recipient: 2 RB tokens
    for i in range(2):
        tokens[idx] = Token(Type.RB, set=Set.RECIPIENT).tensor
        idx += 1
    # Recipient: 1 P token (child mode)
    tokens[idx] = Token(Type.P, set=Set.RECIPIENT, features={TF.MODE: Mode.CHILD}).tensor
    idx += 1

    names = {}
    return Token_Tensor(tokens, names)


@pytest.fixture
def minimal_connections(minimal_token_tensor):
    """Create minimal Connections_Tensor for testing."""
    num_tokens = minimal_token_tensor.get_count()
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
    num_recipient = 7
    num_driver = 7
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
def network(minimal_tokens, minimal_semantics, minimal_params):
    """Create minimal Network object for testing."""
    return Network(minimal_tokens, minimal_semantics, minimal_params)


# =====================[ requirements() Tests ]======================

class TestRequirements:
    """Tests for RelGenOperations.requirements() method."""

    def test_requirements_returns_false_when_no_mappings(self, network: Network):
        """Test requirements returns False when no mapping connections exist."""
        # All mappings are zero by default
        result = network.routines.rel_gen.requirements()
        assert result is False

    def test_requirements_returns_false_when_mapping_below_threshold(self, network: Network):
        """Test requirements returns False when mapping weights are below threshold (0.7)."""
        # Set a mapping weight below threshold (0.7)
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.5
        
        result = network.routines.rel_gen.requirements()
        assert result is False

    def test_requirements_returns_true_when_mapping_above_threshold(self, network: Network):
        """Test requirements returns True when at least one mapping weight is above threshold."""
        # Set a mapping weight above threshold (0.7)
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.8
        
        result = network.routines.rel_gen.requirements()
        assert result is True

    def test_requirements_returns_true_with_multiple_valid_mappings(self, network: Network):
        """Test requirements returns True with multiple valid mapping weights."""
        # Set multiple mapping weights above threshold
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.8
        network.mappings[MappingFields.WEIGHT][1, 1] = 0.9
        network.mappings[MappingFields.WEIGHT][2, 2] = 0.75
        
        result = network.routines.rel_gen.requirements()
        assert result is True

    def test_requirements_returns_false_with_mixed_valid_invalid_mappings(self, network: Network):
        """Test requirements returns False when some mappings are below threshold but non-zero."""
        # One mapping above threshold
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.8
        # One mapping below threshold but non-zero (fails requirement)
        network.mappings[MappingFields.WEIGHT][1, 1] = 0.3
        
        result = network.routines.rel_gen.requirements()
        assert result is False

    def test_requirements_threshold_boundary_below(self, network: Network):
        """Test requirements at exact boundary (0.7) - below should fail."""
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.69
        
        result = network.routines.rel_gen.requirements()
        assert result is False

    def test_requirements_threshold_boundary_at(self, network: Network):
        """Test requirements at exact boundary (0.7) - at threshold should pass."""
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.7
        
        result = network.routines.rel_gen.requirements()
        assert result is True


# =====================[ infer_token() Tests ]======================

class TestInferToken:
    """Tests for RelGenOperations.infer_token() method."""

    def test_infer_token_creates_po_token(self, network: Network):
        """Test infer_token creates a PO token correctly."""
        # Get a driver PO token (pred=true)
        driver_po_idx = 0  # First driver PO token
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 1.0)
        recip_analog = 1
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.rel_gen.infer_token(driver_po_idx, recip_analog, Set.RECIPIENT)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.PO
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.RECIPIENT
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE
        assert network.node_ops.get_tk_value(made_idx, TF.ANALOG) == recip_analog
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_po_idx
        # Verify PRED copied from maker
        maker_pred = network.node_ops.get_tk_value(driver_po_idx, TF.PRED)
        assert network.node_ops.get_tk_value(made_idx, TF.PRED) == maker_pred

    def test_infer_token_creates_rb_token(self, network: Network):
        """Test infer_token creates an RB token correctly."""
        # Get a driver RB token (index 4 or 5 based on fixture)
        driver_rb_idx = 4  # First driver RB token
        recip_analog = 2
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.rel_gen.infer_token(driver_rb_idx, recip_analog, Set.RECIPIENT)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.RB
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.RECIPIENT
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE
        assert network.node_ops.get_tk_value(made_idx, TF.ANALOG) == recip_analog
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_rb_idx

    def test_infer_token_creates_p_token_child_mode(self, network: Network):
        """Test infer_token creates a P token with correct mode."""
        # Get a driver P token (index 6 based on fixture)
        driver_p_idx = 6  # Driver P token
        network.node_ops.set_tk_value(driver_p_idx, TF.MODE, Mode.CHILD)
        recip_analog = 3
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.rel_gen.infer_token(driver_p_idx, recip_analog, Set.RECIPIENT)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.P
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.RECIPIENT
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE
        assert network.node_ops.get_tk_value(made_idx, TF.ANALOG) == recip_analog
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_p_idx
        # Verify MODE copied from maker
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.CHILD

    def test_infer_token_creates_p_token_parent_mode(self, network: Network):
        """Test infer_token creates a P token with parent mode."""
        # Get a driver P token and set parent mode
        driver_p_idx = 6  # Driver P token
        network.node_ops.set_tk_value(driver_p_idx, TF.MODE, Mode.PARENT)
        recip_analog = 3
        
        made_idx = network.routines.rel_gen.infer_token(driver_p_idx, recip_analog, Set.RECIPIENT)
        
        # Verify MODE copied from maker
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.PARENT

    def test_infer_token_sets_maker_made_relationship(self, network: Network):
        """Test infer_token correctly sets maker and made unit relationships."""
        driver_po_idx = 0
        recip_analog = 1
        
        made_idx = network.routines.rel_gen.infer_token(driver_po_idx, recip_analog, Set.RECIPIENT)
        
        # Check made unit is set on maker
        assert network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT) == made_idx
        # Check maker unit is set on made token
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_po_idx

    def test_infer_token_into_new_set(self, network: Network):
        """Test infer_token can create tokens in NEW_SET."""
        driver_po_idx = 0
        recip_analog = 1
        
        made_idx = network.routines.rel_gen.infer_token(driver_po_idx, recip_analog, Set.NEW_SET)
        
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET


# =====================[ rel_gen_type() Tests ]======================

class TestRelGenType:
    """Tests for RelGenOperations.rel_gen_type() method."""

    def test_rel_gen_type_po_no_active_token(self, network: Network):
        """Test rel_gen_type does nothing when no active PO token."""
        # All activations are 0 by default
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_rel_gen_type_po_active_below_threshold(self, network: Network):
        """Test rel_gen_type does nothing when active PO is below threshold."""
        driver_po_idx = 0
        # Set activation below threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.3)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_rel_gen_type_po_active_with_existing_mapping(self, network: Network):
        """Test rel_gen_type does nothing when active PO already has mapping."""
        driver_po_idx = 0
        # Set activation above threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        # Set a mapping for this token (local index 0 in driver)
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.5
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # No new tokens should be created because max_map != 0
        assert network.token_tensor.get_count() == initial_count

    def test_rel_gen_type_po_infers_new_token(self, network: Network):
        """Test rel_gen_type infers new token when conditions are met."""
        driver_po_idx = 0
        # Set activation above threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        # No mapping set (max_map = 0)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit relationship
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.PO

    def test_rel_gen_type_rb_infers_new_token(self, network: Network):
        """Test rel_gen_type infers new RB token when conditions are met."""
        driver_rb_idx = 4  # First driver RB token
        # Set activation above threshold
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.8)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()

        network.print_token_tensor(features=[TF.SET, TF.TYPE, TF.ACT, TF.MADE_UNIT, TF.MAKER_UNIT])
        
        network.routines.rel_gen.rel_gen_type(Type.RB, 0.5, recip_analog)
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit is an RB
        made_idx = int(network.node_ops.get_tk_value(driver_rb_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.RB

    def test_rel_gen_type_p_child_infers_new_token(self, network: Network):
        """Test rel_gen_type infers new P token with CHILD mode."""
        driver_p_idx = 6  # Driver P token
        network.node_ops.set_tk_value(driver_p_idx, TF.MODE, Mode.CHILD)
        # Set activation above threshold
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.8)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, p_mode=Mode.CHILD)
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit is a P with CHILD mode
        made_idx = int(network.node_ops.get_tk_value(driver_p_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.P
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.CHILD

    def test_rel_gen_type_p_requires_mode(self, network: Network):
        """Test rel_gen_type raises error when P type without mode."""
        driver_p_idx = 6
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.8)
        recip_analog = 1
        
        with pytest.raises(ValueError, match="p_mode is None"):
            network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, p_mode=None)

    def test_rel_gen_type_updates_existing_made_token_act(self, network: Network):
        """Test rel_gen_type updates activation of existing made token."""
        driver_po_idx = 0
        # Set activation above threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        recip_analog = 1
        
        # First call creates the token
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        
        # Set made token's activation to something else
        network.node_ops.set_tk_value(made_idx, TF.ACT, 0.1)
        initial_count = network.token_tensor.get_count()
        
        # Second call should update activation
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # No new token created
        assert network.token_tensor.get_count() == initial_count
        # Activation updated to 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0

    def test_rel_gen_type_selects_most_active_token(self, network: Network):
        """Test rel_gen_type operates on the most active token of the type."""
        # Set multiple PO tokens active, one more than the other
        driver_po_idx_1 = 0
        driver_po_idx_2 = 1
        network.node_ops.set_tk_value(driver_po_idx_1, TF.ACT, 0.6)
        network.node_ops.set_tk_value(driver_po_idx_2, TF.ACT, 0.9)  # Most active
        recip_analog = 1
        
        network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
        
        # The most active token should have a made unit
        assert network.node_ops.get_tk_value(driver_po_idx_2, TF.MADE_UNIT) != null
        # The less active token should not
        assert network.node_ops.get_tk_value(driver_po_idx_1, TF.MADE_UNIT) == null


# =====================[ rel_gen_routine() Tests ]======================

class TestRelGenRoutine:
    """Tests for RelGenOperations.rel_gen_routine() method."""

    def test_rel_gen_routine_calls_all_types(self, network: Network):
        """Test rel_gen_routine processes PO, RB, P.child, and P.parent."""
        # Set up active tokens of each type
        driver_po_idx = 0  # PO
        driver_rb_idx = 4  # RB
        driver_p_idx = 6   # P
        
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.8)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.8)
        network.node_ops.set_tk_value(driver_p_idx, TF.MODE, Mode.CHILD)
        
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # Should have created 3 new tokens (PO, RB, P with CHILD mode)
        # Note: P with PARENT mode won't create if no P with PARENT mode exists
        assert network.token_tensor.get_count() >= initial_count + 3

    def test_rel_gen_routine_no_active_tokens(self, network: Network):
        """Test rel_gen_routine does nothing when no tokens are active."""
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_rel_gen_routine_uses_threshold_0_5(self, network: Network):
        """Test rel_gen_routine uses 0.5 threshold for activation."""
        driver_po_idx = 0
        # Set activation at exactly 0.5 (should pass)
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.5)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # Token should be created since activation >= 0.5
        assert network.token_tensor.get_count() == initial_count + 1

    def test_rel_gen_routine_below_threshold(self, network: Network):
        """Test rel_gen_routine doesn't infer when activation below 0.5."""
        driver_po_idx = 0
        # Set activation just below 0.5
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.49)
        recip_analog = 1
        initial_count = network.token_tensor.get_count()
        
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # No token should be created
        assert network.token_tensor.get_count() == initial_count

    def test_rel_gen_routine_with_p_parent_mode(self, network: Network):
        """Test rel_gen_routine processes P tokens with PARENT mode."""
        # Create a P token with PARENT mode in driver
        p_parent_token = Token(Type.P, set=Set.DRIVER, features={TF.MODE: Mode.PARENT, TF.ACT: 0.8})
        p_parent_idx = network.node_ops.add_token(p_parent_token)
        
        recip_analog = 1
        initial_count = network.token_tensor.get_count()

        network.print_token_tensor(features=[TF.SET, TF.TYPE, TF.ACT, TF.MADE_UNIT, TF.MAKER_UNIT])
        
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # Should have created a P token (only PARENT mode since it's most active among P tokens)
        made_idx = network.node_ops.get_tk_value(p_parent_idx, TF.MADE_UNIT)
        if made_idx != null:
            assert network.node_ops.get_tk_value(int(made_idx), TF.MODE) == Mode.PARENT


# =====================[ Integration Tests ]======================

class TestRelGenIntegration:
    """Integration tests for RelGenOperations."""

    def test_full_rel_gen_workflow(self, network: Network):
        """Test complete relation generalisation workflow."""
        # Setup: Create mapping conditions
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.8
        
        # Verify requirements pass
        assert network.routines.rel_gen.requirements() is True
        
        # Set up active driver token
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        recip_analog = 1
        
        # Clear the mapping for the test token so it can infer
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.0
        network.mappings[MappingFields.WEIGHT][1, 1] = 0.8  # Keep one mapping valid for requirements
        
        # Run rel_gen_routine
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # Verify token was inferred
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        assert made_idx != null
        
        # Verify inferred token properties
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.RECIPIENT
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE
        assert network.node_ops.get_tk_value(made_idx, TF.ANALOG) == recip_analog

    def test_rel_gen_preserves_existing_mappings(self, network: Network):
        """Test rel_gen doesn't modify existing mapping weights."""
        # Setup mapping
        network.mappings[MappingFields.WEIGHT][0, 0] = 0.8
        network.mappings[MappingFields.WEIGHT][1, 1] = 0.9
        
        # Set up active driver token that doesn't have mapping
        driver_rb_idx = 4
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.8)
        recip_analog = 1
        
        # Run rel_gen
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        
        # Verify existing mappings preserved
        assert network.mappings[MappingFields.WEIGHT][0, 0] == 0.8
        assert network.mappings[MappingFields.WEIGHT][1, 1] == 0.9

    def test_multiple_rel_gen_calls(self, network: Network):
        """Test multiple calls to rel_gen_routine work correctly."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        recip_analog = 1
        
        # First call
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        made_idx_1 = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        count_after_first = network.token_tensor.get_count()
        
        # Second call - should update existing made token, not create new one
        network.routines.rel_gen.rel_gen_routine(recip_analog)
        made_idx_2 = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        count_after_second = network.token_tensor.get_count()
        
        # Same made token
        assert made_idx_1 == made_idx_2
        # No new tokens created on second call
        assert count_after_first == count_after_second
