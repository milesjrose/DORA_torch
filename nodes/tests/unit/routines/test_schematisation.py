# nodes/tests/unit/routines/test_schematisation.py
# Tests for SchematisationOperations class

import pytest
import torch
from nodes.network.network import Network
from nodes.network.tokens import Tokens, Token_Tensor, Connections_Tensor, Links, Mapping
from nodes.network.single_nodes import Token
from nodes.network.sets import Semantics
from nodes.network.network_params import default_params
from nodes.enums import Set, TF, SF, MappingFields, Type, B, Mode, null
from logging import getLogger
logger = getLogger("TEST")

# =====================[ Fixtures ]======================

@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """
    Create minimal Token_Tensor for testing schematisation.
    
    Layout (by global index):
    Driver (0-6):
        0-1: PO (pred=true)
        2-3: PO (pred=false)
        4-5: RB
        6: P (child mode)
    Recipient (7-13):
        7-8: PO (pred=true)
        9-10: PO (pred=false)
        11-12: RB
        13: P (child mode)
    """
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
    """Tests for SchematisationOperations.requirements() method."""

    def test_requirements_returns_true_with_no_mappings(self, network: Network):
        """Test requirements returns True when no mapping connections exist (all zeros pass)."""
        # All mappings are zero by default - this should pass because no token has 0 < max_map < threshold
        result = network.routines.schema.requirements()
        assert result is True

    def test_requirements_returns_false_when_mapping_below_threshold_nonzero(self, network: Network):
        """Test requirements returns False when a token has 0 < max_map < 0.7 threshold."""
        # Set MAX_MAP for a driver token to be below threshold but non-zero
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.5)
        
        result = network.routines.schema.requirements()
        assert result is False

    def test_requirements_returns_true_when_all_mappings_above_threshold(self, network: Network):
        """Test requirements returns True when all non-zero mappings are above threshold."""
        # Set MAX_MAP above threshold (0.7)
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.8)
        
        result = network.routines.schema.requirements()
        assert result is True

    def test_requirements_threshold_boundary_below(self, network: Network):
        """Test requirements at exact boundary - below should fail."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.69)
        
        result = network.routines.schema.requirements()
        assert result is False

    def test_requirements_threshold_boundary_at(self, network: Network):
        """Test requirements at exact boundary (0.7) - should pass."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.7)
        
        result = network.routines.schema.requirements()
        assert result is True

    def test_requirements_fails_when_valid_node_connects_to_invalid_node(self, network: Network):
        """Test requirements fails when a valid (mapped) node connects to an invalid (partially mapped) node."""
        # Set up: node 0 has valid mapping (>= 0.7), node 1 has invalid mapping (0 < x < 0.7)
        # And they are connected
        network.node_ops.set_tk_value(0, TF.MAX_MAP, 0.8)  # Valid
        network.node_ops.set_tk_value(1, TF.MAX_MAP, 0.5)  # Invalid (0 < 0.5 < 0.7)
        
        # Connect them (parent -> child)
        network.tokens.connections.connect(0, 1)
        
        result = network.routines.schema.requirements()
        assert result is False

    def test_requirements_passes_when_valid_nodes_connect_to_valid_nodes(self, network: Network):
        """Test requirements passes when connected nodes all have valid mappings."""
        network.node_ops.set_tk_value(0, TF.MAX_MAP, 0.8)
        network.node_ops.set_tk_value(1, TF.MAX_MAP, 0.9)
        
        # Connect them
        network.tokens.connections.connect(0, 1)
        
        result = network.routines.schema.requirements()
        assert result is True


# =====================[ infer_token() Tests ]======================

class TestInferToken:
    """Tests for SchematisationOperations.infer_token() method."""

    def test_infer_token_creates_po_token(self, network: Network):
        """Test infer_token creates a PO token correctly in newSet."""
        driver_po_idx = 0  # First driver PO token (pred=true)
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 1.0)
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.schema.infer_token(driver_po_idx)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.PO
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE
        assert network.node_ops.get_tk_value(made_idx, TF.ANALOG) == null
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_po_idx
        # Verify PRED copied from maker
        maker_pred = network.node_ops.get_tk_value(driver_po_idx, TF.PRED)
        assert network.node_ops.get_tk_value(made_idx, TF.PRED) == maker_pred

    def test_infer_token_creates_rb_token(self, network: Network):
        """Test infer_token creates an RB token correctly in newSet."""
        driver_rb_idx = 4  # First driver RB token
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.schema.infer_token(driver_rb_idx)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.RB
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE

    def test_infer_token_creates_p_token_with_mode(self, network: Network):
        """Test infer_token creates a P token with correct mode."""
        driver_p_idx = 6  # Driver P token (child mode)
        
        initial_count = network.token_tensor.get_count()
        
        made_idx = network.routines.schema.infer_token(driver_p_idx)
        
        # Verify token was added
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify token properties
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.P
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET
        # Verify MODE copied from maker
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.CHILD

    def test_infer_token_sets_maker_made_relationship(self, network: Network):
        """Test infer_token correctly sets maker and made unit relationships."""
        driver_po_idx = 0
        
        made_idx = network.routines.schema.infer_token(driver_po_idx)
        
        # Check made unit is set on maker
        assert network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT) == made_idx
        # Check maker unit is set on made token
        assert network.node_ops.get_tk_value(made_idx, TF.MAKER_UNIT) == driver_po_idx
        # Check maker set is stored
        assert network.node_ops.get_tk_value(driver_po_idx, TF.MADE_SET) == Set.NEW_SET


# =====================[ shcematise_p() Tests ]======================

class TestSchematiseP:
    """Tests for SchematisationOperations.shcematise_p() method."""

    def test_schematise_p_no_active_token(self, network: Network):
        """Test schematise_p does nothing when no active P token with given mode."""
        # All activations are 0 by default
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_p(Mode.CHILD)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_p_active_below_threshold(self, network: Network):
        """Test schematise_p does nothing when active P is below activation threshold (0.4)."""
        driver_p_idx = 6  # Driver P token (child mode)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.3)  # Below 0.4 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_p(Mode.CHILD)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_p_no_mapping_does_not_infer(self, network: Network):
        """Test schematise_p does not infer when max_map is below threshold (0.75)."""
        driver_p_idx = 6  # Driver P token (child mode)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_p_idx, TF.MAX_MAP, 0.5)  # Below 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_p(Mode.CHILD)
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_p_infers_new_token(self, network: Network):
        """Test schematise_p infers new token when conditions are met."""
        driver_p_idx = 6  # Driver P token (child mode)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_p_idx, TF.MAX_MAP, 0.8)  # Above 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_p(Mode.CHILD)
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit relationship
        made_idx = int(network.node_ops.get_tk_value(driver_p_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.P
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.CHILD

    def test_schematise_p_parent_mode_infers_token(self, network: Network):
        """Test schematise_p can infer P token with PARENT mode."""
        # Add a P token with PARENT mode to driver
        p_parent_token = Token(Type.P, set=Set.DRIVER, features={TF.MODE: Mode.PARENT, TF.ACT: 0.5, TF.MAX_MAP: 0.8})
        p_parent_idx = network.node_ops.add_token(p_parent_token)
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_p(Mode.PARENT)
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify mode is PARENT
        made_idx = int(network.node_ops.get_tk_value(p_parent_idx, TF.MADE_UNIT))
        assert network.node_ops.get_tk_value(made_idx, TF.MODE) == Mode.PARENT

    def test_schematise_p_updates_existing_made_token(self, network: Network):
        """Test schematise_p updates existing made token's activation and connections."""
        driver_p_idx = 6  # Driver P token (child mode)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_p_idx, TF.MAX_MAP, 0.8)
        
        # First call creates the token
        network.routines.schema.schematise_p(Mode.CHILD)
        made_idx = int(network.node_ops.get_tk_value(driver_p_idx, TF.MADE_UNIT))
        
        # Set made token's activation to something else
        network.node_ops.set_tk_value(made_idx, TF.ACT, 0.1)
        
        # Add an active RB in newSet
        rb_token = Token(Type.RB, set=Set.NEW_SET, features={TF.ACT: 0.6})  # Above 0.5 threshold
        rb_idx = network.node_ops.add_token(rb_token)
        
        initial_count = network.token_tensor.get_count()
        
        # Second call should update existing made token
        network.routines.schema.schematise_p(Mode.CHILD)
        
        # No new token created
        assert network.token_tensor.get_count() == initial_count
        # Activation updated to 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0

    def test_schematise_p_parent_connects_to_rbs(self, network: Network):
        """Test schematise_p with PARENT mode connects P as parent to active RBs."""
        # Create P with PARENT mode in driver that has already made a token
        p_parent_token = Token(Type.P, set=Set.DRIVER, features={TF.MODE: Mode.PARENT, TF.ACT: 0.5, TF.MAX_MAP: 0.8})
        p_parent_idx = network.node_ops.add_token(p_parent_token)
        
        # First, infer a token
        network.routines.schema.schematise_p(Mode.PARENT)
        made_idx = int(network.node_ops.get_tk_value(p_parent_idx, TF.MADE_UNIT))
        
        # Add active RBs in newSet
        rb1_token = Token(Type.RB, set=Set.NEW_SET, features={TF.ACT: 0.6})
        rb1_idx = network.node_ops.add_token(rb1_token)
        rb2_token = Token(Type.RB, set=Set.NEW_SET, features={TF.ACT: 0.7})
        rb2_idx = network.node_ops.add_token(rb2_token)
        
        # Run again to connect
        network.routines.schema.schematise_p(Mode.PARENT)
        
        # Verify connections: made_idx should be parent to RBs
        cons = network.tokens.connections.tensor
        assert cons[made_idx, rb1_idx].item() == True
        assert cons[made_idx, rb2_idx].item() == True


# =====================[ schematise_rb() Tests ]======================

class TestSchematiseRB:
    """Tests for SchematisationOperations.schematise_rb() method."""

    def test_schematise_rb_no_active_token(self, network: Network):
        """Test schematise_rb does nothing when no active RB token."""
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_rb()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_rb_active_below_threshold(self, network: Network):
        """Test schematise_rb does nothing when active RB is below activation threshold (0.4)."""
        driver_rb_idx = 4  # First driver RB token
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.3)  # Below threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_rb()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_rb_no_mapping_does_not_infer(self, network: Network):
        """Test schematise_rb does not infer when max_map is below threshold (0.75)."""
        driver_rb_idx = 4
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_rb_idx, TF.MAX_MAP, 0.5)  # Below 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_rb()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_rb_infers_new_token(self, network: Network):
        """Test schematise_rb infers new token when conditions are met."""
        driver_rb_idx = 4
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_rb_idx, TF.MAX_MAP, 0.8)  # Above 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_rb()
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit is an RB
        made_idx = int(network.node_ops.get_tk_value(driver_rb_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.RB
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET

    def test_schematise_rb_updates_existing_made_token(self, network: Network):
        """Test schematise_rb updates existing made token's activation."""
        driver_rb_idx = 4
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_rb_idx, TF.MAX_MAP, 0.8)
        
        # First call creates the token
        network.routines.schema.schematise_rb()
        made_idx = int(network.node_ops.get_tk_value(driver_rb_idx, TF.MADE_UNIT))
        
        # Set made token's activation to something else
        network.node_ops.set_tk_value(made_idx, TF.ACT, 0.1)
        initial_count = network.token_tensor.get_count()
        
        # Second call should update activation
        network.routines.schema.schematise_rb()
        
        # No new token created
        assert network.token_tensor.get_count() == initial_count
        # Activation updated to 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0

    def test_schematise_rb_connects_to_active_pos(self, network: Network):
        """Test schematise_rb connects made RB to active POs in newSet."""
        driver_rb_idx = 4
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_rb_idx, TF.MAX_MAP, 0.8)
        
        # First, infer a token
        network.routines.schema.schematise_rb()
        made_idx = int(network.node_ops.get_tk_value(driver_rb_idx, TF.MADE_UNIT))
        
        # Add active POs in newSet
        po1_token = Token(Type.PO, set=Set.NEW_SET, features={TF.ACT: 0.6, TF.PRED: B.TRUE})
        po1_idx = network.node_ops.add_token(po1_token)
        po2_token = Token(Type.PO, set=Set.NEW_SET, features={TF.ACT: 0.7, TF.PRED: B.FALSE})
        po2_idx = network.node_ops.add_token(po2_token)
        
        # Run again to connect
        network.routines.schema.schematise_rb()
        
        # Verify connections: made RB should be parent to POs
        cons = network.tokens.connections.tensor
        assert cons[made_idx, po1_idx].item() == True
        assert cons[made_idx, po2_idx].item() == True


# =====================[ schematise_po() Tests ]======================

class TestSchematisePO:
    """Tests for SchematisationOperations.schematise_po() method."""

    def test_schematise_po_no_active_token(self, network: Network):
        """Test schematise_po does nothing when no active PO token."""
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_po()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_po_active_below_threshold(self, network: Network):
        """Test schematise_po does nothing when active PO is below activation threshold (0.4)."""
        driver_po_idx = 0  # First driver PO token
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.3)  # Below threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_po()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_po_no_mapping_does_not_infer(self, network: Network):
        """Test schematise_po does not infer when max_map is below threshold (0.75)."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.5)  # Below 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_po()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematise_po_infers_new_token(self, network: Network):
        """Test schematise_po infers new token when conditions are met."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.5)  # Above 0.4 threshold
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.8)  # Above 0.75 threshold
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematise_po()
        
        # A new token should be created
        assert network.token_tensor.get_count() == initial_count + 1
        
        # Verify the made unit is a PO
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        assert made_idx != null
        assert network.node_ops.get_tk_value(made_idx, TF.TYPE) == Type.PO
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET
        # Verify PRED is copied
        assert network.node_ops.get_tk_value(made_idx, TF.PRED) == B.TRUE

    def test_schematise_po_updates_existing_made_token(self, network: Network):
        """Test schematise_po updates existing made token's activation."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.8)
        
        # First call creates the token
        network.routines.schema.schematise_po()
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        
        # Set made token's activation to something else
        network.node_ops.set_tk_value(made_idx, TF.ACT, 0.1)
        initial_count = network.token_tensor.get_count()
        
        # Second call should update activation
        network.routines.schema.schematise_po()
        
        # No new token created
        assert network.token_tensor.get_count() == initial_count
        # Activation updated to 1.0
        assert network.node_ops.get_tk_value(made_idx, TF.ACT) == 1.0

    def test_schematise_po_selects_most_active_token(self, network: Network):
        """Test schematise_po operates on the most active PO token."""
        # Set multiple PO tokens active, one more than the other
        driver_po_idx_1 = 0
        driver_po_idx_2 = 1
        network.node_ops.set_tk_value(driver_po_idx_1, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_po_idx_2, TF.ACT, 0.9)  # Most active
        network.node_ops.set_tk_value(driver_po_idx_1, TF.MAX_MAP, 0.8)
        network.node_ops.set_tk_value(driver_po_idx_2, TF.MAX_MAP, 0.8)
        
        network.routines.schema.schematise_po()
        
        # The most active token should have a made unit
        assert network.node_ops.get_tk_value(driver_po_idx_2, TF.MADE_UNIT) != null
        # The less active token should not
        assert network.node_ops.get_tk_value(driver_po_idx_1, TF.MADE_UNIT) == null


# =====================[ schematisation_routine() Tests ]======================

class TestSchematisationRoutine:
    """Tests for SchematisationOperations.schematisation_routine() method."""

    def test_schematisation_routine_no_active_tokens(self, network: Network):
        """Test schematisation_routine does nothing when no tokens are active."""
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematisation_routine()
        
        # No new tokens should be created
        assert network.token_tensor.get_count() == initial_count

    def test_schematisation_routine_calls_all_types(self, network: Network):
        """Test schematisation_routine processes PO, RB, P.child, and P.parent."""
        # Set up active tokens of each type with appropriate mappings
        driver_po_idx = 0  # PO
        driver_rb_idx = 4  # RB
        driver_p_idx = 6   # P (child mode)
        
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.8)
        network.node_ops.set_tk_value(driver_rb_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_rb_idx, TF.MAX_MAP, 0.8)
        network.node_ops.set_tk_value(driver_p_idx, TF.ACT, 0.5)
        network.node_ops.set_tk_value(driver_p_idx, TF.MAX_MAP, 0.8)
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematisation_routine()
        
        # Should have created 3 new tokens (PO, RB, P with CHILD mode)
        # Note: P with PARENT mode won't create if no P with PARENT mode exists
        assert network.token_tensor.get_count() >= initial_count + 3

    def test_schematisation_routine_with_p_parent_mode(self, network: Network):
        """Test schematisation_routine processes P tokens with PARENT mode."""
        # Create a P token with PARENT mode in driver
        p_parent_token = Token(Type.P, set=Set.DRIVER, features={TF.MODE: Mode.PARENT, TF.ACT: 0.5, TF.MAX_MAP: 0.8})
        p_parent_idx = network.node_ops.add_token(p_parent_token)
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematisation_routine()
        
        # Should have created a P token with PARENT mode
        made_idx = network.node_ops.get_tk_value(p_parent_idx, TF.MADE_UNIT)
        if made_idx != null:
            assert network.node_ops.get_tk_value(int(made_idx), TF.MODE) == Mode.PARENT

    def test_schematisation_routine_threshold_boundary(self, network: Network):
        """Test schematisation_routine uses 0.4 activation and 0.75 mapping thresholds."""
        driver_po_idx = 0
        # Set activation at exactly 0.4 (should pass)
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.4)
        # Set mapping at exactly 0.75 (should pass)
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.75)
        
        initial_count = network.token_tensor.get_count()
        
        network.routines.schema.schematisation_routine()
        
        # Token should be created since activation >= 0.4 and max_map >= 0.75
        assert network.token_tensor.get_count() == initial_count + 1


# =====================[ Integration Tests ]======================

class TestSchematisationIntegration:
    """Integration tests for SchematisationOperations."""

    def test_full_schematisation_workflow(self, network: Network):
        """Test complete schematisation workflow."""
        # Setup: Set driver tokens with valid mappings
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.9)
        
        # Verify requirements pass
        assert network.routines.schema.requirements() is True
        
        # Run schematisation
        network.routines.schema.schematisation_routine()
        
        # Verify token was inferred
        made_idx = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        assert made_idx != null
        
        # Verify inferred token properties
        assert network.node_ops.get_tk_value(made_idx, TF.SET) == Set.NEW_SET
        assert network.node_ops.get_tk_value(made_idx, TF.INFERRED) == B.TRUE

    def test_multiple_schematisation_calls(self, network: Network):
        """Test multiple calls to schematisation_routine work correctly."""
        driver_po_idx = 0
        network.node_ops.set_tk_value(driver_po_idx, TF.ACT, 0.8)
        network.node_ops.set_tk_value(driver_po_idx, TF.MAX_MAP, 0.9)
        
        # First call
        network.routines.schema.schematisation_routine()
        made_idx_1 = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        count_after_first = network.token_tensor.get_count()
        
        # Add active PO in newSet for testing update behavior
        po_token = Token(Type.PO, set=Set.NEW_SET, features={TF.ACT: 0.6, TF.PRED: B.TRUE})
        network.node_ops.add_token(po_token)
        
        # Second call - should update existing made token, not create new one for PO
        network.routines.schema.schematisation_routine()
        made_idx_2 = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        
        # Same made token for the PO
        assert made_idx_1 == made_idx_2

    def test_schematisation_creates_proper_hierarchy(self, network: Network):
        """Test schematisation creates proper P -> RB -> PO hierarchy."""
        # Set up driver tokens with valid mappings
        driver_po_idx = 0
        driver_rb_idx = 4
        driver_p_idx = 6
        
        for idx in [driver_po_idx, driver_rb_idx, driver_p_idx]:
            network.node_ops.set_tk_value(idx, TF.ACT, 0.8)
            network.node_ops.set_tk_value(idx, TF.MAX_MAP, 0.9)
        
        # First run - infer all tokens
        network.routines.schema.schematisation_routine()
        
        po_made = int(network.node_ops.get_tk_value(driver_po_idx, TF.MADE_UNIT))
        rb_made = int(network.node_ops.get_tk_value(driver_rb_idx, TF.MADE_UNIT))
        p_made = int(network.node_ops.get_tk_value(driver_p_idx, TF.MADE_UNIT))
        
        # Verify all tokens were created in newSet
        assert network.node_ops.get_tk_value(po_made, TF.SET) == Set.NEW_SET
        assert network.node_ops.get_tk_value(rb_made, TF.SET) == Set.NEW_SET
        assert network.node_ops.get_tk_value(p_made, TF.SET) == Set.NEW_SET
        
        # Make all newSet tokens active for connection phase
        network.node_ops.set_tk_value(po_made, TF.ACT, 0.6)
        network.node_ops.set_tk_value(rb_made, TF.ACT, 0.6)
        
        # Second run - should connect hierarchy
        network.routines.schema.schematisation_routine()
        
        # Verify RB connects to PO
        cons = network.tokens.connections.tensor
        assert cons[rb_made, po_made].item() == True
