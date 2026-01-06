# nodes/tests/funct/test_builder.py
# Functional tests for the new network builder

import pytest
import torch
import os
import tempfile

from nodes.builder_new import NetworkBuilder, build_network
from nodes.network import Network
from nodes.enums import Set, Type, TF, SF, B, Mode, MappingFields, null


# =====================[ Test Data / Fixtures ]======================

@pytest.fixture
def simple_sym_props():
    """Simple symProps with one driver and one recipient proposition."""
    return [
        {
            'name': 'lovesMaryTom',
            'RBs': [
                {
                    'pred_name': 'lover',
                    'pred_sem': ['lover1', 'lover2'],
                    'higher_order': False,
                    'object_name': 'Mary',
                    'object_sem': ['mary1', 'mary2'],
                    'P': 'non_exist'
                }
            ],
            'set': 'driver',
            'analog': 0
        },
        {
            'name': 'lovesJohnKathy',
            'RBs': [
                {
                    'pred_name': 'lover',
                    'pred_sem': ['lover1', 'lover2'],
                    'higher_order': False,
                    'object_name': 'John',
                    'object_sem': ['john1', 'john2'],
                    'P': 'non_exist'
                }
            ],
            'set': 'recipient',
            'analog': 0
        }
    ]


@pytest.fixture
def multi_rb_sym_props():
    """symProps with multiple role bindings per proposition."""
    return [
        {
            'name': 'lovesMaryTom',
            'RBs': [
                {
                    'pred_name': 'lover',
                    'pred_sem': ['lover1', 'lover2', 'lover3'],
                    'higher_order': False,
                    'object_name': 'Mary',
                    'object_sem': ['mary1', 'mary2', 'mary3'],
                    'P': 'non_exist'
                },
                {
                    'pred_name': 'beloved',
                    'pred_sem': ['beloved1', 'beloved2', 'beloved3'],
                    'higher_order': False,
                    'object_name': 'Tom',
                    'object_sem': ['tom1', 'tom2', 'tom3'],
                    'P': 'non_exist'
                }
            ],
            'set': 'driver',
            'analog': 0
        }
    ]


@pytest.fixture
def full_test_sym_props():
    """Full test data similar to testsim15.py."""
    return [
        {'name': 'lovesMaryTom', 'RBs': [
            {'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'},
            {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tom3'], 'P': 'non_exist'}
        ], 'set': 'driver', 'analog': 0},
        {'name': 'lovesTomJane', 'RBs': [
            {'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tom3'], 'P': 'non_exist'},
            {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Jane', 'object_sem': ['jane1', 'jane2', 'jane3'], 'P': 'non_exist'}
        ], 'set': 'driver', 'analog': 0},
        {'name': 'lovesJohnKathy', 'RBs': [
            {'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'John', 'object_sem': ['john1', 'john2', 'john3'], 'P': 'non_exist'},
            {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}
        ], 'set': 'recipient', 'analog': 0}
    ]


@pytest.fixture
def memory_sym_props():
    """symProps with memory set propositions."""
    return [
        {
            'name': 'rememberedFact',
            'RBs': [
                {
                    'pred_name': 'knows',
                    'pred_sem': ['knows1', 'knows2'],
                    'higher_order': False,
                    'object_name': 'Fact',
                    'object_sem': ['fact1', 'fact2'],
                    'P': 'non_exist'
                }
            ],
            'set': 'memory',
            'analog': 0
        }
    ]


@pytest.fixture
def sym_file_path(simple_sym_props):
    """Create a temporary sym file for testing file loading."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("simType='sym_file'\n")
        f.write(f"symProps = {simple_sym_props}\n")
        temp_path = f.name
    yield temp_path
    # Cleanup
    os.unlink(temp_path)


# =====================[ NetworkBuilder.__init__ tests ]======================

class TestNetworkBuilderInit:
    """Tests for NetworkBuilder initialization."""

    def test_init_with_symprops(self, simple_sym_props):
        """Test initialization with symProps list."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        
        assert builder.symProps == simple_sym_props
        assert builder.file_path is None
        assert builder.params is not None

    def test_init_with_file_path(self):
        """Test initialization with file path."""
        builder = NetworkBuilder(file_path="/some/path/file.py")
        
        assert builder.file_path == "/some/path/file.py"
        assert builder.symProps is None

    def test_init_with_custom_params(self, simple_sym_props):
        """Test initialization with custom parameters."""
        from nodes.network.network_params import Params
        custom_params = Params({'eta': 0.5})
        builder = NetworkBuilder(symProps=simple_sym_props, params=custom_params)
        
        assert builder.params == custom_params

    def test_init_default_params(self, simple_sym_props):
        """Test that default params are set when not provided."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        
        assert builder.params is not None

    def test_init_set_map(self, simple_sym_props):
        """Test that set_map is correctly initialized."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        
        assert builder.set_map['driver'] == Set.DRIVER
        assert builder.set_map['recipient'] == Set.RECIPIENT
        assert builder.set_map['memory'] == Set.MEMORY
        assert builder.set_map['new_set'] == Set.NEW_SET


# =====================[ NetworkBuilder.build_network tests ]======================

class TestBuildNetwork:
    """Tests for NetworkBuilder.build_network method."""

    def test_build_network_returns_network(self, simple_sym_props):
        """Test that build_network returns a Network object."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        assert isinstance(network, Network)

    def test_build_network_no_input_raises(self):
        """Test that build_network raises error when no symProps or file_path."""
        builder = NetworkBuilder()
        
        with pytest.raises(ValueError, match="No symProps or file_path provided"):
            builder.build_network()

    def test_build_network_from_file(self, sym_file_path):
        """Test building network from sym file."""
        builder = NetworkBuilder(file_path=sym_file_path)
        network = builder.build_network()
        
        assert isinstance(network, Network)
        # Should have tokens from the simple_sym_props
        assert network.token_tensor.get_count() > 0


# =====================[ Token creation tests ]======================

class TestTokenCreation:
    """Tests for token creation during network building."""

    def test_creates_p_tokens(self, simple_sym_props):
        """Test that P tokens are created for each proposition."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Count P tokens
        p_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.P:
                p_count += 1
        
        # Should have 2 P tokens (one per proposition)
        assert p_count == 2

    def test_creates_rb_tokens(self, simple_sym_props):
        """Test that RB tokens are created for each role binding."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Count RB tokens
        rb_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.RB:
                rb_count += 1
        
        # Should have 2 RB tokens (one per RB in symProps)
        assert rb_count == 2

    def test_creates_po_tokens(self, simple_sym_props):
        """Test that PO tokens are created (predicates and objects)."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Count PO tokens
        po_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.PO:
                po_count += 1
        
        # Should have 4 PO tokens (2 predicates + 2 objects, one per RB)
        assert po_count == 4

    def test_multi_rb_creates_correct_tokens(self, multi_rb_sym_props):
        """Test that multiple RBs create correct number of tokens."""
        builder = NetworkBuilder(symProps=multi_rb_sym_props)
        network = builder.build_network()
        
        # Count tokens by type
        p_count = rb_count = po_count = 0
        for i in range(network.token_tensor.get_count()):
            token_type = network.token_tensor.tensor[i, TF.TYPE].item()
            if token_type == Type.P:
                p_count += 1
            elif token_type == Type.RB:
                rb_count += 1
            elif token_type == Type.PO:
                po_count += 1
        
        # 1 proposition = 1 P token
        assert p_count == 1
        # 2 RBs = 2 RB tokens
        assert rb_count == 2
        # 2 RBs * 2 POs (pred + obj) = 4 PO tokens
        assert po_count == 4

    def test_token_names_set_correctly(self, simple_sym_props):
        """Test that token names are set correctly."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        names = list(network.token_tensor.names.values())
        
        # Should contain proposition names
        assert 'lovesMaryTom' in names
        assert 'lovesJohnKathy' in names
        # Should contain predicate/object names
        assert 'lover' in names
        assert 'Mary' in names
        assert 'John' in names


# =====================[ Token set assignment tests ]======================

class TestTokenSetAssignment:
    """Tests for token set assignment."""

    def test_driver_tokens_assigned_correctly(self, simple_sym_props):
        """Test that driver tokens are assigned to DRIVER set."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        driver_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.SET].item() == Set.DRIVER:
                driver_count += 1
        
        # lovesMaryTom has 1 P + 1 RB + 2 POs = 4 tokens in driver
        assert driver_count == 4

    def test_recipient_tokens_assigned_correctly(self, simple_sym_props):
        """Test that recipient tokens are assigned to RECIPIENT set."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        recipient_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.SET].item() == Set.RECIPIENT:
                recipient_count += 1
        
        # lovesJohnKathy has 1 P + 1 RB + 2 POs = 4 tokens in recipient
        assert recipient_count == 4

    def test_memory_tokens_assigned_correctly(self, memory_sym_props):
        """Test that memory tokens are assigned to MEMORY set."""
        builder = NetworkBuilder(symProps=memory_sym_props)
        network = builder.build_network()
        
        memory_count = 0
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.SET].item() == Set.MEMORY:
                memory_count += 1
        
        # rememberedFact has 1 P + 1 RB + 2 POs = 4 tokens in memory
        assert memory_count == 4


# =====================[ Token properties tests ]======================

class TestTokenProperties:
    """Tests for token properties."""

    def test_p_token_has_mode(self, simple_sym_props):
        """Test that P tokens have MODE set."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.P:
                mode = network.token_tensor.tensor[i, TF.MODE].item()
                assert mode == Mode.NEUTRAL

    def test_predicate_po_has_pred_true(self, simple_sym_props):
        """Test that predicate PO tokens have PRED=TRUE."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Find 'lover' token which is a predicate
        for i, name in network.token_tensor.names.items():
            if name == 'lover':
                token_type = network.token_tensor.tensor[i, TF.TYPE].item()
                if token_type == Type.PO:
                    pred_val = network.token_tensor.tensor[i, TF.PRED].item()
                    assert pred_val == B.TRUE

    def test_object_po_has_pred_false(self, simple_sym_props):
        """Test that object PO tokens have PRED=FALSE."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Find 'Mary' token which is an object
        for i, name in network.token_tensor.names.items():
            if name == 'Mary':
                token_type = network.token_tensor.tensor[i, TF.TYPE].item()
                if token_type == Type.PO:
                    pred_val = network.token_tensor.tensor[i, TF.PRED].item()
                    assert pred_val == B.FALSE

    def test_analog_assigned_correctly(self, simple_sym_props):
        """Test that analog values are assigned correctly."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # All tokens should have analog 0 (as specified in symProps)
        for i in range(network.token_tensor.get_count()):
            analog = network.token_tensor.tensor[i, TF.ANALOG].item()
            assert analog == 0


# =====================[ Connection tests ]======================

class TestConnections:
    """Tests for token connections."""

    def test_p_to_rb_connection(self, simple_sym_props):
        """Test that P tokens are connected to their RB tokens."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        connections = network.tokens.connections.connections
        
        # Find P and RB tokens
        p_indices = []
        rb_indices = []
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.P:
                p_indices.append(i)
            elif network.token_tensor.tensor[i, TF.TYPE].item() == Type.RB:
                rb_indices.append(i)
        
        # At least one P->RB connection should exist
        has_p_rb_connection = False
        for p_idx in p_indices:
            for rb_idx in rb_indices:
                if connections[p_idx, rb_idx]:
                    has_p_rb_connection = True
                    break
        
        assert has_p_rb_connection

    def test_rb_to_po_connection(self, simple_sym_props):
        """Test that RB tokens are connected to their PO tokens."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        connections = network.tokens.connections.connections
        
        # Find RB and PO tokens
        rb_indices = []
        po_indices = []
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.RB:
                rb_indices.append(i)
            elif network.token_tensor.tensor[i, TF.TYPE].item() == Type.PO:
                po_indices.append(i)
        
        # At least one RB->PO connection should exist
        has_rb_po_connection = False
        for rb_idx in rb_indices:
            for po_idx in po_indices:
                if connections[rb_idx, po_idx]:
                    has_rb_po_connection = True
                    break
        
        assert has_rb_po_connection

    def test_connections_are_hierarchical(self, multi_rb_sym_props):
        """Test that connections form proper hierarchy (P->RB->PO)."""
        builder = NetworkBuilder(symProps=multi_rb_sym_props)
        network = builder.build_network()
        
        connections = network.tokens.connections.connections
        
        # Verify P has children (RBs)
        p_idx = None
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.P:
                p_idx = i
                break
        
        assert p_idx is not None
        # P should have at least 2 children (2 RBs)
        p_children = connections[p_idx, :].sum().item()
        assert p_children >= 2


# =====================[ Semantics tests ]======================

class TestSemantics:
    """Tests for semantic creation."""

    def test_semantics_created(self, simple_sym_props):
        """Test that semantics are created from symProps."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        # Should have created semantics
        assert network.semantics.get_count() > 0

    def test_unique_semantics(self, full_test_sym_props):
        """Test that duplicate semantic names result in single semantic."""
        builder = NetworkBuilder(symProps=full_test_sym_props)
        network = builder.build_network()
        
        # 'lover1', 'lover2', 'lover3' should each appear once even though used multiple times
        sem_names = list(network.semantics.names.values())
        
        # Check no duplicates
        assert len(sem_names) == len(set(sem_names))

    def test_semantic_names_preserved(self, simple_sym_props):
        """Test that semantic names are preserved."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        sem_names = list(network.semantics.names.values())
        
        # Should contain the semantic names from symProps
        assert 'lover1' in sem_names
        assert 'lover2' in sem_names
        assert 'mary1' in sem_names


# =====================[ Links tests ]======================

class TestLinks:
    """Tests for token-semantic links."""

    def test_links_created(self, simple_sym_props):
        """Test that links are created between tokens and semantics."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        links = network.links.adj_matrix
        
        # At least some links should exist
        assert links.sum() > 0

    def test_po_tokens_have_links(self, simple_sym_props):
        """Test that PO tokens have links to semantics."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        links = network.links.adj_matrix
        
        # Find PO tokens and check they have links
        for i in range(network.token_tensor.get_count()):
            if network.token_tensor.tensor[i, TF.TYPE].item() == Type.PO:
                # PO token should have at least one link to a semantic
                assert links[i, :].sum() > 0

    def test_correct_number_of_links(self, simple_sym_props):
        """Test that correct number of links are created."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        links = network.links.adj_matrix
        
        # Each RB has: predicate (2 sems) + object (2 sems) = 4 links
        # 2 RBs total = 8 links
        # But some semantics are shared (lover1, lover2), so fewer unique semantics
        total_links = (links > 0).sum().item()
        assert total_links == 8  # 4 PO tokens * 2 semantics each


# =====================[ Mapping tests ]======================

class TestMapping:
    """Tests for mapping tensor creation."""

    def test_mapping_created(self, simple_sym_props):
        """Test that mapping tensor is created."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        assert network.mappings is not None
        assert network.mappings.adj_matrix is not None

    def test_mapping_dimensions_correct(self, simple_sym_props):
        """Test that mapping dimensions match driver/recipient counts."""
        builder = NetworkBuilder(symProps=simple_sym_props)
        network = builder.build_network()
        
        mapping = network.mappings.adj_matrix
        
        # Mapping should be [recipient, driver, fields]
        assert len(mapping.shape) == 3
        assert mapping.shape[2] == len(MappingFields)
        
        # Driver count = 4, Recipient count = 4
        assert mapping.shape[1] == 4  # driver dimension
        assert mapping.shape[0] == 4  # recipient dimension


# =====================[ build_network convenience function tests ]======================

class TestBuildNetworkFunction:
    """Tests for the build_network convenience function."""

    def test_build_network_with_props(self, simple_sym_props):
        """Test build_network with props argument."""
        network = build_network(props=simple_sym_props)
        
        assert isinstance(network, Network)
        assert network.token_tensor.get_count() > 0

    def test_build_network_with_file(self, sym_file_path):
        """Test build_network with file argument."""
        network = build_network(file=sym_file_path)
        
        assert isinstance(network, Network)

    def test_build_network_with_dict_params(self, simple_sym_props):
        """Test build_network with dict params."""
        network = build_network(props=simple_sym_props, params={'eta': 0.5})
        
        assert isinstance(network, Network)
        assert network.params.eta == 0.5

    def test_build_network_with_params_object(self, simple_sym_props):
        """Test build_network with Params object."""
        from nodes.network.network_params import Params
        params = Params({'eta': 0.7})
        network = build_network(props=simple_sym_props, params=params)
        
        assert isinstance(network, Network)
        assert network.params.eta == 0.7

    def test_build_network_no_args_raises(self):
        """Test build_network raises error with no arguments."""
        with pytest.raises(ValueError, match="No file or symProps provided"):
            build_network()

    def test_build_network_invalid_params_raises(self, simple_sym_props):
        """Test build_network raises error with invalid params type."""
        with pytest.raises(ValueError, match="Invalid parameters provided"):
            build_network(props=simple_sym_props, params="invalid")


# =====================[ Full integration tests ]======================

class TestFullIntegration:
    """Full integration tests with realistic data."""

    def test_full_testsim15_data(self, full_test_sym_props):
        """Test building network with full test data."""
        network = build_network(props=full_test_sym_props)
        
        # Should create network successfully
        assert isinstance(network, Network)
        
        # Check counts
        # 3 propositions * (1 P + 2 RB + 4 PO) = 21 tokens
        assert network.token_tensor.get_count() == 21
        
        # 2 driver props, 1 recipient prop
        driver_count = 0
        recipient_count = 0
        for i in range(network.token_tensor.get_count()):
            s = network.token_tensor.tensor[i, TF.SET].item()
            if s == Set.DRIVER:
                driver_count += 1
            elif s == Set.RECIPIENT:
                recipient_count += 1
        
        assert driver_count == 14  # 2 props * 7 tokens each
        assert recipient_count == 7  # 1 prop * 7 tokens

    def test_network_is_functional(self, full_test_sym_props):
        """Test that built network can perform basic operations."""
        network = build_network(props=full_test_sym_props)
        
        # Test that we can access sets
        driver_set = network.sets[Set.DRIVER]
        recipient_set = network.sets[Set.RECIPIENT]
        
        assert driver_set is not None
        assert recipient_set is not None
        
        # Test that tokens object works
        assert network.tokens is not None
        assert network.tokens.token_tensor.get_count() > 0

    def test_semantics_init_sdm(self, simple_sym_props):
        """Test that SDM semantics are initialized after network creation."""
        network = build_network(props=simple_sym_props)
        
        # SDM semantics should be initialized
        assert network.semantics.check_sdm_init()


# =====================[ Exact hierarchy/structure tests ]======================

class TestExactHierarchy:
    """Tests verifying exact parent-child connections match symProps structure."""

    def test_p_connected_to_its_rbs(self):
        """Test that each P token is connected to exactly its own RBs."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'pred1', 'pred_sem': ['s1'], 'higher_order': False, 
                     'object_name': 'obj1', 'object_sem': ['s2'], 'P': 'non_exist'},
                    {'pred_name': 'pred2', 'pred_sem': ['s3'], 'higher_order': False, 
                     'object_name': 'obj2', 'object_sem': ['s4'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            },
            {
                'name': 'prop2',
                'RBs': [
                    {'pred_name': 'pred3', 'pred_sem': ['s5'], 'higher_order': False, 
                     'object_name': 'obj3', 'object_sem': ['s6'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 1
            }
        ]
        
        network = build_network(props=symProps)
        connections = network.tokens.connections.connections
        names = network.token_tensor.names
        
        # Find prop1's P token index
        prop1_idx = None
        prop2_idx = None
        for idx, name in names.items():
            if name == 'prop1':
                prop1_idx = idx
            elif name == 'prop2':
                prop2_idx = idx
        
        assert prop1_idx is not None, "prop1 P token not found"
        assert prop2_idx is not None, "prop2 P token not found"
        
        # prop1 should have exactly 2 children (its 2 RBs)
        prop1_children = connections[prop1_idx, :].sum().item()
        assert prop1_children == 2, f"prop1 should have 2 RB children, got {prop1_children}"
        
        # prop2 should have exactly 1 child (its 1 RB)
        prop2_children = connections[prop2_idx, :].sum().item()
        assert prop2_children == 1, f"prop2 should have 1 RB child, got {prop2_children}"

    def test_rb_connected_to_its_pos(self):
        """Test that each RB token is connected to exactly its pred and obj PO."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'predA', 'pred_sem': ['s1'], 'higher_order': False, 
                     'object_name': 'objA', 'object_sem': ['s2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        connections = network.tokens.connections.connections
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        
        # Find RB token (named predA_objA)
        rb_idx = None
        pred_idx = None
        obj_idx = None
        for idx, name in names.items():
            if name == 'predA_objA':
                rb_idx = idx
            elif name == 'predA' and tensor[idx, TF.TYPE].item() == Type.PO:
                pred_idx = idx
            elif name == 'objA' and tensor[idx, TF.TYPE].item() == Type.PO:
                obj_idx = idx
        
        assert rb_idx is not None, "RB token not found"
        assert pred_idx is not None, "Predicate PO token not found"
        assert obj_idx is not None, "Object PO token not found"
        
        # RB should have exactly 2 children
        rb_children = connections[rb_idx, :].sum().item()
        assert rb_children == 2, f"RB should have 2 PO children, got {rb_children}"
        
        # RB should be connected to both predA and objA
        assert connections[rb_idx, pred_idx], "RB not connected to predicate PO"
        assert connections[rb_idx, obj_idx], "RB not connected to object PO"

    def test_hierarchy_chain_p_rb_po(self):
        """Test complete hierarchy: P -> RB -> POs."""
        symProps = [
            {
                'name': 'testProp',
                'RBs': [
                    {'pred_name': 'testPred', 'pred_sem': ['sem1'], 'higher_order': False, 
                     'object_name': 'testObj', 'object_sem': ['sem2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        connections = network.tokens.connections.connections
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        
        # Find all tokens
        p_idx = rb_idx = pred_idx = obj_idx = None
        for idx, name in names.items():
            token_type = tensor[idx, TF.TYPE].item()
            if name == 'testProp' and token_type == Type.P:
                p_idx = idx
            elif name == 'testPred_testObj' and token_type == Type.RB:
                rb_idx = idx
            elif name == 'testPred' and token_type == Type.PO:
                pred_idx = idx
            elif name == 'testObj' and token_type == Type.PO:
                obj_idx = idx
        
        # Verify all tokens found
        assert p_idx is not None, "P token not found"
        assert rb_idx is not None, "RB token not found"
        assert pred_idx is not None, "Predicate PO not found"
        assert obj_idx is not None, "Object PO not found"
        
        # Verify P -> RB connection
        assert connections[p_idx, rb_idx], "P not connected to RB"
        
        # Verify RB -> Pred and RB -> Obj connections
        assert connections[rb_idx, pred_idx], "RB not connected to Predicate"
        assert connections[rb_idx, obj_idx], "RB not connected to Object"
        
        # Verify P is NOT directly connected to POs
        assert not connections[p_idx, pred_idx], "P should not be directly connected to Predicate"
        assert not connections[p_idx, obj_idx], "P should not be directly connected to Object"

    def test_no_cross_proposition_connections(self):
        """Test that tokens from different propositions are not connected."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'p1pred', 'pred_sem': ['s1'], 'higher_order': False, 
                     'object_name': 'p1obj', 'object_sem': ['s2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            },
            {
                'name': 'prop2',
                'RBs': [
                    {'pred_name': 'p2pred', 'pred_sem': ['s3'], 'higher_order': False, 
                     'object_name': 'p2obj', 'object_sem': ['s4'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 1
            }
        ]
        
        network = build_network(props=symProps)
        connections = network.tokens.connections.connections
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        
        # Find prop1's tokens
        prop1_tokens = []
        prop2_tokens = []
        for idx, name in names.items():
            if name in ['prop1', 'p1pred_p1obj', 'p1pred', 'p1obj']:
                prop1_tokens.append(idx)
            elif name in ['prop2', 'p2pred_p2obj', 'p2pred', 'p2obj']:
                prop2_tokens.append(idx)
        
        # No token from prop1 should be connected to any token from prop2
        for p1_idx in prop1_tokens:
            for p2_idx in prop2_tokens:
                assert not connections[p1_idx, p2_idx], \
                    f"Cross-prop connection found: {names[p1_idx]} -> {names[p2_idx]}"
                assert not connections[p2_idx, p1_idx], \
                    f"Cross-prop connection found: {names[p2_idx]} -> {names[p1_idx]}"


class TestExactSemanticLinks:
    """Tests verifying exact PO-semantic links match symProps structure."""

    def test_predicate_linked_to_its_semantics(self):
        """Test that predicate PO is linked to exactly its specified semantics."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'myPred', 'pred_sem': ['predSem1', 'predSem2', 'predSem3'], 
                     'higher_order': False, 
                     'object_name': 'myObj', 'object_sem': ['objSem1'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        links = network.links.adj_matrix
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        sem_names = network.semantics.names
        
        # Find predicate PO token
        pred_idx = None
        for idx, name in names.items():
            if name == 'myPred' and tensor[idx, TF.TYPE].item() == Type.PO:
                pred_idx = idx
                break
        
        assert pred_idx is not None, "Predicate PO not found"
        
        # Find semantic indices
        pred_sem_indices = []
        for sem_id, sem_name in sem_names.items():
            if sem_name in ['predSem1', 'predSem2', 'predSem3']:
                sem_idx = network.semantics.IDs[sem_id]
                pred_sem_indices.append(sem_idx)
        
        assert len(pred_sem_indices) == 3, "Not all predicate semantics found"
        
        # Verify predicate is linked to all its semantics
        for sem_idx in pred_sem_indices:
            assert links[pred_idx, sem_idx] > 0, \
                f"Predicate not linked to semantic at index {sem_idx}"
        
        # Count total links from predicate
        pred_link_count = (links[pred_idx, :] > 0).sum().item()
        assert pred_link_count == 3, f"Predicate should have 3 links, got {pred_link_count}"

    def test_object_linked_to_its_semantics(self):
        """Test that object PO is linked to exactly its specified semantics."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'pred', 'pred_sem': ['ps1'], 'higher_order': False, 
                     'object_name': 'myObject', 'object_sem': ['objS1', 'objS2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        links = network.links.adj_matrix
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        sem_names = network.semantics.names
        
        # Find object PO token
        obj_idx = None
        for idx, name in names.items():
            if name == 'myObject' and tensor[idx, TF.TYPE].item() == Type.PO:
                obj_idx = idx
                break
        
        assert obj_idx is not None, "Object PO not found"
        
        # Find semantic indices
        obj_sem_indices = []
        for sem_id, sem_name in sem_names.items():
            if sem_name in ['objS1', 'objS2']:
                sem_idx = network.semantics.IDs[sem_id]
                obj_sem_indices.append(sem_idx)
        
        assert len(obj_sem_indices) == 2, "Not all object semantics found"
        
        # Verify object is linked to all its semantics
        for sem_idx in obj_sem_indices:
            assert links[obj_idx, sem_idx] > 0, \
                f"Object not linked to semantic at index {sem_idx}"
        
        # Count total links from object
        obj_link_count = (links[obj_idx, :] > 0).sum().item()
        assert obj_link_count == 2, f"Object should have 2 links, got {obj_link_count}"

    def test_shared_semantics_linked_correctly(self):
        """Test that shared semantics are linked to all POs that reference them."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'pred1', 'pred_sem': ['sharedSem', 'uniqueSem1'], 
                     'higher_order': False, 
                     'object_name': 'obj1', 'object_sem': ['sharedSem', 'uniqueSem2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        links = network.links.adj_matrix
        names = network.token_tensor.names
        tensor = network.token_tensor.tensor
        sem_names = network.semantics.names
        
        # Find PO tokens
        pred_idx = obj_idx = None
        for idx, name in names.items():
            if name == 'pred1' and tensor[idx, TF.TYPE].item() == Type.PO:
                pred_idx = idx
            elif name == 'obj1' and tensor[idx, TF.TYPE].item() == Type.PO:
                obj_idx = idx
        
        assert pred_idx is not None, "Predicate not found"
        assert obj_idx is not None, "Object not found"
        
        # Find shared semantic index
        shared_sem_idx = None
        for sem_id, sem_name in sem_names.items():
            if sem_name == 'sharedSem':
                shared_sem_idx = network.semantics.IDs[sem_id]
                break
        
        assert shared_sem_idx is not None, "Shared semantic not found"
        
        # Both pred and obj should be linked to the shared semantic
        assert links[pred_idx, shared_sem_idx] > 0, "Predicate not linked to shared semantic"
        assert links[obj_idx, shared_sem_idx] > 0, "Object not linked to shared semantic"

    def test_non_po_tokens_have_no_semantic_links(self):
        """Test that P and RB tokens do not have semantic links."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'pred1', 'pred_sem': ['sem1'], 'higher_order': False, 
                     'object_name': 'obj1', 'object_sem': ['sem2'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        links = network.links.adj_matrix
        tensor = network.token_tensor.tensor
        
        # Check all P and RB tokens have no links
        for i in range(network.token_tensor.get_count()):
            token_type = tensor[i, TF.TYPE].item()
            if token_type in [Type.P, Type.RB]:
                link_count = (links[i, :] > 0).sum().item()
                assert link_count == 0, \
                    f"Token type {token_type} at index {i} should have no semantic links, got {link_count}"

    def test_link_weights_are_one(self):
        """Test that all semantic link weights are 1.0."""
        symProps = [
            {
                'name': 'prop1',
                'RBs': [
                    {'pred_name': 'pred1', 'pred_sem': ['sem1', 'sem2'], 'higher_order': False, 
                     'object_name': 'obj1', 'object_sem': ['sem3'], 'P': 'non_exist'}
                ],
                'set': 'driver',
                'analog': 0
            }
        ]
        
        network = build_network(props=symProps)
        links = network.links.adj_matrix
        
        # All non-zero links should have weight 1.0
        non_zero_links = links[links > 0]
        assert (non_zero_links == 1.0).all(), "All link weights should be 1.0"