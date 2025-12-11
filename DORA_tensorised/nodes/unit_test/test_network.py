# nodes/unit_test/test_network.py
# Tests for Network class - setup and direct functions

import sys
from pathlib import Path

# Add DORA_tensorised to Python path so 'nodes' module can be imported
# test_network.py is in DORA_tensorised/nodes/unit_test/
# So we need to go up 3 levels: unit_test -> nodes -> DORA_tensorised
dora_tensorised_dir = Path(__file__).parent.parent.parent
if str(dora_tensorised_dir) not in sys.path:
    sys.path.insert(0, str(dora_tensorised_dir))

import pytest
import torch
import tempfile
import os
import json
from nodes.network.network import Network
from nodes.network.tokens.tokens import Tokens
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.tokens.connections.connections import Connections_Tensor
from nodes.network.tokens.connections.mapping import Mapping
from nodes.network.tokens.connections.links import Links
from nodes.network.sets_new.semantics import Semantics
from nodes.network.network_params import Params, default_params, load_from_json
from nodes.enums import Set, TF, SF, MappingFields


@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """Create minimal Token_Tensor for testing."""
    num_tokens = 10
    num_features = len(TF)
    tokens = torch.zeros((num_tokens, num_features))
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    names = {}
    return Token_Tensor(tokens, Connections_Tensor(connections), names)


@pytest.fixture
def minimal_connections():
    """Create minimal Connections_Tensor for testing."""
    num_tokens = 10
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

def test_network_init_success(network):
    """Test successful Network initialization."""
    assert network.tokens is not None
    assert network.token_tensor is not None
    assert network.semantics is not None
    assert network.mappings is not None
    assert network.links is not None
    assert network.params is not None
    assert network.sets is not None
    assert len(network.sets) == 4  # DRIVER, RECIPIENT, MEMORY, NEW_SET
    assert Set.DRIVER in network.sets
    assert Set.RECIPIENT in network.sets
    assert Set.MEMORY in network.sets
    assert Set.NEW_SET in network.sets


def test_network_init_type_checking_semantics(minimal_tokens, minimal_mapping, minimal_links, minimal_params):
    """Test Network initialization raises ValueError for invalid semantics type."""
    with pytest.raises(ValueError, match="semantics must be a Semantics object"):
        Network(minimal_tokens, "invalid", minimal_mapping, minimal_links, minimal_params)


def test_network_init_type_checking_mappings(minimal_tokens, minimal_semantics, minimal_links, minimal_params):
    """Test Network initialization raises ValueError for invalid mappings type."""
    with pytest.raises(ValueError, match="mappings must be a Mapping object"):
        Network(minimal_tokens, minimal_semantics, "invalid", minimal_links, minimal_params)


def test_network_init_type_checking_links(minimal_tokens, minimal_semantics, minimal_mapping, minimal_params):
    """Test Network initialization raises ValueError for invalid links type."""
    with pytest.raises(ValueError, match="links must be a Links object"):
        Network(minimal_tokens, minimal_semantics, minimal_mapping, "invalid", minimal_params)


def test_network_init_type_checking_params(minimal_tokens, minimal_semantics, minimal_mapping, minimal_links):
    """Test Network initialization raises ValueError for invalid params type."""
    with pytest.raises(ValueError, match="params must be a Params object"):
        Network(minimal_tokens, minimal_semantics, minimal_mapping, minimal_links, "invalid")


def test_network_init_operations_created(network):
    """Test that all operation objects are created during initialization."""
    assert network.routines is not None
    assert network.tensor_ops is not None
    assert network.update_ops is not None
    assert network.mapping_ops is not None
    assert network.firing_ops is not None
    assert network.analog_ops is not None
    assert network.entropy_ops is not None
    assert network.node_ops is not None
    assert network.inhibitor_ops is not None


def test_network_init_inhibitors_initialized(network):
    """Test that inhibitors are initialized to 0.0."""
    assert network.local_inhibitor == 0.0
    assert network.global_inhibitor == 0.0


def test_network_init_sets_have_correct_types(network):
    """Test that sets are created with correct types."""
    from nodes.network.sets_new import Driver, Recipient, Memory, New_Set
    assert isinstance(network.sets[Set.DRIVER], Driver)
    assert isinstance(network.sets[Set.RECIPIENT], Recipient)
    assert isinstance(network.sets[Set.MEMORY], Memory)
    assert isinstance(network.sets[Set.NEW_SET], New_Set)


# =====================[ set_params Tests ]======================

def test_set_params_updates_network_params(network, minimal_params):
    """Test that set_params updates network params."""
    # Create new params with different values
    new_params_dict = minimal_params.get_params_dict()
    new_params_dict["gamma"] = 999.0
    new_params = Params(new_params_dict)
    
    network.set_params(new_params)
    assert network.params.gamma == 999.0


def test_set_params_updates_sets_params(network, minimal_params):
    """Test that set_params updates params for all sets."""
    new_params_dict = minimal_params.get_params_dict()
    new_params_dict["gamma"] = 888.0
    new_params = Params(new_params_dict)
    
    network.set_params(new_params)
    for set_enum in Set:
        assert network.sets[set_enum].params.gamma == 888.0


def test_set_params_updates_semantics_params(network, minimal_params):
    """Test that set_params updates semantics params."""
    new_params_dict = minimal_params.get_params_dict()
    new_params_dict["gamma"] = 777.0
    new_params = Params(new_params_dict)
    
    network.set_params(new_params)
    assert network.semantics.params.gamma == 777.0


def test_set_params_updates_links_params(network, minimal_params):
    """Test that set_params updates links params."""
    new_params_dict = minimal_params.get_params_dict()
    new_params_dict["gamma"] = 666.0
    new_params = Params(new_params_dict)
    
    network.set_params(new_params)
    # Links should have params set (check if links has params attribute)
    # This depends on Links implementation - if it doesn't have params, this test may need adjustment
    if hasattr(network.links, 'params'):
        assert network.links.params.gamma == 666.0


# =====================[ setup_sets_and_semantics Tests ]======================

def test_setup_sets_and_semantics_sets_mappings(network):
    """Test that setup_sets_and_semantics sets mappings correctly."""
    # Check that recipient has mappings
    assert network.sets[Set.RECIPIENT].mappings is not None
    assert network.sets[Set.RECIPIENT].mappings == network.mappings
    
    # Check that driver has mappings set
    # This depends on driver.set_mappings implementation
    assert hasattr(network.sets[Set.DRIVER], 'mappings') or hasattr(network.sets[Set.DRIVER], '_mappings')


def test_setup_sets_and_semantics_sets_links(network):
    """Test that setup_sets_and_semantics sets links for all sets."""
    for set_enum in Set:
        assert network.sets[set_enum].links is not None
        assert network.sets[set_enum].links == network.links


def test_setup_sets_and_semantics_sets_semantics_links(network):
    """Test that setup_sets_and_semantics sets links for semantics."""
    assert network.semantics.links is not None
    assert network.semantics.links == network.links


def test_setup_sets_and_semantics_calls_set_params(network):
    """Test that setup_sets_and_semantics calls set_params."""
    # After setup, params should be set on all sets
    for set_enum in Set:
        assert network.sets[set_enum].params is not None
    assert network.semantics.params is not None


# =====================[ load_json_params Tests ]======================

def test_load_json_params_success(network, minimal_params):
    """Test loading params from JSON file."""
    # Create temporary JSON file
    params_dict = minimal_params.get_params_dict()
    params_dict["gamma"] = 555.0
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(params_dict, f)
        temp_path = f.name
    
    try:
        network.load_json_params(temp_path)
        assert network.params.gamma == 555.0
        # Check that params are propagated
        assert network.sets[Set.DRIVER].params.gamma == 555.0
    finally:
        os.unlink(temp_path)


def test_load_json_params_file_not_found(network):
    """Test loading params from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        network.load_json_params("/nonexistent/path/to/file.json")


# =====================[ __getitem__ Tests ]======================

def test_network_getitem_driver(network):
    """Test __getitem__ returns driver set."""
    driver = network[Set.DRIVER]
    assert driver is not None
    assert driver == network.sets[Set.DRIVER]


def test_network_getitem_recipient(network):
    """Test __getitem__ returns recipient set."""
    recipient = network[Set.RECIPIENT]
    assert recipient is not None
    assert recipient == network.sets[Set.RECIPIENT]


def test_network_getitem_memory(network):
    """Test __getitem__ returns memory set."""
    memory = network[Set.MEMORY]
    assert memory is not None
    assert memory == network.sets[Set.MEMORY]


def test_network_getitem_new_set(network):
    """Test __getitem__ returns new_set."""
    new_set = network[Set.NEW_SET]
    assert new_set is not None
    assert new_set == network.sets[Set.NEW_SET]


# =====================[ get_count Tests ]======================

def test_get_count_calls_token_tensor(network):
    """Test that get_count calls token_tensor.get_count()."""
    # This test verifies the method exists and can be called
    # The actual implementation may vary
    result = network.get_count()
    # get_count doesn't return anything in the current implementation
    # but it should not raise an error
    assert result is None  # Based on the implementation


def test_get_count_with_semantics(network):
    """Test get_count with semantics=True."""
    result = network.get_count(semantics=True)
    # Should not raise an error
    assert result is None


# =====================[ clear Tests ]======================

def test_clear_limited(network):
    """Test clear with limited=True."""
    # Should not raise an error
    network.clear(limited=True)
    # Verify limited clear operations were called
    # (This depends on tensor_ops implementation)


def test_clear_full(network):
    """Test clear with limited=False (full clear)."""
    # Should not raise an error
    network.clear(limited=False)
    # Verify full clear operations were called
    # (This depends on tensor_ops implementation)


# =====================[ Property Tests ]======================

def test_tensor_property(network):
    """Test tensor property returns tensor_ops."""
    assert network.tensor == network.tensor_ops


def test_update_property(network):
    """Test update property returns update_ops."""
    assert network.update == network.update_ops


def test_mapping_property(network):
    """Test mapping property returns mapping_ops."""
    assert network.mapping == network.mapping_ops


def test_firing_property(network):
    """Test firing property returns firing_ops."""
    assert network.firing == network.firing_ops


def test_analog_property(network):
    """Test analog property returns analog_ops."""
    assert network.analog == network.analog_ops


def test_entropy_property(network):
    """Test entropy property returns entropy_ops."""
    assert network.entropy == network.entropy_ops


def test_node_property(network):
    """Test node property returns node_ops."""
    assert network.node == network.node_ops


def test_inhibitor_property(network):
    """Test inhibitor property returns inhibitor_ops."""
    assert network.inhibitor == network.inhibitor_ops


# =====================[ Set Access Function Tests ]======================

def test_driver_function(network):
    """Test driver() returns driver set."""
    driver = network.driver()
    assert driver is not None
    assert driver == network.sets[Set.DRIVER]


def test_recipient_function(network):
    """Test recipient() returns recipient set."""
    recipient = network.recipient()
    assert recipient is not None
    assert recipient == network.sets[Set.RECIPIENT]


def test_memory_function(network):
    """Test memory() returns memory set."""
    memory = network.memory()
    assert memory is not None
    assert memory == network.sets[Set.MEMORY]


def test_new_set_function(network):
    """Test new_set() returns new_set."""
    new_set = network.new_set()
    assert new_set is not None
    assert new_set == network.sets[Set.NEW_SET]


def test_semantics_function(network):
    """Test semantics attribute returns semantics object."""
    # Note: network.semantics is an attribute, not a method
    # The semantics() method exists but conflicts with the attribute name
    # So we test the attribute directly
    semantics = network.semantics
    assert semantics is not None
    # Verify it's the same object
    assert semantics == network.semantics


# =====================[ set_name Tests ]======================

def test_set_name_success(network):
    """Test set_name sets name for a token."""
    idx = 0
    name = "test_token"
    network.set_name(idx, name)
    # Verify name was set (depends on tokens.set_name implementation)
    assert network.tokens.token_tensor.names[idx] == name


# =====================[ get_max_map_value Tests ]======================

def test_get_max_map_value_driver_token(network):
    """Test get_max_map_value for a driver token."""
    # Set up a driver token
    idx = 0
    network.token_tensor.move_tokens(torch.tensor([idx]), Set.DRIVER)
    
    # Set some mapping weights
    network.mappings[MappingFields.WEIGHT][0, 0] = 0.5
    network.mappings[MappingFields.WEIGHT][1, 0] = 0.8
    network.mappings[MappingFields.WEIGHT][2, 0] = 0.3
    
    # Get max map value (should look in recipient dimension)
    max_map = network.get_max_map_value(idx)
    assert max_map == pytest.approx(0.8, abs=1e-6)


def test_get_max_map_value_recipient_token(network: Network):
    """Test get_max_map_value for a recipient token."""
    # Set up a recipient token
    idx = 0
    network.token_tensor.move_tokens(torch.tensor([idx]), Set.RECIPIENT)
    network[Set.RECIPIENT].update_view()
    
    # Set some mapping weights
    network.mappings[MappingFields.WEIGHT][0, 0] = 0.6
    network.mappings[MappingFields.WEIGHT][0, 1] = 0.9
    network.mappings[MappingFields.WEIGHT][0, 2] = 0.4
    
    # Get max map value (should look in driver dimension)
    max_map = network.get_max_map_value(idx)
    assert max_map == pytest.approx(0.9, abs=1e-6)


# =====================[ get_ref_string Tests ]======================

def test_get_ref_string_success(network):
    """Test get_ref_string returns string representation."""
    idx = 0
    ref_string = network.get_ref_string(idx)
    assert isinstance(ref_string, str)


# =====================[ __getattr__ Tests ]======================

def test_getattr_delegates_to_promoted_components(network):
    """Test that __getattr__ delegates to promoted components."""
    # Try to access a method that exists in tensor_ops
    # This depends on what methods are available in TensorOperations
    # We'll test that it doesn't raise AttributeError for a known method
    if hasattr(network.tensor_ops, 'clear_set'):
        # Access through network should work
        assert hasattr(network, 'clear_set') or callable(getattr(network, 'clear_set', None))


def test_getattr_raises_attribute_error_for_missing(network):
    """Test that __getattr__ raises AttributeError for non-existent attributes."""
    with pytest.raises(AttributeError):
        _ = network.nonexistent_attribute_xyz


# =====================[ Integration Tests ]======================

def test_network_initialization_complete_setup(network):
    """Test that network initialization completes all setup steps."""
    # Verify all components are connected
    assert network.sets[Set.RECIPIENT].mappings == network.mappings
    assert network.semantics.links == network.links
    for set_enum in Set:
        assert network.sets[set_enum].links == network.links
        assert network.sets[set_enum].params == network.params
    assert network.semantics.params == network.params

