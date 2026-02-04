# DORA_bridge/tests/test_load_network_from_json.py
# Tests for NetworkLoader class and load_network_from_json functions

import pytest
import json
from pathlib import Path

import torch


from ..new_net import NewNetworkStateGenerator, NetworkLoader
from nodes.network import Network
from nodes.enums import TF, SF, Type, Set, B, MappingFields, OntStatus, null

def load_network_from_json(file_path) -> Network:
    """
    Convenience function to load a Network from a JSON state file.
    
    Args:
        file_path: Path to the JSON state file
        
    Returns:
        Network: The reconstructed Network object
        
    Example:
        >>> network = load_network_from_json('test_data/state.json')
    """
    return NetworkLoader().load(file_path)

def load_from_state(state) -> Network:
    """
    Load a Network from a state dictionary.
    
    Args:
        state: State dictionary
    
    Returns:
        Network: The reconstructed Network object
    """
    return NetworkLoader().load_from_state(state)

# =====================[ NetworkLoader Initialization Tests ]======================

class TestNetworkLoaderInit:
    """Tests for NetworkLoader initialization."""

    def test_init_creates_instance(self):
        """Test that NetworkLoader can be instantiated."""
        loader = NetworkLoader()
        assert loader is not None

    def test_init_state_is_none(self):
        """Test that initial state is None."""
        loader = NetworkLoader()
        assert loader._state is None

    def test_init_set_map_populated(self):
        """Test that set map is correctly populated."""
        loader = NetworkLoader()
        
        assert 'driver' in loader._set_map
        assert 'recipient' in loader._set_map
        assert 'memory' in loader._set_map
        assert 'newSet' in loader._set_map
        assert 'new_set' in loader._set_map
        
        assert loader._set_map['driver'] == Set.DRIVER
        assert loader._set_map['recipient'] == Set.RECIPIENT
        assert loader._set_map['memory'] == Set.MEMORY

    def test_init_type_map_populated(self):
        """Test that type map is correctly populated."""
        loader = NetworkLoader()
        
        assert 'P' in loader._type_map
        assert 'RB' in loader._type_map
        assert 'PO' in loader._type_map
        
        assert loader._type_map['P'] == Type.P
        assert loader._type_map['RB'] == Type.RB
        assert loader._type_map['PO'] == Type.PO

    def test_init_ont_map_populated(self):
        """Test that ont_status map is correctly populated."""
        loader = NetworkLoader()
        
        assert 'state' in loader._ont_map
        assert 'value' in loader._ont_map
        assert 'sdm' in loader._ont_map
        assert 'ho' in loader._ont_map


# =====================[ Load from File Tests ]======================

class TestLoadFromFile:
    """Tests for load method (file loading)."""

    def test_load_from_json_file(self, testsim_path, tmp_output_dir):
        """Test loading network from JSON file."""
        # First create a JSON state file
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        # Load it with NetworkLoader
        loader = NetworkLoader()
        network = loader.load(json_path)
        
        assert network is not None
        assert isinstance(network, Network)

    def test_load_nonexistent_file_raises(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        loader = NetworkLoader()
        
        with pytest.raises(FileNotFoundError, match="State file not found"):
            loader.load("nonexistent_file.json")

    def test_load_with_path_object(self, testsim_path, tmp_output_dir):
        """Test loading with Path object."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        loader = NetworkLoader()
        network = loader.load(Path(json_path))
        
        assert network is not None

    def test_load_with_string_path(self, testsim_path, tmp_output_dir):
        """Test loading with string path."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        loader = NetworkLoader()
        network = loader.load(str(json_path))
        
        assert network is not None


# =====================[ Load from State Tests ]======================

class TestLoadFromState:
    """Tests for load_from_state method."""

    def test_load_from_state_dict(self, testsim_path):
        """Test loading network from state dictionary."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        assert network is not None
        assert isinstance(network, Network)

    def test_load_from_state_without_state_raises(self):
        """Test that load_from_state raises error when no state provided."""
        loader = NetworkLoader()
        
        with pytest.raises(ValueError, match="No state/file provided"):
            loader.load_from_state()

    def test_load_from_state_preserves_token_counts(self, testsim_path):
        """Test that loading preserves token counts."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Count non-deleted tokens by type
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        num_ps = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        num_rbs = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        num_pos = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        
        assert num_ps == len(state['tokens']['Ps'])
        assert num_rbs == len(state['tokens']['RBs'])
        assert num_pos == len(state['tokens']['POs'])

    def test_load_from_state_preserves_semantic_counts(self, testsim_path):
        """Test that loading preserves semantic counts."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Count semantics
        num_sems = len(network.semantics.IDs)
        
        assert num_sems == len(state['semantics'])


# =====================[ Token Tensor Building Tests ]======================

class TestBuildTokenTensor:
    """Tests for _build_token_tensor method."""

    def test_token_tensor_has_correct_shape(self, testsim_path):
        """Test that token tensor has correct shape."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        total_tokens = (
            len(state['tokens']['Ps']) +
            len(state['tokens']['RBs']) +
            len(state['tokens']['POs'])
        )
        
        assert network.token_tensor.tensor.shape[0] == total_tokens
        assert network.token_tensor.tensor.shape[1] == len(TF)

    def test_token_types_correct(self, testsim_path):
        """Test that token types are correctly set."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        # Each type should be present
        p_count = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        rb_count = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        po_count = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        
        assert p_count > 0
        assert rb_count > 0
        assert po_count > 0

    def test_token_sets_correct(self, testsim_path):
        """Test that token sets (driver/recipient) are correctly set."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        # Should have driver and recipient tokens
        driver_count = ((token_tensor.tensor[:, TF.SET] == Set.DRIVER) & non_deleted).sum().item()
        recipient_count = ((token_tensor.tensor[:, TF.SET] == Set.RECIPIENT) & non_deleted).sum().item()
        
        assert driver_count > 0
        assert recipient_count > 0

    def test_token_names_preserved(self, testsim_path):
        """Test that token names are preserved."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Check that names dict is populated
        assert len(network.token_tensor.names) > 0
        
        # Check some known names from testsim
        all_names = list(network.token_tensor.names.values())
        
        # P names from testsim
        p_names = [t['name'] for t in state['tokens']['Ps']]
        for name in p_names:
            assert name in all_names, f"P name {name} not found"


# =====================[ Connections Building Tests ]======================

class TestBuildConnections:
    """Tests for _build_connections method."""

    def test_connections_has_correct_shape(self, testsim_path):
        """Test that connections tensor has correct shape."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        total_tokens = (
            len(state['tokens']['Ps']) +
            len(state['tokens']['RBs']) +
            len(state['tokens']['POs'])
        )
        
        assert network.tokens.connections.tensor.shape[0] == total_tokens
        assert network.tokens.connections.tensor.shape[1] == total_tokens

    def test_p_to_rb_connections_exist(self, testsim_path):
        """Test that P to RB connections are created."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Should have some True connections
        total_connections = network.tokens.connections.tensor.sum().item()
        assert total_connections > 0

    def test_rb_to_po_connections_exist(self, testsim_path):
        """Test that RB to PO connections are created."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Number of connections should match state
        expected_connections = (
            len(state['connections']['P_to_RB']) +
            len(state['connections']['RB_to_PO']) +
            len(state['connections']['RB_to_childP'])
        )
        
        actual_connections = network.tokens.connections.tensor.sum().item()
        assert actual_connections == expected_connections


# =====================[ Links Building Tests ]======================

class TestBuildLinks:
    """Tests for _build_links method."""

    def test_links_has_correct_shape(self, testsim_path):
        """Test that links tensor has correct shape."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        total_tokens = (
            len(state['tokens']['Ps']) +
            len(state['tokens']['RBs']) +
            len(state['tokens']['POs'])
        )
        num_sems = len(state['semantics'])
        
        # Links tensor shape is [num_tokens, num_semantics]
        assert network.tokens.links.adj_matrix.shape[0] == total_tokens
        assert network.tokens.links.adj_matrix.shape[1] == num_sems

    def test_links_weights_populated(self, testsim_path):
        """Test that link weights are populated."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Should have non-zero links
        total_weight = network.tokens.links.adj_matrix.sum().item()
        assert total_weight > 0

    def test_links_match_state(self, testsim_path):
        """Test that number of non-zero links matches state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        gen.network.tokens.print_links(use_names=False)
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Count non-zero links
        non_zero_links = (network.tokens.links.adj_matrix != 0).sum().item()
        expected_links = len(state['links']['links_list'])
        for link in state['links']['links_list']:
            print(link['po_index'], link['po_name'], link['sem_index'], link['sem_name'], link['weight'])

        network.tokens.print_links(use_names=False)
        
        assert non_zero_links == expected_links


# =====================[ Mapping Building Tests ]======================

class TestBuildMapping:
    """Tests for _build_mapping method."""

    def test_mapping_has_correct_dimensions(self, testsim_path):
        """Test that mapping tensor has correct dimensions."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Mapping should be [recipient_count, driver_count, num_fields]
        assert network.tokens.mapping.adj_matrix.shape[2] == len(MappingFields)

    def test_mapping_with_existing_mappings(self, testsim_path, tmp_output_dir):
        """Test loading mapping with existing mappings."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        # Run mapping to create some mappings
        gen.network.mapping.update_mapping_hyps()
        gen.network.mapping.update_mapping_connections()
        gen.network.mapping.get_max_maps()
        
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Should load successfully
        assert network is not None


# =====================[ Semantics Building Tests ]======================

class TestBuildSemantics:
    """Tests for _build_semantics method."""

    def test_semantics_has_correct_count(self, testsim_path):
        """Test that semantics has correct count."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        assert len(network.semantics.IDs) == len(state['semantics'])

    def test_semantics_names_preserved(self, testsim_path):
        """Test that semantic names are preserved."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Check that names are populated
        assert len(network.semantics.names) > 0
        
        # Check some known semantic names
        state_sem_names = [s['name'] for s in state['semantics']]
        loaded_names = list(network.semantics.names.values())
        
        for name in state_sem_names:
            assert name in loaded_names, f"Semantic {name} not found"

    def test_semantics_ont_status_loaded(self, testsim_path):
        """Test that ont_status is loaded correctly."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Check if any semantics with ont_status are present
        sems_with_ont = [s for s in state['semantics'] if s.get('ont_status')]
        
        if sems_with_ont:
            # Verify ont_status was loaded
            sem_nodes = network.semantics.nodes
            for i, sem in enumerate(state['semantics']):
                if sem.get('ont_status') == 'sdm':
                    ont_val = sem_nodes[i, SF.ONT].item()
                    assert ont_val == OntStatus.SDM or ont_val != null


# =====================[ Params Building Tests ]======================

class TestBuildParams:
    """Tests for _build_params method."""

    def test_params_loaded(self, testsim_path):
        """Test that params are loaded."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        assert network.params is not None

    def test_params_values_match(self, testsim_path):
        """Test that parameter values match state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        state_params = state['metadata']['parameters']
        
        # Check some key parameters
        if 'asDORA' in state_params:
            assert network.params.as_DORA == state_params['asDORA']
        if 'gamma' in state_params:
            assert network.params.gamma == state_params['gamma']


# =====================[ Convenience Function Tests ]======================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_load_network_from_json_function(self, testsim_path, tmp_output_dir):
        """Test load_network_from_json convenience function."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        network = load_network_from_json(json_path)
        
        assert network is not None
        assert isinstance(network, Network)

    def test_load_from_state_function(self, testsim_path):
        """Test load_from_state convenience function."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        state = gen.get_state()
        
        network = load_from_state(state)
        
        assert network is not None
        assert isinstance(network, Network)


# =====================[ Edge Cases Tests ]======================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_tokens_state(self):
        """Test loading state with no tokens."""
        empty_state = {
            'tokens': {'Ps': [], 'RBs': [], 'POs': []},
            'semantics': [],
            'links': {'links_list': []},
            'mappings': {'all_mappings': []},
            'connections': {'P_to_RB': [], 'RB_to_PO': [], 'RB_to_childP': []},
            'metadata': {'parameters': {}}
        }
        
        loader = NetworkLoader()
        network = loader.load_from_state(empty_state)
        
        assert network is not None

    def test_empty_semantics_state(self, simple_props):
        """Test loading state with no semantics (edge case)."""
        gen = NewNetworkStateGenerator()
        gen.load_props(simple_props)
        state = gen.get_state()
        
        # Artificially empty semantics
        state['semantics'] = []
        state['links']['links_list'] = []
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        # Should create with placeholder semantics
        assert network is not None

    def test_state_with_null_analog(self, simple_props):
        """Test loading state where analog is null."""
        gen = NewNetworkStateGenerator()
        gen.load_props(simple_props)
        state = gen.get_state()
        
        # Set some analogs to null
        for token in state['tokens']['Ps']:
            token['analog'] = None
        
        loader = NetworkLoader()
        network = loader.load_from_state(state)
        
        assert network is not None


# =====================[ Roundtrip Tests ]======================

class TestRoundtrip:
    """Tests for save/load roundtrip consistency."""

    def test_roundtrip_preserves_structure(self, testsim_path, tmp_output_dir):
        """Test that save/load roundtrip preserves network structure."""
        # Create and save original
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        original_state = gen.get_state()
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        # Load with NetworkLoader
        network = load_network_from_json(json_path)
        
        # Verify counts match
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        num_ps = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        num_rbs = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        num_pos = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        
        assert num_ps == original_state['metadata']['token_counts']['Ps']
        assert num_rbs == original_state['metadata']['token_counts']['RBs']
        assert num_pos == original_state['metadata']['token_counts']['POs']

    def test_roundtrip_preserves_driver_recipient(self, testsim_path, tmp_output_dir):
        """Test that driver/recipient assignment is preserved."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        original_state = gen.get_state()
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        network = load_network_from_json(json_path)
        
        # Count driver/recipient
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        driver_count = ((token_tensor.tensor[:, TF.SET] == Set.DRIVER) & non_deleted).sum().item()
        recipient_count = ((token_tensor.tensor[:, TF.SET] == Set.RECIPIENT) & non_deleted).sum().item()
        
        original_driver = (
            original_state['driver']['counts']['Ps'] +
            original_state['driver']['counts']['RBs'] +
            original_state['driver']['counts']['POs']
        )
        original_recipient = (
            original_state['recipient']['counts']['Ps'] +
            original_state['recipient']['counts']['RBs'] +
            original_state['recipient']['counts']['POs']
        )
        
        assert driver_count == original_driver
        assert recipient_count == original_recipient

    def test_roundtrip_with_activations(self, testsim_path, tmp_output_dir):
        """Test that activation values are preserved in roundtrip."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        # Set some activations
        token_tensor = gen.network.token_tensor
        po_mask = (token_tensor.tensor[:, TF.TYPE] == Type.PO) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        po_indices = torch.where(po_mask)[0]
        
        if len(po_indices) > 0:
            test_activation = 0.75
            first_po_idx = po_indices[0].item()
            token_tensor.tensor[first_po_idx, TF.ACT] = test_activation
        
        # Save state
        state = gen.get_state()
        json_path = tmp_output_dir / "state_with_act.json"
        gen.save_state_json(json_path)
        
        # Load with NetworkLoader
        network = load_network_from_json(json_path)
        
        # Find the same PO token and verify activation
        if len(po_indices) > 0:
            loaded_tensor = network.token_tensor
            loaded_po_mask = (loaded_tensor.tensor[:, TF.TYPE] == Type.PO) & (loaded_tensor.tensor[:, TF.DELETED] == B.FALSE)
            loaded_po_indices = torch.where(loaded_po_mask)[0]
            
            # Find token with matching activation
            found_activation = False
            for idx in loaded_po_indices:
                act = loaded_tensor.tensor[idx, TF.ACT].item()
                if abs(act - test_activation) < 0.01:
                    found_activation = True
                    break
            
            assert found_activation, "Activation value not preserved in roundtrip"


# =====================[ Integration Tests ]======================

class TestIntegration:
    """Integration tests with real data files."""

    def test_load_compnew_json(self):
        """Test loading the compnew.json test data file."""
        json_path = Path(__file__).parent.parent.parent / "test_data" / "compnew.json"
        
        if json_path.exists():
            network = load_network_from_json(json_path)
            
            assert network is not None
            assert isinstance(network, Network)
            
            # Verify it has tokens
            non_deleted = network.token_tensor.tensor[:, TF.DELETED] == B.FALSE
            assert non_deleted.sum().item() > 0

    def test_loaded_network_can_recache(self, testsim_path, tmp_output_dir):
        """Test that loaded network can recache successfully."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        network = load_network_from_json(json_path)
        
        # Should be able to recache without error
        network.recache()

    def test_loaded_network_usable_for_operations(self, testsim_path, tmp_output_dir):
        """Test that loaded network can perform operations."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)
        
        json_path = tmp_output_dir / "state.json"
        gen.save_state_json(json_path)
        
        network = load_network_from_json(json_path)
        
        # Should be able to run mapping operations
        network.mapping.update_mapping_hyps()
        network.mapping.update_mapping_connections()
        network.mapping.get_max_maps()

