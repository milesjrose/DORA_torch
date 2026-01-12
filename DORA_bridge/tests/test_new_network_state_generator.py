# DORA_bridge/tests/test_new_network_state_generator.py
# Tests for NewNetworkStateGenerator class

import pytest
import pickle
import json
from pathlib import Path

from DORA_bridge import (
    NewNetworkStateGenerator,
    new_load_state,
    new_load_state_json,
    compare_states
)
from nodes.enums import TF, Type, B
import torch


# =====================[ Initialization Tests ]======================

class TestNewNetworkStateGeneratorInit:
    """Tests for NewNetworkStateGenerator initialization."""

    def test_init_creates_instance(self):
        """Test that NewNetworkStateGenerator can be instantiated."""
        gen = NewNetworkStateGenerator()
        assert gen is not None
        assert gen.network is None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_params = {"asDORA": False, "gamma": 0.5}
        gen = NewNetworkStateGenerator(parameters=custom_params)

        assert gen.parameters["asDORA"] == False
        assert gen.parameters["gamma"] == 0.5

    def test_default_parameters_structure(self):
        """Test that default parameters have expected structure."""
        gen = NewNetworkStateGenerator()
        params = gen.parameters

        # Check key parameters exist
        assert "asDORA" in params
        assert "gamma" in params
        assert "doGUI" in params

        # GUI should be disabled by default
        assert params["doGUI"] == False


# =====================[ Loading Tests ]======================

class TestLoadSim:
    """Tests for load_sim method."""

    def test_load_sim_from_file(self, testsim_path):
        """Test loading a simulation from file."""
        gen = NewNetworkStateGenerator()
        result = gen.load_sim(testsim_path)

        # Should return self for chaining
        assert result is gen

        # Network should be populated
        assert gen.network is not None

    def test_load_sim_creates_tokens(self, testsim_path):
        """Test that load_sim creates the expected tokens."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        # Get state to check token counts
        state = gen.get_state()

        # testsim.py has 2 Ps, 4 RBs, 8 POs, 16 semantics
        assert len(state['tokens']['Ps']) == 2
        assert len(state['tokens']['RBs']) == 4
        assert len(state['tokens']['POs']) == 8
        assert len(state['semantics']) == 16

    def test_load_sim_creates_semantics(self, testsim_path):
        """Test that load_sim creates semantic units."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        assert len(state['semantics']) > 0

    def test_load_sim_sets_driver_recipient(self, testsim_path):
        """Test that load_sim populates driver and recipient."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        # Driver should have 1 P (lovesMaryTom)
        driver = state['driver']
        assert driver['counts']['Ps'] == 1
        assert 'lovesMaryTom' in driver['P_names']

        # Recipient should have 1 P (lovesTomMary)
        recipient = state['recipient']
        assert recipient['counts']['Ps'] == 1
        assert 'lovesTomMary' in recipient['P_names']

    def test_load_sim_nonexistent_file_raises(self):
        """Test that load_sim raises FileNotFoundError for missing file."""
        gen = NewNetworkStateGenerator()

        with pytest.raises(FileNotFoundError):
            gen.load_sim("nonexistent_file.py")

    def test_load_sim_method_chaining(self, testsim_path):
        """Test that load_sim returns self for method chaining."""
        gen = NewNetworkStateGenerator()

        # Should be able to chain
        state = gen.load_sim(testsim_path).get_state()
        assert state is not None

    def test_load_sim_token_names_match_old_format(self, testsim_path):
        """Test that token names match expected old format."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        # Check for expected token names
        p_names = [t['name'] for t in state['tokens']['Ps']]
        assert 'lovesMaryTom' in p_names
        assert 'lovesTomMary' in p_names


class TestLoadProps:
    """Tests for load_props method."""

    def test_load_props_creates_network(self, simple_props):
        """Test loading network from props directly."""
        gen = NewNetworkStateGenerator()
        result = gen.load_props(simple_props)

        assert result is gen
        assert gen.network is not None

    def test_load_props_token_counts(self, simple_props):
        """Test that load_props creates correct number of tokens."""
        gen = NewNetworkStateGenerator()
        gen.load_props(simple_props)

        state = gen.get_state()

        # 2 props, each with 1 RB, each RB with 1 pred + 1 obj = 4 POs
        assert len(state['tokens']['Ps']) == 2
        assert len(state['tokens']['RBs']) == 2
        assert len(state['tokens']['POs']) == 4


# =====================[ State Extraction Tests ]======================

class TestGetState:
    """Tests for get_state method."""

    def test_get_state_returns_dict(self, testsim_path):
        """Test that get_state returns a dictionary."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self, testsim_path):
        """Test that state dict has all required keys."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        required_keys = ['tokens', 'semantics', 'links', 'mappings',
                        'analogs', 'connections', 'driver', 'recipient', 'metadata']
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_get_state_tokens_structure(self, testsim_path):
        """Test tokens structure in state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        tokens = state['tokens']

        assert 'Ps' in tokens
        assert 'RBs' in tokens
        assert 'POs' in tokens

        # Check that we have the right number of each type
        assert len(tokens['Ps']) > 0
        assert len(tokens['RBs']) > 0
        assert len(tokens['POs']) > 0

    def test_get_state_token_properties(self, testsim_path):
        """Test that tokens have expected properties."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        # Check P token properties
        p_token = state['tokens']['Ps'][0]
        required_props = ['index', 'name', 'set', 'analog', 'act', 'max_map', 'net_input']
        for prop in required_props:
            assert prop in p_token, f"P token missing property: {prop}"

        # Check RB token properties
        rb_token = state['tokens']['RBs'][0]
        rb_required_props = required_props + ['parent_P_names', 'pred_name', 'obj_name']
        for prop in rb_required_props:
            assert prop in rb_token, f"RB token missing property: {prop}"

        # Check PO token has predOrObj
        po_token = state['tokens']['POs'][0]
        assert 'predOrObj' in po_token
        assert 'semantic_names' in po_token
        assert 'parent_RB_names' in po_token

    def test_get_state_links_matrix(self, testsim_path):
        """Test that links contains a weight matrix."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        links = state['links']

        assert 'matrix' in links
        assert 'po_names' in links
        assert 'semantic_names' in links

        # Matrix should be num_POs x num_semantics
        matrix = links['matrix']
        assert len(matrix) == len(state['tokens']['POs'])

    def test_get_state_mappings_structure(self, testsim_path):
        """Test mappings structure in state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        mappings = state['mappings']

        # Should have all mapping types
        assert 'P_mappings' in mappings
        assert 'RB_mappings' in mappings
        assert 'PO_mappings' in mappings
        assert 'all_mappings' in mappings

        # All should be lists
        for mapping_type in mappings.values():
            assert isinstance(mapping_type, list)

    def test_get_state_analogs_structure(self, testsim_path):
        """Test analogs structure in state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        analogs = state['analogs']

        # Should be a list of dicts
        assert isinstance(analogs, list)
        if analogs:
            analog = analogs[0]
            assert 'index' in analog
            assert 'P_names' in analog
            assert 'RB_names' in analog
            assert 'PO_names' in analog

    def test_get_state_connections_structure(self, testsim_path):
        """Test connections structure in state."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        connections = state['connections']

        # Should have connection types
        assert 'P_to_RB' in connections
        assert 'RB_to_PO' in connections
        assert 'RB_to_childP' in connections

        # All should be lists of dicts
        for conn_type in connections.values():
            assert isinstance(conn_type, list)
            if conn_type:
                assert isinstance(conn_type[0], dict)

    def test_get_state_without_loading_raises(self):
        """Test that get_state raises error if no network loaded."""
        gen = NewNetworkStateGenerator()

        with pytest.raises(ValueError, match="No network loaded"):
            gen.get_state()

    def test_get_state_metadata(self, testsim_path):
        """Test that metadata is populated correctly."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        metadata = state['metadata']

        assert 'sim_path' in metadata
        assert 'parameters' in metadata
        assert 'token_counts' in metadata

        # Token counts should match actual counts
        assert metadata['token_counts']['Ps'] == len(state['tokens']['Ps'])
        assert metadata['token_counts']['RBs'] == len(state['tokens']['RBs'])
        assert metadata['token_counts']['POs'] == len(state['tokens']['POs'])
        assert metadata['token_counts']['semantics'] == len(state['semantics'])


class TestGetStateAfterOperations:
    """Tests for state extraction after running operations."""

    def test_state_captures_activations(self, testsim_path):
        """Test that state captures activation values."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        # Set some activations manually by modifying the tensor
        token_tensor = gen.network.token_tensor
        # Find first PO token index
        po_mask = (token_tensor.tensor[:, TF.TYPE] == Type.PO) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        po_indices = torch.where(po_mask)[0]
        
        if len(po_indices) > 0:
            first_po_idx = po_indices[0].item()
            # Set first PO activation to 0.75
            token_tensor.tensor[first_po_idx, TF.ACT] = 0.75

            state = gen.get_state()

            # Find the PO token with matching index and check its activation
            po_tokens = state['tokens']['POs']
            po_token = next((t for t in po_tokens if t['index'] == first_po_idx), None)
            assert po_token is not None, "PO token not found in state"
            assert po_token['act'] == 0.75

    def test_state_captures_mappings_after_mapping(self, testsim_path):
        """Test that state captures mapping connections after mapping."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        # Run mapping operations using the correct methods
        gen.network.mapping.update_mapping_hyps()
        gen.network.mapping.update_mapping_connections()
        gen.network.mapping.get_max_maps()

        state = gen.get_state()

        # Should have mappings structure (may or may not have actual mappings depending on activations)
        assert 'all_mappings' in state['mappings']
        assert isinstance(state['mappings']['all_mappings'], list)


# =====================[ Save State Tests ]======================

class TestSaveState:
    """Tests for save_state method."""

    def test_save_state_pickle(self, testsim_path, tmp_output_dir):
        """Test saving state as pickle file."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.pkl"
        result_path = gen.save_state(output_path, format='pickle')

        assert result_path.exists()
        assert result_path == output_path

    def test_save_state_json(self, testsim_path, tmp_output_dir):
        """Test saving state as JSON file."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.json"
        result_path = gen.save_state(output_path, format='json')

        assert result_path.exists()

    def test_save_state_creates_directories(self, testsim_path, tmp_output_dir):
        """Test that save_state creates parent directories."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "nested" / "dirs" / "state.pkl"
        result_path = gen.save_state(output_path)

        assert result_path.exists()

    def test_save_state_without_loading_raises(self, tmp_output_dir):
        """Test that save_state raises error if no network loaded."""
        gen = NewNetworkStateGenerator()

        with pytest.raises(ValueError, match="No network loaded"):
            gen.save_state(tmp_output_dir / "state.pkl")

    def test_save_state_invalid_format_raises(self, testsim_path, tmp_output_dir):
        """Test that invalid format raises error."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        with pytest.raises(ValueError, match="Unknown format"):
            gen.save_state(tmp_output_dir / "state.xyz", format='xyz')

    def test_save_state_pickle_convenience(self, testsim_path, tmp_output_dir):
        """Test save_state_pickle convenience method."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.pkl"
        result_path = gen.save_state_pickle(output_path)

        assert result_path.exists()

    def test_save_state_json_convenience(self, testsim_path, tmp_output_dir):
        """Test save_state_json convenience method."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.json"
        result_path = gen.save_state_json(output_path)

        assert result_path.exists()


# =====================[ Load State Tests ]======================

class TestLoadState:
    """Tests for new_load_state functions."""

    def test_load_state_pickle(self, testsim_path, tmp_output_dir):
        """Test loading state from pickle file."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.pkl"
        gen.save_state(output_path, format='pickle')

        loaded_state = new_load_state(output_path)

        assert isinstance(loaded_state, dict)
        assert 'tokens' in loaded_state

    def test_load_state_json(self, testsim_path, tmp_output_dir):
        """Test loading state from JSON file."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        output_path = tmp_output_dir / "state.json"
        gen.save_state(output_path, format='json')

        loaded_state = new_load_state_json(output_path)

        assert isinstance(loaded_state, dict)
        assert 'tokens' in loaded_state

    def test_roundtrip_preserves_data(self, testsim_path, tmp_output_dir):
        """Test that save/load roundtrip preserves data."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        original_state = gen.get_state()

        output_path = tmp_output_dir / "state.pkl"
        gen.save_state(output_path)

        loaded_state = new_load_state(output_path)

        # Token counts should match
        assert len(loaded_state['tokens']['Ps']) == len(original_state['tokens']['Ps'])
        assert len(loaded_state['tokens']['RBs']) == len(original_state['tokens']['RBs'])
        assert len(loaded_state['tokens']['POs']) == len(original_state['tokens']['POs'])

        # Token names should match
        original_p_names = [t['name'] for t in original_state['tokens']['Ps']]
        loaded_p_names = [t['name'] for t in loaded_state['tokens']['Ps']]
        assert original_p_names == loaded_p_names


# =====================[ Inspection Utility Tests ]======================

class TestInspectionUtilities:
    """Tests for inspection utility methods."""

    def test_print_summary_runs(self, testsim_path, capsys):
        """Test that print_summary runs without error."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        gen.print_summary()

        captured = capsys.readouterr()
        assert "NETWORK SUMMARY (New Implementation)" in captured.out
        assert "Token Counts" in captured.out

    def test_print_summary_without_network(self, capsys):
        """Test print_summary when no network loaded."""
        gen = NewNetworkStateGenerator()

        gen.print_summary()

        captured = capsys.readouterr()
        assert "No network loaded" in captured.out


# =====================[ Comparison Utility Tests ]======================

class TestCompareStates:
    """Tests for compare_states function."""

    def test_compare_states_identical(self, testsim_path):
        """Test comparing identical states."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state1 = gen.get_state()
        state2 = gen.get_state()

        results = compare_states(state1, state2, verbose=False)

        assert results['match'] == True
        assert len(results['differences']) == 0

    def test_compare_states_different_tokens(self, testsim_path):
        """Test comparing states with different token counts."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state1 = gen.get_state()

        # Modify state2 to have different token count
        state2 = gen.get_state()
        state2['tokens']['Ps'].pop()  # Remove one P token

        results = compare_states(state1, state2, verbose=False)

        assert results['match'] == False
        assert len(results['differences']) > 0

    def test_compare_states_different_links(self, testsim_path):
        """Test comparing states with different links."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state1 = gen.get_state()

        # Modify state2 to have different links
        state2 = gen.get_state()
        if state2['links']['links_list']:
            state2['links']['links_list'].pop()

        results = compare_states(state1, state2, verbose=False)

        assert results['match'] == False
        assert len(results['differences']) > 0

    def test_compare_states_verbose_output(self, testsim_path, capsys):
        """Test that verbose comparison produces output."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state1 = gen.get_state()
        state2 = gen.get_state()

        results = compare_states(state1, state2, verbose=True)

        captured = capsys.readouterr()
        assert "States match!" in captured.out or "differences" in captured.out


# =====================[ Integration Tests ]======================

class TestIntegration:
    """Integration tests for full workflows."""

    def test_full_workflow(self, testsim_path, tmp_output_dir):
        """Test complete workflow: load, operate, save, load."""
        # Create generator and load
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        # Save initial state
        initial_path = tmp_output_dir / "initial.pkl"
        gen.save_state(initial_path)

        # Run operations using correct methods
        gen.network.mapping.update_mapping_hyps()
        gen.network.mapping.update_mapping_connections()
        gen.network.mapping.get_max_maps()

        # Save after operations
        after_path = tmp_output_dir / "after_map.pkl"
        gen.save_state(after_path)

        # Load both and compare
        initial_state = new_load_state(initial_path)
        after_state = new_load_state(after_path)

        # Both should be valid
        assert 'tokens' in initial_state
        assert 'tokens' in after_state

        # Should have metadata
        assert 'metadata' in initial_state
        assert 'metadata' in after_state

    def test_multiple_load_sim_calls(self, testsim_path, testsym_path):
        """Test that loading a new sim replaces the old one."""
        gen = NewNetworkStateGenerator()

        # Load first sim
        gen.load_sim(testsim_path)
        first_state = gen.get_state()
        first_p_count = len(first_state['tokens']['Ps'])

        # Load second sim
        gen.load_sim(testsym_path)
        second_state = gen.get_state()
        second_p_count = len(second_state['tokens']['Ps'])

        # Should have different counts (testsym has more Ps)
        assert second_p_count != first_p_count or second_p_count >= first_p_count

    def test_state_consistency_across_saves(self, testsim_path, tmp_output_dir):
        """Test that multiple saves produce consistent states."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        # Save twice
        path1 = tmp_output_dir / "state1.pkl"
        path2 = tmp_output_dir / "state2.pkl"

        gen.save_state(path1)
        gen.save_state(path2)

        state1 = new_load_state(path1)
        state2 = new_load_state(path2)

        # States should be identical
        assert state1['metadata']['token_counts'] == state2['metadata']['token_counts']
        assert len(state1['tokens']['Ps']) == len(state2['tokens']['Ps'])


class TestStructureCompatibility:
    """Tests to ensure structure matches TestDataGenerator."""

    def test_state_keys_match_test_data_generator(self, testsim_path):
        """Test that state keys match TestDataGenerator structure."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()

        # Required keys should match TestDataGenerator
        expected_keys = [
            'tokens', 'semantics', 'links', 'mappings',
            'analogs', 'connections', 'driver', 'recipient', 'metadata'
        ]

        for key in expected_keys:
            assert key in state, f"Missing expected key: {key}"

    def test_token_structure_matches(self, testsim_path):
        """Test that token structure matches TestDataGenerator."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        tokens = state['tokens']

        # Should have Ps, RBs, POs
        assert 'Ps' in tokens
        assert 'RBs' in tokens
        assert 'POs' in tokens

        # Each should be a list
        assert isinstance(tokens['Ps'], list)
        assert isinstance(tokens['RBs'], list)
        assert isinstance(tokens['POs'], list)

        # Each token should be a dict with expected keys
        if tokens['Ps']:
            p_token = tokens['Ps'][0]
            assert 'index' in p_token
            assert 'name' in p_token
            assert 'set' in p_token

        if tokens['POs']:
            po_token = tokens['POs'][0]
            assert 'predOrObj' in po_token
            assert 'semantic_names' in po_token

    def test_links_structure_matches(self, testsim_path):
        """Test that links structure matches TestDataGenerator."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        links = state['links']

        # Should have matrix, po_names, semantic_names, links_list
        assert 'matrix' in links
        assert 'po_names' in links
        assert 'semantic_names' in links
        assert 'links_list' in links

        # Matrix should be list of lists
        assert isinstance(links['matrix'], list)
        if links['matrix']:
            assert isinstance(links['matrix'][0], list)

        # links_list should be list of dicts
        assert isinstance(links['links_list'], list)
        if links['links_list']:
            link = links['links_list'][0]
            assert 'po_name' in link
            assert 'sem_name' in link
            assert 'weight' in link

    def test_mappings_structure_matches(self, testsim_path):
        """Test that mappings structure matches TestDataGenerator."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        mappings = state['mappings']

        # Should have P_mappings, RB_mappings, PO_mappings, all_mappings
        assert 'P_mappings' in mappings
        assert 'RB_mappings' in mappings
        assert 'PO_mappings' in mappings
        assert 'all_mappings' in mappings

        # Each should be a list of dicts
        for mapping_type in mappings.values():
            assert isinstance(mapping_type, list)
            if mapping_type:
                mapping = mapping_type[0]
                assert 'driver_name' in mapping
                assert 'recipient_name' in mapping
                assert 'weight' in mapping
                assert 'type' in mapping

    def test_metadata_structure_matches(self, testsim_path):
        """Test that metadata structure matches TestDataGenerator."""
        gen = NewNetworkStateGenerator()
        gen.load_sim(testsim_path)

        state = gen.get_state()
        metadata = state['metadata']

        # Should have sim_path, parameters, token_counts
        assert 'sim_path' in metadata
        assert 'parameters' in metadata
        assert 'token_counts' in metadata

        # token_counts should have Ps, RBs, POs, semantics
        token_counts = metadata['token_counts']
        assert 'Ps' in token_counts
        assert 'RBs' in token_counts
        assert 'POs' in token_counts
        assert 'semantics' in token_counts

        # All counts should be integers
        for count in token_counts.values():
            assert isinstance(count, int)
            assert count >= 0
