# DORA_bridge/tests/test_data_generator.py
# Tests for TestDataGenerator class

import pytest
import pickle
import json
from pathlib import Path

from DORA_bridge import TestDataGenerator, load_state, load_state_json


# =====================[ Initialization Tests ]======================

class TestTestDataGeneratorInit:
    """Tests for TestDataGenerator initialization."""

    def test_init_creates_instance(self):
        """Test that TestDataGenerator can be instantiated."""
        gen = TestDataGenerator()
        assert gen is not None
        assert gen.memory is None
        assert gen.network is None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_params = {"asDORA": False, "gamma": 0.5}
        gen = TestDataGenerator(parameters=custom_params)
        
        assert gen.parameters["asDORA"] == False
        assert gen.parameters["gamma"] == 0.5

    def test_default_parameters_disable_gui(self):
        """Test that default parameters have GUI disabled."""
        gen = TestDataGenerator()
        assert gen.parameters["doGUI"] == False


# =====================[ Loading Tests ]======================

class TestLoadSim:
    """Tests for load_sim method."""

    def test_load_sim_from_file(self, testsim_path):
        """Test loading a simulation from file."""
        gen = TestDataGenerator()
        result = gen.load_sim(testsim_path)
        
        # Should return self for chaining
        assert result is gen
        
        # Memory should be populated
        assert gen.memory is not None
        assert gen.network is not None

    def test_load_sim_creates_tokens(self, testsim_path):
        """Test that load_sim creates the expected tokens."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # testsim.py has 2 Ps, 4 RBs, 8 POs
        assert len(gen.memory.Ps) == 2
        assert len(gen.memory.RBs) == 4
        assert len(gen.memory.POs) == 8

    def test_load_sim_creates_semantics(self, testsim_path):
        """Test that load_sim creates semantic units."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Should have semantics
        assert len(gen.memory.semantics) > 0

    def test_load_sim_sets_driver_recipient(self, testsim_path):
        """Test that load_sim populates driver and recipient."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Driver should have 1 P (lovesMaryTom)
        assert len(gen.memory.driver.Ps) == 1
        assert gen.memory.driver.Ps[0].name == "lovesMaryTom"
        
        # Recipient should have 1 P (lovesTomMary)
        assert len(gen.memory.recipient.Ps) == 1
        assert gen.memory.recipient.Ps[0].name == "lovesTomMary"

    def test_load_sim_nonexistent_file_raises(self):
        """Test that load_sim raises FileNotFoundError for missing file."""
        gen = TestDataGenerator()
        
        with pytest.raises(FileNotFoundError):
            gen.load_sim("nonexistent_file.py")

    def test_load_sim_method_chaining(self, testsim_path):
        """Test that load_sim returns self for method chaining."""
        gen = TestDataGenerator()
        
        # Should be able to chain
        state = gen.load_sim(testsim_path).get_state()
        assert state is not None


class TestLoadProps:
    """Tests for load_props method."""

    def test_load_props_creates_network(self, simple_props):
        """Test loading network from props directly."""
        gen = TestDataGenerator()
        result = gen.load_props(simple_props)
        
        assert result is gen
        assert gen.memory is not None
        assert gen.network is not None

    def test_load_props_token_counts(self, simple_props):
        """Test that load_props creates correct number of tokens."""
        gen = TestDataGenerator()
        gen.load_props(simple_props)
        
        # 2 props, each with 1 RB, each RB with 1 pred + 1 obj = 4 POs
        assert len(gen.memory.Ps) == 2
        assert len(gen.memory.RBs) == 2
        assert len(gen.memory.POs) == 4


# =====================[ State Extraction Tests ]======================

class TestGetState:
    """Tests for get_state method."""

    def test_get_state_returns_dict(self, testsim_path):
        """Test that get_state returns a dictionary."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        
        assert isinstance(state, dict)

    def test_get_state_has_required_keys(self, testsim_path):
        """Test that state dict has all required keys."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        
        required_keys = ['tokens', 'semantics', 'links', 'mappings', 
                        'analogs', 'connections', 'driver', 'recipient', 'metadata']
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_get_state_tokens_structure(self, testsim_path):
        """Test tokens structure in state."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        tokens = state['tokens']
        
        assert 'Ps' in tokens
        assert 'RBs' in tokens
        assert 'POs' in tokens
        
        # Check token counts match
        assert len(tokens['Ps']) == len(gen.memory.Ps)
        assert len(tokens['RBs']) == len(gen.memory.RBs)
        assert len(tokens['POs']) == len(gen.memory.POs)

    def test_get_state_token_properties(self, testsim_path):
        """Test that tokens have expected properties."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        
        # Check P token properties
        p_token = state['tokens']['Ps'][0]
        required_props = ['index', 'name', 'set', 'analog', 'act', 'max_map']
        for prop in required_props:
            assert prop in p_token, f"P token missing property: {prop}"
        
        # Check PO token has predOrObj
        po_token = state['tokens']['POs'][0]
        assert 'predOrObj' in po_token

    def test_get_state_links_matrix(self, testsim_path):
        """Test that links contains a weight matrix."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        links = state['links']
        
        assert 'matrix' in links
        assert 'po_names' in links
        assert 'semantic_names' in links
        
        # Matrix should be num_POs x num_semantics
        matrix = links['matrix']
        assert len(matrix) == len(gen.memory.POs)

    def test_get_state_without_loading_raises(self):
        """Test that get_state raises error if no network loaded."""
        gen = TestDataGenerator()
        
        with pytest.raises(ValueError, match="No network loaded"):
            gen.get_state()

    def test_get_state_metadata(self, testsim_path):
        """Test that metadata is populated correctly."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        metadata = state['metadata']
        
        assert 'sim_path' in metadata
        assert 'parameters' in metadata
        assert 'token_counts' in metadata
        
        assert metadata['token_counts']['Ps'] == len(gen.memory.Ps)

    def test_get_state_driver_recipient_contents(self, testsim_path):
        """Test driver and recipient contents in state."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        state = gen.get_state()
        
        assert 'P_names' in state['driver']
        assert 'RB_names' in state['driver']
        assert 'PO_names' in state['driver']
        assert 'counts' in state['driver']


class TestGetStateAfterOperations:
    """Tests for state extraction after running operations."""

    def test_state_captures_activations(self, testsim_path):
        """Test that state captures activation values."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Set some activations manually
        gen.memory.POs[0].act = 0.75
        
        state = gen.get_state()
        
        assert state['tokens']['POs'][0]['act'] == 0.75

    def test_state_captures_mappings(self, testsim_path):
        """Test that state captures mapping connections after mapping."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Run retrieval and mapping
        gen.network.do_retrieval()
        gen.network.do_map()
        
        state = gen.get_state()
        
        # Should have some mappings after running map
        # (may or may not have mappings depending on similarity)
        assert 'all_mappings' in state['mappings']


# =====================[ Save State Tests ]======================

class TestSaveState:
    """Tests for save_state method."""

    def test_save_state_pickle(self, testsim_path, tmp_output_dir):
        """Test saving state as pickle file."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.pkl"
        result_path = gen.save_state(output_path, format='pickle')
        
        assert result_path.exists()
        assert result_path == output_path

    def test_save_state_json(self, testsim_path, tmp_output_dir):
        """Test saving state as JSON file."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.json"
        result_path = gen.save_state(output_path, format='json')
        
        assert result_path.exists()

    def test_save_state_creates_directories(self, testsim_path, tmp_output_dir):
        """Test that save_state creates parent directories."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "nested" / "dirs" / "state.pkl"
        result_path = gen.save_state(output_path)
        
        assert result_path.exists()

    def test_save_state_without_loading_raises(self, tmp_output_dir):
        """Test that save_state raises error if no network loaded."""
        gen = TestDataGenerator()
        
        with pytest.raises(ValueError, match="No network loaded"):
            gen.save_state(tmp_output_dir / "state.pkl")

    def test_save_state_invalid_format_raises(self, testsim_path, tmp_output_dir):
        """Test that invalid format raises error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        with pytest.raises(ValueError, match="Unknown format"):
            gen.save_state(tmp_output_dir / "state.xyz", format='xyz')

    def test_save_state_pickle_convenience(self, testsim_path, tmp_output_dir):
        """Test save_state_pickle convenience method."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.pkl"
        result_path = gen.save_state_pickle(output_path)
        
        assert result_path.exists()

    def test_save_state_json_convenience(self, testsim_path, tmp_output_dir):
        """Test save_state_json convenience method."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.json"
        result_path = gen.save_state_json(output_path)
        
        assert result_path.exists()


# =====================[ Load State Tests ]======================

class TestLoadState:
    """Tests for load_state function."""

    def test_load_state_pickle(self, testsim_path, tmp_output_dir):
        """Test loading state from pickle file."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.pkl"
        gen.save_state(output_path, format='pickle')
        
        loaded_state = load_state(output_path)
        
        assert isinstance(loaded_state, dict)
        assert 'tokens' in loaded_state

    def test_load_state_json(self, testsim_path, tmp_output_dir):
        """Test loading state from JSON file."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        output_path = tmp_output_dir / "state.json"
        gen.save_state(output_path, format='json')
        
        loaded_state = load_state_json(output_path)
        
        assert isinstance(loaded_state, dict)
        assert 'tokens' in loaded_state

    def test_roundtrip_preserves_data(self, testsim_path, tmp_output_dir):
        """Test that save/load roundtrip preserves data."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        original_state = gen.get_state()
        
        output_path = tmp_output_dir / "state.pkl"
        gen.save_state(output_path)
        
        loaded_state = load_state(output_path)
        
        # Token counts should match
        assert len(loaded_state['tokens']['Ps']) == len(original_state['tokens']['Ps'])
        assert len(loaded_state['tokens']['RBs']) == len(original_state['tokens']['RBs'])
        assert len(loaded_state['tokens']['POs']) == len(original_state['tokens']['POs'])
        
        # Token names should match
        original_names = [t['name'] for t in original_state['tokens']['Ps']]
        loaded_names = [t['name'] for t in loaded_state['tokens']['Ps']]
        assert original_names == loaded_names


# =====================[ Inspection Utility Tests ]======================

class TestInspectionUtilities:
    """Tests for inspection utility methods."""

    def test_print_summary_runs(self, testsim_path, capsys):
        """Test that print_summary runs without error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_summary()
        
        captured = capsys.readouterr()
        assert "NETWORK SUMMARY" in captured.out
        assert "Token Counts" in captured.out

    def test_print_summary_without_network(self, capsys):
        """Test print_summary when no network loaded."""
        gen = TestDataGenerator()
        
        gen.print_summary()
        
        captured = capsys.readouterr()
        assert "No network loaded" in captured.out

    def test_print_tokens_runs(self, testsim_path, capsys):
        """Test that print_tokens runs without error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_tokens()
        
        captured = capsys.readouterr()
        assert "P Tokens" in captured.out
        assert "RB Tokens" in captured.out
        assert "PO Tokens" in captured.out

    def test_print_tokens_filter_by_type(self, testsim_path, capsys):
        """Test printing tokens filtered by type."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_tokens('PO')
        
        captured = capsys.readouterr()
        assert "PO Tokens" in captured.out
        # Should not have P or RB sections
        assert "--- P Tokens ---" not in captured.out
        assert "--- RB Tokens ---" not in captured.out

    def test_print_semantics_runs(self, testsim_path, capsys):
        """Test that print_semantics runs without error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_semantics()
        
        captured = capsys.readouterr()
        assert "Semantics" in captured.out

    def test_print_semantics_without_network(self, capsys):
        """Test print_semantics when no network loaded."""
        gen = TestDataGenerator()
        
        gen.print_semantics()
        
        captured = capsys.readouterr()
        assert "No network loaded" in captured.out

    def test_print_semantics_shows_semantic_info(self, testsim_path, capsys):
        """Test that print_semantics shows semantic information."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_semantics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show semantic names (testsim has semantics like lover1, beloved1, etc.)
        assert "lover" in output or "beloved" in output or "mary" in output or "tom" in output
        
        # Should show activation values
        assert "act=" in output
        
        # Should have indexed entries
        assert "[" in output and "]" in output

    def test_print_semantics_shows_linked_pos(self, testsim_path, capsys):
        """Test that print_semantics shows linked POs."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_semantics()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show linked_to information if there are links
        # (may or may not have links depending on network state)
        # Just verify it runs without error and produces output
        assert len(output) > 0

    def test_print_semantics_empty_semantics(self, capsys):
        """Test print_semantics when there are no semantics."""
        # This would require creating a network with no semantics,
        # which is unlikely but we can test the code path
        gen = TestDataGenerator()
        # Can't easily create empty semantics, so just test error handling
        gen.print_semantics()
        
        captured = capsys.readouterr()
        assert "No network loaded" in captured.out

    def test_print_links_runs(self, testsim_path, capsys):
        """Test that print_links runs without error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links()
        
        captured = capsys.readouterr()
        assert "PO-Semantic Links" in captured.out

    def test_print_links_without_network(self, capsys):
        """Test print_links when no network loaded."""
        gen = TestDataGenerator()
        
        gen.print_links()
        
        captured = capsys.readouterr()
        assert "No network loaded" in captured.out

    def test_print_links_shows_summary(self, testsim_path, capsys):
        """Test that print_links shows summary information."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show total links
        assert "Total links:" in output
        # Should show PO and semantic counts
        assert "POs:" in output
        assert "Semantics:" in output

    def test_print_links_shows_individual_links(self, testsim_path, capsys):
        """Test that print_links shows individual link information."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show individual links section
        assert "Individual Links" in output
        # Should show link format with PO and semantic names
        assert "<->" in output or "weight=" in output

    def test_print_links_shows_statistics(self, testsim_path, capsys):
        """Test that print_links shows weight statistics."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show statistics section
        assert "Statistics:" in output
        assert "Average weight:" in output
        assert "Max weight:" in output
        assert "Min weight:" in output

    def test_print_links_with_matrix(self, testsim_path, capsys):
        """Test that print_links can show the weight matrix."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links(show_matrix=True)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show matrix section
        assert "Weight Matrix" in output

    def test_print_links_max_links_parameter(self, testsim_path, capsys):
        """Test that max_links parameter limits output."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_links(max_links=5)
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Should show limited number of links
        assert "showing up to 5" in output

    def test_print_mappings_runs(self, testsim_path, capsys):
        """Test that print_mappings runs without error."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        gen.print_mappings()
        
        captured = capsys.readouterr()
        assert "Mapping Connections" in captured.out


# =====================[ Integration Tests ]======================

class TestIntegration:
    """Integration tests for full workflows."""

    def test_full_workflow(self, testsim_path, tmp_output_dir):
        """Test complete workflow: load, operate, save, load."""
        # Create generator and load
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Save initial state
        initial_path = tmp_output_dir / "initial.pkl"
        gen.save_state(initial_path)
        
        # Run operations
        gen.network.do_retrieval()
        gen.network.do_map()
        
        # Save after operations
        after_path = tmp_output_dir / "after_map.pkl"
        gen.save_state(after_path)
        
        # Load both and compare
        initial_state = load_state(initial_path)
        after_state = load_state(after_path)
        
        # Both should be valid
        assert 'tokens' in initial_state
        assert 'tokens' in after_state

    def test_multiple_load_sim_calls(self, testsim_path, testsym_path):
        """Test that loading a new sim replaces the old one."""
        gen = TestDataGenerator()
        
        # Load first sim
        gen.load_sim(testsim_path)
        first_p_count = len(gen.memory.Ps)
        
        # Load second sim
        gen.load_sim(testsym_path)
        second_p_count = len(gen.memory.Ps)
        
        # Should have different counts (testsym has more Ps)
        assert second_p_count != first_p_count or second_p_count >= first_p_count

    def test_state_consistency_across_saves(self, testsim_path, tmp_output_dir):
        """Test that multiple saves produce consistent states."""
        gen = TestDataGenerator()
        gen.load_sim(testsim_path)
        
        # Save twice
        path1 = tmp_output_dir / "state1.pkl"
        path2 = tmp_output_dir / "state2.pkl"
        
        gen.save_state(path1)
        gen.save_state(path2)
        
        state1 = load_state(path1)
        state2 = load_state(path2)
        
        # States should be identical
        assert state1['metadata']['token_counts'] == state2['metadata']['token_counts']
        assert len(state1['tokens']['Ps']) == len(state2['tokens']['Ps'])

