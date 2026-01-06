import pytest
import torch
import tempfile
import os

from nodes.builder import NetworkBuilder
from nodes.file_ops import save_network, load_network_new
from nodes.enums import Set, MappingFields

from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()
"""
def test_save_and_load_network(network):
    # Test saving toa newwork and loading it back
    with tempfile.TemporaryDirectory() as temp_dir:
        # ------ Setup ------
        file_path = os.path.join(temp_dir, "test_network.sym")
        save_network(network, file_path)                # Save to file
        assert os.path.exists(file_path)                # Check file created
        loaded_network = load_network_new(file_path)    # Load from file
        
        # --- Assertions ----
        
        # 1. Check Params
        assert network.params.__dict__ == loaded_network.params.__dict__    
        
        # 2. Check Inhibitors
        assert network.local_inhibitor == loaded_network.local_inhibitor
        assert network.global_inhibitor == loaded_network.global_inhibitor
        
        # 3. Check Semantics
        assert torch.equal(network.semantics.nodes, loaded_network.semantics.nodes)
        assert torch.equal(network.semantics.connections, loaded_network.semantics.connections)
        assert len(network.semantics.IDs.keys()) == len(loaded_network.semantics.IDs.keys())
        assert network.semantics.IDs == loaded_network.semantics.IDs
        assert network.semantics.names == loaded_network.semantics.names
        
        # 4. Check each Set
        for s in Set:
            original_set = network.sets[s]
            loaded_set = loaded_network.sets[s]
            
            # Check nodes/connections tensors
            assert torch.equal(original_set.nodes, loaded_set.nodes)
            assert torch.equal(original_set.connections, loaded_set.connections)
            
            # Check ID/names
            assert original_set.IDs == loaded_set.IDs
            assert original_set.names == loaded_set.names

        # 5. Check Links
        for s in Set:
            original_links = network.links[s]
            loaded_links = loaded_network.links[s]
            assert torch.equal(original_links, loaded_links)
            
        # 6. Check Mappings
        for s in Set:
            if not s == Set.NEW_SET and not s == Set.DRIVER:
                original_mappings = network.mappings[s]
                loaded_mappings = loaded_network.mappings[s]
                assert torch.equal(original_mappings.adj_matrix, loaded_mappings.adj_matrix)
"""