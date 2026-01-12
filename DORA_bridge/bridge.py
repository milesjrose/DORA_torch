"""Bridge module for comparing old (currVers) and new (nodes/network) DORA implementations.

This module provides utilities for loading simulations into both implementations
and comparing their states to validate the new tensorized implementation.
"""

from .load_network_from_json import NetworkLoader
from .test_data_generator import TestDataGenerator
from .new_network_state_generator import NewNetworkStateGenerator
import sys
from pathlib import Path
from .utils import *
from nodes.enums import B


class Bridge:
    """Bridge between old (currVers) and new (nodes/network) DORA implementations.
    
    Provides a unified interface for loading simulations into both the legacy
    currVers implementation and the new tensorised nodes/network implementation,
    extracting their states, and comparing them for validation.
    
    Attributes:
        NetworkLoader: Loader for reconstructing networks from JSON state files.
        TestDataGenerator: Generator for extracting state from the old implementation.
        NewNetworkStateGenerator: Generator for extracting state from the new implementation.
    
    Example:
        >>> bridge = Bridge()
        >>> # Load the same simulation into both implementations
        >>> old_net = bridge.load_sim_old('sims/testsim15.py')
        >>> new_net = bridge.load_sim_new('sims/testsim15.py')
        >>> # Compare their states
        >>> differences = bridge.compare_states()
        >>> if not differences:
        ...     print("Implementations match!")
    """

    def __init__(self):
        """Initialize the Bridge with fresh generator instances."""
        self.NetworkLoader = NetworkLoader()
        self.TestDataGenerator = TestDataGenerator()
        self.NewNetworkStateGenerator = NewNetworkStateGenerator()
        currvers_dir = Path(__file__).parent / 'currVers'
        if str(currvers_dir) not in sys.path:
            sys.path.insert(0, str(currvers_dir))

    def load_sim_old(self, sim_path: str):
        """Load a simulation file into the old (currVers) implementation.
        
        Args:
            sim_path: Path to the simulation file (.py format).
        
        Returns:
            The network object from the old implementation.
        """
        self.TestDataGenerator.load_sim(sim_path)
        return self.TestDataGenerator.network
    
    def load_sim_new(self, sim_path: str):
        """Load a simulation file into the new (nodes/network) implementation.
        
        Args:
            sim_path: Path to the simulation file (.py format).
        
        Returns:
            Network: The Network object from the new implementation.
        """
        network = load_net_from_sim(sim_path)
        self.NewNetworkStateGenerator.network = network
        return network
    
    
    def get_state_old(self) -> dict:
        """Extract the current state from the old implementation.
        
        Returns:
            A dictionary containing the complete state of the old network,
            including tokens, semantics, links, mappings, and metadata.
        
        Raises:
            ValueError: If no simulation has been loaded into the old implementation.
        """
        return self.TestDataGenerator.get_state()
    
    def get_state_new(self) -> dict:
        """Extract the current state from the new implementation.
        
        Returns:
            A dictionary containing the complete state of the new network,
            including tokens, semantics, links, mappings, and metadata.
        
        Raises:
            ValueError: If no simulation has been loaded into the new implementation.
        """
        return self.NewNetworkStateGenerator.get_state()

    def compare_states(self) -> dict:
        """Compare the states of both loaded implementations.
        
        Extracts states from both the old and new implementations and
        performs a detailed comparison to identify any differences.
        
        Returns:
            A dictionary containing any differences found between the two
            implementations. Empty dict if implementations match exactly.
        
        Raises:
            ValueError: If simulations haven't been loaded into both implementations.
        """
        compared = compare_states(self.get_state_old(), self.get_state_new())
        return compared
    
    def compare_states_arg(self, old_state: dict, new_state: dict) -> dict:
        """Compare two state dictionaries directly.
        
        Useful for comparing saved state files without needing to reload
        simulations.
        
        Args:
            old_state: State dictionary from the old implementation.
            new_state: State dictionary from the new implementation.
        
        Returns:
            A dictionary containing any differences found between the two
            states. Empty dict if states match exactly.
        """
        return compare_states(old_state, new_state)
    
    def compare_connections(self) -> bool:
        """Compare the connections of both loaded implementations.
        
        Returns:
            True if the connections match, False otherwise.
        """
        old_state = self.get_state_old()
        new_state = self.get_state_new()
        connections_match = compare_connections(old_state, new_state)
        links_connections_match = compare_links_connections(old_state, new_state)
        links_weights_match = compare_links_weights(old_state, new_state)
        mappings_connections_match = compare_mappings_connections(old_state, new_state)
        mappings_weights_match = compare_mappings_weights(old_state, new_state)
        all_match = connections_match and links_connections_match and links_weights_match and mappings_connections_match and mappings_weights_match
        return all_match
    
    def print_summary_old(self):
        """Print a summary of the network loaded in the old implementation.
        
        Displays token counts, set contents, and other relevant statistics.
        """
        self.TestDataGenerator.print_summary()
    
    def print_summary_new(self):
        """Print a summary of the network loaded in the new implementation.
        
        Displays token counts, set contents, and other relevant statistics.
        """
        self.NewNetworkStateGenerator.print_summary()