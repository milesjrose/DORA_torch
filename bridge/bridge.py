from .old_net import OldNet
from .new_net import NewNet
from .compare_states import CompareStates, Diff
import sys
from pathlib import Path
from nodes.utils.printer.print_table import OutputType
from logging import getLogger
logger = getLogger("BRIDGE")

class Bridge:
    """Bridge between old (currVers) and new (nodes/network) DORA implementations.
    
    Provides a unified interface for loading simulations into both the legacy
    currVers implementation and the new tensorised nodes/network implementation,
    extracting their states, and comparing them for validation.
    
    Attributes:
        old: OldNet object for the old implementation.
        new: NewNet object for the new implementation.
    """

    def __init__(self):
        """Initialize the Bridge with fresh generator instances."""
        currvers_dir = Path(__file__).parent / 'currVers'
        if str(currvers_dir) not in sys.path:
            sys.path.insert(0, str(currvers_dir))
        
        self.old = OldNet()
        self.new = NewNet()
        self.comp_states = CompareStates(self.old.state, self.new.state)
    
    def load_both(self, sim_path: str, use_builder: bool = False):
        """ Load the simulation into both the old and new implementations. """
        self.old.load_sim(sim_path)
        if use_builder: # Build based on sim file
            self.new.load_sim(sim_path)
        else:   # Build based of the old net state.
            self.new.set_state(self.old.get_state())
            self.new.build_network()
    
    def load_new_from_old(self):
        """ Load the new network from the old network. """
        state = self.old.get_state()
        self.new.set_state(state)
        self.new.build_network()
        self.new.state.metadata['sim_path'] = self.old.state.metadata['sim_path']
    
    def update_states(self):
        """ Update the states of both the old and new implementations to match their saved networks. """
        self.old.get_state()
        self.new.get_state()

    def set_print_output_type(self, output_type: OutputType):
        """ Set the output type for the printer. """
        self.old.printer.output_type = output_type
        self.new.printer.output_type = output_type
    
    def compare_states(self, output_diffs: bool = True, verbose: bool = False, return_diffs: bool = False) -> tuple[bool, list[Diff]]:
        """Compare the states of both loaded implementations.
        
        Extracts states from both the old and new implementations and
        performs a detailed comparison to identify any differences.

        Args:
            verbose (bool, optional): Whether to print a summary of the differences between the two states. Default False.
        Returns:
            Tuple containing:
                - match (bool): Whether states match
                - diffs (list[Diff]): list of differences
        """
        self.update_states()
        match, diffs = self.comp_states.compare(verbose=verbose)
        if output_diffs:
            self.print_diffs(diffs)
        else:
            result = "States match" if match else "States do not match"
            logger.info(result)
        if return_diffs:
            return match, diffs
        else:
            return match
        
    def compare_states_arg(self, old_state: dict, new_state: dict) -> dict:
        """Compare two state dictionaries directly.
        
        Useful for comparing saved state files without needing to reload
        simulations.
        
        Args:
            verbose (bool, optional): Whether to print a summary of the differences between the two states. Default False.
        Returns:
            Tuple containing:
                - match (bool): Whether states match
                - diffs (list[Diff]): list of differences
        """
        new_comp = CompareStates(old_state, new_state)
        return new_comp.compare()

    def print_diffs(self, diffs:list[Diff]):
        """ Print the differences. """
        self.comp_states.print_diffs(diffs)