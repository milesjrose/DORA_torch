# nodes/network/routines/schematisation.py
# Schematisation routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
import torch

if TYPE_CHECKING:
    from ...network import Network

class SchematisationOperations:
    """
    Schematisation operations for the Network class.
    Handles schematisation routines.
    """
    
    def __init__(self, network):
        """
        Initialize SchematisationOperations with reference to Network.
        """
        self.network: 'Network' = network
        self.debug = False
    
    def requirements(self):
        """
        Check requirments for schematisation:
        - All driver and recepient mapping connections are above threshold (=.7)
        - Parents/Children of these mapped tokens are mapped with weight above threshold
        """
        threshold = 0.7
        # Check recipient nodes

        def check_set(self, set: 'Set'):
            tensor = self.network.sets[set]
            max_maps = tensor.nodes[:, TF.MAX_MAP]
            cons = tensor.connections
            valid_mask = max_maps >= threshold
            invalid_mask = ~valid_mask

            # Check for any nodes with 0 < max_map < threshold
            if torch.any((max_maps > 0) & (max_maps < threshold)):
                raise ValueError(f"Nodes with 0 < max_map < threshold found in {set} set")

            # Check for connections to invalid nodes
            invalid_child = torch.matmul(
                cons,
                invalid_mask.float()
            )

            invalid_parent = torch.matmul(
                torch.t(cons),              # Transpose to get parent->child connections.
                invalid_mask.float()
            )

            invalid_connections = (invalid_child > 0) | (invalid_parent > 0) # Get all nodes that connect to an invalid node
            fail_nodes = valid_mask & invalid_connections                    # Get all nodes that are valid but connect to an invalid node

            if torch.any(fail_nodes):
                raise ValueError(f"Failing nodes found in {set} set")

        try:
            check_set(self, Set.DRIVER)
            check_set(self, Set.RECIPIENT)
            return True
        except ValueError as e:
            if self.debug:
                print(e)
            return False

    def schematisation_routine(self):
        """
        Run the schematisation routine.
        """
        pass