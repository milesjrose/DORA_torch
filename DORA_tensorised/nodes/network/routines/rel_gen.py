# nodes/network/routines/rel_gen.py
# Relation generalisation routines for Network class

from ...enums import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...network import Network
    from ..connections import Mappings

class RelGenOperations:
    """
    RelGen operations for the Network class.
    Handles relation generalisation routines.
    """
    def __init__(self, network):
        """
        Initialize RelGenOperations with reference to Network.
        """
        self.network: 'Network' = network

    def requirements(self):
        """
        Checks requirements for relation generalisation:
        - At least one driver unit maps to a recipient unit.
        - All driver units that have mapping connections have weight > threshold (=.7)
        """

        def check_maps(self):
            threshold = 0.7
            mappings: 'Mappings' = self.network.mappings[Set.RECIPIENT] # Driver -> Recipient mappings 

            # Check that at least one driver unit maps to a recipient unit
            map_cons = mappings[MappingFields.CONNECTIONS]
            if not (map_cons == 1).any():
                raise ValueError("No driver units map to a recipient unit")

            # Check that all map weights are above threshold
            map_weights = mappings[MappingFields.WEIGHT]
            masked_weights = map_weights[map_cons == 1] # Mask only active connections
            if (masked_weights < threshold).any():
                raise ValueError("Some driver units have mapping connections with weight below threshold")
            
        try:
            check_maps(self)
            return True
        except ValueError as e:
            if self.debug:
                print(e)
            return False

    def rel_gen_routine(self):
        """
        Run the relation generalisation routine.
        """
        pass