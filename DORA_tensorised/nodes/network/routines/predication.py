# nodes/network/routines/predication.py
# Predication routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Recipient, Driver
    from ..connections import Mappings

class PredicationOperations:
    """
    Predication operations for the Network class.
    Handles predication routines.
    """
    
    def __init__(self, network):
        """
        Initialize PredicationOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
    
    def requirements(self):
        """
        Checks requirements for predication:
        - All driver POs map to units in the recipient that don't have RBs
        - All driver POs map to a recipient PO with weight above threshold (=.8)
        """
        # Helper functions
        def check_rb_po_connections(self):
            """
            Chceks that all driver POs map to units in the recipient that don't have RBs
            Returns:
                bool: True if passes check, False o.w.
            """
            driver: 'Driver' = self.network.driver()
            recipient: 'Recipient' = self.network.recipient()
            mappings: 'Mappings' = self.network.mappings[Set.RECIPIENT]

            d_po = driver.get_mask(Type.PO)
            r_po = recipient.get_mask(Type.PO)
            
            # Get mask of recipient POs that are mapped to by driver POs
            map_cons = mappings[MappingFields.CONNECTIONS]
            mapped_r_po = (map_cons[r_po][:, d_po]== 1).any(dim=1)
            mapped_r_po = tOps.sub_union(r_po, mapped_r_po)

            # Use mask to find RBs connected to mapped recipient POs
            r_rb_mask = recipient.get_mask(Type.RB)
            r_connected_rbs = (recipient.connections[mapped_r_po][:, r_rb_mask] == 1)
            return not bool(r_connected_rbs.any())
    
        def check_weights(self):
            """
            Checks that all driver POs map to a recipient PO with weight above threshold (=.8)
            Returns:
                bool: True if passes check, False o.w.
            """
            threshold = 0.8
            mappings: 'Mappings' = self.network.mappings[Set.RECIPIENT]
            recipient: 'Recipient' = self.network.recipient()
            driver: 'Driver' = self.network.driver()

            # Get masks
            d_po = driver.get_mask(Type.PO)
            r_po = recipient.get_mask(Type.PO)

            # Check that mapped recipient nodes are all POs
            map_cons = mappings[MappingFields.CONNECTIONS]
            mapped_r_mask = (map_cons[:, d_po] == 1).any(dim=1)  # Which recipient nodes are mapped to
            # Check if any mapped recipient nodes are NOT POs
            if (mapped_r_mask & ~r_po).any():
                raise ValueError("Mapped recipient nodes are not all POs")
            
            # Check that all the mapped weights are above 0.8
            map_weights = mappings[MappingFields.WEIGHT]
            driver_po_mask = driver.get_mask(Type.PO)
            active_maps = map_cons[:, driver_po_mask] == 1
            active_weights = map_weights[:, driver_po_mask][active_maps]

            min_weight = min(active_weights.tolist())
            return bool(min_weight >= threshold)
    
        try:
            return check_rb_po_connections(self) and check_weights(self)
        except ValueError as e:
            if self.debug:
                print(e)
            return False

    def predication_routine(self):
        """
        Run the predication routine.
        """
        pass