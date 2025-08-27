#nodes/requirements/predication.py
# Checks requirements for predication

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
import torch

if TYPE_CHECKING:
    from ..network import Network
    from ..connections import Mappings
    from ..sets import Recipient, Driver

class Requirements(object):
    """
    Requirements object for the network.
    Functions:
    - predication() -> Checks requirements for predication
    - rel_form() -> Checks requirements for relation formation 
    - schema() -> Checks requirements for schematisation 
    - rel_gen() -> Checks requirements for relation generalisation
    """
    def __init__(self, network: 'Network'):
        """
        Sets the network to check requirements for
        Args:
            network (Network): Network to check requirements for
        """
        self.network = network
        self.debug = True

    def predication(self):
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
        
    
    def rel_form(self):
        """
        Checks requirements for relation formation:
        - There are at least 2 RBs in the recipient that both map to RBs in the driver with mapping connections above 0.8, and that are NOT already connected to a P unit.
        """
        def check_rbs(self):
            threshold = 0.8
            recipient: 'Recipient' = self.network.recipient()
            mappings: 'Mappings' = self.network.mappings[Set.RECIPIENT]
            # Get mask of recipient RBs that don't connect to a P unit (Parent P).
            r_rb = recipient.get_mask(Type.RB)
            if r_rb.sum() < 2:
                raise ValueError(f"Only {r_rb.sum()} RBs in recipient (required at least 2)")
            r_p = recipient.get_mask(Type.P)
            t_cons = torch.t(recipient.connections)                 # Transpose to get child->parent connections.
            r_noP_rb = (t_cons[r_rb][:, r_p] == 0).all(dim=1)       # Mask of RBs that don't connect to a p unit
            if r_noP_rb.sum() < 2:
                raise ValueError(f"Only {r_noP_rb.sum()} RBs in recipient that don't connect to a P unit (required at least 2)")
            r_noP_rb = tOps.sub_union(r_rb, r_noP_rb)               # Expand mask to be size of recipient node tensor

            # Find mapping connections to RBs in the driver that are above 0.8
            map_cons = mappings[MappingFields.CONNECTIONS]
            map_weights = mappings[MappingFields.WEIGHT]
            d_rb = self.network.driver().get_mask(Type.RB)

            map_cons = map_cons[r_noP_rb][:, d_rb]                  # Get just (valid recipient_RB) -> driver_RB mappings
            map_weights = map_weights[r_noP_rb][:, d_rb]
            active_weights = map_cons * map_weights                 # NOTE: Not sure if this is required. If mapping weights are only > 0 for active connections, then this can be removed
            active_weights = active_weights[active_weights > threshold]   # Find number of connections that are above threshold
        
            if len(active_weights) < 2:
                raise ValueError(f"Only {len(active_weights)} RBs in recipient that map to RBs in the driver with mapping connections above 0.8 (required at least 2)")
        
        try:
            check_rbs(self)
            return True
        except ValueError as e:
            if self.debug:
                print(e)
            return False
        
    def schema(self):
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

            invalid_connections = (invalid_child > 0) | (invalid_parent > 0)
            fail_nodes = valid_mask & invalid_connections

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
    
    
    def rel_gen(self):
        """
        Checks requirements for relation generalisation
        """


