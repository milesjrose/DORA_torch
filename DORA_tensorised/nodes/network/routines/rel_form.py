# nodes/network/routines/rel_form.py
# Relation formation routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
from ..single_nodes import Token, Ref_Token
import torch

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Recipient
    from ..connections import Mappings

class RelFormOperations:
    """
    RelForm operations for the Network class.
    Handles relation formation routines.
    """
    
    def __init__(self, network):
        """
        Initialize RelFormOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
        " Reference to the Network object. "
        self.debug: bool = False
        " Debug flag. "
        self.inferred_new_p: bool = False
        " Flag to indicate if a new P was inferred. "
        self.inferred_p: Ref_Token = None
        " Reference to the inferred P token. "
    
    def requirements(self):
        """
        Checks requirements for relation formation:
        - There are at least 2 RBs in the recipient that both map to RBs in the driver with mapping connections above 0.8, and that are NOT already connected to a P unit.
        """
        def check_rbs(self):
            threshold = 0.8
            recipient: 'Recipient' = self.network.recipient()
            mappings: 'Mappings' = self.network.mappings
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

    def rel_form_routine(self):
        """
        Run the relation formation routine:
        - If new P has been inferred, connect it to RBs with act >= threshold (0.8).
        - Else, infer a new P in recipient
        """
        if self.inferred_new_p:
            # Connect new P to RBs with act >= threshold
            if self.inferred_p is None:
                raise ValueError("Inferred P token is not set.")
            threshold = 0.8
            rb_mask = self.network.recipient().get_mask(Type.RB)
            active_mask = self.network.recipient().nodes[:, TF.ACT] >= threshold
            rb_to_connect = rb_mask & active_mask
            infered_p_index = self.network.get_index(self.inferred_p)
            self.network.recipient().token_op.connect_idx(infered_p_index, rb_to_connect)
        else:
            new_p_name = "" # Name should be RB1+RB2+...RBx. For now leave blank and name after phase set. NOTE: Why?
            new_p = Token(Type.P, {TF.SET: Set.RECIPIENT, TF.INFERRED: B.TRUE})
            ref_new_p = self.network.add_token(new_p)
            if ref_new_p is None:
                raise ValueError("Failed to add new P token to recipient.")
            self.network.set_name(ref_new_p, new_p_name)
            self.inferred_new_p = True
            self.inferred_p = ref_new_p
    
    def name_inferred_p(self):
        """Give the inferred p a name baseed on its RBs."""
        if self.inferred_p is None:
            raise ValueError("Inferred P token is not set.")
        rbs = self.network.recipient().get_connected_tokens(self.inferred_p)
        if len(rbs) == 0:
            raise ValueError("Hey, you got a an error awhile ago that you were unable to reproduce. Basically, it seems you learned a P unit with no RBs (or something to that effect). You added a try/except to catch it in case it popped up again. It has. You will want to look very carefully at what happened with the latest P unit that has been made.") # Yoinked this debug message from runDORA.py
        name_string = self.network.node_ops.get_name(rbs[0])
        for rb in rbs[1:]:
            name_string += "+" + self.network.node_ops.get_name(rb)
        self.network.set_name(self.inferred_p, name_string)
        