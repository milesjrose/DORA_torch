# nodes/network/routines/rel_form.py
# Relation formation routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
from ..single_nodes import Token, Ref_Token
import torch

from logging import getLogger
logger = getLogger(__name__)

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Recipient, Driver
    from ..tokens.connections import Mapping
    from ..tokens.tensor.token_tensor import Token_Tensor

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
        self.inferred_p: int|None = None
        " Index of the inferred P token. "
    
    def requirements(self) -> bool:
        """
        Checks requirements for relation formation, there must be at least 2 RBs in the recipient that both:
        - Do not have connections to a P token (where P is the parent).
        - Map to RBs in the driver with mapping connections above threshold (0.8).

        Returns:
            bool: True if requirements are met, False o.w.
        """
        threshold = 0.8
        net: 'Network' = self.network
        mappings: 'Mapping' = net.mappings
        tk_tensor: 'Token_Tensor' = net.token_tensor
        cons = tk_tensor.connections.tensor

        # 1). Find RBs in the recipient that have no parent P.
        r_rb = tk_tensor.cache.get_arbitrary_mask({TF.TYPE: Type.RB, TF.SET: Set.RECIPIENT})
        if r_rb.sum() < 2:
            logger.debug(f"Only {torch.sum(r_rb)} RBs in recipient (required at least 2)")
            return False
        r_p = tk_tensor.cache.get_arbitrary_mask({TF.TYPE: Type.P, TF.SET: Set.RECIPIENT})
        if torch.any(r_p): # Only need to check if there are Ps in the recipient.
            # P is parent, RB is child, so take dim=0? TODO: Check which dim I need
            r_rb_no_p = (cons[r_p][:, r_rb] == True).all(dim=0) # Mask of RBs that don't connect to a P unit
            if r_rb_no_p.sum() < 2:
                logger.debug(f"Only {r_rb_no_p.sum()} RBs in recipient that don't connect to a P unit (required at least 2)")
                return False
            r_rb_no_p = tOps.sub_union(r_rb, r_rb_no_p) # Expand mask to size of token tensor.
            r_rb_no_p = r_rb_no_p[net.recipient().lcl._indices] # Shrink mask to be the size of the recipient, to index into mappings.

        # 2). Find at least 2 of these RBs that map to driver RBs above threshold.
        map_weights = mappings[MappingFields.WEIGHT]
        d_rb = self.network.driver().tensor_op.get_mask(Type.RB)
        passing_maps = map_weights[r_rb_no_p][:, d_rb] > threshold
        if len(passing_maps) < 2:
            logger.debug(f"Only {len(passing_maps)} RBs in recipient that map to RBs in the driver with mapping connections above 0.8 (required at least 2)")
            return False
        return True

    def rel_form_routine(self):
        """
        Run the relation formation routine:
        - If new P has been inferred, connect it to RBs with act >= threshold (0.8).
        - Else, infer a new P in recipient
        """
        if self.inferred_new_p: # Connect new P to RBs with act >= threshold
            recipient: 'Recipient' = self.network.recipient()
            tk_tensor: 'Token_Tensor' = self.network.token_tensor
            if self.inferred_p is None:
                raise ValueError("Inferred P is not set.")
            threshold = 0.8
            rb_mask = recipient.tensor_op.get_mask(Type.RB)
            active_mask = recipient.lcl[:, TF.ACT] >= threshold
            rb_to_connect = rb_mask & active_mask
            if rb_to_connect.sum() == 0:
                logger.critical("No RBs to connect new P to.")
                return
            else:
                rb_to_connect = self.network.to_global(torch.where(rb_to_connect)[0], Set.RECIPIENT) # convert to glbl indices.
                tk_tensor.connections.connect_multiple(self.inferred_p, rb_to_connect)
        else: # Infer a new P in recipient
            new_p_name = "" # Name should be RB1+RB2+...RBx. For now leave blank and name after phase set. NOTE: Why? Connections change?
            new_p_token = Token(Type.P, {TF.SET: Set.RECIPIENT, TF.INFERRED: B.TRUE}, name=new_p_name)
            new_p = self.network.node_ops.add_token(new_p_token)
            self.inferred_new_p = True
            self.inferred_p = new_p
    
    def name_inferred_p(self):
        """Give the inferred p a name baseed on its RBs."""
        if self.inferred_p is None:
            raise ValueError("Inferred P is not set.")
        rbs = self.network.token_tensor.connections.get_children(self.inferred_p).tolist()
        if len(rbs) == 0:
            # Debug message from runDORA.py, kept in case still needed.
            raise ValueError("Hey, you got a an error awhile ago that you were unable to reproduce. Basically, it seems you learned a P unit with no RBs (or something to that effect). You added a try/except to catch it in case it popped up again. It has. You will want to look very carefully at what happened with the latest P unit that has been made.")
        name_string = self.network.get_name(int(rbs[0]))
        for rb in rbs[1:]:
            name_string += "+" + self.network.get_name(int(rb))
        self.network.set_name(self.inferred_p, name_string)