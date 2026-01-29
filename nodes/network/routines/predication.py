# nodes/network/routines/predication.py
# Predication routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
import torch
from ..single_nodes.token import Token
from logging import getLogger
logger = getLogger("rtn")

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Recipient, Driver
    from ..tokens.connections import Mapping
    from ..single_nodes.token import Ref_Token

class PredicationOperations:
    """
    Implements DORA's predication learning mechanism, which learns single-place predicates 
    from objects by discovering their shared features through comparison. All the driver POs map
    strongly to recipient units that are not already bound to RBs. We look at the most active recipient
    PO, and if it strongly maps to a driver PO (and both POs have high activation), we can infer a new
    predicate, and an RB to connect it to the recipient PO. If we have inferred a new predicate, 
    we can then update the predicates links to semantics based on the coactivation of the semantics 
    and the predicate. This allows the predicate to learn to respond to the common features of 
    the two mapped POs.
    
    Method:

    If no new predicate has been inferred, and the most active recipient PO meets requirments 
    (object with act > 0.6, max_map > 0.75, max_map_unit act > 0.6):
        - Copy the object token to newSet
        - Infer a new predicate (PO with pred=True) and RB token
        - Connect the RB to both the predicate and the object
        - Update semantic connections for the new predicate based on active semantics
    
    If a new predicate has been inferred:
        - Refine the predicate by updating its semantic links
          based on the coactivation of semantics and the predicate
    
    Requirements:
        - Driver POs must map to recipient POs 
        - Mapping connections must be above threshold (0.8)
    """
    
    def __init__(self, network: 'Network'):
        """
        Initialize PredicationOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
        self.debug: bool = False
        self.made_new_pred: bool = False
        self.inferred_pred: Ref_Token = None
    
    def requirements(self):
        """
        Checks requirements for predication:
        - All driver POs map to units in the recipient that don't have RBs
        - All driver POs map to a recipient PO with weight above threshold (=.8)
        """
        net: 'Network' = self.network
        driver: 'Driver' = net.driver()
        recipient: 'Recipient' = net.recipient()
        mappings: 'Mapping' = net.mappings

        threshold = 0.8

        # Get the mappings from driver POs to recipient POs
        d_po = driver.tensor_op.get_mask(Type.PO)
        r_po = recipient.tensor_op.get_mask(Type.PO)
        if not torch.any(d_po):
            logger.debug("PredReq Failed: No driver POs")
            return False
        if not torch.any(r_po):
            logger.debug("PredReq Failed: No recipient POs")
            return False
        p_maps = mappings[MappingFields.WEIGHT][r_po][:, d_po]
    
        # 1). Check that all driver POs have a mapping above the threshold.
        max_maps = p_maps.max(dim=0)
        if torch.any(max_maps <= threshold):
            logger.debug("PredReq Failed: Driver POs do not all have a mapping above threshold")
            return False
        
        # 2). Check that all recipient POs that are mapped to are not already connected to RBs.
        mapped_r_po = (p_maps > threshold).any(dim=1)
        mapped_r_po = tOps.sub_union(r_po, mapped_r_po) # Expand mask to size of recipient
        r_rb = recipient.tensor_op.get_mask(Type.RB)
        rec_cons = recipient.get_connections(custom_view=False)
        r_to_rb = rec_cons[mapped_r_po][:, r_rb] == 1
        if r_to_rb.any():
            logger.debug("PredReq Failed: Mapped Recipient POs already connected to RBs.")
            return False
        else:
            return True
    
    def check_po_requirements(self, po: int):
        """
        Check that a PO meets the requirements for predication:
        - PO is an object
        - act > 0.6
        - mapping connection > 0.75
        - driver token act > 0.6
        """
        tokens = self.network.token_tensor.tensor
        
        if tokens[po, TF.PRED] == B.TRUE:   # Check that PO is an object
            return False 
        if tokens[po, TF.ACT] <= 0.6: # Check act
            return False

        # Get max map for PO
        max_map_unit_index = int(tokens[po, TF.MAX_MAP_UNIT])
        max_map_value = tokens[po, TF.MAX_MAP]
        
        if max_map_value <= 0.75:
            return False
        if tokens[max_map_unit_index, TF.ACT] <= 0.6:
            return False
        return True

    def predication_routine(self):
        """
        Run the predication routine.
        """
        if self.made_new_pred:
            self.predication_routine_made_new_pred()
        else:
            self.predication_routine_no_new_pred()

    def predication_routine_made_new_pred(self):
        """
        Run the predication routine when a new pred has been made.
        """
        pred = self.inferred_pred

        # Update the links between new pred and active semantics (sem act>0)
        # Get active semantics, their acts, and weight of links to them
        sems = self.network.semantics.nodes
        active_sem_mask = sems[:, SF.ACT]>0
        sem_acts = sems[active_sem_mask, SF.ACT]
        link_weights = self.network.links[pred, active_sem_mask]
        # Update weights
        new_weights = 1 * (sem_acts - link_weights) * self.network.params.gamma
        self.network.links[pred, active_sem_mask] += new_weights

    def predication_routine_no_new_pred(self):
        """
        Run the predication routine when no new pred has been made.

        If most active PO meets requirements, copy PO to newSet, infer new pred and RB, and connect the new RB to the copied/inferred PO tokens.
        """
        # Get the most active recipient PO. If no active POs, return.
        # NOTE: switching between local and global indices here a bunch, probably should just add a method in the network for most active token.
        most_active_po = self.network.recipient().token_op.get_most_active_token(Type.PO)
        if most_active_po is None:
            return
        most_active_po = self.network.to_global(most_active_po, Set.RECIPIENT)

        # Check requirement for PO:
        if self.check_po_requirements(most_active_po): # If meets -> copy PO, infer new pred and RB.
            tk_tensor = self.network.token_tensor
            old_po_name = tk_tensor.get_name(most_active_po)
            
            # 1). copy the recipient object token into newSet
            new_po = self.network.token_tensor.copy_tokens(most_active_po, Set.NEW_SET)
            new_po = int(new_po[0].item())
            # Set features for new PO, and copy over name.
            tk_tensor.set_feature(new_po, TF.MAKER_UNIT, most_active_po)
            tk_tensor.set_feature(new_po, TF.INFERRED, B.TRUE)
            self.network.set_name(new_po, old_po_name)
            # Set made unit for old PO
            tk_tensor.set_feature(most_active_po, TF.MADE_UNIT, new_po)

            # 2). infer new predicate and RB tokens
            # - add tokens to newSet
            new_pred = Token(Type.PO, {TF.SET: Set.NEW_SET, TF.PRED: B.TRUE, TF.INFERRED: B.TRUE})
            new_rb = Token(Type.RB, {TF.SET: Set.NEW_SET, TF.INFERRED: B.TRUE})
            new_pred_ref = self.network.node_ops.add_token(new_pred)
            new_rb_ref = self.network.node_ops.add_token(new_rb)
            # - give new PO name 'nil' + len(memory.POs)+1
            po_count = tk_tensor.cache.get_type_mask(Type.PO).sum()
            tk_tensor.set_name(new_pred_ref, "nil" + str(po_count+1))
            # - give new RB name 'nil' + len(memory.POs)+1 + '+' + active_rec_PO.name
            tk_tensor.set_name(new_rb_ref, "nil" + str(po_count+1) + "+" + old_po_name)
            # NOTE: Doesn't seem to set these in old code? Not sure if needed?
            #tk_tensor.set_feature(new_pred_ref, TF.MADE_UNIT, new_po)
            #tk_tensor.set_feature(new_rb_ref, TF.MADE_UNIT, new_po)

            # 3). connect POs to RB
            tk_tensor.connections.connect(new_rb, new_pred)
            tk_tensor.connections.connect(new_rb, new_po)

            # 4). Update state
            self.made_new_pred = True
            self.inferred_pred = new_pred