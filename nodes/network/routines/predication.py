# nodes/network/routines/predication.py
# Predication routines for Network class

from ...enums import *

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
import torch
from ..single_nodes.token import Token

if TYPE_CHECKING:
    from ...network import Network
    from ..sets import Recipient, Driver
    from ..tokens.connections import Mapping
    from ..single_nodes.token import Ref_Token

class PredicationOperations:
    """
    Predication operations for the Network class.
    Handles predication routines.
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
        # Helper functions
        def check_rb_po_connections(self):
            """
            Checks that all driver POs map to units in the recipient that don't have RBs
            Returns:
                bool: True if passes check, False o.w.
            """
            net: 'Network' = self.network
            driver: 'Driver' = net.driver()
            recipient: 'Recipient' = net.recipient()
            mappings: 'Mapping' = net.mappings

            # TODO: Check all the edge cases. Don't know if all driver POs have to be mapped or not.
            # Get masks
            d_po = driver.tensor_op.get_mask(Type.PO)
            if not torch.any(d_po): return True  # No driver POs so they can't map to anything -> True?
            r_po = recipient.tensor_op.get_mask(Type.PO)
            if not torch.any(r_po): return False # No recipient POs, so driver POs can't map to them -> False
            
            # Get mask of recipient POs that are mapped to by driver POs
            map_cons = mappings[MappingFields.WEIGHT]
            mapped_r_po = (map_cons[r_po][:, d_po]> 0.0).any(dim=1)
            mapped_r_po = tOps.sub_union(r_po, mapped_r_po)
            if not torch.any(mapped_r_po): return False # No recipient POs mapped to, so false.

            # Use mask to find RBs connected to mapped recipient POs
            r_rb_mask = recipient.tensor_op.get_mask(Type.RB)
            r_connected_rbs = (recipient.get_connections()[mapped_r_po][:, r_rb_mask] == 1)
            return not bool(r_connected_rbs.any())
    
        def check_weights(self):
            """
            Checks that all driver POs map to a recipient PO with weight above threshold (=.8)
            Returns:
                bool: True if passes check, False o.w.
            """
            threshold = 0.8
            net: 'Network' = self.network
            mappings: 'Mapping' = net.mappings
            recipient: 'Recipient' = net.recipient()
            driver: 'Driver' = net.driver()

            # Get masks
            d_po = driver.tensor_opget_mask(Type.PO)
            r_po = recipient.tensor_op.get_mask(Type.PO)

            # Check that mapped recipient nodes are all POs
            map_cons = mappings[MappingFields.CONNECTIONS]
            mapped_r_mask = (map_cons[:, d_po] == 1).any(dim=1)  # Which recipient nodes are mapped to
            # Check if any mapped recipient nodes are NOT POs
            if (mapped_r_mask & ~r_po).any():
                raise ValueError("Mapped recipient nodes are not all POs")
            
            # Check that all the mapped weights are above 0.8
            map_weights = mappings[MappingFields.WEIGHT]
            driver_po_mask = driver.tensor_op.get_mask(Type.PO)
            active_maps = map_cons[:, driver_po_mask] == 1
            active_weights = map_weights[:, driver_po_mask][active_maps]

            min_weight = min(active_weights.tolist())
            return bool(min_weight >= threshold)

        # No idea why I did the checks using assertions, feels like this is a bad way to do it.
        try:
            return check_rb_po_connections(self) and check_weights(self)
        except ValueError as e:
            if self.debug:
                print(e)
            return False
    
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