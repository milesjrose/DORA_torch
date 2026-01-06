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
    from ..connections import Mappings
    from ..single_nodes.token import Ref_Token

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
            Chceks that all driver POs map to units in the recipient that don't have RBs
            Returns:
                bool: True if passes check, False o.w.
            """
            driver: 'Driver' = self.network.driver()
            recipient: 'Recipient' = self.network.recipient()
            mappings: 'Mappings' = self.network.mappings

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
            mappings: 'Mappings' = self.network.mappings
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
    
    def check_po_requirements(self, po: 'Ref_Token'):
        """
        Check that a PO meets the requirements for predication:
        - PO is an object
        - act > 0.6
        - mapping connection > 0.75
        - driver token act > 0.6
        """
        po_index = self.network.recipient().token_op.get_index(po)
        
        if self.network.recipient().nodes[po_index, TF.PRED] == B.TRUE:   # Check that PO is an object
            return False 
        if self.network.recipient().nodes[po_index, TF.ACT] <= 0.6: # Check act
            return False

        # Get max map for PO
        max_map_unit_index = int(self.network.recipient().nodes[po_index, TF.MAX_MAP_UNIT])
        max_map_value = self.network.recipient().nodes[po_index, TF.MAX_MAP]
        
        if max_map_value <= 0.75:
            return False
        if self.network.driver().nodes[max_map_unit_index, TF.ACT] <= 0.6:
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
        pred_set = self.inferred_pred.set
        inferred_pred_index = self.network.sets[pred_set].token_op.get_index(self.inferred_pred)

        # Update the links between new pred and active semantics (sem act>0)
        # Get active semantics, their acts, and weight of links to them
        active_sem_mask = self.network.semantics.nodes[:, SF.ACT]>0
        sem_acts = self.network.semantics.nodes[active_sem_mask, SF.ACT]
        link_weights = self.network.links.sets[pred_set][inferred_pred_index, active_sem_mask]
        # Update weights
        new_weights = 1 * (sem_acts - link_weights) * self.network.params.gamma
        self.network.links.sets[pred_set][inferred_pred_index, active_sem_mask] += new_weights

    def predication_routine_no_new_pred(self):
        """
        Run the predication routine when no new pred has been made.
        """
        # Get the most active recipient PO. If no active POs, return.
        most_active_po = self.network.recipient().token_op.get_most_active_token()
        if most_active_po is None:
            return
        
        self.check_po_requirements(most_active_po)

        # Check requirement for PO:
        if self.check_po_requirements(most_active_po): # If meets -> copy PO, infer new pred and RB.
            # 1). copy the recipient object token into newSet
            rec_po_copy = self.network.node_ops.get_token(most_active_po)   # Get recipient PO
            rec_po_copy.tensor[TF.SET] = Set.NEW_SET                        # Set set for new PO
            
            most_active_po_index = self.network.recipient().token_op.get_index(most_active_po)
            rec_po_copy.tensor[TF.MAKER_UNIT] = most_active_po_index        # Set maker unit for new PO
            
            rec_po_copy.tensor[TF.INFERRED] = B.TRUE                        # Set inferred to True
            new_po_ref = self.network.node_ops.add_token(rec_po_copy)       # Add new PO to newSet
            old_po_name = self.network.recipient().token_op.get_name(most_active_po)
            self.network.new_set().token_op.set_name(new_po_ref, old_po_name) # Set name to be same as copied object
            self.network.node_ops.set_value(most_active_po, TF.MADE_UNIT, new_po_ref.ID) # Set made unit field for most active PO
            self.network.node_ops.set_value(most_active_po, TF.MADE_SET, new_po_ref.set) # Set made set field for most active PO
            
            # 2). infer new predicate and RB tokens
            # - add tokens to newSet
            new_pred = Token(Type.PO, {TF.SET: Set.NEW_SET, TF.PRED: B.TRUE, TF.INFERRED: B.TRUE})
            new_rb = Token(Type.RB, {TF.SET: Set.NEW_SET, TF.INFERRED: B.TRUE})
            new_pred_ref = self.network.node_ops.add_token(new_pred)
            new_rb_ref = self.network.node_ops.add_token(new_rb)
            # - give new PO name 'nil' + len(memory.POs)+1
            self.network.new_set().token_op.set_name(new_pred_ref, "nil" + str(new_pred_ref.ID))
            # - give new RB name 'nil' + len(memory.POs)+1 + '+' + active_rec_PO.name
            self.network.new_set().token_op.set_name(new_rb_ref, "nil" + str(new_rb_ref.ID) + "+" + old_po_name)
            # NOTE: Doesn't seem to set these in old code? Not sure if needed?
            #self.network.node_ops.set_value(new_pred_ref, TF.MADE_UNIT, new_po_ref.ID)
            #self.network.node_ops.set_value(new_rb_ref, TF.MADE_UNIT, new_po_ref.ID)

            # 3). connect POs to RB
            new_pred_index = self.network.sets[new_pred_ref.set].token_op.get_index(new_pred_ref)
            new_rb_index = self.network.sets[new_rb_ref.set].token_op.get_index(new_rb_ref)
            new_po_index = self.network.sets[new_po_ref.set].token_op.get_index(new_po_ref)

            self.network.new_set().token_op.connect_idx(new_rb_index, new_pred_index)
            self.network.new_set().token_op.connect_idx(new_rb_index, new_po_index)

            self.made_new_pred = True
            self.inferred_pred = new_pred_ref