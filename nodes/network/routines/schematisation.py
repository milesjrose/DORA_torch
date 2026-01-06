# nodes/network/routines/schematisation.py
# Schematisation routines for Network class

from ...enums import *
import logging

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
from ..single_nodes import Token
import torch

if TYPE_CHECKING:
    from ...network import Network

logger = logging.getLogger(__name__)

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
            logger.debug("Schematisation:requirements_passed")
            return True
        except ValueError as e:
            if self.debug:
                print(e)
            logger.debug("Schematisation:requirements_failed")
            return False
    
    def shcematise_p(self, mode):
        """
        Perform schematisation for p tokens with given mode.
        """
        logger.debug("Schematisating P tokens")
        token_mask = self.network.driver().get_mask(Type.P)
        token_mask = tOps.refine_mask(self.network.driver().nodes, token_mask, TF.MODE, mode)
        ref_active = self.network.driver().token_op.get_most_active_token(mask=token_mask)
        threshold = 0.4
        # Check if most active token is active above threshold.
        if ref_active is not None and self.network.get_value(ref_active, TF.ACT) >= threshold: # NOTE: original code had both > and >=, assume combining these is fine.
            ref_made = self.network.node_ops.get_made_unit_ref(ref_active)
            logger.debug(f"- {self.network.get_ref_string(ref_active)}:made_unit={self.network.get_ref_string(ref_made)}")
            # Token has caused a token to be inferred.
            if ref_made != None:
                logger.debug(f"- {self.network.get_ref_string(ref_active)}:made_unit_exists({self.network.get_ref_string(ref_made)}) -> updating made unit (act = 1.0), connecting RBs")
                # update made (newSet) unit (act = 1.0, connect to active newSet RBs).
                self.network.set_value(ref_made, TF.ACT, 1.0)
                # Connect to any active newSet RBs
                rb_thresh = 0.5
                rb_mask = self.network.new_set().get_mask(Type.RB)
                active_mask = self.network.new_set().nodes[:, TF.ACT] >= rb_thresh
                rb_to_connect = rb_mask & active_mask
                idx_made = self.network.get_index(ref_made)
                if mode == Mode.PARENT: # Connect as parent
                    self.network.new_set().connections[idx_made, rb_to_connect] = B.TRUE
                else: # Connect as child
                    self.network.new_set().connections[rb_to_connect, idx_made] = B.TRUE  
            # Token has not caused a token to be inferred.
            else:
                # if act (already checked) and map to rec token above threshold, infer a newSet token
                max_map = self.network.get_max_map_value(ref_active, map_set=Set.RECIPIENT)
                map_threshold = 0.75
                if max_map >= map_threshold:
                    logger.debug(f"- {self.network.get_ref_string(ref_active)}:map_above_threshold({max_map}>={map_threshold}) -> inferring token")
                    self.infer_token(ref_active)
                else:
                    logger.debug(f"- {self.network.get_ref_string(ref_active)}:map_below_threshold({max_map}<{map_threshold}) -> not inferring token")
        else:
            logger.debug(f"- {self.network.get_ref_string(ref_active)}:not_active({self.network.get_value(ref_active, TF.ACT)}<{threshold}) -> not inferring token")

    def schematise_rb(self):
        """
        Perform schematisation for rb tokens.
        """
        logger.debug("Schematising RB tokens")
        token_mask = self.network.driver().get_mask(Type.RB)
        ref_most_active_token = self.network.driver().token_op.get_most_active_token(mask=token_mask)
        # Check most active RB is active.
        if ref_most_active_token is not None:
            ref_made_unit = self.network.node_ops.get_made_unit_ref(ref_most_active_token)
            logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:made_unit={self.network.get_ref_string(ref_made_unit)}")
            # Rb has caused a token to be inferred.
            if ref_made_unit != None:
                logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:made_unit_exists({self.network.get_ref_string(ref_made_unit)}) -> updating made unit (act = 1.0), connecting POs")
                # Set activation to 1.0 for made unit
                self.network.set_value(ref_made_unit, TF.ACT, 1.0)
                # Connect to any active newSet POs
                po_mask = self.network.new_set().get_mask(Type.PO)
                active_mask = self.network.new_set().nodes[:, TF.ACT] >= po_thresh
                po_to_connect = po_mask & active_mask
                po_thresh = 0.5
                made_unit_index = self.network.get_index(ref_made_unit)
                self.network.new_set().connections[made_unit_index, po_to_connect] = B.TRUE
            # Rb has not caused a token to be inferred.
            else:
                # if act and map to rec token above threshold, infer a newSet token
                active_thresh = 0.4
                map_thresh = 0.75
                if self.act_and_map_above_threshold(ref_most_active_token, active_thresh, map_thresh):
                    logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:act_and_map_above_threshold({self.network.get_value(ref_most_active_token, TF.ACT)}>={active_thresh} and {self.network.get_max_map_value(ref_most_active_token, map_set=Set.RECIPIENT)}>={map_thresh}) -> inferring token")
                    self.infer_token(ref_most_active_token)
                else:
                    logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:act_or_map_below_threshold({self.network.get_value(ref_most_active_token, TF.ACT)}<{active_thresh} or {self.network.get_max_map_value(ref_most_active_token, map_set=Set.RECIPIENT)}<{map_thresh}) -> not inferring token")
    
    def schematise_po(self):
        """
        Perform schematisation for po tokens.
        """
        logger.debug("Schematising PO tokens")
        token_mask = self.network.driver().get_mask(Type.PO)
        ref_most_active_token = self.network.driver().token_op.get_most_active_token(mask=token_mask)
        # Check most active PO is active.
        if ref_most_active_token is not None:
            ref_made_unit = self.network.node_ops.get_made_unit_ref(ref_most_active_token)
            logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:made_unit={self.network.get_ref_string(ref_made_unit)}")
            # Po has caused a token to be inferred.
            if ref_made_unit != None:
                # Update made unit activation to 1.0
                logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:made_unit_exists({self.network.get_ref_string(ref_made_unit)}) -> updating made unit (act = 1.0), updating link weights")
                self.network.set_value(ref_made_unit, TF.ACT, 1.0)
                # Get semantics connected to made unit
                made_unit_index = self.network.get_index(ref_made_unit)
                # Get shared semantics between active token and its made unit
                active_token_index = self.network.get_index(ref_most_active_token)
                active_token_sems = self.network.links[Set.NEW_SET][active_token_index, :] > 0
                made_token_sems = self.network.links[Set.NEW_SET][made_unit_index, :] > 0
                shared_sems = active_token_sems & made_token_sems
                # If shared semantics, update their link weights
                if torch.any(shared_sems):
                    self.network.links.update_link_weights(ref_made_unit, mask=shared_sems)
                else:
                    # Otherwise, update connection to any active semantics
                    active_sems = self.network.links[Set.NEW_SET][made_unit_index, SF.ACT] > 0
                    self.network.links.update_link_weights(ref_made_unit, mask=active_sems)
            # Po has not caused a token to be inferred.
            else:
                # Check if active above threshold, and map to recipient token above threshold
                act_thresh = 0.4
                map_thresh = 0.75
                if self.act_and_map_above_threshold(ref_most_active_token, act_thresh, map_thresh):
                    logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:act_and_map_above_threshold({self.network.get_value(ref_most_active_token, TF.ACT)}>={act_thresh} and {self.network.get_max_map_value(ref_most_active_token, map_set=Set.RECIPIENT)}>={map_thresh}) -> inferring token")
                    self.infer_token(ref_most_active_token)
                else:
                    logger.debug(f"- {self.network.get_ref_string(ref_most_active_token)}:act_or_map_below_threshold({self.network.get_value(ref_most_active_token, TF.ACT)}<{act_thresh} or {self.network.get_max_map_value(ref_most_active_token, map_set=Set.RECIPIENT)}<{map_thresh}) -> not inferring token")
    
    def infer_token(self, ref_maker):
        """
        Infer a newSet token (act = 1.0)
        """
        type = self.network.get_value(ref_maker, TF.TYPE)
        idx_maker = self.network.get_index(ref_maker)
        base_features = {
            TF.SET: Set.NEW_SET,
            TF.INFERRED: B.TRUE,
            TF.ACT: 1.0,
            TF.ANALOG: null,
            TF.MAKER_UNIT: idx_maker,
            TF.MAKER_SET: ref_maker.set
        }
        # Infer token
        if type == Type.P:
            base_features[TF.MODE] = self.network.get_value(ref_maker, TF.MODE)
        elif type == Type.PO:
            base_features[TF.PRED] = self.network.get_value(ref_maker, TF.PRED)
        elif type != Type.RB:
            raise ValueError(f"Invalid token type: {type}")
        new_token = Token(type, base_features)
        
        # Add token to new set and set maker/made unit features
        ref_made = self.network.add_token(new_token)
        idx_made = self.network.get_index(ref_made)
        self.network.node_ops.set_value(ref_maker, TF.MADE_UNIT, idx_made)
        self.network.node_ops.set_value(ref_maker, TF.MADE_SET, ref_made.set)
        logger.info(f"- {self.network.get_ref_string(ref_maker)} -> inferred -> maker={self.network.get_ref_string(ref_made)}")
        return ref_made

    def act_and_map_above_threshold(self, ref_token, act_thresh, map_thresh):
        """
        Check if token is active above threshold, and maps to recipient token above threshold.
        """
        act_value = self.network.get_value(ref_token, TF.ACT)
        map_value = self.network.get_max_map_value(ref_token, map_set=Set.RECIPIENT)
        return act_value >= act_thresh and map_value >= map_thresh

    def schematisation_routine(self):
        """
        Run the schematisation routine.
        """
        logger.info('Running schematisation routine')
        self.schematise_p(Mode.PARENT)
        self.schematise_p(Mode.CHILD)
        self.schematise_rb()
        self.schematise_po()