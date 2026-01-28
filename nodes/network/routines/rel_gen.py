# nodes/network/routines/rel_gen.py
# Relation generalisation routines for Network class

import torch
import logging

from ...enums import *
from ..single_nodes import Token, Ref_Token, Ref_Analog

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.network import Network
    from nodes.network.tokens import Mapping
    from nodes.network.sets import Driver

logger = logging.getLogger("rtn")

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
        threshold = 0.7
        mappings: 'Mapping' = self.network.mappings # Driver -> Recipient mappings 

        # Check that at least one driver unit maps to a recipient unit
        if not (mappings[MappingFields.WEIGHT] > 0.0).any():
            logger.debug("RelGen req failed : No mapping connections")
            return False

        # Check that all active mapping connections are above threshold
        map_weights = mappings[MappingFields.WEIGHT]
        below_thresh = map_weights < threshold
        non_zero = map_weights > 0.0
        if (below_thresh & non_zero).any():
            logger.debug("RelGen req failed : found mapping weights between 0.0 and threshold")
            return False
        logger.debug("RelGen req passed :)")
        return True

    def infer_token(self, maker: int, recip_analog: int, set: Set) -> int:
        """
        Infer a token in the given set, with act = 1.0, inferred = True, and maker/made unit features set.
        Copies type specific features (P mode/PO pred) from maker to inferred token.
        NOTE: Old code puts inferred token into both new_set and recipient, but can only assign token to one set with tensors.
              Currently infers one token into the recipient - as driver token can only hold one made token at a time.
        TODO: See what I should do about this??

        Args:
            maker (int): The maker token index.
            recip_analog (int): The recipient analog reference.
            set (Set): The set to infer the token in.
        Returns:
            (int): The index of the inferred token.
        """
        # Create new token
        type = self.network.node_ops.get_tk_value(maker, TF.TYPE)
        base_features = {
            TF.SET: set,
            TF.INFERRED: B.TRUE,
            TF.ACT: 1.0,
            TF.ANALOG: recip_analog,
            TF.MAKER_UNIT: maker,
            TF.MAKER_SET: self.network.node_ops.get_tk_value(maker, TF.SET)
        }
        match type:
            case Type.P:
                base_features[TF.MODE] = self.network.node_ops.get_tk_value(maker, TF.MODE)
            case Type.PO:
                base_features[TF.PRED] = self.network.node_ops.get_tk_value(maker, TF.PRED)
            case Type.RB:
                pass
            case _:
                raise ValueError(f"Invalid token type: {type}")
        new_token = Token(type, base_features)
        
        # Add token to network and set maker/made unit features
        made = self.network.node_ops.add_token(new_token)
        self.network.node_ops.set_tk_value(maker, TF.MADE_UNIT, made)
        # NOTE: Don't think the maker set is needed anymore, since indexes are global now.
        self.network.node_ops.set_tk_value(maker, TF.MADE_SET, self.network.node_ops.get_tk_value(maker, TF.SET))
        logger.info(f"- {self.network.get_ref_string(made)} -> inferred -> maker={self.network.get_ref_string(maker)}")
        return made

    def rel_gen_type(self, type: Type, threshold: float, recip_analog: int, p_mode:Mode = None):
        """
        Run relational generalisation for a given token type:
          - Find the most active driver unit of given type.
          - Check if this token has created a token in the recipient:
                - True -> Update the inferred token's act to 1 and update connections to lower nodes.
                - False -> Infer a new token in the recipient with act = 1.0
        
        Args:
            type (Type): The type of token to perform rel gen for.
            threshold (float): The threshold for the active token to be considered active.
            recip_analog (int): The recipient analog number.
            p_mode (Mode): The mode of the P token to perform rel gen for.
        """
        net: 'Network' = self.network
        driver: 'Driver' = net.driver()
        # Mask tokens
        if type == Type.P:
            if p_mode is None:
                logger.critical("p_mode is not set for rel gen type >:[")
                raise ValueError("p_mode is None")
            token_mask = driver.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: p_mode})
        else:
            token_mask = driver.tensor_op.get_mask(type)

        # Get most active token, if no active token, return
        active_in_mask_idx = driver.token_op.get_most_active_token(local_mask=token_mask)
        # Check if no active token found (must check before indexing, as None index unsqueezes tensors)
        if active_in_mask_idx is None:
            if p_mode is not None:
                logger.debug(f"No active {p_mode.name} {type.name} token found")
            else:
                logger.debug(f"No active {type.name} token found")
            return
        # The idx is the position in the mask, so we need to convert to the local driver index
        mask_indices = torch.where(token_mask)[0]
        active_lcl = mask_indices[active_in_mask_idx]

        # check if active above threshold and max map is 0.0
        act = driver.token_op.get_feature(active_lcl, TF.ACT)
        max_map = net.get_max_map_value_local(active_lcl, Set.DRIVER)
        if not (act >= threshold and max_map == 0.0):
            logger.debug(f"- Driver[{active_lcl}]:active_below_threshold({act}<threshold) or max_map_is_not_zero({max_map}!=0.0) -> not updating made unit")
            return
        
        # If active unit inferred token: F-> Infer new token, T-> Update made unit act, 
        made_glbl = driver.tkop.get_feature(active_lcl, TF.MADE_UNIT)
        if made_glbl == null:
            logger.debug(f"- Driver[{active_lcl}]:no_made_token -> inferring new token")
            # infer a new token in the recipient, and new_set TODO: Check how to correctly port the old code putting tokens into both sets.
            active_glbl = net.to_global(active_lcl, Set.DRIVER)
            made_recipient = self.infer_token(active_glbl, recip_analog, Set.RECIPIENT)
            #made_new_set = self.infer_token(active_glbl, recip_analog, Set.NEW_SET)
        else:
            # Set act of inferred token to 1.0
            logger.debug(f"- Driver[{active_lcl}]:made_unit_exists({net.get_ref_string(made_glbl)}) -> updating made unit (act = 1.0), connecting tokens")
            net.node_ops.set_tk_value(made_glbl, TF.ACT, 1.0)

            # Update inferred tokens connections/links to nodes below it. Note type of inferred token is the same as the active token.
            match type:
                case Type.PO:
                    # Update semantic connections.
                    net.semantics.update_link_weights(made_glbl)
                case Type.RB:
                    # Get most active PO.
                    active_po_lcl = net.recipient().token_op.get_most_active_token(token_type=Type.PO)
                    if active_po_lcl is None: # This isn't handled in old code, so assume not possible. Double check just in case.
                        logger.critical("No active PO found for rel gen, no idea how to handle this :/")
                        return
                    active_po_glbl = net.to_global(active_po_lcl, Set.RECIPIENT)
                    # If act >= 0.7, then connect (as child)
                    if net.node_ops.get_tk_value(active_po_glbl, TF.ACT) >= 0.7:
                        net.tokens.connections.connect(made_glbl, active_po_glbl)
                case Type.P:
                    # Get most active RB.
                    active_rb_lcl = net.recipient().token_op.get_most_active_token(token_type=Type.RB)
                    if active_rb_lcl is None: # This isn't handled in old code, so assume not possible. Double check just in case
                        logger.critical("No active RB found for rel gen, no idea how to handle this :/")
                        return
                    active_rb_glbl = net.to_global(active_rb_lcl, Set.RECIPIENT)
                    rb_act = net.node_ops.get_tk_value(active_rb_glbl, TF.ACT)
                    # Update connections.
                    if p_mode == Mode.CHILD:
                        # if act >= 0.7, then connect (P as child)
                        if rb_act >= 0.7:
                            net.tokens.connections.connect(active_rb_glbl, made_glbl)
                    elif p_mode == Mode.PARENT:
                        # Check if RB already has a P as a parent.
                        parent_p = net.tokens.arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.PARENT})
                        rb_has_parent_p = net.tokens.connections.tensor[parent_p, active_rb_glbl].any()
                        # if act >= 0.5, then connect (P as parent)
                        # and RB does not already have a P as parent.
                        if rb_act >= 0.5 and not rb_has_parent_p:
                            net.tokens.connections.connect(made_glbl, active_rb_glbl)

    def rel_gen_routine(self, recip_analog: int):
        """
        Run the relational generalisation routine.

        Note: Only RBs in the driver from the analog that contains mapped tokens are firing.

        If active driver token maps to no unit in the recipient -> infer a token in recipient with act = 1.0 and
        connect it to active nodes above and below itself (e.g RB to Ps & POs; PO to RBs & Sems; etc) NOTE: Does this happen?
        
        Mark that the new unit is inferred from the active driver unit (update made and maker features)

        Inferred tokens are assigned to the recipient analog mapped to the driver tokens driving rel gen.
        (e.g Driver analog 1, maps to recipient analog 3, then inferred token is assigned to analog 3)

        - For each token type (PO, RB, P.child, P.parent):
          - Find the most active driver unit of that type.
          - If this token has created a unit:
            - True -> Update the unit's act to 1 and update connections to lower tokens.
            - False -> Infer a new unit in the recipient
        
        Args:
            recip_analog (int):The recipient analog reference.
        """
        logger.debug("RelGen routine started")
        self.rel_gen_type(Type.PO, 0.5, recip_analog)
        self.rel_gen_type(Type.RB, 0.5, recip_analog)
        self.rel_gen_type(Type.P, 0.5, recip_analog, Mode.CHILD)
        self.rel_gen_type(Type.P, 0.5, recip_analog, Mode.PARENT)
