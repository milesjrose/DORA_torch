# nodes/network/routines/rel_gen.py
# Relation generalisation routines for Network class

import torch
import logging

from ...enums import *
from ..single_nodes import Token, Ref_Token, Ref_Analog

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...network import Network
    from ..connections import Mappings

logger = logging.getLogger(__name__)

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
        self.debug = False

    def requirements(self):
        """
        Checks requirements for relation generalisation:
        - At least one driver unit maps to a recipient unit.
        - All driver units that have mapping connections have weight > threshold (=.7)
        """

        def check_maps(self):
            threshold = 0.7
            mappings: 'Mappings' = self.network.mappings # Driver -> Recipient mappings 

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

    def infer_token(self, ref_maker, recip_analog: Ref_Analog, set: Set):
        """
        Infer a recepint token (act = 1.0)
        NOTE: Currently only infers token in recipient set. 
        Should also be in new_set, but with tensor structure this would mean duplicating tokens.
        TODO: Check to see if duplicating tokens into new_set could cause an issue later.
        """
        type = self.network.get_value(ref_maker, TF.TYPE)
        idx_maker = self.network.get_index(ref_maker)
        base_features = {
            TF.SET: set,
            TF.INFERRED: B.TRUE,
            TF.ACT: 1.0,
            TF.ANALOG: recip_analog.analog_number,
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
        self.network.set_value(ref_maker, TF.MADE_UNIT, idx_made)
        self.network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
        logger.info(f"- {self.network.get_ref_string(ref_maker)} -> inferred -> maker={self.network.get_ref_string(ref_made)}")
        return ref_made

    def rel_gen_type(self, type: Type, threshold: float, recip_analog: Ref_Analog, p_mode:Mode = None):
        """Perform rel gen routine for a given token type."""
        # Get most active token
        token_mask = self.network.driver().get_mask(type)
        if type == Type.P:
            if p_mode is None:
                logger.critical("p_mode is None")
                raise ValueError("p_mode is None")
            token_mask = self.network.driver().get_mask(type, p_mode)
        active_unit = self.network.driver().token_op.get_most_active_token(mask=token_mask)
        if active_unit is None:
            logger.debug(f"- No active {type.name} token found")
            return

        # check if active above threshold and max map is 0.0
        act = self.network.get_value(active_unit, TF.ACT)
        max_map = self.network.get_max_map_value(active_unit, map_set=Set.RECIPIENT)
        if not (act >= threshold and max_map == 0.0):
            logger.debug(f"- {self.network.get_ref_string(active_unit)}:active_below_threshold({act}<threshold) or max_map_is_not_zero({max_map}!=0.0) -> not updating made unit")
            return
        
        # check if active unit has made a token
        made_unit = self.network.node_ops.get_made_unit_ref(active_unit)
        if made_unit is None:
            logger.debug(f"- {self.network.get_ref_string(active_unit)}:no_made_unit -> inferring new unit")
            # infer a new unit in the recipient, and new_set TODO: check this
            ref_made = self.infer_token(active_unit, recip_analog, Set.RECIPIENT)
            ref_made_new_set = self.infer_token(active_unit, recip_analog, Set.NEW_SET)
        else:
            # Set act of inferred unit
            logger.debug(f"- {self.network.get_ref_string(active_unit)}:made_unit_exists({self.network.get_ref_string(made_unit)}) -> updating made unit (act = 1.0), connecting tokens")
            self.network.set_value(made_unit, TF.ACT, 1.0)
            # Update inferred units connections
            rec_mask = self.network.recipient().get_mask(type)
            match type:
                case Type.PO:
                    # update semantic connections
                    self.network.links.update_link_weights(made_unit)
                case Type.RB:
                    # Get most active PO
                    # If act >= 0.7, then connect (as child)
                    ref_active_po = self.network.recipient().token_op.get_most_active_token(mask=rec_mask)
                    if self.network.get_value(ref_active_po, TF.ACT) >= 0.7:
                        self.network.recipient().token_op.connect(made_unit, ref_active_po)
                case Type.P:
                    # Get most active RB
                    ref_active_rb = self.network.recipient().token_op.get_most_active_token(mask=rec_mask)
                    if p_mode == Mode.CHILD:
                        # if act >= 0.7, then connect (as child)
                        self.network.recipient().token_op.connect(made_unit, ref_active_rb)
                    elif p_mode == Mode.PARENT:
                        # if act >= 0.5, then connect (as parent)
                        self.network.recipient().token_op.connect(ref_active_rb, made_unit)

    def rel_gen_routine(self, recip_analog: Ref_Analog):
        """
        Run the relation generalisation routine:
        - For each token type (PO, RB, P.child, P.parent):
          - Find the most active driver unit of that type.
          - If this token has created a unit:
            - True -> Update the unit's act to 1 and update connections to lower tokens.
            - False -> Infer a new unit in the recipient
        """
        logger.debug("Running relation generalisation routine")
        self.rel_gen_type(Type.PO, 0.5, recip_analog)
        self.rel_gen_type(Type.RB, 0.5, recip_analog)
        self.rel_gen_type(Type.P, 0.5, recip_analog, Mode.CHILD)
        self.rel_gen_type(Type.P, 0.5, recip_analog, Mode.PARENT)
        

        