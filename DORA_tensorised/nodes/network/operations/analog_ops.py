# nodes/network/operations/analog_ops.py
# Analog operations for Network class

from ...enums import *
import torch
from ..single_nodes import Ref_Analog, Analog, Ref_Token
from ...utils import tensor_ops as tOps

from typing import TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..network import Network

class AnalogOperations:
    """
    Analog operations for the Network class.
    Handles analog management and related functionality.
    """
    
    def __init__(self, network):
        """
        Initialize AnalogOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
    
    def copy(self, analog: Ref_Analog, to_set: Set):
        """
        Copy an analog from one set to another.

        Args:
            analog (Ref_Analog): The analog to copy.
            to_set (Set): The set to copy the analog to.

        Returns:
            Ref_Analog: Reference to the new analog.
        """
        analog_obj: Analog = self.get_analog(analog)       # get copy of analog
        analog_obj.set = to_set                            # set the set of the analog
        return self.add_analog(analog_obj)                 # add to new set, return the new analog reference

    def delete(self, analog: Ref_Analog):
        """
        Delete an analog from a set.

        Args:
            analog (Ref_Analog): The analog to delete.
        """
        indices = self.get_analog_indices(analog)                               # Get indices of tokens in the analog
        self.network.sets[analog.set].tensor_op.del_token_indicies(indices)     # Delete the tokens - this will also delete the links to semantics/connections

    def move(self, analog: Ref_Analog, to_set: Set):
        """
        Move an analog from one set to another, deleting the old analog.

        Args:
            analog (Ref_Analog): The analog to move.
            to_set (Set): The set to move the analog to.
        
        Returns: 
            Ref_Analog: Reference to the new analog.
        """
        new_analog = self.copy(analog, to_set)
        self.delete(analog)
        return new_analog

    def check_set_match(self):
        """
        Check that the tokens in an analog are from the correct set

        Returns:
            Lists of analog references that do not match the set.
        """
        results = []
        for set in Set:
            analogs = self.network.sets[set].token_op.get_analogs_where_not(TF.SET, set)
            results.append(analogs)
        return results
    
    def check_for_copy(self):
        """
        Check for analogs in memory that have set != memory.

        Returns:
            List[Ref_Analog]: References to the analogs that have set != memory.
        """
        analogs = self.network.sets[Set.MEMORY].token_op.get_analogs_where_not(TF.SET, Set.MEMORY)
        return analogs
    
    def clear_set(self, analog: Ref_Analog):
        """
        Clear the set feature to "memory" for tokens in an analog.
        NOTE: Doesn't move the analog to memory set.

        Args:
            analog (Ref_Analog): The analog to clear the set feature for.
        """
        self.set_analog_features(analog, TF.SET, Set.MEMORY)

    def make_AM_copy(self):
        """
        Copy any analogs with set != memory to AM.

        Returns:
            List[Ref_Analog]: References to the analogs that were copied to AM.
        """
        # TODO: Still need to check if should delete from memory set after copy - as think when am set is cleared, ill move then back to memory anyway.
        analogs = self.check_for_copy()
        copied_analogs = []
        for analog in analogs:
            analog_obj = self.get_analog(analog)                  # Get the analog object
            analog_obj.retrieve_lower_tokens()                    # Set lower tokens to same set as analog
            analog_obj.remove_memory_tokens()                     # Remove memory tokens
            new_ref = self.add_analog(analog_obj)                 # Add the analog to the new set
            copied_analogs.append(new_ref)
        return copied_analogs

    def get_analog(self, analog: Ref_Analog):
        """ Get an analog from the network. """
        return self.network.sets[analog.set].get_analog(analog)
    
    def add_analog(self, analog: Analog):
        """
        Add an analog to the network, based on objects set field.

        Args:
            analog (Analog): The analog to add.

        Returns:
            Ref_Analog: Reference to the new analog.
        """
        return self.network.sets[analog.set].add_analog(analog)
    
    def get_analog_indices(self, analog: Ref_Analog):
        """ Get the indices of the tokens in an analog. """
        return self.network.sets[analog.set].token_op.get_analog_indices(analog)
    
    def set_analog_features(self, analog: Ref_Analog, feature: TF, value):
        """ Set a feature of the tokens in an analog. """
        indices = self.get_analog_indices(analog)
        self.network.sets[analog.set].token_op.set_features(indices, feature, value)
    
    def find_mapped_analog(self, set:Set):
        """
        Find the analog in a set that is mapped to - used in rel_gen.
        """
        # Find the a po that has max_map > 0.0, then return its analog.
        self.network.mapping_ops.get_max_maps()
        po_mask = self.network.sets[set].get_mask(Type.PO)
        map_pos = self.network.sets[set].nodes[po_mask, TF.MAX_MAP] > 0.0
        full_map_pos = tOps.sub_union(po_mask, map_pos)
        indices = torch.nonzero(full_map_pos)
        if indices.shape[0] == 0:
            logger.error("No POs with max_map > 0.0 in recipient.")
            return None
        else:
            ref_po = self.network.sets[set].token_op.get_reference(index=indices[0])
            logger.debug(f"Recip_analog_token:{self.network.get_ref_string(ref_po)}")
            analog_number = self.network.node_ops.get_value(ref_po, TF.ANALOG)
            return Ref_Analog(analog_number, set)
    
    def find_mapping_analog(self) -> list[Ref_Analog]:
        """
        Find analogs that have a mapping connection in the recipient
        TODO: need to add test
        """
        self.network.mapping_ops.get_max_maps(set=Set.RECIPIENT) # update max_map for recipient tokens
        all_tks = self.network.recipient().tensor_op.get_all_nodes_mask()
        map_tokens = self.network.recepient().nodes[all_tks, TF.MAX_MAP] > 0 # Get tokens with mapping connections
        if map_tokens.sum() == 0:
            logger.debug("RECIPIENT: No tokens with mapping connections")
        else:
            return self.network.recipient().tensor_op.get_analog_ref_list(map_tokens)
    
    def move_mapping_analogs_to_new(self) -> Ref_Analog:
        """
        Move any analogs in the recipient that have a mapping connection to a new analog
        TODO: add test

        Returns:
        - Ref_Analog: the new analog
        """
        map_analogs = self.find_mapping_analog()
        new_id = float(self.network.recipient().tensor_ops.get_new_analog_id())
        for analog in map_analogs:
            self.set_analog_features(analog, TF.ANALOG, new_id)
        self.network.recipient().tensor_op.analog_node_count() # Update analog counts

    def new_set_to_analog(self):
        """
        Put the tokens in the new set into their own analog.
        """
        # Putting all newSet tokens into analog, so just set all analog to 1
        self.network.new_set().token_ops.set_features_all(TF.ANALOG, 1.0)

    def print_analog(self, analog: Ref_Analog):
        """
        print the names of the tokens in an analog
        """
        self.network.sets[analog.set].print_analog(analog)
