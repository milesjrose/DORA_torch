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
    
    def copy(self, analog: int, to_set: Set) -> int:
        """
        Copy an analog from one set to another.

        Args:
            analog (int): The number of the analog to copy.
            to_set (Set): The set to copy the analog to.

        Returns:
            int: The number of the new analog.
        """
        return self.network.tokens.analog_ops.copy_analog(analog, to_set)

    def delete(self, analog: int):
        """
        Delete an analog from a set.

        Args:
            analog (int): The number of the analog to delete.
        """
        self.network.tokens.analog_ops.delete_analog(analog)

    def move(self, analog: int, to_set: Set):
        """
        Move an analog from one set to another

        Args:
            analog (int): The number of the analog to move.
            to_set (Set): The set to move the analog to.
        """
        self.network.tokens.analog_ops.move_analog(analog, to_set)
    
    def check_for_copy(self) -> torch.Tensor:
        """
        Check for analogs in memory that have set != memory.

        Returns:
            Torch.tensor: The analogs that have set != memory.
        """
        analogs = self.network.sets[Set.MEMORY].analog_ops.get_analogs_where_not(TF.SET, Set.MEMORY)
        return analogs
    
    def clear_set(self, analog: int):
        """
        Clear the set feature to "memory" for tokens in an analog.

        Args:
            analog (int): The number of the analog to clear the set feature for.
        """
        self.set_analog_features(analog, TF.SET, Set.MEMORY)


    #Checking subtokens not done fully I dont think. At least for moving into recipient.
    def make_AM_copy(self):
        """
        Copy any analogs with set != memory to AM.

        Returns:
            List[int]: The new analog numbers of the copied analogs.
        """
        tokens = self.network.tokens
        # NOTE: Don't delete from memory after copy.
        analogs = self.check_for_copy()
        new_analogs = []
        for analog in analogs:
            # first get list of all analog indicies
            all_analog_indices = tokens.analog_ops.get_analog_indices(analog)
            # then find all tokens in the analog that have set != memory, and what that value is
            non_memory_tokens = tokens.token_tensor.get_tokens_where_not(TF.SET, Set.MEMORY, all_analog_indices)
            if len(non_memory_tokens) == 0:
                continue  # skip if no non-memory tokens - shouldn't happen but just in case
            to_set = Set(int(tokens.token_tensor.get_feature(non_memory_tokens, TF.SET)[0].item()))
            # get their lower tokens
            lower_tokens = tokens.connections.get_children_recursive(non_memory_tokens)
            # combine to get all indices to copy
            indices = torch.cat([non_memory_tokens, lower_tokens])
            # remove duplicates - shouldn't be any but just in case
            indices = torch.unique(indices)
            # copy to AM
            new_indices = tokens.copy_tokens(indices, to_set, connect_to_copies=True)
            # update with new analog number
            new_analog_number = tokens.analog_ops.new_analog_id()
            tokens.token_tensor.set_feature(new_indices, TF.ANALOG, new_analog_number)
            new_analogs.append(new_analog_number)
        self.network.cache_sets()
        self.network.cache_analogs()
        return new_analogs
    
    def make_AM_move(self):
        """
        Move any analogs with set != memory to AM.
        """
        analogs = self.check_for_copy()
        # Find all tokens not in memory, set their sub-tokens to the same set.
        for check_set in [Set.DRIVER, Set.RECIPIENT]:
            set_tokens = self.network.tokens.token_tensor.get_tokens_where(TF.SET, check_set)
            if not torch.any(set_tokens):
                continue  # skip if no tokens in set
            lower_tokens = self.network.tokens.connections.get_children_recursive(set_tokens)
            self.network.tokens.token_tensor.set_features(lower_tokens, TF.SET, check_set)
        self.network.cache_sets()
        self.network.cache_analogs()

    def get_analog(self, analog: Ref_Analog):
        """ Get an analog from the network. """
        # Don't think this is needed anymore
        raise NotImplementedError("get_analog is not implemented for AnalogOperations")
    
    def add_analog(self, analog: Analog):
        """
        Add an analog to the network, based on objects set field.

        Args:
            analog (Analog): The analog to add.

        Returns:
            Ref_Analog: Reference to the new analog.
        """
        # Don't think this is needed anymore
        raise NotImplementedError("add_analog is not implemented for AnalogOperations")
    
    def get_analog_indices(self, analog: int) -> torch.Tensor:
        """ Get the indices of the tokens in an analog. """
        return self.network.tokens.analog_ops.get_analog_indices(analog)
    
    def set_analog_features(self, analog: int, feature: TF, value):
        """ Set a feature of the tokens in an analog. """
        indices = self.get_analog_indices(analog)
        self.network.tokens.token_tensor.set_features(indices, feature, value)
    
    def find_mapped_analog(self, set:Set) -> int:
        """
        Find the analog in a set that is mapped to - used in rel_gen.
        """
        self.network.mapping_ops.get_max_maps(set=[set]) # update max_map for set tokens
        # Find the a po that has max_map > 0.0, then return its analog.
        mapped_pos = self.network.sets[set].token_op.get_mapped_pos()
        if not torch.any(mapped_pos):
            raise ValueError(f"No mapped POs in set {set.name}")
        analog = int(self.network.tokens.token_tensor.get_feature(mapped_pos[0], TF.ANALOG).item())
        return analog
    
    def find_mapping_analog(self) -> torch.Tensor:
        """
        Find analogs that have a mapping connection in the recipient
        """
        self.network.mapping_ops.get_max_maps(set=[Set.RECIPIENT]) # update max_map for recipient tokens
        map_tokens = self.network.driver().lcl[:, TF.MAX_MAP] > 0.0
        analogs = self.network.driver().lcl[map_tokens, TF.ANALOG]
        if not torch.any(analogs):
            logger.debug(f"RECIPIENT: No tokens with mapping connections")
            return None
        return torch.unique(analogs)

    def move_mapping_analogs_to_new(self) -> int:
        """
        Move any analogs in the recipient that have a mapping connection to a new analog

        Returns:
        - Ref_Analog: the new analog
        """
        map_analogs = self.find_mapping_analog()
        if map_analogs is None:
            return None # No mapping analogs found
        indices = self.network.tokens.analog_ops.get_analog_indices_multiple(map_analogs)
        new_id = self.network.tokens.analog_ops.new_analog_id()
        self.network.tokens.token_tensor.set_feature(indices, TF.ANALOG, new_id)
        self.network.cache_sets()
        self.network.cache_analogs()
        return new_id

    def new_set_to_analog(self) -> int:
        """
        Put the tokens in the new set into their own analog.
        Returns:
            int: The new analog number.
        """
        new_id = self.network.tokens.analog_ops.new_analog_id()
        self.network.sets[Set.NEW_SET].token_op.set_features_all(TF.ANALOG, new_id)
        return new_id
