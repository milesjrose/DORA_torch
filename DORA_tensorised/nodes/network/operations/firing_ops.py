# nodes/network/operations/firing_ops.py
# Firing operations for Network class
from typing import TYPE_CHECKING
from ...enums import *
import random
import torch

if TYPE_CHECKING: # For autocomplete/hover-over docs
    from ..network import Network
    from ..sets import Driver
    from ..single_nodes import Ref_Token

class FiringOperations:
    """
    Firing operations for the Network class.
    Handles firing order management.
    """
    
    def __init__(self, network):
        """
        Initialize FiringOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network

    def make_firing_order(self, rule: str = "by_top_random") -> list[int]:
        """
        Create firing order of nodes.
            - "by_top_random": Randomise order of highest token nodes in driver. 
            - "totally_random": Randomise order of all nodes in driver.

        Args:
            rule (str): The rule to use to create the firing order

        Returns:
            A list of indices representing the firing order
        """
        # Implementation using network.sets
        pass 

    def by_top_random(self) -> list[int]:
        """
        Create a firing order of nodes in the driver.
        Randomise order of highest token nodes in driver. 
        Then, add children of these nodes to the firing order.
        If there are no nodes of a given type, return an empty list.
        
        Returns:
            A list of indices representing the firing order
        """
        highest_token_type = self.network.driver().token_ops.get_highest_token_type()
        firing_order = [] # List of indices
        match highest_token_type:
            case Type.GROUP:
                groups = self.get_random_order_of_type(Type.GROUP)  # Order the groups randomly
                pos = self.get_all_children_firing_order(groups)    # Get Ps of Groups
                rbs = self.get_all_children_firing_order(pos)       # Get RBs of the Ps 
                firing_order = groups + pos + rbs
            case Type.P:
                pos = self.get_random_order_of_type(Type.P)         # Order the Ps randomly
                rbs = self.get_all_children_firing_order(pos)       # Get RBs of the Ps 
                firing_order = pos + rbs
            case Type.RB:
                rbs = self.get_random_order_of_type(Type.RB)        # Order the RBs randomly
                firing_order = rbs
            case Type.PO:
                pos = self.get_random_order_of_type(Type.PO)        # Order the POs randomly
                firing_order = pos
            case _:
                pass # (No tokens in driver)
        return firing_order

    def get_children_firing_order(self, index: int) -> list[int]:
        """
        Get the firing order of the children of a given token.

        Args:
            index (int): The index of the token to get the children of

        Returns:
            A list of indices representing the firing order of the children of the given token
        """
        # Get children using the token operations
        return self.network.driver().token_ops.get_child_indices(index)

    def get_all_children_firing_order(self, indices: list[int]):
        """
        Get the firing order of the children of a given list of tokens.

        Args:
            indices (list[int]): A list of indices of tokens to get the children of

        Returns:
            A list of indices representing the firing order of the children of the given tokens
        """
        children = []
        for index in indices:
            children.extend(self.get_children_firing_order(index))
        return children
    
    def get_random_order_of_type(self, type: Type):
        """
        Get a random order of the given type.

        Args:
            type (Type): The type of token to get a random order of

        Returns:
            A list of indices representing the random order of the given type
        """
        mask = self.network.driver().tensor_op.get_mask(type)   # Get mask of type
        indices = torch.where(mask)[0].tolist()                 # convert to list of indices
        random.shuffle(indices)                                 # randomise indices
        return indices

    def totally_random(self):
        """
        Create a firing order by randomly shuffling either RB nodes or PO nodes.
        If RB nodes exist, they are shuffled and added to the firing order.
        Otherwise, PO nodes are shuffled and added to the firing order.
        If there are no nodes of a given type, return an empty list.

        Returns:
            A list of indices representing the firing order
        """
        firing_order = []
        if self.network.driver().tensor_op.get_mask(Type.RB).sum() > 0:
            firing_order = self.get_random_order_of_type(Type.RB)
        elif self.network.driver().tensor_op.get_mask(Type.PO).sum() > 0:
            firing_order = self.get_random_order_of_type(Type.PO)
        return firing_order