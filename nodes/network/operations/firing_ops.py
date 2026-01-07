# nodes/network/operations/firing_ops.py
# Firing operations for Network class
from typing import TYPE_CHECKING

from ...enums import *
from random import shuffle
import torch
from logging import getLogger
from ..tokens.tensor_view import TensorView
logger = getLogger(__name__)

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
        Initialise FiringOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
        self.firing_order = None

    def make_firing_order(self, rule: str = None) -> list[int]:
        """
        Create firing order of driver tokens, using specified rule (uses rule in network params if not specified).
            - "by_top_random": Randomise order of highest token nodes in driver. 
            - "totally_random": Randomise order of all nodes in driver.

        Args:
            rule (str, optional): Rule used to create order. If not specified in arg or params, defaults to "by_top_random".

        Returns:
            list[int]: A list of global indices representing the firing order // NOTE: Could change to local, depending on usage.
        """
        if rule is None:
            rule = self.network.params.firing_order_rule
        if rule is None:
            rule = "by_top_random"
        logger.debug(f"Creating firing order with rule: {rule}")
        match rule:
            case "by_top_random":
                return self.by_top_random()
            case "totally_random":
                return self.totally_random()
            case _:
                logger.error(f"Invalid firing order rule: {rule}, if rule not specified in call, check network params.")
                raise ValueError(f"Invalid firing order rule: {rule}")

    def by_top_random(self) -> list[int]:
        """
        Create a firing order of nodes in the driver.
        Randomise order of highest token nodes in driver.
        Then, add children of these nodes to the firing order.
        If there are no nodes of a given type, return an empty list.
        
        Returns:
            list[int]: A list of global indices representing the firing order
        """
        driver = self.network.driver()
        cons = self.network.tokens.get_view(TensorTypes.CON, Set.DRIVER) # get local view of connections in driver
        highest_token_type = driver.token_op.get_highest_token_type()
        self.firing_order = []
        # Get the indices of the tokens at each level
        match highest_token_type:
            case Type.GROUP:
                 # randomly order groups
                groups = self.get_random_order_of_type(Type.GROUP)
                # get ordered list of children
                pos = self.get_lcl_child_idxs(groups, cons, Type.PO)
                rbs = self.get_lcl_child_idxs(pos, cons, Type.RB)
                self.firing_order = groups + pos + rbs
            case Type.P:
                # randomly order Ps
                pos = self.get_random_order_of_type(Type.PO)
                # get ordered list of children
                rbs = self.get_lcl_child_idxs(pos, cons, Type.RB)       # Get RBs of the Ps 
                self.firing_order = pos + rbs
            case Type.RB:
                # randomly order RBs
                rbs = self.get_random_order_of_type(Type.RB)
                self.firing_order = rbs
            case Type.PO:
                # randomly order POs
                pos = self.get_random_order_of_type(Type.PO)
                self.firing_order = pos
            case _:
                logger.error(f"Tried to make firing order, but no tokens in driver.")
        return self.network.to_global(self.firing_order).tolist()
    
    def totally_random(self) -> list[int]:
        """
        Create a firing order by randomly shuffling either RB nodes or PO nodes.
        If RB nodes exist, they are shuffled and added to the firing order.
        Otherwise, PO nodes are shuffled and added to the firing order.
        If there are no nodes of a given type, return an empty list.

        Returns:
            list[int]: A list of global indices representing the firing order
        """
        driver = self.network.driver()
        if driver.tensor_op.get_count(Type.RB) > 0:
            self.firing_order = self.get_random_order_of_type(Type.RB)
        elif driver.tensor_op.get_count(Type.PO) > 0:
            self.firing_order = self.get_random_order_of_type(Type.PO)
        else:
            logger.error(f"Tried to make firing order, but no tokens in driver.")
            self.firing_order = []
        return self.network.to_global(self.firing_order).tolist()
    
    def get_random_order_of_type(self, type: Type) -> list[int]:
        """
        Get randomly shuffled list of local indices of tokens of the given type.
        Args:
            type: Type - The type of the tokens to get the order of.
        Returns:
            list[int]: A list of local indices representing the order of the tokens.
        """
        mask = self.network.driver().tensor_op.get_mask(type)
        indices = torch.where(mask)[0].tolist()
        shuffle(indices)
        return indices

    def get_lcl_child_idxs(self, indices: torch.Tensor, local_cons: TensorView, child_type: Type, set=Set.DRIVER) -> list[int]:
        """
        Get the local indices of the children of the given indices. Filters out higher token children (i.e RB with child P.).
        Returns the chilren in the order of the input indices. (i.e first index of output is the child of the first index of input.)
        Args:
            indices: torch.Tensor - The indices of the tokens to get the children of.
            local_cons: TensorView - The local connections tensor.
            child_type: Type - The type of the children to get.
            set: Set - The set to get the children from.
        Returns:
            torch.Tensor - The local indices of the children of the given indices.
        """
        # NOTE: Loop seems a bit inefficient, but not sure how to maintain order otherwise.
        type_mask = self.network.sets[set].tensor_op.get_mask(child_type)
        output = []
        for index in indices:
            cons_mask = local_cons[index, :] == True # mask of connections to children of index
            output.extend(torch.where(cons_mask & type_mask)[0].tolist())
        return output
