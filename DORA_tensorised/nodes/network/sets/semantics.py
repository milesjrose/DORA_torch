# nodes/network/sets/semantics.py
# Represents the semantics set of tokens.

import torch

from ...enums import *
from ...utils import tensor_ops as tOps

from ..single_nodes import Ref_Semantic
from ..connections import Links
from ..network_params import Params
from ..single_nodes import Semantic

from .base_set import Base_Set

class Semantics(object):
    """
    A class for representing semantics nodes.

    Attributes:
        IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
        names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
        nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
        links (Links): A Links object containing links from token sets to semantics.
        params (Params): An object containing shared parameters. Defaults to None.
    """
    def __init__(self, nodes, connections, IDs: dict[int, int], names: dict[int, str] = None):
        """
        Initialise a Semantics object

        Args:
            nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
            connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
            IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
            names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
        Raises:
            ValueError: If the number of semantics in nodes, connections, and links do not match.
            ValueError: If the number of features in nodes does not match the number of features in SF enum.
        """
        if nodes.size(dim=0) != connections.size(dim=0):
            raise ValueError("nodes and connections must have the same number of semantics.")
        if nodes.size(dim=1) != len(SF):
            raise ValueError("nodes must have number of features listed in SF enum.")
        if names is not None:
            if not isinstance(names, dict):
                raise ValueError(f"names must be a dictionary, not {type(names)}.")
            if not all(isinstance(name, str) for name in names.values()):
                # get types that are not strings
                non_strings = [type(name) for name in names.values() if type(name) != str]
                raise ValueError(f"names must be a dictionary of strings, not {non_strings}.")
        self.names = names 
        """Map ID to name string"""
        self.nodes: torch.Tensor = nodes
        """Semantic nodes tensor"""
        self.connections: torch.Tensor = connections
        """Same-set connections for semantics"""
        self.links: Links = None
        """Semantic links to each token set"""
        self.IDs = IDs
        """Map ID to index in tensor"""
        self.params = None
        """Shared parameters"""
        self.expansion_factor = 1.1
        """Factor to expand when adding sem to full tensor"""
    
    def add_semantic(self, semantic: Semantic):
        """
        Add a semantic to the semantics tensor.

        Args:
            semantic (Semantic): The semantic to add.
        """
        deleted_mask = self.nodes[:, SF.DELETED] == B.TRUE          # find all deleted semantics in nodes tensor
        if not deleted_mask.any():                                  # if no deleted semantics, expand tensor
            self.expand_tensor()
        empty_rows = torch.where(deleted_mask)[0]                   # find all empty rows in nodes tensor
        empty_row = empty_rows[0]                                   # find first empty row
        self.nodes[empty_row, :] = semantic.nodes                   # add semantic to empty row
        new_id = self.IDs.keys()[-1] + 1                            # get new id
        self.IDs[new_id] = empty_row                                # add id to IDs
        if semantic.name is None:
            semantic.name = f"Semantic {new_id}"
        self.names[new_id] = semantic.name                          # add name to names
        self.nodes[empty_row, SF.ID] = new_id                       # set node id feature

    def expand_tensor(self):
        """
        Expand the nodes, connections, and links tensors by the expansion factor.
        """
        current_size = self.nodes.size(dim=0)
        new_size = int(current_size * self.expansion_factor)        # calculate new size
        new_nodes = torch.zeros(new_size, self.nodes.size(dim=1))   # create new nodes tensor
        new_nodes[current_size:, SF.DELETED] = 1                    # set all deleted to 1 for all new nodes
        new_nodes[:current_size, :] = self.nodes                    # copy over old nodes
        self.nodes = new_nodes                                      # update nodes

        new_cons = torch.zeros(new_size, new_size)                  # create new connections tensor
        new_cons[:current_size, :current_size] = self.connections   # copy over old connections
        self.connections = new_cons                                 # update connections

        if self.links is not None:
            for set in Set:                                         # expand links for each set
                self.expand_links(set, new_size)

    def expand_links(self, set: Set, new_size: int):
        """
        Expand the links tensors to new_size
        """
        links = self.links[set]
        current_num_token = links.size(dim=0)                       # links is Token x Semantics tensor
        current_num_sem = links.size(dim=1)                         # expand by expansion factor
        new_links = torch.zeros(current_num_token, new_size)        # create new links tensor (same number of tokens, new size for semantics)
        new_links[:current_num_token, :current_num_sem] = links     # copy over old links
        self.links[set] = new_links                                 # update links

    def del_semantic(self, ID):                                     # Delete a semantic from the semantics tensor. TODO: Remove connections and links.
        """
        Delete a semantic from the semantics tensor.
        """
        self.nodes[self.IDs[ID], SF.DELETED] = B.TRUE
        self.IDs.pop(ID)
        self.names.pop(ID)
        
        if self.links is not None:
            for set in Set:
                self.links[set][:, self.IDs[ID]] = 0.0

        self.connections[self.IDs[ID], :] = 0.0
        self.connections[:, self.IDs[ID]] = 0.0

    def get_count(self):
        """Get the number of semantics in the semantics tensor."""
        return self.nodes.shape[0]
    
    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get(self, ref_semantic: Ref_Semantic, feature):
        """
        Get a feature for a semantic with a given ID.
        
        Args:
            ref_semantic (Ref_Semantic): The semantic to get the feature for.
            feature (TF): The feature to get.

        Returns:
            The feature for the semantic.
        """

        try:
            return self.nodes[self.get_index(ref_semantic), feature]
        except:
            raise ValueError("Invalid reference semantic or feature.")

    def set(self, ref_semantic: Ref_Semantic, feature, value):
        """
        Set a feature for a semantic with a given ID.
        
        Args:
            ref_semantic (Ref_Semantic): The semantic to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.

        Raises:
            TypeError: If the feature is not a TF enum.
            ValueError: If the ID or feature is invalid.
        """

        try:
            self.nodes[self.get_index(ref_semantic), feature] = float(value)
        except:
            raise ValueError("Invalid reference semantic or feature.")
        
    def get_index(self, ref_semantic: Ref_Semantic):
        """
        Get the index for a semantic based on a reference semantic.

        Args:
            ref_semantic (Ref_Semantic): The reference semantic.

        Returns:
            The index of the semantic.

        Raises:
            ValueError: If the reference semantic is invalid.
        """
        try:
            return self.IDs[ref_semantic.ID]
        except:
            raise ValueError("Invalid reference semantic.")
    
    def get_reference(self, id=None, index=None, name=None):
        """
        Get the reference for a semantic using any of the following:
        - ID
        - index
        - name

        Args:
            id (int, optional): The ID of the semantic.
            index (int, optional): The index of the semantic.
            name (str, optional): The name of the semantic.

        Returns:
            A Ref_Semantic object.

        Raises:
            ValueError: If the ID, index, or name is invalid. Or if none are provided.
        """
        if index is not None:
            try:
                id = self.nodes[index, SF.ID]
            except:
                raise ValueError("Invalid index.")
        elif name is not None:
            try:
                # I feel like there is a better way to do this
                dict_index = list(self.names.values()).index(name)
                id = list(self.names.keys())[dict_index]
            except:
                raise ValueError("Invalid name.")
        elif id is not None:
            try:
                self.get_reference(id=id)                           # check if id is valid
            except:
                raise ValueError("Invalid ID.")
        else:
            raise ValueError("No ID, index, or name provided.")
        
        try:
            name = self.names[id]
        except:
            name = None

        return Ref_Semantic(id, name)
    
    def get_single_semantic(self, ref_semantic: Ref_Semantic, copy=True):
        """
        Get a single semantic from the semantics tensor.

        - If copy is set to False, changes to the returned semantic will affect the semantic set tensor.

        Args:
            ref_semantic (Ref_Semantic): The reference semantic.
            copy (bool, optional): Whether to use a copy of the semantic sub-tensor. Defaults to True.

        Returns:
            A Semantic object.
        
        Raises:
            ValueError: If the reference semantic is invalid.
        """
        tensor = self.nodes[self.get_index(ref_semantic), :]
        sem = Semantic(self.names[ref_semantic.ID], {SF.TYPE: Type.SEMANTIC})
        if copy:
            sem.tensor = tensor.clone()
        else:
            sem.tensor = tensor
        return sem
    # --------------------------------------------------------------

    # ===================[ SEMANTIC FUNCTIONS ]=====================
    def init_sem(self):                                             # Set act and input to 0 TODO: Check how used
        """Initialise the semantics """
        self.nodes[:, SF.ACT] = 0.0
        self.nodes[:, SF.INPUT] = 0.0

    def init_input(self, refresh):                                  # Set nodes to refresh value TODO: Check how used
        """Initialise the input of the semantics """
        self.nodes[:, SF.INPUT] = refresh

    def set_max_input(self, max_input):                             # set max input of all semantics
        """Set the max input of the semantics """
        self.nodes[:, SF.MAX_INPUT] = max_input
    
    def get_max_input(self):                                        # Get the max input in semantics
        """Get the maximum input in semantics """
        return self.nodes[:, SF.INPUT].max()

    def update_act(self):                                           # Update act of all sems
        """Update the acts of the semantics """
        sem_mask = self.nodes[:, SF.MAX_INPUT] > 0                  # Get sem where max_input > 0
        input = self.nodes[sem_mask, SF.INPUT]
        max_input = self.nodes[sem_mask, SF.MAX_INPUT]
        self.nodes[sem_mask, SF.ACT] = input / max_input            # - Set act of sem to input/max_input
        sem_mask = self.nodes[:, SF.MAX_INPUT] == 0                 # Get sem where max_input == 0       
        self.nodes[sem_mask, SF.ACT] = 0.0                          #  -  Set act of sem to 0
    
    def update_input(self, driver, recipient, memory = None, ignore_obj=False):
        """
        Update the input of the semantics
        - Note, if memory is not provided, equivalent to "ignore_mem = True"
        
        Args:
            driver (Base_Set): The driver set.
            recipient (Base_Set): The recipient set.
            memory (Base_Set, optional): The memory set. Defaults to None.
            ignore_obj (bool, optional): Whether to ignore the object set. Defaults to False.

        Raises:
            ValueError: If ignore_obj is set to False and no memory is provided.
        """
        self.update_input_from_set(driver, Set.DRIVER, ignore_obj)
        self.update_input_from_set(recipient, Set.RECIPIENT, ignore_obj)
        if memory is not None:
            self.update_input_from_set(memory, Set.MEMORY, ignore_obj)

    def update_input_from_set(self, tensor: Base_Set, set: Set, ignore_obj=False):
        """Update the input of the semantics from a set of tokens """
        if self.links is None:
            raise ValueError("Links not initialised. Should be set when network is created.")
        
        # Get mask of POs
        if ignore_obj:
            po_mask = tOps.refine_mask(po_mask, tensor.get_mask(Type.PO), TF.PRED, B.TRUE) # Get mask of POs non object POs
        else:
            po_mask = tensor.get_mask(Type.PO)
        #group_mask = tensor.get_mask(Type.GROUP)
        #token_mask = torch.bitwise_or(po_mask, group_mask)             # In case groups used in future

        # Update based on linked tokens
        links: torch.Tensor = self.links[set]
        connected_nodes_sub = (links[po_mask, :] != 0).any(dim=1)       # Mask all PO that have a link to a sem
        connected_nodes = tOps.sub_union(po_mask, connected_nodes_sub)  # Resize mask to full tensor size
        connected_sem = (links[po_mask, :] != 0).any(dim=0)             # Mask all sems that have a link to a PO

        links_cons = torch.transpose(links[connected_nodes][:, connected_sem], 0, 1)

        sem_input = torch.matmul(                                       # Get sum of act * link_weight for all connected nodes and sems
            links_cons,                                                 # connected_sem x connected_nodes matrix of link weights
            tensor.nodes[connected_nodes, TF.ACT]                       # connected_nodes x 1 matrix of node acts
        )
        self.nodes[connected_sem, SF.INPUT] += sem_input                # Update input of connected sems
    # --------------------------------------------------------------

