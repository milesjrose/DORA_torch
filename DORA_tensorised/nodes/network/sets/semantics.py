# nodes/network/sets/semantics.py
# Represents the semantics set of tokens.

import torch

from nodes.enums import *
from nodes.utils import tensorOps as tOps

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
    def __init__(self, nodes, connections, links: Links, IDs: dict[int, int], names= None, params: Params = None):
        """
        Initialise a Semantics object

        Args:
            nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
            connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
            links (Links): A Links object containing links from token sets to semantics.
            IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
            names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
            params (Params, optional): An object containing shared parameters. Defaults to None.
        Raises:
            ValueError: If the number of semantics in nodes, connections, and links do not match.
            ValueError: If the number of features in nodes does not match the number of features in SF enum.
            ValueError: If the number of semantics in links tensors do not match the number of semantics in nodes.
        """
        if nodes.size(dim=0) != connections.size(dim=0):
            raise ValueError("nodes and connections must have the same number of semantics.")
        if nodes.size(dim=1) != len(SF):
            raise ValueError("nodes must have number of features listed in SF enum.")
        if links.driver.size(dim=1) != nodes.size(dim=0):
            raise ValueError("links tensors must have same number of semantics as semantics.nodes")
        self.names = names 
        """Map ID to name string"""
        self.nodes: torch.Tensor = nodes
        """Semantic nodes tensor"""
        self.connections: torch.Tensor = connections
        """Same-set connections for semantics"""
        self.links: Links = links
        """Semantic links to each token set"""
        self.IDs = IDs
        """Map ID to index in tensor"""
        self.params = params
        self.expansion_factor = 1.1
    
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

        for set in Set:                                             # expand links for each set
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
        
        for set in Set:
            self.links[set][:, self.IDs[ID]] = 0.0
        self.connections[self.IDs[ID], :] = 0.0
        self.connections[:, self.IDs[ID]] = 0.0

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get(self, ID, feature):
        """
        Get a feature for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to get the feature for.
            feature (TF): The feature to get.

        Returns:
            The feature for the semantic with the given ID.
        """
        try:
            return self.nodes[self.IDs[ID], feature]
        except:
            raise ValueError("Invalid ID or feature.")

    def set(self, ID, feature, value):
        """
        Set a feature for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.
        """
        if type(feature) != TF:
            raise TypeError("Feature must be a TF enum.")
        try:
            self.nodes[self.IDs[ID], feature] = float(value)
        except:
            raise ValueError("Invalid ID or feature.")

    def get_name(self, ID):
        """
        Get the name for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to get the name for.
        """
        return self.names[ID]

    def set_name(self, ID, name):
        """
        Set the name for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to set the name for.
            name (str): The name to set the semantic to.
        """
        self.names[ID] = name
    
    def get_ID(self, name):
        """
        Get the ID for a semantic with a given name.
        
        Args:
            name (str): The name of the semantic to get the ID for.
        """
        try:
            return self.IDs.keys()[self.IDs.values().index(name)]
        except:
            raise ValueError("Invalid name.")
    # --------------------------------------------------------------

    # ===================[ SEMANTIC FUNCTIONS ]=====================
    def intitialise_sem(self):                                      # Set act and input to 0 TODO: Check how used
        """Initialise the semantics """
        self.nodes[:, SF.ACT] = 0.0
        self.nodes[:, SF.INPUT] = 0.0

    def initialise_input(self, refresh):                            # Set nodes to refresh value TODO: Check how used
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
    
    def update_input(self, driver, recipient, memory = None, ignore_obj=False, ignore_mem=False):
        """Update the input of the semantics """
        self.update_input_from_set(driver, Set.DRIVER, ignore_obj)
        self.update_input_from_set(recipient, Set.RECIPIENT, ignore_obj)
        if not ignore_mem:
            self.update_input_from_set(memory, Set.MEMORY, ignore_obj)

    def update_input_from_set(self, tensor: Base_Set, set: Set, ignore_obj=False):
        """Update the input of the semantics from a set of tokens """
        if ignore_obj:
            po_mask = tOps.refine_mask(po_mask, tensor.get_mask(Type.PO), TF.PRED, B.TRUE) # Get mask of POs non object POs
        else:
            po_mask = tensor.get_mask(Type.PO)
        #group_mask = tensor.get_mask(Type.GROUP)
        #token_mask = torch.bitwise_or(po_mask, group_mask)         # In case groups used in future

        links: torch.Tensor = self.links[set]
        connected_nodes = (links[:, po_mask] != 0).any(dim=1)       # Get mask of nodes linked to a sem
        connected_sem = (links != 0).any(dim=0)                     # Get mask of sems linked to a node

        sem_input = torch.matmul(                                   # Get sum of act * link_weight for all connected nodes and sems
            links[connected_sem, connected_nodes],                  # connected_sem x connected_nodes matrix of link weights
            tensor.nodes[connected_nodes, TF.ACT]                   # connected_nodes x 1 matrix of node acts
        )
        self.nodes[connected_sem, SF.INPUT] += sem_input            # Update input of connected sems
    # --------------------------------------------------------------

