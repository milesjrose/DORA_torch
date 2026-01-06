# nodes/builder/build_connections.py
# Builds connections for each set.

import numpy as np
from ..enums import *

from .inter_sets import *
import torch
class Build_connections(object):                        # Builds links and connections for each set
    """
    A class for building the links and connections for each set.    

    Attributes:
        token_sets (dict): A dictionary of token sets, mapping set to token set.
        sems (Sem_set): The semantic set object.
    """
    def __init__(self, token_sets: dict[Set, Token_set], sems: Sem_set):
        """
        Initialise the build connections with token_sets and sems.

        Args:
            token_sets (dict): A dictionary of token sets, mapping set to token_set object.
            sems (Sem_set): The semantic set object.
        """
        self.token_sets = token_sets
        self.sems = sems

    def build_connections_links(self):
        """
        Build the connections and links for each set.
        """
        for set in Set:
            token_set = self.token_sets[set]
            self.build_set_connections(token_set)
            self.build_set_links(token_set)

    def build_set_connections(self, token_set: Token_set):  # Returns matrix of all connections for a given set
        """
        Build the connections matrix for a given set.
        
        Returns:
            connections (torch.Tensor): The NxN connections matrix for the set.
        """
        num_tks = token_set.num_tokens
        token_set.connections = torch.zeros((num_tks, num_tks), dtype=tensor_type)  # len tokens x len tokens matrix for connections.
        for type in Type:
            if type != Type.PO:
                for node in token_set.tokens[type]:
                    for child in node.children:
                        token_set.connections[node.ID][child] = 1.0
    
    def build_set_links(self, token_set: Token_set):        # Returns matrix of all po -> sem links for a given set
        """
        Build the links matrix for a given set.

        Returns:
            links (torch.Tensor): The NxM links matrix for the set.
        """
        num_tks = token_set.num_tokens
        num_sems = self.sems.num_sems
        token_set.links = torch.zeros((num_tks, num_sems), dtype=tensor_type)    # Len tokens x len sems matrix for links.
        for po in token_set.tokens[Type.PO]:
            for child in po.children:
                token_set.links[po.ID][child] = 1.0


# ====================[ RUN FUNCTION ]======================
def build_con_tensors(token_sets: dict[Set, Token_set], sems: Sem_set):
    """Build the connections tensors for the inter set."""
    builder = Build_connections(token_sets, sems)
    for set in Set:
        builder.build_set_connections(token_sets[set])

def build_links_tensors(token_sets: dict[Set, Token_set], sems: Sem_set):
    """Build the links tensors for the inter set."""
    builder = Build_connections(token_sets, sems)
    for set in Set:
        builder.build_set_links(token_sets[set])
