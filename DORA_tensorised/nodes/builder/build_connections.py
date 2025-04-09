# nodes/builder/build_connections.py
# Builds connections for each set.

import numpy as np
from nodes.enums import *

from .intermediate_types import *

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
            token_set.connections = self.build_set_connections(token_set)
            token_set.links = self.build_set_links(token_set)

    def build_set_connections(self, token_set: Token_set):  # Returns matrix of all connections for a given set
        """
        Build the connections matrix for a given set.
        
        Returns:
            connections (np.ndarray): The NxN connections matrix for the set.
        """
        num_tks = token_set.num_tokens
        connections = np.zeros((num_tks, num_tks))          # len tokens x len tokens matrix for connections.
        for type in Type:
            if type != Type.PO:
                for node in token_set.tokens[type]:
                    for child in node.children:
                        connections[node.ID][child] = 1
        return connections
    
    def build_set_links(self, token_set: Token_set):        # Returns matrix of all po -> sem links for a given set
        """
        Build the links matrix for a given set.

        Returns:
            links (np.ndarray): The NxM links matrix for the set.
        """
        num_tks = token_set.num_tokens
        num_sems = self.sems.num_sems
        links = np.zeros((num_tks, num_sems))               # Len tokens x len sems matrix for links.
        for po in token_set.tokens[Type.PO]:
            for child in po.children:
                links[po.ID][child] = 1
        return links
