# nodes/sets/connections/links.py
# Weighted connections between nodes and semantics.
# TODO: Implement add_links

import torch

from nodes.enums import *

class Links(object): 
    """
    A class for representing weighted connections between token sets and semantics.
    """
    def __init__(self, driver_links, recipient_links, memory_links, new_set_links, semantics):  # Takes weighted adjacency matrices
        """
        Initialize the Links object.

        Args:
            driver_links (torch.Tensor): A tensor of weighted connections from the driver set to semantics.
            recipient_links (torch.Tensor): A tensor of weighted connections from the recipient set to semantics.
            memory_links (torch.Tensor): A tensor of weighted connections from the memory set to semantics.
            semantics (Semantics): The semantics that links connect to.
        
        Raises:
            TypeError: If the link tensors are not torch.Tensor.
            ValueError: If the number of semantics (columns) in the link tensors are not the same.
        """
        if type(driver_links) != torch.Tensor:
            raise TypeError("Driver links must be torch.Tensor.")
        if type(recipient_links) != torch.Tensor:
            raise TypeError("Recipient links must be torch.Tensor.")
        if type(memory_links) != torch.Tensor:
            raise TypeError("Memory links must be torch.Tensor.")
        if type(new_set_links) != torch.Tensor:
            raise TypeError("New set links must be torch.Tensor.")
        if driver_links.size(dim=1) != recipient_links.size(dim=1) or driver_links.size(dim=1) != memory_links.size(dim=1) or driver_links.size(dim=1) != new_set_links.size(dim=1):
            raise ValueError("All link tensors must have the same number of semantics (columns).")
    
        self.driver: torch.Tensor = driver_links
        self.recipient: torch.Tensor = recipient_links
        self.memory: torch.Tensor = memory_links
        self.new_set: torch.Tensor = new_set_links
        self.semantics = semantics
        self.sets = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory,
            Set.NEW_SET: self.new_set
        }
    
    def update_link(self, set, token_index, semantic_index, weight):
        """
        Update the link between a token and a semantic.
        """
        self.sets[set][token_index, semantic_index] = weight

    def __getitem__(self, key):                                     # Allows for links[set], instead of links.sets[set]
        return self.sets[key]

    def __setitem__(self, key, value):                              # Allows for links[set] = tensor, instead of links.sets[set] = tensor
        self.sets[key] = value
