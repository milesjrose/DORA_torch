# nodes/sets/connections/links.py
# Weighted connections between nodes and semantics.
# TODO: Implement add_links

import torch

from ...enums import *

class Links(object): 
    """
    A class for representing weighted connections between token sets and semantics.
    """
    def __init__(self, links: dict[Set, torch.Tensor], semantics):  # Takes weighted adjacency matrices
        """
        Initialize the Links object.

        Args:
            links (dict[Set, torch.Tensor]): A dictionary of weighted connections from each set to semantics.
            semantics (Semantics): The semantics that links connect to.
        
        Raises:
            TypeError: If the link tensors are not torch.Tensor (torch.float)
            ValueError: If the number of semantics (columns) in the link tensors are not the same.
        """
        # check input
        sem_count = semantics.nodes.size(dim=0)
        for set in Set:
                tensor = links[set]
                # Check type
                if type(tensor) != torch.Tensor:
                    raise TypeError(f"Link for {set} must be torch.Tensor.")
                # Check float
                if tensor.dtype != torch.float:
                    raise TypeError(f"Link for {set} must be torch.float.")
                # Check correct number of semantics
                if tensor.size(dim=1) != sem_count:
                    raise ValueError(f"{set} links tensor has {tensor.size(dim=1)} semantics, but semantics has {sem_count}.")
        # initialise
        self.semantics = semantics
        self.sets = links
    
    def update_link(self, set, token_index, semantic_index, weight):
        """
        Update the link between a token and a semantic.
        """
        self.sets[set][token_index, semantic_index] = weight

    def __getitem__(self, key):                                     # Allows for links[set], instead of links.sets[set]
        return self.sets[key]

    def __setitem__(self, key, value):                              # Allows for links[set] = tensor, instead of links.sets[set] = tensor
        self.sets[key] = value

    def del_small_link(self, threshold: float):
        """
        Delete links below threshold.
        """
        # set any values in links tensor below threshold to 0.0
        for set in Set:
            self.sets[set] = torch.where(self.sets[set] < threshold, 0.0, self.sets[set])
    
    def round_big_link(self, threshold: float):
        """
        Round links above threshold to 1.0.
        """
        # set any values in links tensor above threshold to 1.0
        for set in Set:
            self.sets[set] = torch.where(self.sets[set] > threshold, 1.0, self.sets[set])
