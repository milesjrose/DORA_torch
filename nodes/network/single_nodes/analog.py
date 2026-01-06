# nodes/network/single_nodes/analog.py
# Representation of an analog

import torch
from ...enums import *

class Analog:
    """
    Holds the data for an analog.
    - token tensor: [n_tokens, len_tf]
    - connections tensor: [n_tokens, n_tokens]
    - links tensor: [n_tokens, n_semantics]
    - name_dict: {id: name} for tokens in the analog

    Args:
        tokens (torch.Tensor): The tokens in the analog.
        connections (torch.Tensor): The connections in the analog.
        links (torch.Tensor): The links in the analog.
        name_dict (dict[int, str]): A dictionary of the names of the tokens in the analog.

    """

    def __init__(self, tokens: torch.Tensor, connections: torch.Tensor, links: torch.Tensor, name_dict: dict[int, str]):
        self.tokens: torch.Tensor = tokens
        self.connections = connections
        self.links: torch.Tensor = links
        self.name_dict: dict[int, str] = name_dict
        self.analog_number: int = self.tokens[0, TF.ANALOG].item()
        self.set = self.get_set()

    def get_set(self):
        """
        Find any tokens where set != memory.
        """
        mask = self.tokens[:, TF.SET] != Set.MEMORY
        if torch.any(mask):
            first_non_memory_idx = torch.where(mask)[0][0]      # Get the first token that is not in memory
            return self.tokens[first_non_memory_idx, TF.SET].item()
        else:
            return Set.MEMORY

    def retrieve_lower_tokens(self):
        """
        Set sub tokens to same set as analog.
        """
        if self.set == Set.MEMORY:
            return # (All tokens are in memory)
        
        mask = self.tokens[:, TF.SET] != Set.MEMORY             # Find tokens that have set != memory
        last_mask_sum = 0
        while torch.sum(mask) > last_mask_sum:                  # Find all tokens connected to these tokens recursively
            last_mask_sum = torch.sum(mask)
            connected_mask = torch.any(self.connections[mask], dim=0)
            mask = torch.logical_or(mask, connected_mask)
    
        self.tokens[mask, TF.SET] = self.set                    # Set these tokens to the set of the analog
    
    def remove_memory_tokens(self):
        """
        Remove tokens that have set == memory.
        """
        mask = self.tokens[:, TF.SET] == Set.MEMORY
        self.tokens[mask, TF.DELETED] = B.TRUE
        self.connections[mask, :] = B.FALSE
        self.connections[:, mask] = B.FALSE
        self.links[mask, :] = B.FALSE
        del_ids = self.tokens[mask, TF.ID].tolist()
        for id in del_ids:
            del self.name_dict[id]

    def copy(self):
        """
        Create a deep copy of this analog.
        
        Returns:
            Analog: A new Analog object with copied data.
        """
        return Analog(
            tokens=self.tokens.clone(),
            connections=self.connections.clone(),
            links=self.links.clone(),
            name_dict=self.name_dict.copy()
        )

class Ref_Analog:
    """ Reference to an analog in a set. """

    def __init__(self, analog_number: int, set: Set):
        self.analog_number = analog_number
        self.set = set

    def __eq__(self, other):
        if not isinstance(other, Ref_Analog):
            return NotImplemented
        return self.analog_number == other.analog_number and self.set == other.set