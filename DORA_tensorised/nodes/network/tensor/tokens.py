import torch
from ..single_nodes import Token, Ref_Token
from ...enums import *
from ..connections import Mappings
from typing import List

class Tokens:
    """
    A class for holding all the tokens in the network.
    #NOTE: Moving from seperate set tensors to a single tensor.
    """
    def __init__(self, tokens: torch.Tensor, connections: torch.Tensor, names: dict[int, str]):
        """
        Initialize the Tokens object.
        Args:
            tokens: torch.Tensor - The tensor of tokens.
            connections: torch.Tensor - The tensor of connections.
            names: dict[int, str] - The dictionary of names.
        """
        self.tokens: torch.Tensor = tokens
        self.connections: torch.Tensor = connections
        self.names: dict[int, str] = names # idx -> name
        self.expansion_factor = 1.1
    
    def mask_set(self, set: Set) -> torch.Tensor:
        """
        Get a mask for the tokens in the given set.
        """
        mask = (self.tokens[:, TF.SET] == set)
        return mask
    
    def idx_set(self, set: Set) -> torch.Tensor:
        """
        Get the indicies of the tokens in the given set.
        """
        indices = torch.where(self.mask_set(set))[0]
        return indices
    
    def add_tokens(self, tokens: torch.Tensor, names: List[str]) -> tuple[list[Set], torch.Tensor]:
        """
        add a tokens to the tensor.
        Args:
            tokens: torch.Tensor - The tensor of tokens to add.
            names: List[str] - The list of names to add.
        Returns:
            tuple[list[Set], torch.Tensor] - The sets that were changed and the indices that were replaced.
        """
        num_to_add = tokens.size(dim=0)
        if num_to_add > (self.tokens[:, TF.DELETED]==B.TRUE).count():
            self.expand_tensor(num_to_add)
        # Add tokens to deleted indices
        deleted_mask = self.tokens[:, TF.DELETED] == B.TRUE
        deleted_idxs = torch.where(deleted_mask)[0]
        replace_idxs = deleted_idxs[:num_to_add]
        self.tokens[replace_idxs, :] = tokens
        # Add names to names dictionary
        for idx, name in zip(replace_idxs, names):
            self.names[idx] = name
        # Get sets that were changed to update masks
        sets_changed = [Set(int(set)) for set in torch.unique(tokens[replace_idxs, TF.SET])]
        return sets_changed, replace_idxs
    
    def expand_tensor(self, min_expansion: int = 5):
        """
        Expand the tensor by the expansion factor.
        """
        # Get new size
        new_size = max(min_expansion, int(self.tokens.size(dim=0) * self.expansion_factor))
        # Create new tensor
        new_tokens = torch.full((new_size, len(TF)), null, dtype=tensor_type)
        new_tokens[:, TF.DELETED] = B.TRUE
        # Copy over old tokens
        new_tokens[:self.tokens.size(dim=0), :] = self.tokens
        # Update tokens
        self.tokens = new_tokens
        # Create new connections
        new_connections = torch.zeros(new_size, new_size)
        new_connections[:self.connections.size(dim=0), :self.connections.size(dim=1)] = self.connections
        # Update connections
        self.connections = new_connections
