import torch
from ..single_nodes import Token, Ref_Token
from ...enums import *
from ..connections import Mappings
from typing import List
from .cache import Cache
from ...utils import tensor_ops as tOps

class Token_Tensor:
    """
    A class for holding all the tokens in the network.
    """
    def __init__(self, tokens: torch.Tensor, connections: torch.Tensor, names: dict[int, str]):
        """
        Initialize the Tokens object.
        Args:
            tokens: torch.Tensor - The tensor of tokens.
            connections: torch.Tensor - The tensor of connections.
            names: dict[int, str] - The dictionary of names.
        """
        self.tensor: torch.Tensor = tokens
        self.connections: torch.Tensor = connections
        self.names: dict[int, str] = names # idx -> name
        self.expansion_factor = 1.1
        self.cache = Cache(tokens)
        """holds the cache object"""
    
    def add_tokens(self, tokens: torch.Tensor, names: List[str]) -> torch.Tensor:
        """
        add a tokens to the tensor.
        Args:
            tokens: torch.Tensor - The tensor of tokens to add.
            names: List[str] - The list of names to add.
        Returns:
            tuple[list[Set], torch.Tensor] - The sets that were changed and the indices that were replaced.
        """
        num_to_add = tokens.size(dim=0)
        num_deleted = (self.tensor[:, TF.DELETED]==B.TRUE).sum()
        if num_to_add > num_deleted:
            # Expand to ensure we have enough deleted slots
            # We need at least (num_to_add - num_deleted) more slots
            min_additional_slots = num_to_add - num_deleted
            self.expand_tensor(min_additional_slots)
        # Add tokens to deleted indices (recalculate after potential expansion)
        deleted_mask = self.tensor[:, TF.DELETED] == B.TRUE
        deleted_idxs = torch.where(deleted_mask)[0]
        replace_idxs = deleted_idxs[:num_to_add]
        self.tensor[replace_idxs, :] = tokens
        # Add names to names dictionary
        for idx, name in zip(replace_idxs, names):
            self.names[idx.item()] = name
        # Get sets that were changed to update masks
        sets_changed = [Set(int(set)) for set in torch.unique(tokens[:, TF.SET])]
        self.cache.cache_sets(sets_changed)
        return replace_idxs
    
    def expand_tensor(self, min_expansion: int = 5):
        """
        Expand the tensor by the expansion factor.
        Args:
            min_expansion: Minimum number of additional slots to add (not total size).
        """
        current_size = self.tensor.size(dim=0)
        # Calculate new size: at least current_size + min_expansion, or expanded by factor
        expanded_size = int(current_size * self.expansion_factor)
        new_size = max(current_size + min_expansion, expanded_size)
        # Create new tensor
        new_tokens = torch.full((new_size, len(TF)), null, dtype=tensor_type)
        new_tokens[:, TF.DELETED] = B.TRUE
        # Copy over old tokens
        new_tokens[:current_size, :] = self.tensor
        # Update tokens
        self.tensor = new_tokens
    
    def move_tokens(self, indices: torch.Tensor, to_set: Set):
        """
        Move the tokens at the given indices to the given set.
        Args:
            indices: torch.Tensor - The indices of the tokens to move.
            to_set: Set - The set to move the tokens to.
        """
        self.tensor[indices, TF.SET] = to_set
        self.cache.cache_sets([to_set])
    
    def copy_tokens(self, indices: torch.Tensor, to_set: Set) -> torch.Tensor:
        """
        Copy the tokens at the given indices to the given set.
        Args:
            indices: torch.Tensor - The indices of the tokens to copy.
            to_set: Set - The set to copy the tokens to.
        Returns:
            torch.Tensor - The indices of the tokens that were replaced.
        """
        tensor = self.tensor[indices, :].clone()
        # Update SET field to the target set
        tensor[:, TF.SET] = to_set
        replace_idxs = self.add_tokens(tensor, [self.names[idx.item()] for idx in indices])
        return replace_idxs
    
    def get_mapped_pos(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the indices of the POs that are mapped to.
        Args:
            idxs: torch.Tensor - The indices of the tokens to get the mapped POs of.
        Returns:
            torch.Tensor - The indices of the mapped POs.
        """
        pos = self.tensor[idxs, TF.TYPE] == Type.PO
        mapped_pos = self.tensor[idxs, TF.MAX_MAP] > 0.0
        mapped_pos = tOps.sub_union(pos, mapped_pos)
        return torch.where(mapped_pos)[0]
    
    def get_ref_string(self, idx: int) -> str:
        """
        Get the string to reference a token at the given index.
        Args:
            idx: int - The index of the token to get the string representation of.
        Returns:
            str - The string representation of the token.
        """
        return f"{self.tensor[idx, TF.SET]}[{idx}]"