from logging import getLogger

from torch.fx.experimental.symbolic_shapes import Int
from ...single_nodes import Token, Ref_Token, Ref_Analog, Pairs, get_default_features
from ....utils import tensor_ops as tOps
from ....enums import *
from typing import TYPE_CHECKING
import torch


if TYPE_CHECKING:
    from ..base_set import Base_Set

logger = getLogger(__name__)

class TokenOperations:
    """
    Token operations for the Base_Set class.
    """
    def __init__(self, base_set):
        """
        Initialize TokenOperations with reference to Base_Set.
        """
        self.base_set: 'Base_Set' = base_set
    
    def get_features(self, idxs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Get the features for the given indices.
        """
        return self.base_set.glbl.get_features(self.base_set.lcl.to_global(idxs), features)
    
    def set_features(self, idxs: torch.Tensor, features: torch.Tensor, values: torch.Tensor):
        """
        Set the features for the given indices.
        """
        self.base_set.glbl.set_features(self.base_set.lcl.to_global(idxs), features, values)
    
    def set_features_all(self, feature: TF, value: float):
        """
        Set the features for all tokens in the set.
        """
        self.base_set.lcl[:, feature] = value

    def get_name(self, idx: int) -> str:
        """
        Get the name for the given index (local index).
        """
        global_idx_tensor = self.base_set.lcl.to_global(idx)
        global_idx = global_idx_tensor[0].item()
        return self.base_set.glbl.get_name(global_idx)
    
    def set_name(self, idx: int, name: str):
        """
        Set the name for the given index (local index).
        """
        global_idx_tensor = self.base_set.lcl.to_global(idx)
        global_idx = global_idx_tensor[0].item()
        self.base_set.glbl.set_name(global_idx, name)

    def get_index(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the indices for the given indices.
        """
        return self.base_set.lcl.to_global(idxs)
    
    def get_single_token(self, idx: int) -> Token:
        """
        Get a single token from the tensor
        """
        return Token(tensor=self.base_set.lcl[idx, :].clone())
    
    def get_max_acts(self):
        """
        Set max_act for all tokens in the set
        """
        self.set_features_all(TF.MAX_ACT, self.base_set.lcl[:, TF.ACT].max())
    
    def get_highest_token_type(self) -> Type:
        """
        Get the highest token type in the set
        """
        return Type(self.base_set.lcl[:, TF.TYPE].max().item())
    
    def get_child_idxs(self, idx: int) -> torch.Tensor:
        """
        Get the indicies of the children of the given token
        """
        global_idx = self.base_set.lcl.to_global(idx)
        indicies = self.base_set.glbl.connections.get_children(global_idx)
        return self.base_set.lcl.to_local(indicies)
    
    def get_most_active_token(self) -> int:
        """
        Get the index of the most active token in the set (returns local index)
        """
        local_idxs = self.base_set.glbl.cache.get_set_indices(self.base_set.tk_set)
        if len(local_idxs) == 0:
            return None
        max_idx_in_list, max_val = self.base_set.glbl.get_max(TF.ACT, local_idxs)
        if max_val == 0.0:
            return None
        else:
            global_idx = local_idxs[max_idx_in_list]
            # indexing into tensor returns 0-d tensor, wrap it
            global_idx_tensor = global_idx.unsqueeze(0) if global_idx.dim() == 0 else global_idx
            local_idx_tensor = self.base_set.lcl.to_local(global_idx_tensor)
            return local_idx_tensor[0].item()
    
    def connect(self, parent_idx: int, child_idx: int, value=B.TRUE):
        """
        Connect a token at parent_idx to a token at child_idx.
        """
        parent_global_idx = self.base_set.lcl.to_global(parent_idx)
        child_global_idx = self.base_set.lcl.to_global(child_idx)
        self.base_set.glbl.connections.connect(parent_global_idx, child_global_idx, value)
    
    def connect_multiple(self, parent_idxs: torch.Tensor, child_idxs: torch.Tensor, value=B.TRUE):
        """
        Connect a list of tokens at parent_idxs to a list of tokens at child_idxs.
        """
        parent_global_idxs = self.base_set.lcl.to_global(parent_idxs)
        child_global_idxs = self.base_set.lcl.to_global(child_idxs)
        self.base_set.glbl.connections.connect(parent_global_idxs, child_global_idxs, value)
    
    def get_ref_string(self, idx: int) -> str:
        """
        Get a string representation of a token at the given index. (Mainly for debugging)
        """
        global_idx_tensor = self.base_set.lcl.to_global(idx)
        global_idx_val = global_idx_tensor[0].item()
        return f"{self.base_set.tk_set.name}[{idx}](glbl[{global_idx_val}])"
    
    def reset_inferences(self):
        """
        Reset the inferences of all tokens in the set.
        """
        self.base_set.lcl[:, TF.INFERRED] = B.FALSE
        self.base_set.lcl[:, TF.MAKER_UNIT] = null
        self.base_set.lcl[:, TF.MADE_UNIT] = null
    
    def reset_maker_made_units(self):
        """
        Reset the maker and made units of all tokens in the set.
        """
        self.base_set.lcl[:, TF.MAKER_UNIT] = null
        self.base_set.lcl[:, TF.MADE_UNIT] = null
    
    def get_mapped_pos(self) -> list[Ref_Token]:
        """
        get all Pos that are mapped to.
        """
        cache = self.base_set.glbl.cache
        pos_mask = cache.get_type_mask(Type.PO)
        set_mask = cache.get_set_mask(self.base_set.tk_set)
        set_pos = torch.where(pos_mask & set_mask)[0]
        return self.base_set.glbl.get_mapped_pos(set_pos)
    
    def set_max_maps(self, max_maps: torch.Tensor):
        """
        Set the max maps for all tokens in the set.
        """
        self.base_set.lcl[:, TF.MAX_MAP] = max_maps
    
    def set_max_map_units(self, max_map_units: torch.Tensor):
        """
        Set the max map units for all tokens in the set.
        """
        self.base_set.lcl[:, TF.MAX_MAP_UNIT] = max_map_units
    
        