from logging import getLogger
from threading import local

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

        Args:
            idxs (torch.Tensor): The indices of the tokens to get the features of.
            features (torch.Tensor): The features to get the values of.
        Returns:
            torch.Tensor: The features for the given indices.
        """
        return self.base_set.lcl[idxs, features]
    
    def get_feature(self, idx: int|torch.Tensor, feature: TF) -> float:
        """ Get the value of a feature for a given idx
        
        Args:
            idx (int|torch.Tensor): The index of the token to get the feature value of.
            feature (TF): The feature to get the value of.
        Returns:
            float: The value of the feature for the given index
        """
        return self.base_set.lcl[idx, feature]
    
    def set_features(self, idxs: torch.Tensor, features: torch.Tensor, values: torch.Tensor):
        """
        Set the features for the given indices.

        Args:
            idxs (torch.Tensor): The indices of the tokens to set the features of.
            features (torch.Tensor): The features to set the values of.
            values (torch.Tensor): The values to set the features to.
        """
        # Apparently tensor view class doesn't like assigning sub-tensors. So, use the global tensor to set instead.
        glb_idxs = self.base_set.lcl.to_global(idxs)
        self.base_set.glbl.set_features(glb_idxs, features, values)
    
    def set_feature(self, idx: int|torch.Tensor, feature: TF, value: float):
        """
        Set the value of a feature for a given index.

        Args:
            idx (int|torch.Tensor): The index of the token to set the feature value of.
            feature (TF): The feature to set the value of.
            value (float): The value to set the feature to.
        """
        self.base_set.lcl[idx, feature] = value
    
    def set_features_all(self, feature: TF, value: float):
        """
        Set the features for all tokens in the set.

        Args:
            feature (TF): The feature to set the values of.
            value (float): The value to set the features to.
        """
        self.base_set.lcl[:, feature] = value

    def get_name(self, idx: int) -> str:
        """
        Get the name for the given index (local index).

        Args:
            idx (int): The index of the token to get the name of.
        Returns:
            str: The name of the token at the given index.
        """
        global_idx_tensor = self.base_set.lcl.to_global(idx)
        global_idx = global_idx_tensor[0].item()
        return self.base_set.glbl.get_name(global_idx)
    
    def set_name(self, idx: int, name: str):
        """
        Set the name for the given index (local index).

        Args:
            idx (int): The index of the token to set the name of.
            name (str): The name to set the token to.
        """
        global_idx_tensor = self.base_set.lcl.to_global(idx)
        global_idx = global_idx_tensor[0].item()
        self.base_set.glbl.set_name(global_idx, name)

    def get_index(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the indices for the given indices.

        Args:
            idxs (torch.Tensor): The indices of the tokens to get the global indices of.
        Returns:
            torch.Tensor: The global indices of the given indices.
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
        indicies = self.base_set.tokens.connections.get_children(global_idx)
        return self.base_set.lcl.to_local(indicies)
    
    def get_most_active_token(self, token_type: Type = None, local_mask: torch.Tensor=None) -> int:
        """
        Get the index of the most active token in the set (returns local index).
        Must provide either token_type or local_mask.

        Args:
            token_type (Type): The type of the token to get the most active of.
            local_mask (torch.Tensor): The local mask to use to filter the tokens.
        Returns:
            (int): The local index of the most active token.
        """
        # Check any tokens in the set
        if self.base_set.get_count() == 0:
            return None
        # Get local mask (if not provided)
        if local_mask is None:
            # Use token_type to get local mask
            if token_type is None:
                # No filter, use all tokens
                local_mask = torch.ones(len(self.base_set.lcl), dtype=torch.bool)
            else:
                local_mask = self.base_set.tensor_op.get_mask(token_type)
        # Check we have any tokens to look at
        if not torch.any(local_mask):
            return None
        # The custom tensor view implementation does not support torch.max afaik, so we need to do manually.
        set_idxs = self.base_set.lcl._indices
        local_view = self.base_set.glbl.tensor[set_idxs]
        max_val, max_idx = local_view[local_mask, TF.ACT].max(dim=0)
        # Check the max value is greater than 0.0
        if max_val.item() >= 0.01:
            return max_idx.item()
        else:
            return None
    
    def connect(self, parent_idx: int, child_idx: int, value=B.TRUE):
        """
        Connect a token at parent_idx to a token at child_idx.
        """
        parent_global_idx = self.base_set.lcl.to_global(parent_idx)
        child_global_idx = self.base_set.lcl.to_global(child_idx)
        self.base_set.tokens.connections.connect(parent_global_idx, child_global_idx, value)
    
    def connect_multiple(self, parent_idxs: torch.Tensor, child_idxs: torch.Tensor, value=B.TRUE):
        """
        Connect a list of tokens at parent_idxs to a list of tokens at child_idxs.
        """
        parent_global_idxs = self.base_set.lcl.to_global(parent_idxs)
        child_global_idxs = self.base_set.lcl.to_global(child_idxs)
        self.base_set.tokens.connections.connect(parent_global_idxs, child_global_idxs, value)
    
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
    
    def get_mapped_pos(self) -> torch.Tensor:
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
    
        