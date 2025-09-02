# nodes/network/sets/base_set_operations/token_operations.py
# Token operations for Base_Set class

import torch
from typing import TYPE_CHECKING

from ...single_nodes import Token, Ref_Token, Ref_Analog
from ....enums import *

if TYPE_CHECKING:
    from ..base_set import Base_Set

class TokenOperations:
    """
    Token operations for the Base_Set class.
    Handles token operations.
    """
    
    def __init__(self, base_set):
        """
        Initialize TokenOperations with reference to Base_Set.
        
        Args:
            base_set: Reference to the Base_Set object
        """
        self.base_set: 'Base_Set' = base_set

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================

    def get_feature(self, ref_token: Ref_Token, feature: TF):        # Get feature of single node
        """
        Get a feature for a referenced token.
        
        Args:
            ref_token (Ref_Token): Reference of token.
            feature (TF): The feature to get.

        Returns:
            The feature for the referenced token.

        Raises:
            ValueError: If the referenced token or feature is invalid.
        """
        try:
            return self.base_set.nodes[self.get_index(ref_token), feature]
        except:
            raise ValueError("Invalid reference token or feature.")

    def set_feature(self, ref_token: Ref_Token, feature: TF, value): # Set feature of single node
        """
        Set a feature for a referenced token.
        
        Args:
            ref_token (Ref_Token): Reference of token.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.

        Raises:
            ValueError: If the referenced token, feature, or value is invalid.
        """
        try:
            self.base_set.nodes[self.get_index(ref_token), feature] = float(value)
        except:
            raise ValueError("Invalid reference token, feature, or value.")

    def set_features(self, indices: list[int], feature: TF, value: float):
        """
        Set a feature for a list of tokens.

        Args:
            indices (list[int]): The indices of the tokens to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.

        Raises:
            ValueError: If the indices, feature, or value is invalid.
        """
        try:
            self.base_set.nodes[indices, feature] = float(value)
        except:
            raise ValueError("Invalid indices, feature, or value.")

    def get_name(self, ref_token: Ref_Token):                       # Get name of node by reference token
        """
        Get the name for a referenced token.
        
        Args:
            ref_token (Ref_Token): The token to get the name for.
        
        Returns:
            The name of the token.

        Raises:
            ValueError: If the referenced token is invalid.
        """
        try:
            return self.base_set.names[ref_token.ID]
        except:
            raise ValueError("Invalid reference token.")

    def set_name(self, ref_token: Ref_Token, name):                 # Set name of node by reference token
        """
        Set the name for a referenced token.
        
        Args:
            ref_token (Ref_Token): The token to set the name for.
            name (str): The name to set the token to.

        Raises:
            ValueError: If the referenced token is invalid.
        """
        self.base_set.names[ref_token.ID] = name
    
    def get_index(self, ref_token: Ref_Token):                      # Get index in tensor of reference token
        """
        Get index in tensor of referenced token.

        Args:
            ref_token (Ref_Token): The token to get the index for.

        Returns:
            The index of the token in the tensor.

        Raises:
            ValueError: If the referenced token is invalid.
        """
        try:
            return self.base_set.IDs[ref_token.ID]
        except:
            raise ValueError(f"[{self.base_set.token_set.name}] : Token ID {ref_token.ID} not found in set {ref_token.set.name}.")

    def get_indices(self, ref_tokens: list[Ref_Token]) -> list[int]:
        """
        Get indicies of a list of reference tokens.

        Args:
            ref_tokens (list[Ref_Token]): The tokens to get the indices for.

        Returns:
            A list of indices of the tokens.
        """
        indices = []
        for tk in ref_tokens:
            indices.append(self.get_index(tk))
        return indices

    def get_reference(self, id=None, index=None, name=None) -> Ref_Token:        # Get reference to token with given ID, index, or name
        """
        Get a reference to a token with a given ID, index, or name.

        Args:
            id (int, optional): The ID of the token to get the reference for.
            index (int, optional): The index of the token to get the reference for.
            name (str, optional): The name of the token to get the reference for.

        Returns:
            A Ref_Token object.

        Raises:
            ValueError: If the ID, index, or name is invalid. Or if none are provided.
        """
        source = ""
        if index is not None:
            try:
                id = self.base_set.nodes[index, TF.ID].item()
                source = "index"    
            except:
                raise ValueError("Invalid index.")
        elif name is not None:
            try:
                # I feel like there is a better way to do this...
                dict_index = list(self.base_set.names.values()).index(name)
                id = list(self.base_set.names.keys())[dict_index]
                source = "name"
            except:
                raise ValueError("Invalid name.")
        elif id is not None:
            try:
                # Check if ID exists in the IDs dictionary instead of recursive call
                if id not in self.base_set.IDs:
                    raise ValueError("Invalid ID.")
                source = "id"
            except:
                raise ValueError("Invalid ID.")
        else:
            raise ValueError("No ID, index, or name provided.")
        
        try:
            name = self.base_set.names[id]
        except:
            name = None

        return Ref_Token(self.base_set.token_set, id, name)
    
    def get_single_token(self, ref_token: Ref_Token, copy=True) -> Token:    # Get a single token from the tensor
        """
        Get a single token from the tensor.

        - If copy is set to False, changes to the returned token will affect the tensor.

        Args:
            ref_token (Ref_Token): The token to get.
            copy (bool, optional): Whether to return a copy of the token. Defaults to True.

        Returns:
            Token: The token object.
        """
        tensor = self.base_set.nodes[self.get_index(ref_token), :]
        token = Token(self.base_set.token_set, {TF.PRED: tensor[TF.PRED]})
        if copy:
            token.tensor = tensor.clone()
        else:
            token.tensor = tensor
        return token
    
    def get_reference_multiple(self, mask=None, types: list[Type] = None) -> list[Ref_Token]:  # Get references to tokens in tensor
        """
        Get references to tokens in the tensor. Must provide either mask or types.

        Args:
            mask (torch.Tensor, optional): A mask to apply to the tensor. Defaults to None.
            types (list[Type], optional): A list of types to filter by. Defaults to None.
        Returns:
            A list of references to the tokens in the tensor.
        """
        if types is not None:
            mask = self.base_set.tensor_op.get_combined_mask(types)
        elif mask is None:
            raise ValueError("No mask or types provided.")
        
        indices = torch.where(mask)[0]
        references = [self.get_reference(index=i) for i in indices]
        return references
    
    def get_analog_indices(self, analog: Ref_Analog) -> list[int]:
        """
        Get indices of tokens in an analog.

        Args:
            analog (Ref_Analog): The analog to get the indices for.

        Returns:
            A list of indices of tokens in the analog.
        """
        if not isinstance(analog, Ref_Analog):
            raise TypeError("analog must be a Ref_Analog object.")
        if analog.set != self.base_set.token_set:
            raise ValueError(f"Analog {analog.analog_number} is not in the set {self.base_set.token_set}.")

        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        return torch.where(self.base_set.nodes[all_nodes_mask, TF.ANALOG] == analog.analog_number)[0].tolist()
    
    def get_analogs_where(self, feature: TF, value) -> list[Ref_Analog]:
        """
        Get any analogs that contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            List[Ref_Analog]: References to the analogs that contain a token with the given feature and value.
        """

        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()           # Only non-deleted tokens
        matching_tokens = (self.base_set.nodes[all_nodes_mask, feature] == value)
        if not torch.any(matching_tokens):
            return [] # (No matching tokens)
        
        matching_analog_ids = self.base_set.nodes[matching_tokens, TF.ANALOG]   # Get the analog IDs of the matching tokens
        unique_analog_ids = torch.unique(matching_analog_ids).tolist()          # Convert to unique list of analog IDs

        analogs = []
        for analog_id in unique_analog_ids:
            analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
        return analogs
   
    def get_analogs_where_not(self, feature: TF, value) -> list[Ref_Analog]:
        """
        Get any analogs that do not contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            List[Ref_Analog]: References to the analogs that do not contain a token with the given feature and value.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()               # Only non-deleted tokens
        non_matching_tokens = (self.base_set.nodes[all_nodes_mask, feature] != value)
        non_matching_analog_ids = self.base_set.nodes[non_matching_tokens, TF.ANALOG] 
        unique_analog_ids = torch.unique(non_matching_analog_ids).tolist()          # Convert to unique list of analog IDs

        analogs = []
        for analog_id in unique_analog_ids:
            analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
        return analogs
    
    def get_analogs_active(self, ids=False) -> list[Ref_Analog]:
        """
        Get all analogs that have at least one active token.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        active_tokens = (self.base_set.nodes[all_nodes_mask, TF.ACT] > 0.0)
        active_analog_ids = self.base_set.nodes[active_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(active_analog_ids).tolist()

        if ids:
            return unique_analog_ids
        else: # Convert to ref objects
            analogs = []
            for analog_id in unique_analog_ids:
                analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
            return analogs
    
    def get_max_acts(self):
        """
        Set max_act for all tokens in the set
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        nodes_to_update = self.base_set.nodes[:, TF.ACT] > self.base_set.nodes[:, TF.MAX_ACT]
        update_mask = nodes_to_update & all_nodes_mask
        self.base_set.nodes[update_mask, TF.MAX_ACT] = self.base_set.nodes[update_mask, TF.ACT]
    
    def get_highest_token_type(self) -> Type:
        """
        Get the highest token type in the set.

        Returns:
            Type: The highest token type in the set.
        """
        if self.base_set.nodes.shape[0] == 0:
            return None
        
        return self.base_set.nodes[self.base_set.tensor_op.get_all_nodes_mask(), TF.TYPE].max().item()
    
    def get_child_indices(self, index: int) -> list[int]:
        """
        Get the indices of the children of a given token.

        Args:
            index (int): The index of the token to get the children for.

        Returns:
            A list of indices of the children of the token.
        """
        # Check connections, get nodes that connect to (connections: parent -> child)
        indices = torch.where(self.base_set.connections[index, :] == B.TRUE)[0].tolist()
        return indices
    # --------------------------------------------------------------