# nodes/network/sets/base_set_operations/token_operations.py
# Token operations for Base_Set class

import torch

from ...single_nodes import Token, Ref_Token
from ....enums import *

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
        self.base_set = base_set

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
            raise ValueError("Invalid ID.")
        
    def get_reference(self, id=None, index=None, name=None):        # Get reference to token with given ID, index, or name
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
    
    def get_single_token(self, ref_token: Ref_Token, copy=True):    # Get a single token from the tensor
        """
        Get a single token from the tensor.

        - If copy is set to False, changes to the returned token will affect the tensor.

        Args:
            ref_token (Ref_Token): The token to get.
            copy (bool, optional): Whether to return a copy of the token. Defaults to True.

        """
        tensor = self.base_set.nodes[self.get_index(ref_token), :]
        token = Token(self.base_set.token_set, {TF.PRED: tensor[TF.PRED]})
        if copy:
            token.tensor = tensor.clone()
        else:
            token.tensor = tensor
        return token
    
    def get_reference_multiple(self, mask=None, types: list[Type] = None):  # Get references to tokens in tensor
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
    
    def get_analog_indices(self, analog_id):
        """
        Get indices of tokens in an analog.

        Args:
            analog_id (int): The analog ID to get the indices for.

        Returns:
            A list of indices of tokens in the analog.
        """
        return torch.where(self.base_set.nodes[:, TF.ANALOG] == analog_id)[0].tolist()
    
    # --------------------------------------------------------------