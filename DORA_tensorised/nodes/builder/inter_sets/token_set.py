# nodes/builder/sets/token_set.py
# Token set class for the builder.
import torch
import numpy as np

from ...enums import *

from ..inter_nodes import Inter_Token

class Token_set(object):
    """
    An intermediate class for representing a set of tokens.

    Attributes:
        set (Set): The set of the tokens.
        tokens (dict): A dictionary of tokens, mapping type to list of tokens.
        name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
        id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
        num_tokens (int): The number of tokens in the set.
        connections (np.ndarray): A matrix of connections between tokens.
        links (np.ndarray): A matrix of links between tokens and semantics.
    """
    def __init__(self, set: Set, tokens: dict[Type, list[Inter_Token]], name_dict: dict[str, Inter_Token], id_dict: dict[int, Inter_Token]):
        """
        Initialise the token set with set, tokens, name_dict, and id_dict.

        Args:
            set (Set): The set of the tokens.
            tokens (dict): A dictionary of tokens, mapping type to list of tokens.
            name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
            id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
        """
        self.set = set
        self.tokens = tokens
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_tokens = sum([len(self.tokens[type]) for type in Type])
        self.connections = None
        self.links = None
    
    def get_token(self, name):
        """
        Get a token from the token set by name.

        Args:
            name (str): The name of the token.
        """
        return self.name_dict[name]

    def get_token_by_id(self, ID):
        """
        Get a token from the token set by ID.

        Args:
            ID (int): The ID of the token.
        """
        return self.id_dict[ID] 
    
    def get_token_tensor(self):
        """
        Get the token tensor for the token set.
        """
        token_tensor = torch.zeros((self.num_tokens, len(TF)), dtype=tensor_type)
        for type in Type:
            for token in self.tokens[type]:
                token.floatate_features()
                token_tensor[token.ID] = torch.tensor(token.features, dtype=tensor_type)
        return token_tensor
    
    def tensorise(self):
        """
        Tensorise the token set, creating a tensor of tokens, and tensors of connections and links to semantics.
        """
        self.token_tensor = self.get_token_tensor()