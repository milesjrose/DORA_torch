# nodes/sets/new_set.py
# Represents a new set of tokens.

import torch
from nodes.sets import Tokens
from nodes.sets.connections import Links
from nodes.enums import *
from nodes import Params

class New_Set(Tokens):
    """
    A class for representing a new set of tokens.
    """
    def __init__(self, floatTensor, connections, links: Links, IDs: dict[int, int], names: dict[int, str] = {}, params: Params = None):
        super().__init__(floatTensor, connections, links, IDs, names, params)
        self.token_set = Set.NEW_SET
    
    def update_input_type(self, n_type: Type):
        """
        Update the input for a given type of token.
        """ 
