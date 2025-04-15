# nodes/network/sets/new_set.py
# Represents a new set of tokens.

from ...enums import *
from ..connections import Links
from ..network_params import Params
from .base_set import Base_Set

class New_Set(Base_Set):
    """
    A class for representing a new set of tokens.
    """
    def __init__(self, floatTensor, connections, IDs: dict[int, int], names: dict[int, str] = {}):
        super().__init__(floatTensor, connections, IDs, names)
        self.token_set = Set.NEW_SET
    
    def update_input(self):
        """
        Update the input for the new set.
        """
        for n_type in Type:
            self.update_input_type(n_type)
    
    def update_input_type(self, n_type: Type):
        """
        Update the input for a given type of token.
        """ 
        pass
