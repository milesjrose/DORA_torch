from ..tokens.tensor.token_tensor import Token_Tensor
from ..network_params import Params
from ...enums import Set
from .base_set import Base_Set

class New_Set(Base_Set):
    """
    A class for representing a new set of tokens.
    """
    def __init__(self, tokens: Token_Tensor, params: Params):
        super().__init__(tokens, Set.NEW_SET, params)
        
    def update_input(self):
        """ Update the input of the new set.
        This is a placeholder function.
        """
        pass