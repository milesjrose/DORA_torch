import torch
from ...enums import *
from ..tokens.tensor.token_tensor import Token_Tensor
from ..tokens.tensor_view import TensorView
class Base_Set:
    """
    Base class for token sets.
    """
    def __init__(self, tokens: torch.Tensor, token_set: Set):
        self.tokens: Token_Tensor = tokens
        self.token_set: Set = token_set
        self.tensor = self.tokens.get_set_view(self.token_set)
    
    def get_tensor(self) -> torch.Tensor:
        return self.tensor
    
    def get_token_set(self) -> Set:
        return self.token_set
    
    def update_view(self) -> TensorView:
        self.tensor = self.tokens.get_set_view(self.token_set)
        return self.tensor

    
    