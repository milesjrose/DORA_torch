import torch
from ..network_params import Params
from ...enums import *
from ..tokens.tensor.token_tensor import Token_Tensor
from ..tokens.tensor_view import TensorView
from .base_set_ops import TensorOperations, UpdateOperations, AnalogOperations, KludgeyOperations, TokenOperations

class Base_Set:
    """
    Base class for token sets.
    """
    def __init__(self, tokens: torch.Tensor, token_set: Set, params: Params):
        self.glbl: Token_Tensor = tokens
        """Token_Tensor: Global view of the tensor"""
        self.tk_set: Set = token_set
        """Set: Token set"""
        self.lcl = self.glbl.get_set_view(self.tk_set)
        """TensorView: Local view of the tensor"""
        self.params = params
        """Params: Parameters for the set"""
        self.tensor_op = TensorOperations(self)
        self.tnsr = self.tensor_op
        """TensorOperations: Operations for the tensor"""
        self.update_op = UpdateOperations(self)
        self.updt = self.update_op
        """UpdateOperations: Operations for the update"""
        self.analog_op = AnalogOperations(self)
        self.anlg = self.analog_op
        """AnalogOperations: Operations for the analogs"""
        self.kludgey_op = KludgeyOperations(self)
        self.klud = self.kludgey_op
        """KludgeyOperations: Operations for the kludgey"""
        self.token_op = TokenOperations(self)
        self.tkop = self.token_op
        """TokenOperations: Operations for the tokens"""
    
    def get_tensor(self) -> torch.Tensor:
        return self.lcl
    
    def get_token_set(self) -> Set:
        return self.tk_set
    
    def update_view(self) -> TensorView:
        self.lcl = self.glbl.get_set_view(self.tk_set)
        return self.lcl

    
    