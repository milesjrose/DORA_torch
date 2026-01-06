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
    def __init__(self, tokens: Token_Tensor, token_set: Set, params: Params):
        self.glbl: Token_Tensor = tokens
        """Token_Tensor: Global view of the tensor"""
        self.tk_set: Set = token_set
        """Set: Token set"""
        self.lcl: TensorView = self.glbl.get_set_view(self.tk_set)
        """TensorView: Local view of the tensor"""
        self.params: Params = params
        """Params: Parameters for the set"""
        self.tensor_op: TensorOperations = TensorOperations(self)
        """TensorOperations: Operations for the tensor.
            Functions:
            - get_mask(token_type: Type) -> torch.Tensor
            - get_combined_mask(token_types: list[Type]) -> torch.Tensor
            - get_count(token_type: Type=None, mask: torch.Tensor = None) -> int
            - print(f_types=None)
            - print_tokens(f_types=None)
        """
        self.tnop = self.tensor_op
        self.update_op: UpdateOperations = UpdateOperations(self)
        """UpdateOperations: Operations for the update.
            Functions:
            - init_float(n_type: list[Type], features: list[TF], value: float = 0.0)
            - init_input(n_type: list[Type], refresh: float)
            - init_act(n_type: list[Type])
            - init_state(n_type: list[Type])
            - update_act()
            - zero_laternal_input(n_type: list[Type])
            - update_inhibitor_input(n_type: list[Type])
            - reset_inhibitor(n_type: list[Type])
            - update_inhibitor_act(n_type: list[Type])
            - p_initialise_mode()
            - p_get_mode()
            - po_get_weight_length()
            - po_get_max_semantic_weight()
        """
        self.upop = self.update_op
        self.analog_op: AnalogOperations = AnalogOperations(self)
        """AnalogOperations: Operations for the analogs.
            Functions:
            - get_analog_indices(analog: int) -> torch.Tensor
            - get_analogs_where(feature: TF, value: float) -> torch.Tensor
            - get_analogs_where_not(feature: TF, value: float) -> torch.Tensor
            - get_analogs_active() -> torch.Tensor
            - get_analog_ref_list(mask) -> torch.Tensor
        """
        self.anop = self.analog_op
        self.kludgey_op: KludgeyOperations = KludgeyOperations(self)
        """KludgeyOperations: Operations for the kludgey.
            Functions:
            - get_pred_rb_no_ps(pairs: Pairs) -> Pairs
            - get_pred_rb_shared_p(pairs: Pairs) -> Pairs
        """
        self.klud = self.kludgey_op
        self.token_op: TokenOperations = TokenOperations(self)
        """TokenOperations: Operations for the tokens.
            Functions:
            - get_features(idxs: torch.Tensor, features: torch.Tensor) -> torch.Tensor
            - set_features(idxs: torch.Tensor, features: torch.Tensor, values: torch.Tensor)
            - set_features_all(feature: TF, value: float)
            - get_name(idx: int) -> str
            - set_name(idx: int, name: str)
            - get_index(idxs: torch.Tensor) -> torch.Tensor
            - get_single_token(idx: int) -> Token
            - get_max_acts()
            - get_highest_token_type() -> Type
            - get_child_idxs(idx: int) -> torch.Tensor
            - get_most_active_token() -> int
            - connect(parent_idx: int, child_idx: int, value=B.TRUE)
            - connect_multiple(parent_idxs: torch.Tensor, child_idxs: torch.Tensor, value=B.TRUE)
            - get_ref_string(idx: int) -> str
            - reset_inferences()
            - reset_maker_made_units()
            - get_mapped_pos() -> list[Ref_Token]
        """
        self.tkop = self.token_op
    
    def get_tensor(self) -> torch.Tensor:
        return self.lcl
    
    def get_token_set(self) -> Set:
        return self.tk_set
    
    def update_view(self) -> TensorView:
        self.lcl = self.glbl.get_set_view(self.tk_set)
        return self.lcl
    
    def get_count(self) -> int:
        """
        Get the count of tokens in the set.
        """
        return self.lcl.get_count()

    
    