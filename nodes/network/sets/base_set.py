# nodes/network/sets/base_set.py
# Base class for all set classes.

import torch
import logging

from ...enums import *
from ...utils import tensor_ops as tOps

from ..connections import Links, Mappings
from ..network_params import Params
from ..single_nodes import Token, Ref_Token, Analog, Ref_Analog

from .base_set_ops.set_token import TokenOperations
from .base_set_ops.set_tensor import TensorOperations
from .base_set_ops.set_update import UpdateOperations

logger = logging.getLogger(__name__)

class Base_Set(object):
    """
    A class for holding a tensor of tokens, and interfacing with low-level tensor operations.

    Using set.function() is deprecated, use set.op.function() instead.

    Op Attributes:
        - token_op: TokenOperations object for the set.
        - tensor_op: TensorOperations object for the set.
        - update_op: UpdateOperations object for the set.
    
    Attributes:
        - names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        - nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        - analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        - analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        - links (Links): A shared Links object containing interset links from tokens to semantics.
        - connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        - masks (torch.Tensor): A Tensor of masks for the tokens in this set.
        - IDs (dict): A dictionary mapping token IDs to index in the tensor.
        - params (Params): An object containing shared parameters.
        - token_set (Set): This set's enum, used to access links and mappings for this set in shared mem objects.
    """
    def __init__(self, nodes, connections, IDs: dict[int, int], names: dict[int, str] = {}):
        """
        Initialize the TokenTensor object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to empty dict.
        Raises:
            TypeError: If connections, or floatTensor are not torch.Tensor.
            ValueError: If the number of tokens in floatTensor, connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
        """
        # Initialize operations objects
        self.token_op = TokenOperations(self)
        self.tensor_op = TensorOperations(self)
        self.update_op = UpdateOperations(self)

        # For __get_attr__ (deprecated)
        self._promoted_components = [
            self.token_op, 
            self.tensor_op, 
            self.update_op
        ]

        # check types
        c_type = type(connections)
        if c_type != torch.Tensor:
            raise TypeError(f"Connections must be torch.Tensor, not {c_type}.")
        f_type = type(nodes)
        if f_type != torch.Tensor:
            raise TypeError(f"floatTensor must be torch.Tensor, not {f_type}.")
        # check sizes
        f_size = nodes.size(dim=0)
        c_size = connections.size(dim=0)
        if f_size != c_size:
            raise ValueError(f"floatTensor and connections must have same number of tokens. {f_size} != {c_size}")
        f_features = nodes.size(dim=1)
        if f_features != len(TF):
            raise ValueError(f"floatTensor must have number of features listed in TF enum. {f_features} != {len(TF)}")
        if nodes.dtype != torch.float:
            raise TypeError(f"floatTensor must be torch.float, not {nodes.dtype}.")
        if connections.dtype != torch.float:
            raise TypeError(f"connections must be torch.float, not {connections.dtype}.")
        
        # intialise attributes
        self.names = names
        "Dict ID -> Name"
        self.nodes: torch.Tensor = nodes
        "NxTF Tensor: Tokens"
        self.cache_masks()
        self.analogs = None
        "Ax1 Tensor: Analogs in the set"
        self.analog_counts = None
        "Ax1 Tensor: Node count for each analog in self.analogs"
        self.analog_activations = None
        "Ax1 Tensor: Total activation for each analog in self.analogs"
        self.tensor_op.get_analog_activation_counts_scatter() # Get the analogs, counts, and activations
        self.links = None
        """Links object for the set.
            - Links[set] gives set's links to semantics
        """
        self.connections = connections.float()
        "NxN tensor: Connections from parent to child"
        self.IDs = IDs
        "Dict ID -> index"
        self.params = None
        "Params object, holding parameters for tensor functions"
        self.expansion_factor = 1.1
        """Factor used in expanding tensor. 
            E.g: expansion_factor = 1.1 -> 10 percent increase in size on expansion
        """
        self.token_set = None
        """Set: This sets set type"""
    
    def __getattr__(self, name):
        for component in self._promoted_components:
            if hasattr(component, name):
                logger.warning(f"Deprecated: Use set.op.{name} instead of set.{name}")
                return getattr(component, name)
        raise AttributeError(f"Base_Set object has no attribute '{name}'")

    @property
    def token_ops(self) -> 'TokenOperations':
        """Token operations object for the set.
            Functions:
            - get_feature(ref_token, feature) -> float
            - set_feature(ref_token, feature, value)
            - get_name(ref_token) -> str
            - set_name(ref_token, name)
            - get_index(ref_token) -> int
            - get_reference(id=None, index=None, name=None) -> Ref_Token
            - get_single_token(ref_token, copy=True) -> Token
            - get_reference_multiple(mask=None, types: list[Type] = None) -> list[Ref_Token]
            - get_analog_indices(analog_id) -> list[int]
        """
        return self.token_op
    
    @property
    def tensor_ops(self) -> 'TensorOperations':
        """Tensor operations object for the set.
            Functions:
            - cache_masks(types_to_recompute: list[Type] = None)
            - compute_mask(token_type: Type) -> torch.Tensor
            - get_mask(token_type: Type) -> torch.Tensor
            - get_combined_mask(n_types: list[Type]) -> torch.Tensor
            - get_all_nodes_mask() -> torch.Tensor
            - add_token(token: Token) -> Ref_Token
            - expand_tensor()
            - expand_tensor_by_count(count: int)
            - del_token(ref_tokens: Ref_Token)
            - del_connections(ref_token: Ref_Token)
            - get_analog(analog: int) -> Analog
            - add_analog(analog: Analog) -> int
            - analog_node_count()
            - print(f_types=None)
            - get_count() -> int
        """
        return self.tensor_op
    
    @property
    def update_ops(self) -> 'UpdateOperations':
        """Update operations object for the set.
            Functions:
            - initialise_float(n_type: list[Type], features: list[TF])
            - initialise_input(n_type: list[Type], refresh: float)
            - initialise_act(n_type: list[Type])
            - initialise_state(n_type: list[Type])
            - update_act()
            - zero_lateral_input(n_type: list[Type])
            - update_inhibitor_input(n_type: list[Type])
            - reset_inhibitor(n_type: list[Type])
            - update_inhibitor_act(n_type: list[Type])
            - p_initialise_mode()
            - p_get_mode()
            - po_get_weight_length()
            - po_get_max_semantic_weight()
        """
        return self.update_op

