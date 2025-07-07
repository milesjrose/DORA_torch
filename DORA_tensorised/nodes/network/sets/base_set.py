# nodes/network/sets/base_set.py
# Base class for all set classes.

import torch

from ...enums import *
from ...utils import tensor_ops as tOps

from ..connections import Links, Mappings
from ..network_params import Params
from ..single_nodes import Token, Ref_Token, Analog

from .base_set_operations.token_operations import TokenOperations
from .base_set_operations.tensor_operations import TensorOperations
from .base_set_operations.update_operations import UpdateOperations

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
    def __init__(self, floatTensor, connections, IDs: dict[int, int], names: dict[int, str] = {}):
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
        self.tensor_op = TensorOperations(self)
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
        self.update_op = UpdateOperations(self)
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

        # check types
        c_type = type(connections)
        if c_type != torch.Tensor:
            raise TypeError(f"Connections must be torch.Tensor, not {c_type}.")
        f_type = type(floatTensor)
        if f_type != torch.Tensor:
            raise TypeError(f"floatTensor must be torch.Tensor, not {f_type}.")
        # check sizes
        f_size = floatTensor.size(dim=0)
        c_size = connections.size(dim=0)
        if f_size != c_size:
            raise ValueError(f"floatTensor and connections must have same number of tokens. {f_size} != {c_size}")
        f_features = floatTensor.size(dim=1)
        if f_features != len(TF):
            raise ValueError(f"floatTensor must have number of features listed in TF enum. {f_features} != {len(TF)}")
        if floatTensor.dtype != torch.float:
            raise TypeError(f"floatTensor must be torch.float, not {floatTensor.dtype}.")
        if connections.dtype != torch.float:
            raise TypeError(f"connections must be torch.float, not {connections.dtype}.")
        
        # intialise attributes
        self.names = names
        "Dict ID -> Name"
        self.nodes: torch.Tensor = floatTensor
        "NxTF Tensor: Tokens"
        self.cache_masks()
        self.analogs = None
        "Ax1 Tensor: Analogs in the set"
        self.analog_counts = None
        "Ax1 Tensor: Node count for each analog in self.analogs"
        self.analog_node_count()
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
        return self.token_op.get_feature(ref_token, feature)

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
        self.token_op.set_feature(ref_token, feature, value)

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
        return self.token_op.get_name(ref_token)

    def set_name(self, ref_token: Ref_Token, name):                 # Set name of node by reference token
        """
        Set the name for a referenced token.
        
        Args:
            ref_token (Ref_Token): The token to set the name for.
            name (str): The name to set the token to.
        """
        self.token_op.set_name(ref_token, name)
    
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
        return self.token_op.get_index(ref_token)
        
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
        return self.token_op.get_reference(id, index, name)
    
    def get_single_token(self, ref_token: Ref_Token, copy=True):    # Get a single token from the tensor
        """
        Get a single token from the tensor.

        - If copy is set to False, changes to the returned token will affect the tensor.

        Args:
            ref_token (Ref_Token): The token to get.
            copy (bool, optional): Whether to return a copy of the token. Defaults to True.

        """
        return self.token_op.get_single_token(ref_token, copy)
    
    def get_reference_multiple(self, mask=None, types: list[Type] = None):  # Get references to tokens in tensor
        """
        Get references to tokens in the tensor. Must provide either mask or types.

        Args:
            mask (torch.Tensor, optional): A mask to apply to the tensor. Defaults to None.
            types (list[Type], optional): A list of types to filter by. Defaults to None.
        Returns:
            A list of references to the tokens in the tensor.
        """
        return self.token_op.get_reference_multiple(mask, types)
    
    def get_analog_indices(self, analog_id):
        """
        Get indices of tokens in an analog.

        Args:
            analog_id (int): The analog ID to get the indices for.

        Returns:
            A list of indices of tokens in the analog.
        """
        return self.token_op.get_analog_indices(analog_id)
    
    # --------------------------------------------------------------

    # ====================[ TENSOR FUNCTIONS ]======================
    def cache_masks(self, types_to_recompute: list[Type] = None):   # Compute and cach masks for given types
        """
        Compute and cache masks
        
        Args:
            types_to_recompute (list[Type], optional): The types to recompute the mask for. Defaults to All types.
        """
        self.tensor_op.cache_masks(types_to_recompute)
    
    def compute_mask(self, token_type: Type):                       # Compute the mask for a token type
        """
        Compute the mask for a token type
        
        Args:
            token_type (Type): The type to get the mask for.

        Returns:
            A mask of nodes with given type.   
        """
        return self.tensor_op.compute_mask(token_type)
    
    def get_mask(self, token_type: Type):                           # Returns mask for given token type
        """
        Return cached mask for given token type
        
        Args:
            token_type (Type): The type to get the mask for.

        Returns:
            The cached mask for the given token type.
        """
        return self.tensor_op.get_mask(token_type)                   

    def get_combined_mask(self, n_types: list[Type]):               # Returns combined mask of give types
        """
        Return combined mask of given types

        Args:
            n_types (list[Type]): The types to get the mask for.

        Returns:
            A mask of the given types.

        Raises:
            TypeError: If n_types is not a list.
        """
        return self.tensor_op.get_combined_mask(n_types)

    def get_all_nodes_mask(self):                                   # Returns a mask for all nodes (Exluding empty or deleted rows)
        """Return mask for all non-deleted nodes"""
        return (self.nodes[:, TF.DELETED] == B.FALSE)

    def add_token(self, token: Token):                              # Add a token to the tensor
        """
        Add a token to the tensor. If tensor is full, expand it first.

        Args:
            token (Token): The token to add.
            name (str, optional): The name of the token. Defaults to None.

        Returns:
            Ref_Token: Reference to the token that was added.

        Raises:
            ValueError: If the token is invalid.
        """
        return self.tensor_op.add_token(token)
    
    def expand_tensor(self):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes expansion factor. Minimum expansion is 5.
        Expands nodes, connections, links and mappings tensors.
        """
        self.tensor_op.expand_tensor()
    
    def expand_tensor_by_count(self, count: int):                   # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes by count.
        Expands nodes, connections, links and mappings tensors.
        """
        self.tensor_op.expand_tensor_by_count(count)
    
    def del_token(self, ref_tokens: Ref_Token):                     # Delete nodes from tensor   
        """
        Delete tokens from tensor. Pass in a list of Ref_Tokens to delete multiple tokens at once.
        
        Args:
            ref_tokens (Ref_Token): The token(s) to delete. 
        """
        self.tensor_op.del_token_ref(ref_tokens)

    def del_connections(self, ref_token: Ref_Token):
        """
        Delete connection from tensor.
        
        Args:
            ref_token (Ref_Token): Refence to token, to delete connections from.
        """
        self.tensor_op.del_connections_ref(ref_token)
    
    def get_analog(self, analog: int):
        """
        Get an analog from the set.
        
        Args:
            analog (int): The analog ID to get.
        
        Returns:
            Analog: The analog object containing tokens, connections, links, and names.
            
        Raises:
            ValueError: If the analog doesn't exist in the set.
        """
        return self.tensor_op.get_analog(analog)
            
    def add_analog(self, analog: Analog):
        """
        Add an analog to the set.

        Args:
            analog (Analog): The analog to add.
        """
        return self.tensor_op.add_analog(analog)

    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        """Update list of analogs in tensor, and their node counts"""
        self.tensor_op.analog_node_count()
   
    def print(self, f_types=None):                                  # Here for testing atm
        """
        Print the set.

        Args:
            f_types (list[TF], optional): The features to print.

        Raises:
            ValueError: If nodePrinter is not found.
        """
        self.tensor_op.print(f_types)
    
    def get_count(self):
        """Get the number of nodes in the set."""
        return self.tensor_op.get_count()
    # --------------------------------------------------------------

    # ====================[ UPDATE FUNCTIONS ]=======================
    def initialise_float(self, n_type: list[Type], features: list[TF]): # Initialise given features
        """
        Initialise given features
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            features (list[TF]): The features to initialise.
        """
        self.update_op.initialise_float(n_type, features)
    
    def initialise_input(self, n_type: list[Type], refresh: float):     # Initialize inputs to 0, and td_input to refresh.
        """ 
        Initialize inputs to 0, and td_input to refresh
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            refresh (float): The value to set the td_input to.
        """
        self.update_op.initialise_input(n_type, refresh)

    def initialise_act(self, n_type: list[Type]):                       # Initialize act to 0.0,  and call initialise_inputs
        """Initialize act to 0.0,  and call initialise_inputs
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        self.update_op.initialise_act(n_type)

    def initialise_state(self, n_type: list[Type]):                     # Set self.retrieved to false, and call initialise_act
        """Set self.retrieved to false, and call initialise_act
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        self.update_op.initialise_state(n_type)
        
    def update_act(self):                                               # Update act of nodes
        """Update act of nodes. Based on params.gamma, params.delta, and params.HebbBias."""
        self.update_op.update_act()

    def zero_lateral_input(self, n_type: list[Type]):                   # Set lateral_input to 0 
        """
        Set lateral_input to 0;
        to allow synchrony at different levels by 0-ing lateral inhibition at that level 
        (e.g., to bind via synchrony, 0 lateral inhibition in POs).
        
        Args:
            n_type (list[Type]): The types of nodes to set lateral_input to 0.
        """
        self.update_op.zero_lateral_input(n_type)
    
    def update_inhibitor_input(self, n_type: list[Type]):               # Update inputs to inhibitors by current activation for nodes of type n_type
        """
        Update inputs to inhibitors by current activation for nodes of type n_type
        
        Args:
            n_type (list[Type]): The types of nodes to update inhibitor inputs.
        """
        self.update_op.update_inhibitor_input(n_type)

    def reset_inhibitor(self, n_type: list[Type]):                      # Reset the inhibitor input and act to 0.0 for given type
        """
        Reset the inhibitor input and act to 0.0 for given type
        
        Args:
            n_type (list[Type]): The types of nodes to reset inhibitor inputs and acts.
        """
        self.update_op.reset_inhibitor(n_type)
    
    def update_inhibitor_act(self, n_type: list[Type]):                 # Update the inhibitor act for given type
        """
        Update the inhibitor act for given type
        
        Args:
            n_type (list[Type]): The types of nodes to update inhibitor acts.
        """
        self.update_op.update_inhibitor_act(n_type)
    # --------------------------------------------------------------

    # =======================[ P FUNCTIONS ]========================
    def p_initialise_mode(self):                                        # Initialize all p.mode back to neutral.
        """Initialize mode to neutral for all P units."""
        self.update_op.p_initialise_mode()

    def p_get_mode(self):                                               # Set mode for all P units
        """Set mode for all P units"""
        # Pmode = Parent: child RB act> parent RB act / Child: parent RB act > child RB act / Neutral: o.w
        self.update_op.p_get_mode()

    # ---------------------------------------------------------------

    # =======================[ PO FUNCTIONS ]======================== # TODO: Can move out of tensor to save memory, as shared values.
    def po_get_weight_length(self):                                     # Sum value of links with weight > 0.1 for all PO nodes
        """Sum value of links with weight > 0.1 for all PO nodes - Used for semNormalisation"""
        self.update_op.po_get_weight_length()
            
    def po_get_max_semantic_weight(self):                               # Get max link weight for all PO nodes
        """Get max link weight for all PO nodes - Used for semNormalisation"""
        self.update_op.po_get_max_semantic_weight()
        
    # ---------------------------------------------------------------
