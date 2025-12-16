from .tensor.token_tensor import Token_Tensor
from .connections.connections import Connections_Tensor
from .connections.links import Links, LD
from .connections.mapping import Mapping, MD
from .tensor.analogs import Analog_ops
from .tensor_view import TensorView
import torch
from ...enums import *

class TensorTypes():
    """
    Enum to access the type of view to get.
    """
    SET = 0
    """ Get tokens for a specific set """
    CON = 1
    """ Get connections for a specific set """
    LINK = 2
    """ Get links for a specific set """
    MAP = 3
    """ Get mappings"""
    

class Tokens:
    """
    Class to hold all the tokens in the network.
    Provides functions that perform operations that 
    include multiple objects (e.g deleting tokens in token tensor and connections objects.)
    """
    def __init__(self, token_tensor: Token_Tensor, connections: Connections_Tensor, links: Links, mapping: Mapping):
        """
        Initialize the Tokens object.
        Args:
            token_tensor: Token_Tensor - The token tensor object. Shape: [tokens, features]
            connections: Connections_Tensor - The connections object. Shape: [tokens, tokens]
            links: Links - The links object. Shape: [tokens, semantics]
            mapping: Mapping - The mapping object. Shape: [driver, recipient, mapping fields]
        """
        self.token_tensor: Token_Tensor = token_tensor
        """holds the token tensor"""
        self.connections: Connections_Tensor = connections
        """holds the connections tensor"""
        self.analog_ops: Analog_ops = Analog_ops(self.token_tensor)
        """holds the analog operations"""
        self.links: Links = links
        """holds the links tensor"""
        self.mapping: Mapping = mapping
        """holds the mapping tensor, None if not driver or recipient"""
    
    def check_count(self) -> int:
        """
        Check the number of tokens in the tensor is the same as connections, links, and mapping tensors, etc
        If token count is greater, expand the tensors to match the token count.
        If token count is less, delete the tokens from the tensors.
        """
        resized = []
        token_count = self.token_tensor.get_count()
        connections_count = self.connections.get_count()
        links_count = self.links.get_count(LD.TK)
        if token_count > connections_count:
            resized.append(TensorTypes.CON)
            self.connections.expand_to(token_count)
        if token_count > links_count:
            resized.append(TensorTypes.LINK)
            self.links.expand_to(token_count, LD.TK)
        # Check mapping counts for driver and recipient
        driver_count = self.token_tensor.get_set_count(Set.DRIVER)
        recipient_count = self.token_tensor.get_set_count(Set.RECIPIENT)
        if driver_count > self.mapping.get_driver_count():
            resized.append(TensorTypes.MAP)
            self.mapping.expand(driver_count, MD.DRI)
        if recipient_count > self.mapping.get_recipient_count():
            resized.append(TensorTypes.MAP)
            self.mapping.expand(recipient_count, MD.REC)
        return resized
    
    def delete_tokens(self, idxs: torch.Tensor):
        """
        Delete the tokens at the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to delete.
        """
        self.token_tensor.del_tokens(idxs)
        self.connections.del_connections(idxs)
        self.links.del_links(idxs)
        # Delete mappings if required.
        driver_mask = torch.where(self.token_tensor.tensor[idxs, TF.SET] == Set.DRIVER)[0]
        if torch.any(driver_mask):
            self.mapping.del_driver_mappings(idxs[driver_mask])
        recipient_mask = torch.where(self.token_tensor.tensor[idxs, TF.SET] == Set.RECIPIENT)[0]
        if torch.any(recipient_mask):
            self.mapping.del_recipient_mappings(idxs[recipient_mask])
        # Make sure the counts are correct (Shouldn't be needed, but just in case)
        self.check_count()
    
    def add_tokens(self, tokens: torch.Tensor, names: list[str]):
        """
        Add the tokens to the token tensor.
        Args:
            tokens: torch.Tensor - The tokens to add.
            names: list[str] - The names of the tokens.
        """
        new_indicies = self.token_tensor.add_tokens(tokens, names)
        self.check_count()
        return new_indicies
    
    def copy_tokens(self, indices: torch.Tensor, to_set: Set, connect_to_copies: bool = False) -> torch.Tensor:
        """
        Copy the tokens at the given indices to the given set.
        Args:
            indices: torch.Tensor - The indices of the tokens to copy.
            to_set: Set - The set to copy the tokens to.
            connect_to_copies: bool - Whether to connect the new tokens to the copies of the original tokens.
        Returns:
            torch.Tensor - The indices of the tokens that were replaced.
        """
        copy_indicies =  self.token_tensor.copy_tokens(indices, to_set)
        self.check_count()
        if connect_to_copies:
            internal_connections = self.connections.connections[indices, indices].clone()
            self.connections.connections[copy_indicies, copy_indicies] = internal_connections
        return copy_indicies
    
    def get_view(self, view_type: TensorTypes, set: Set = None) -> TensorView | torch.Tensor:
        """
        Get a view of the tokens, connections, links, or mappings for the given set.
        Args:
            view_type: ViewTypes - The type of view to get.
            set: Set - The set to get the view for.
        Returns:
            TensorView - A view-like object that maps operations back to the original tensor.
        """
        set_indices = self.token_tensor.cache.get_set_indices(set)
        match view_type:
            case TensorTypes.SET:
                return self.token_tensor.get_view(set_indices)
            case TensorTypes.CON:
                return self.connections.get_view(set_indices)
            case TensorTypes.LINK:
                return self.links.get_view(set_indices)
            case TensorTypes.MAP:
                if set not in [Set.DRIVER, Set.RECIPIENT, None]:
                    raise ValueError(f"Invalid set for mapping view: {set}")
                return self.mapping.adj_matrix # only one view for mappings, so can just return the whole tensor
            case _:
                raise ValueError(f"Invalid view type: {view_type}")
    
    def set_name(self, idx: int, name: str):
        """
        Set the name of the token at the given index.
        Args:
            idx: int - The index of the token to set the name of.
            name: str - The name to set the token to.
        """
        self.token_tensor.set_name(idx, name)
    
    def get_name(self, idx: int) -> str:
        """
        Get the name of the token at the given index.
        Args:
            idx: int - The index of the token to get the name of.
        Returns:
            str - The name of the token.
        """
        return self.token_tensor.get_name(idx)