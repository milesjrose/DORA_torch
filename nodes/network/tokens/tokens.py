from .tensor.token_tensor import Token_Tensor
from .connections.connections import Connections_Tensor
from .connections.links import Links, LD
from .connections.mapping import Mapping, MD
from .tensor.analogs import Analog_ops
from .tensor_view import TensorView
import torch
from ...enums import *
from logging import getLogger
logger = getLogger(__name__)

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
        assert isinstance(token_tensor, Token_Tensor), f"token_tensor must be a Token_Tensor, not {type(token_tensor)}"
        self.token_tensor: Token_Tensor = token_tensor
        """holds the token tensor, token_tensor[token, feature] = value"""
        assert isinstance(connections, Connections_Tensor), f"connections must be a Connections_Tensor, not {type(connections)}"
        self.connections: Connections_Tensor = connections
        """holds the connections tensor, connections.connections[parent, child] = True if connected"""
        assert isinstance(links, Links), f"links must be a Links, not {type(links)}"
        self.analog_ops: Analog_ops = Analog_ops(self.token_tensor)
        """holds the analog operations"""
        assert isinstance(links, Links), f"links must be a Links, not {type(links)}"
        self.links: Links = links
        """holds the links tensor, tokens -> semantics"""
        assert isinstance(mapping, Mapping), f"mapping must be a Mapping, not {type(mapping)}"
        self.mapping: Mapping = mapping
        d_count = self.token_tensor.cache.get_set_count(Set.DRIVER)
        r_count = self.token_tensor.cache.get_set_count(Set.RECIPIENT)
        m_d_count = self.mapping.get_driver_count()
        m_r_count = self.mapping.get_recipient_count()
        if d_count < m_d_count or r_count < m_r_count:
            logger.critical(f"Malformed token/map data at init: d_count: {d_count} < m_d_count: {m_d_count} or r_count: {r_count} < m_r_count: {m_r_count} -> clearing mappings")
            self.mapping.init_tensor(r_count, d_count)
        """holds the mapping tensor, driver tokens -> recipient tokens"""
        self.recache()
    
    def recache(self):
        """
        Recache the token information, and ensure the sizes of the connections (con, link, map) tensors are correct.
        """
        old_set_idxs = {}
        for set in Set:
            old_set_idxs[set] = self.token_tensor.cache.get_set_indices(set)
        self.token_tensor.cache.cache_sets()
        self.token_tensor.cache.cache_analogs()
        # Check the connecting tensor sizes.
        token_count = self.token_tensor.get_count()

        # Check connections tensor size.
        con_count = self.connections.get_count()
        if token_count > con_count:
            self.connections.expand_to(token_count)
        if token_count < con_count:
            logger.error(f"Connections tensor size is smaller than token count, no way to handle this atm.")

        # Check links tensor size.
        link_count = self.links.get_count(LD.TK)
        if token_count > link_count: 
            self.links.expand_to(token_count, LD.TK)
        if token_count < link_count:
            logger.error(f"Links tensor size is smaller than token count, no way to handle this atm.")

        # Check mapping tensor size.
        driver_count = self.token_tensor.get_set_count(Set.DRIVER)
        recipient_count = self.token_tensor.get_set_count(Set.RECIPIENT)
        if driver_count > self.mapping.get_driver_count():
            self.mapping.expand(driver_count, MD.DRI)
        elif driver_count < self.mapping.get_driver_count():
            # Need to shrink the mapping tensor.
            # First find the indices of the tokens that are no longer in the driver set.
            mask = ~torch.isin(old_set_idxs[Set.DRIVER], self.token_tensor.cache.get_set_indices(Set.DRIVER))
            idx_to_delete = torch.where(mask)[0]
            self.mapping.shrink(idx_to_delete, MD.DRI)
            # NOTE: this just a double check, should remove later:
            if driver_count != self.mapping.get_driver_count():
                logger.critical(f"Driver map count mismatch: {driver_count} != {self.mapping.get_driver_count()}")
                logger.debug(f"old_set_idxs[Set.DRIVER]: {old_set_idxs[Set.DRIVER]}, new_set_idxs: {self.token_tensor.cache.get_set_indices(Set.DRIVER)}")
                logger.debug(f"mask: {mask}")
                raise ValueError(f"Driver map shrink unsuccessful.")
        if recipient_count > self.mapping.get_recipient_count():
            self.mapping.expand(recipient_count, MD.REC)
        elif recipient_count < self.mapping.get_recipient_count():
            # Need to shrink the mapping tensor.
            # First find the indices of the tokens that are no longer in the recipient set.
            mask = ~torch.isin(old_set_idxs[Set.RECIPIENT], self.token_tensor.cache.get_set_indices(Set.RECIPIENT))
            idx_to_delete = torch.where(mask)[0]
            self.mapping.shrink(idx_to_delete, MD.REC)
            # NOTE: this just a double check, should remove later:
            if recipient_count != self.mapping.get_recipient_count():
                logger.critical(f"Recipient map count mismatch: {recipient_count} != {self.mapping.get_recipient_count()}")
                logger.debug(f"old_set_idxs[Set.RECIPIENT]: {old_set_idxs[Set.RECIPIENT]}, new_set_idxs: {self.token_tensor.cache.get_set_indices(Set.RECIPIENT)}")
                logger.debug(f"mask: {mask}")
                raise ValueError(f"Recipient map shrink unsuccessful.")
        
        logger.debug(f"Recached: tk={token_count}, con={con_count}, link={link_count}, map={driver_count}x{recipient_count}")
    
    def check_count(self) -> int:
        """
        Check the number of tokens in the tensor is the same as connections, links, and mapping tensors, etc
        If token count is greater, expand the tensors to match the token count.
        If token count is less, delete the tokens from the tensors.
        """
        self.recache()
    
    def delete_tokens(self, idxs: torch.Tensor):
        """
        Delete the tokens at the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to delete.
        """
        self.token_tensor.del_tokens(idxs)
        self.connections.del_connections(idxs)
        self.links.del_links(idxs)
        self.recache()
    
    def add_tokens(self, tokens: torch.Tensor, names: list[str]):
        """
        Add the tokens to the token tensor.
        Args:
            tokens: torch.Tensor - The tokens to add.
            names: list[str] - The names of the tokens.
        """
        new_indicies = self.token_tensor.add_tokens(tokens, names)
        self.recache()
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
        self.recache()
        if connect_to_copies:
            internal_connections = self.connections.tensor[indices, indices].clone()
            self.connections.tensor[copy_indicies, copy_indicies] = internal_connections
        return copy_indicies
    
    def move_tokens(self, indices: torch.Tensor, to_set: Set):
        """
        Move the tokens at the given indices to the given set.
        """
        self.token_tensor.move_tokens(indices, to_set)
        self.recache()
    
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