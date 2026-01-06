# nodes/network/operations/node_ops.py
# Node operations for Network class
import logging
import torch

from ...enums import *
from ..single_nodes import Token, Semantic

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING, Optional, Tuple
if TYPE_CHECKING:
    from ..network import Network

class NodeOperations:
    """
    Node operations for the Network class.
    Handles node management using global indexes.
    """
    
    def __init__(self, network):
        """
        Initialize NodeOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network

    # =====================[ P Mode Operations ]======================
    
    def get_pmode(self):
        """
        Get parent mode of P units in driver and recipient. Used in time steps activations.
        """
        self.network.sets[Set.DRIVER].update_op.p_get_mode()
        self.network.sets[Set.RECIPIENT].update_op.p_get_mode()
    
    def initialise_p_mode(self, tk_set: Set = Set.RECIPIENT):
        """
        Initialise p_mode in the given set.

        Args:
            tk_set (Set, optional): The set to initialise p_mode in. (Defaults to recipient)
        """
        self.network.sets[tk_set].update_op.p_initialise_mode()
    
    def get_weight_lengths(self):
        """
        Get weight lengths of PO units. Used in run initialisation.
        (driver, recipient, memory, new_set)
        """
        for tk_set in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY, Set.NEW_SET]:
            self.network.sets[tk_set].update_op.po_get_weight_length()

    # =====================[ Token Operations ]======================
    
    def add_token(self, token: Token) -> int:
        """
        Add a token to the network.

        Args:
            token (Token): The token to add. Set is determined by TF.SET feature.
        
        Returns:
            int: The global index of the added token.
        
        Raises:
            ValueError: If the token set feature is not a valid set.
        """
        add_set = int(token.tensor[TF.SET])
        if add_set not in [s.value for s in Set]:
            raise ValueError("Invalid set in token feature.")
        
        name = token.name if hasattr(token, 'name') and token.name else ""
        new_indices = self.network.tokens.add_tokens(token.tensor.unsqueeze(0), [name])
        self.network.recache()
        return new_indices[0].item() if len(new_indices) > 0 else None

    def del_token(self, idx: int):
        """
        Delete a token from the network by global index.

        Args:
            idx (int): The global index of the token to delete.
        """
        self.network.tokens.delete_tokens(torch.tensor([idx]))
        self.network.recache()
    
    def get_token(self, idx: int) -> Token:
        """
        Get a token from the network by global index.

        Args:
            idx (int): The global index of the token.
            
        Returns:
            Token: A Token object with the token's features.
        """
        tensor = self.network.token_tensor.tensor[idx, :].clone()
        name = self.network.token_tensor.get_name(idx)
        return Token(tensor=tensor, name=name)
    
    def get_tk_value(self, idx: int, feature: TF) -> float:
        """
        Get the value of a feature for a token.

        Args:
            idx (int): The global index of the token.
            feature (TF): The feature to get.

        Returns:
            float: The value of the feature.
        """
        return self.network.token_tensor.get_feature(idx, feature).item()
    
    def set_tk_value(self, idx: int, feature: TF, value: float):
        """
        Set the value of a feature for a token.

        Args:
            idx (int): The global index of the token.
            feature (TF): The feature to set.
            value (float): The value to set.
        """
        self.network.token_tensor.set_feature(torch.tensor([idx]), feature, float(value))
    
    def get_tk_name(self, idx: int) -> str:
        """
        Get the name of a token by global index.

        Args:
            idx (int): The global index of the token.
            
        Returns:
            str: The name of the token.
        """
        return self.network.token_tensor.get_name(idx)
    
    def set_tk_name(self, idx: int, name: str):
        """
        Set the name of a token by global index.

        Args:
            idx (int): The global index of the token.
            name (str): The name to set.
        """
        self.network.token_tensor.set_name(idx, name)

    def get_tk_set(self, idx: int) -> Set:
        """
        Get the set of a token by global index.

        Args:
            idx (int): The global index of the token.
            
        Returns:
            Set: The set the token belongs to.
        """
        return Set(int(self.network.token_tensor.get_feature(idx, TF.SET).item()))

    # =====================[ Semantic Operations ]======================

    def add_semantic(self, semantic: Semantic) -> int:
        """
        Add a semantic to the network.

        Args:
            semantic (Semantic): The semantic to add.
        
        Returns:
            int: The index of the added semantic.
            
        Raises:
            ValueError: If provided semantic is not semantic type.
        """
        if semantic.tensor[SF.TYPE] != Type.SEMANTIC:
            raise ValueError("Cannot add non-semantic to semantic set.")
        
        return self.network.semantics.add_semantic(semantic)

    def del_semantic(self, idx: int):
        """
        Delete a semantic from the network by index.
        
        Args:
            idx (int): The index of the semantic to delete.
        """
        self.network.semantics.del_semantic(idx)
    
    def get_sem_value(self, idx: int, feature: SF) -> float:
        """
        Get the value of a feature for a semantic.

        Args:
            idx (int): The index of the semantic.
            feature (SF): The feature to get.

        Returns:
            float: The value of the feature.
        """
        return self.network.semantics.get(idx, feature)
    
    def set_sem_value(self, idx: int, feature: SF, value: float):
        """
        Set the value of a feature for a semantic.

        Args:
            idx (int): The index of the semantic.
            feature (SF): The feature to set.
            value (float): The value to set.
        """
        self.network.semantics.set(idx, feature, value)

    def set_sem_max_input(self):
        """
        Set the maximum input for all semantics.
        """
        max_input = self.network.semantics.get_max_input()
        self.network.semantics.set_max_input(max_input)

    # =====================[ Most Active Token ]======================
    
    def get_most_active_token(self, sets: list[Set] = None) -> dict[Set, int]:
        """
        Get the most active token in each set.

        Args:
            sets (list[Set], optional): List of sets to check. Defaults to all sets.

        Returns:
            dict[Set, int]: Dictionary mapping set to global index of most active token.
        """
        tokens = dict()
        sets_to_check = sets if sets else list(Set)
        
        for tk_set in sets_to_check:
            local_idx = self.network.sets[tk_set].token_op.get_most_active_token()
            if local_idx is not None:
                # Convert local index to global index
                global_idx = self.network.sets[tk_set].lcl.to_global(torch.tensor([local_idx]))[0].item()
                tokens[tk_set] = global_idx
        
        return tokens

    # =====================[ Made/Maker Unit Operations ]======================
    
    def get_made_unit(self, idx: int) -> Optional[Tuple[int, Set]]:
        """
        Get the made unit for a token.

        Args:
            idx (int): The global index of the token.
            
        Returns:
            Optional[Tuple[int, Set]]: Tuple of (local_index, set) of the made unit, or None if no made unit.
        """
        made_unit_index = self.get_tk_value(idx, TF.MADE_UNIT)
        made_unit_set_val = self.get_tk_value(idx, TF.MADE_SET)
        
        if made_unit_index == null:
            logger.debug(f"No made unit for token at index {idx}")
            return None
        
        made_unit_set = Set(int(made_unit_set_val))
        return (int(made_unit_index), made_unit_set)
    
    def get_maker_unit(self, idx: int) -> Optional[Tuple[int, Set]]:
        """
        Get the maker unit for a token.

        Args:
            idx (int): The global index of the token.
            
        Returns:
            Optional[Tuple[int, Set]]: Tuple of (local_index, set) of the maker unit, or None if no maker unit.
        """
        maker_unit_index = self.get_tk_value(idx, TF.MAKER_UNIT)
        maker_unit_set_val = self.get_tk_value(idx, TF.MAKER_SET)
        
        if maker_unit_index == null:
            logger.debug(f"No maker unit for token at index {idx}")
            return None
        
        maker_unit_set = Set(int(maker_unit_set_val))
        return (int(maker_unit_index), maker_unit_set)

    # =====================[ Kludgey Comparitor ]======================
    
    def kludgey_comparitor(self, tk_set: Set, po1: int, po2: int):
        """
        Kludgey comparitor for the network.
        Based on Hummel & Biederman, 1992.
        Looks at highest weight linked semantics for two POs.
        If they encode the same dimension, attach the comparative semantics to these POs.
        
        Args:
            tk_set (Set): The set containing the PO tokens.
            po1 (int): Local index of first PO token.
            po2 (int): Local index of second PO token.
        """
        logger.debug(f"Kludgey comparitor for {tk_set.name}[{po1}] and {tk_set.name}[{po2}]")
        # Make sure the comparative semantics exist
        self.network.semantics.init_sdm()
        # Get the highest weight semantics for each PO
        po1_sem = self.network.links.get_max_linked_sem_idx(tk_set, po1)
        po2_sem = self.network.links.get_max_linked_sem_idx(tk_set, po2)
        # Check for common dimension
        sem1_dim = self.network.semantics.get_dim(po1_sem)
        sem2_dim = self.network.semantics.get_dim(po2_sem)
        if sem1_dim == sem2_dim:
            logger.debug(f"Common dimension found: {sem1_dim}")
            # Compare literal values
            sem1_value = self.network.semantics.get(po1_sem, SF.AMOUNT)
            sem2_value = self.network.semantics.get(po2_sem, SF.AMOUNT)
            if sem1_value > sem2_value:
                logger.debug(f"More value found: {sem1_value} > {sem2_value}")
                self.network.links.connect_comparitive(tk_set, po1, SDM.MORE)
                self.network.links.connect_comparitive(tk_set, po2, SDM.LESS)
            elif sem1_value < sem2_value:
                logger.debug(f"Less value found: {sem1_value} < {sem2_value}")
                self.network.links.connect_comparitive(tk_set, po1, SDM.LESS)
                self.network.links.connect_comparitive(tk_set, po2, SDM.MORE)
            else:
                logger.debug(f"Same value found: {sem1_value} == {sem2_value}")
                self.network.links.connect_comparitive(tk_set, po1, SDM.SAME)
                self.network.links.connect_comparitive(tk_set, po2, SDM.SAME)
        else:
            logger.debug(f"No common dimension found: {sem1_dim} != {sem2_dim}")

    # =====================[ Index Lookup Operations ]======================
    
    def get_index_by_name(self, tk_set: Set, name: str) -> int:
        """
        Get the global index of a token by name.
        
        Args:
            tk_set (Set): The set to search in.
            name (str): The name of the token.
            
        Returns:
            int: The global index of the token.
            
        Raises:
            ValueError: If token not found.
        """
        set_indices = self.network.token_tensor.cache.get_set_indices(tk_set)
        for global_idx in set_indices:
            if self.network.token_tensor.get_name(global_idx.item()) == name:
                return global_idx.item()
        raise ValueError(f"Token with name '{name}' not found in set {tk_set.name}")
    
    def get_index_by_id(self, tk_set: Set, token_id: int) -> int:
        """
        Get the global index of a token by ID.
        
        Args:
            tk_set (Set): The set to search in.
            token_id (int): The ID of the token.
            
        Returns:
            int: The global index of the token.
            
        Raises:
            ValueError: If token not found.
        """
        set_indices = self.network.token_tensor.cache.get_set_indices(tk_set)
        for global_idx in set_indices:
            if int(self.network.token_tensor.tensor[global_idx, TF.ID].item()) == token_id:
                return global_idx.item()
        raise ValueError(f"Token with ID {token_id} not found in set {tk_set.name}")

    def local_to_global(self, tk_set: Set, local_idx: int) -> int:
        """
        Convert a local index to a global index.
        
        Args:
            tk_set (Set): The set the local index is in.
            local_idx (int): The local index.
            
        Returns:
            int: The global index.
        """
        return self.network.sets[tk_set].lcl.to_global(torch.tensor([local_idx]))[0].item()
    
    def global_to_local(self, idx: int) -> Tuple[int, Set]:
        """
        Convert a global index to a local index.
        
        Args:
            idx (int): The global index.
            
        Returns:
            Tuple[int, Set]: The local index and set.
        """
        tk_set = self.get_tk_set(idx)
        local_idx = self.network.sets[tk_set].lcl.to_local(torch.tensor([idx]))[0].item()
        return (local_idx, tk_set)
