# nodes/network/sets/base_set_operations/token_operations.py
# Token operations for Base_Set class

import torch
import logging
from typing import TYPE_CHECKING

from ...single_nodes import Token, Ref_Token, Ref_Analog, Pairs, get_default_features
from ....utils import tensor_ops as tOps
from ....enums import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if TYPE_CHECKING:
    from ..base_set import Base_Set

class TokenOperations:
    """
    Token operations for the Base_Set class.
    Handles token operations.
    """
    
    def __init__(self, base_set):
        """
        Initialize TokenOperations with reference to Base_Set.
        
        Args:
            base_set: Reference to the Base_Set object
        """
        self.base_set: 'Base_Set' = base_set

    # ----------------------------[TOKEN FUNCTIONS]-------------------------------

    def get_feature(self, ref_token: Ref_Token, feature: TF):        # Get feature of single node
        """
        Get a feature for a referenced token.
        
        Args:
            ref_token (Ref_Token): Reference of token.
            feature (TF): The feature to get.

        Returns:
            The feature for the referenced token.

        Raises:
            IndexError: If the referenced token or feature is invalid.
            KeyError: If the feature value is not a valid type.
            TypeError: If the feature value is not a valid type.
            ValueError: If the feature value is invalid.
        """
        idx = self.get_index(ref_token)
        try:
            feature_value = self.base_set.nodes[idx, feature].item()
            casted_value = TF_type(feature)(feature_value) if feature_value != null else null
            return casted_value
        except IndexError as e:
            logger.critical(f"Invalid reference token or feature: {ref_token.set.name}[{idx}] {feature.name}")
            raise ValueError("Invalid reference token or feature.") from e
        except KeyError as e:
            logger.critical(f"No type for feature: {ref_token.set.name}[{idx}] {feature.name}")
            raise ValueError("No type for feature.") from e
        except (ValueError, TypeError) as e:
            logger.critical(f"Invalid value for feature: {ref_token.set.name}[{idx}] {feature.name}, value: {feature_value}")
            logger.critical(f"Token:\n {Token(tensor=self.base_set.nodes[idx, :]).get_string()}")
            raise ValueError("Invalid value for feature.") from e

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
        idx = self.get_index(ref_token)
        try:
            self.base_set.nodes[idx, feature] = float(value)
            logger.debug(f"Set feature: {ref_token.set.name}[{idx}] {feature.name} -> {value}")
            logger.debug(f"Token:\n{Token(tensor=self.base_set.nodes[idx, :]).get_string()}")
        except:
            raise ValueError("Invalid reference token, feature, or value.")

    def set_features(self, indices: list[int], feature: TF, value: float):
        """
        Set a feature for a list of tokens.

        Args:
            indices (list[int]): The indices of the tokens to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.

        Raises:
            ValueError: If the indices, feature, or value is invalid.
        """
        try:
            self.base_set.nodes[indices, feature] = float(value)
        except:
            raise ValueError("Invalid indices, feature, or value.")

    def set_features_all(self, feature: TF, value: float):
        """
        Set a feature for all tokens in the set.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        self.base_set.nodes[all_nodes_mask, feature] = float(value)

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
        try:
            return self.base_set.names[ref_token.ID]
        except:
            raise ValueError("Invalid reference token.")
    
    def set_name(self, ref_token: Ref_Token, name: str):
        """
        Set the name for a referenced token.
        
        Args:
            ref_token (Ref_Token): The token to set the name for.
            name (str): The name to set the token to.

        Raises:
            ValueError: If the referenced token is invalid.
        """
        try:
            self.base_set.names[ref_token.ID] = name
        except:
            raise ValueError("Invalid reference token or name.")
    
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
        id = int(ref_token.ID)
        logger.debug(f"Get index for {ref_token.set.name}[{id}] in {self.base_set.token_set.name}: {self.base_set.nodes.shape}")
        try:
            return self.base_set.IDs[id]
        except Exception as e:
            logger.critical(f"[{self.base_set.token_set.name}] : Token ID {id} not in {self.base_set.IDs}.")
            raise e

    def get_indices(self, ref_tokens: list[Ref_Token]) -> list[int]:
        """
        Get indicies of a list of reference tokens.

        Args:
            ref_tokens (list[Ref_Token]): The tokens to get the indices for.

        Returns:
            A list of indices of the tokens.
        """
        indices = []
        for tk in ref_tokens:
            indices.append(self.get_index(tk))
        return indices

    def get_reference(self, id:int=None, index:int=None, name:str=None) -> Ref_Token:        # Get reference to token with given ID, index, or name
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
        source = ""
        if index is not None:
            try:
                id = self.base_set.nodes[index, TF.ID].item()
                source = "index"    
            except:
                logger.critical(f"Invalid index: {index} in {self.base_set.token_set.name} with shape: {self.base_set.nodes.shape}")
                raise ValueError("Invalid index.")
        elif name is not None:
            try:
                # I feel like there is a better way to do this...
                dict_index = list(self.base_set.names.values()).index(name)
                id = list(self.base_set.names.keys())[dict_index]
                source = "name"
            except:
                logger.critical(f"Invalid name: {name} in {self.base_set.token_set.name} with names: {self.base_set.names}")
                raise ValueError("Invalid name.")
        elif id is not None:
            try:
                # Check if ID exists in the IDs dictionary instead of recursive call
                if id not in self.base_set.IDs:
                    raise ValueError("Invalid ID.")
                source = "id"
            except Exception as e:
                logger.critical(f"Invalid ID: {id} in {self.base_set.token_set.name} with IDs: {self.base_set.IDs}")
                raise e
        else:
            raise ValueError("No ID, index, or name provided.")
        
        try:
            name = self.base_set.names[id]
        except:
            name = None

        return Ref_Token(self.base_set.token_set, id, name)
    
    def get_single_token(self, ref_token: Ref_Token, copy=True) -> Token:    # Get a single token from the tensor
        """
        Get a single token from the tensor.

        - If copy is set to False, changes to the returned token will affect the tensor.

        Args:
            ref_token (Ref_Token): The token to get.
            copy (bool, optional): Whether to return a copy of the token. Defaults to True.

        Returns:
            Token: The token object.
        """
        if copy:
            tensor = self.base_set.nodes[self.get_index(ref_token), :].clone()
        else:
            tensor = self.base_set.nodes[self.get_index(ref_token), :]
        return Token(tensor=tensor)
    
    def get_reference_multiple(self, mask=None, types: list[Type] = None) -> list[Ref_Token]:  # Get references to tokens in tensor
        """
        Get references to tokens in the tensor. Must provide either mask or types.

        Args:
            mask (torch.Tensor, optional): A mask to apply to the tensor. Defaults to None.
            types (list[Type], optional): A list of types to filter by. Defaults to None.
        Returns:
            A list of references to the tokens in the tensor.
        """
        if types is not None:
            mask = self.base_set.tensor_op.get_combined_mask(types)
        elif mask is None:
            raise ValueError("No mask or types provided.")
        
        indices = torch.where(mask)[0]
        references = [self.get_reference(index=i) for i in indices]
        return references
    
    def get_max_acts(self):
        """
        Set max_act for all tokens in the set
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        nodes_to_update = self.base_set.nodes[:, TF.ACT] > self.base_set.nodes[:, TF.MAX_ACT]
        update_mask = nodes_to_update & all_nodes_mask
        self.base_set.nodes[update_mask, TF.MAX_ACT] = self.base_set.nodes[update_mask, TF.ACT]
    
    def get_highest_token_type(self) -> Type:
        """
        Get the highest token type in the set.

        Returns:
            Type: The highest token type in the set.
        """
        if self.base_set.nodes.shape[0] == 0:
            return None
        
        return self.base_set.nodes[self.base_set.tensor_op.get_all_nodes_mask(), TF.TYPE].max().item()
    
    def get_child_indices(self, index: int) -> list[int]:
        """
        Get the indices of the children of a given token.

        Args:
            index (int): The index of the token to get the children for.

        Returns:
            A list of indices of the children of the token.
        """
        # Check connections, get nodes that connect to (connections: parent -> child)
        indices = torch.where(self.base_set.connections[index, :] == B.TRUE)[0].tolist()
        return indices
    
    def get_most_active_token(self, mask=None, id=False):
        """
        Get the most active token in the set.

        Args:
            mask (torch.Tensor, optional): A mask to apply to the tensor. Defaults to None.
            id (bool, optional): Whether to return the ID or Ref_Token of the most active token. Defaults to False.

        Returns:
            Ref_Token: The most active token in the set.
            Or Index of the most active token in the set.
        """
        if mask is None:
            mask = self.base_set.tensor_op.get_all_nodes_mask()

        max_act_value, relative_index = torch.max(self.base_set.nodes[mask, TF.ACT], dim=0)
        masked_indices = torch.where(mask)[0]
        absolute_index = masked_indices[relative_index]
        # if most active token is not active, return None
        if max_act_value == 0.0:
            return None
        # O.w return the most active token.
        if id:
            return self.base_set.nodes[absolute_index, TF.ID].item()
        else:
            return self.get_reference(index=absolute_index)
    
    def connect(self, parent: Ref_Token, child: Ref_Token, value=B.TRUE):
        """
        Connect a parent token to a child token.
        """
        logger.debug(f"Connecting {self.get_ref_string(parent)} -> {self.get_ref_string(child)}")
        idx_from = self.get_index(parent)
        idx_to = self.get_index(child)
        self.base_set.connections[idx_from, idx_to] = value
    
    def connect_idx(self, from_idx: int, to_idx: int, value=B.TRUE):
        """
        Connect a token at from_idx to a token at to_idx.
        """
        self.base_set.connections[from_idx, to_idx] = value
    
    def connect_idxs(self, from_idxs: torch.Tensor, to_idxs: torch.tensor, value=B.TRUE):
        """
        Take a tensor of indices, e.g from torch.where, and connect them to each other.
        """
        self.base_set.connections[from_idxs, to_idxs] = value
    
    def connect_mask(self, mask_from: torch.Tensor, mask_to: torch.Tensor, value=B.TRUE):
        """
        Connect tokens in mask_from to tokens in mask_to.
        """
        self.base_set.connections[mask_from, mask_from] = value
    
    def get_connected_tokens(self, ref_token: Ref_Token) -> list[Ref_Token]:
        """
        Get all tokens that are connected to a given token.
        """
        return self.get_reference_multiple(mask=self.base_set.connections[self.get_index(ref_token), :] == B.TRUE)

    def get_ref_string(self, ref_token: Ref_Token):
        """
        Get a string representation of a reference token.
        """
        return f"{self.base_set.token_set.name}[{self.get_index(ref_token)}]({ref_token.ID})"
    
    def reset_inferences(self):
        """
        Reset the inferences of all tokens in the set.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        self.base_set.nodes[all_nodes_mask, TF.INFERRED] = B.FALSE
        self.base_set.nodes[all_nodes_mask, TF.MADE_UNIT] = null
        self.base_set.nodes[all_nodes_mask, TF.MAKER_UNIT] = null
    
    def get_children(self, token_mask: torch.Tensor) -> torch.Tensor:
        """get a mask of the children of a token"""
        children = self.base_set.connections[token_mask, :] == B.TRUE
        children_mask = children.any(dim=0)
        return children_mask

    def get_mapped_pos(self) -> list[Ref_Token]:
        """
        Get all POs that are mapped to.
        """
        pos = self.base_set.tensor_op.get_mask(Type.PO)
        mapped_pos = self.base_set.nodes[pos, TF.MAX_MAP] > 0.0
        mapped_pos = tOps.sub_union(pos, mapped_pos)
        return self.get_reference_multiple(mask=mapped_pos)

    # ----------------------------[ANALOG FUNCTIONS]-------------------------------
    def get_analog_indices(self, analog: Ref_Analog) -> list[int]:
        """
        Get indices of tokens in an analog.

        Args:
            analog (Ref_Analog): The analog to get the indices for.

        Returns:
            A list of indices of tokens in the analog.
        """
        if not isinstance(analog, Ref_Analog):
            raise TypeError("analog must be a Ref_Analog object.")
        if analog.set != self.base_set.token_set:
            raise ValueError(f"Analog {analog.analog_number} is not in the set {self.base_set.token_set}.")

        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        return torch.where(self.base_set.nodes[all_nodes_mask, TF.ANALOG] == analog.analog_number)[0].tolist()
    
    def get_analogs_where(self, feature: TF, value) -> list[Ref_Analog]:
        """
        Get any analogs that contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            List[Ref_Analog]: References to the analogs that contain a token with the given feature and value.
        """

        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()           # Only non-deleted tokens
        matching_tokens = (self.base_set.nodes[all_nodes_mask, feature] == value)
        if not torch.any(matching_tokens):
            return [] # (No matching tokens)
        
        matching_analog_ids = self.base_set.nodes[matching_tokens, TF.ANALOG]   # Get the analog IDs of the matching tokens
        unique_analog_ids = torch.unique(matching_analog_ids).tolist()          # Convert to unique list of analog IDs

        analogs = []
        for analog_id in unique_analog_ids:
            analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
        return analogs
   
    def get_analogs_where_not(self, feature: TF, value) -> list[Ref_Analog]:
        """
        Get any analogs that do not contain a token with a given feature and value.

        Args:
            feature (TF): The feature to check for.
            value (float): The value to check for.

        Returns:
            List[Ref_Analog]: References to the analogs that do not contain a token with the given feature and value.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()               # Only non-deleted tokens
        non_matching_tokens = (self.base_set.nodes[all_nodes_mask, feature] != value)
        non_matching_analog_ids = self.base_set.nodes[non_matching_tokens, TF.ANALOG] 
        unique_analog_ids = torch.unique(non_matching_analog_ids).tolist()          # Convert to unique list of analog IDs

        analogs = []
        for analog_id in unique_analog_ids:
            analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
        return analogs
    
    def get_analogs_active(self, ids=False) -> list[Ref_Analog]:
        """
        Get all analogs that have at least one active token.
        """
        all_nodes_mask = self.base_set.tensor_op.get_all_nodes_mask()
        active_tokens = (self.base_set.nodes[all_nodes_mask, TF.ACT] > 0.0)
        active_analog_ids = self.base_set.nodes[active_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(active_analog_ids).tolist()

        if ids:
            return unique_analog_ids
        else: # Convert to ref objects
            analogs = []
            for analog_id in unique_analog_ids:
                analogs.append(Ref_Analog(analog_id, self.base_set.token_set))
            return analogs

    def print_analog(self, analog: Ref_Analog):
        """
        print the names of the tokens in an analog
        """
        indices = self.get_analog_indices(analog)
        for index in indices:
            print(self.base_set.names[index])

    # ----------------------------[KLUDGEY COMPARATOR FUNCTIONS]-------------------------------
    def get_pred_rb_no_ps(self, pairs: Pairs) -> Pairs:
        """
        Get all pairs of preds that are connected to RBs that are not connected to any P
        - only neeed one non_p rb to be counted as valid for this.
        """
        # get masks
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        p = self.base_set.tensor_op.get_mask(Type.P)
        pred = self.base_set.tensor_op.get_mask(Type.PO) & (self.base_set.nodes[:, TF.PRED] == B.TRUE)
        # get indices
        rb_indices = torch.where(rb)[0]
        p_indices = torch.where(p)[0]
        pred_indices = torch.where(pred)[0]
        
        # connections from rb -> p
        connections_to_p = self.base_set.connections[rb_indices, :][:, p_indices] # TODO: check if correct direction
        rb_no_p_connections = torch.sum(connections_to_p, dim=1) == 0
        # convert to full-size mask
        rb_no_p_mask = torch.zeros_like(rb, dtype=torch.bool)
        rb_no_p_mask[rb_indices[rb_no_p_connections]] = True

        # connections from pred -> rb_no_p
        rb_no_p_indices = torch.where(rb_no_p_mask)[0]
        connections_to_rb_no_p = self.base_set.connections[pred_indices, :][:, rb_no_p_indices]
        preds_connected_to_rb_no_p = torch.sum(connections_to_rb_no_p, dim=1) > 0
        # convert to full-size mask
        pred_rb_no_p_mask = torch.zeros_like(pred, dtype=torch.bool)
        pred_rb_no_p_mask[pred_indices[preds_connected_to_rb_no_p]] = True
        
        # get pairs
        row = pred_rb_no_p_mask.unsqueeze(1)
        col = pred_rb_no_p_mask.unsqueeze(0)
        pred_rb_no_p_pairs = torch.bitwise_and(row, col)
        # remove duplicates (below the diagonal + the diagonal)
        pred_rb_no_p_pairs = torch.triu(pred_rb_no_p_pairs, diagonal=1)
        
        # create list of pair indices in pairs obj
        pair_indices = torch.where(pred_rb_no_p_pairs)
        for i, j in zip(pair_indices[0], pair_indices[1]):
            pairs.add(i.item(), j.item())
        return pairs
        
    
    def get_pred_rb_shared_p(self, pairs: Pairs) -> Pairs:
        """
        Get all pairs of preds that are connected to the same P.
        (i.e preds connected to RBs that are connected to the same P)
        """
        # get masks for different token types
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        p = self.base_set.tensor_op.get_mask(Type.P)
        pred = self.base_set.tensor_op.get_mask(Type.PO) & (self.base_set.nodes[:, TF.PRED] == B.TRUE)
        # convert to indices
        rb_indices = torch.where(rb)[0]
        p_indices = torch.where(p)[0]
        pred_indices = torch.where(pred)[0]
        
        # get rb s.t rb -> p
        rb_to_p_connections = self.base_set.connections[rb_indices, :][:, p_indices]
        rb_with_p_mask_local = torch.sum(rb_to_p_connections, dim=1) > 0
        rb_with_p_indices = rb_indices[rb_with_p_mask_local]
        if len(rb_with_p_indices) == 0:
            return pairs
        # get specific connections from rb_with_p -> p
        rb_with_p_to_p_connections = self.base_set.connections[rb_with_p_indices, :][:, p_indices]

        # matrix of rb_with_p, with true if they share a p, false o.w
        num_rbs_with_p = len(rb_with_p_indices)
        rb_shared_p = torch.zeros((num_rbs_with_p, num_rbs_with_p), dtype=torch.bool)
        for i in range(num_rbs_with_p):
            for j in range(i + 1, num_rbs_with_p):
                shared_p = torch.bitwise_and(rb_with_p_to_p_connections[i].bool(), rb_with_p_to_p_connections[j].bool()).sum() > 0
                rb_shared_p[i, j] = shared_p

        # get specific connections from pred -> rb_with_p
        pred_to_rb_connections = self.base_set.connections[pred_indices, :][:, rb_with_p_indices]
        num_preds = len(pred_indices)
        if num_preds == 0:
            return pairs

        # create matrix of preds, with true if they share a p, false o.w 
        # NOTE: not even vaguely efficient, but should be a low number of tokens involved afaik.
        #       should probs try improve this, but don't seem worth it for now.
        pred_shared_p = torch.zeros((num_preds, num_preds), dtype=torch.bool)
        for i in range(num_preds):
            for j in range(i + 1, num_preds):
                pred_i_rbs = pred_to_rb_connections[i]
                pred_j_rbs = pred_to_rb_connections[j]
                
                shared = False
                for rb_i_idx in range(num_rbs_with_p):
                    if pred_i_rbs[rb_i_idx]:
                        for rb_j_idx in range(num_rbs_with_p):
                            if pred_j_rbs[rb_j_idx] and rb_shared_p[rb_i_idx, rb_j_idx]:
                                shared = True
                                break
                        if shared:
                            break
                
                pred_shared_p[i, j] = shared
        
        # create list of pair indices in pairs obj
        pair_indices_local = torch.where(pred_shared_p)
        for i, j in zip(pair_indices_local[0], pair_indices_local[1]):
            global_i = pred_indices[i].item()
            global_j = pred_indices[j].item()
            pairs.add(global_i, global_j)
            
        return pairs
        