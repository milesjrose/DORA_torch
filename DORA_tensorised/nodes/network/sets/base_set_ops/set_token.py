# nodes/network/sets/base_set_operations/token_operations.py
# Token operations for Base_Set class

import torch
import logging
from typing import TYPE_CHECKING

from ...single_nodes import Token, Ref_Token, Ref_Analog
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
            ValueError: If the referenced token or feature is invalid.
        """
        try:
            feature_value = self.base_set.nodes[self.get_index(ref_token), feature].item()
        except Exception as e:
            logger.critical(f"Invalid reference token or feature: {ref_token.set.name}[{ref_token.ID}] {feature.name}")
            raise ValueError("Invalid reference token or feature.")
        # Convert to correct type:
        f_type = feature_type(feature)
        return f_type(feature_value)

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
        try:
            self.base_set.nodes[self.get_index(ref_token), feature] = float(value)
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
        tensor = self.base_set.nodes[self.get_index(ref_token), :]
        token = Token(self.base_set.token_set, {TF.PRED: tensor[TF.PRED]})
        if copy:
            token.tensor = tensor.clone()
        else:
            token.tensor = tensor
        return token
    
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
    
    def connect(self, parent: Ref_Token, child: Ref_Token):
        """
        Connect a parent token to a child token.
        """
        logger.debug(f"Connecting {self.get_ref_string(parent)} -> {self.get_ref_string(child)}")
        self.base_set.connections[self.get_index(parent), self.get_index(child)] = B.TRUE
    
    def get_connected_tokens(self, ref_token: Ref_Token) -> list[Ref_Token]:
        """
        Get all tokens that are connected to a given token. TODO: Need to add test for this.
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
    def get_pred_rb_no_ps(self, pair_dict: dict[int, list[Ref_Token]]) -> dict[int, list[Ref_Token]]:
        """
        Get all pairs of preds that are connected to RBs that are not connected to any P
        - only neeed one non_p rb to be counted as valid for this.
        TODO: Test
        """
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        p = self.base_set.tensor_op.get_mask(Type.P)
        rb_no_p = rb & (self.base_set.connections[rb][:, p] == 0)
        pred = self.base_set.tensor_op.get_mask(Type.PO) & (self.base_set.nodes[:, TF.PRED] == B.TRUE)
        # Mat mul rb_no_p with connections to get preds that are connected to rbs_with_no_ps
        pred_rb_no_p = torch.matmul(
            rb_no_p,
            self.base_set.connections[rb][:, pred]
        )
        # now take g.t to get a mask 
        pred_rb_no_p = torch.gt(pred_rb_no_p, 0).bool()
        # now need a predxpred tensor to get pairs. so we want 2d tensor of logical an of the po tensor in two dimensions.
        row = pred_rb_no_p.unsqueeze(1)
        col = pred_rb_no_p.unsqueeze(0)
        pred_rb_no_p = torch.bitwise_and(row, col)
        # remove duplicates (i.e remove anything below the diagonal + the diagonal)
        pred_rb_no_p = torch.triu(pred_rb_no_p, diagonal=1)
        # now get the indices of the pairs
        indices = torch.where(pred_rb_no_p == 1).tolist()
        # now get the references to the pairs
        for i,j in indices:
            ref_i = self.get_reference(index=i)
            ref_j = self.get_reference(index=j)
            pair_dict[self.pair_hash(i,j)] = [ref_i, ref_j]
        return pair_dict
        
    
    def get_pred_rb_shared_p(self, pair_dict: dict[int, list[Ref_Token]]) -> dict[int, list[Ref_Token]]:
        """
        Get all pairs of preds that are connected to the same P.
        (i.e preds connected to RBs that are connected to the same P)
        TODO: test
        """
        # Get masks for different token types
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        p = self.base_set.tensor_op.get_mask(Type.P)
        pred = self.base_set.tensor_op.get_mask(Type.PO) & (self.base_set.nodes[:, TF.PRED] == B.TRUE)
        
        # Get RBs that are connected to P units
        rb_with_p = rb & (self.base_set.connections[rb][:, p].sum(dim=1) > 0)
        
        # Get which P each RB is connected to
        rb_to_p_connections = self.base_set.connections[rb_with_p][:, p]
        
        # Create a matrix to track which RBs share the same P
        # rb_shared_p[i,j] = 1 if RB i and RB j are connected to the same P
        num_rbs_with_p = rb_with_p.sum().item()
        if num_rbs_with_p == 0:
            return []
            
        rb_shared_p = torch.zeros((num_rbs_with_p, num_rbs_with_p), dtype=torch.bool)
        
        # For each pair of RBs, check if they share any P
        for i in range(num_rbs_with_p):
            for j in range(i + 1, num_rbs_with_p):
                # Check if RB i and RB j share any P unit
                shared_p = torch.bitwise_and(rb_to_p_connections[i], rb_to_p_connections[j]).sum() > 0
                rb_shared_p[i, j] = shared_p
        
        # Get which preds are connected to which RBs
        pred_to_rb_connections = self.base_set.connections[pred][:, rb_with_p]
        
        # Create pred x pred matrix for pairs that share RBs with shared Ps
        num_preds = pred.sum().item()
        if num_preds == 0:
            return []
            
        pred_shared_p = torch.zeros((num_preds, num_preds), dtype=torch.bool)
        
        # For each pair of preds, check if they are connected to RBs that share a P
        for i in range(num_preds):
            for j in range(i + 1, num_preds):
                # Get RBs connected to pred i and pred j
                pred_i_rbs = pred_to_rb_connections[i]
                pred_j_rbs = pred_to_rb_connections[j]
                
                # Check if any RB connected to pred i shares a P with any RB connected to pred j
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
        
        # Get the indices of the pairs
        indices = torch.where(pred_shared_p == 1)
        
        # Convert to references
        pred_indices = torch.where(pred)[0]
        pairs = []
        for i, j in zip(indices[0], indices[1]):
            pred_i_ref = self.get_reference(index=pred_indices[i].item())
            pred_j_ref = self.get_reference(index=pred_indices[j].item())
            pair_dict[self.pair_hash(i,j)] = [pred_i_ref, pred_j_ref]
        return pairs
    
    def pair_hash(self, po1, po2):
        """
        Get a hash that is the same regardless of orientation of pos
        """
        if po1>po2:
            return int(str(po1)+str(po2))
        else:
            return int(str(po2)+str(po1))
        