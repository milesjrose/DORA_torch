# nodes/network/sets/base_set_operations/tensor_operations.py
# Tensor operations for Base_Set class

import torch

from ....enums import *
from ...single_nodes import Token, Analog, Ref_Token

class TensorOperations:
    """
    Tensor operations for the Base_Set class.
    Handles tensor expansion and contraction.
    """
    
    def __init__(self, base_set):
        """
        Initialize TensorOperations with reference to Base_Set.
        
        Args:
            base_set: Reference to the Base_Set object
        """
        self.base_set = base_set

# ====================[ TENSOR FUNCTIONS ]======================

    def cache_masks(self, types_to_recompute: list[Type] = None):   # Compute and cach masks for given types
        """
        Compute and cache masks
        
        Args:
            types_to_recompute (list[Type], optional): The types to recompute the mask for. Defaults to All types.
        """
        if types_to_recompute is None:                              #  If no type specified, recompute all
            types_to_recompute = [Type.PO, Type.RB, Type.P, Type.GROUP]

        masks = []
        for token_type in [Type.PO, Type.RB, Type.P, Type.GROUP]:
            if token_type in types_to_recompute:
                masks.append(self.compute_mask(token_type))         # Recompute mask
            else:
                masks.append(self.base_set.masks[token_type])       # Use cached mask from base_set

        # Update the base_set masks
        self.base_set.masks = torch.stack(masks, dim=0)
    
    def compute_mask(self, token_type: Type):                       # Compute the mask for a token type
        """
        Compute the mask for a token type
        
        Args:
            token_type (Type): The type to get the mask for.

        Returns:
            A mask of nodes with given type.   
        """
        mask = (self.base_set.nodes[:, TF.TYPE] == token_type) & (self.base_set.nodes[:, TF.DELETED] == B.FALSE)
        return mask
    
    def get_mask(self, token_type: Type):                           # Returns mask for given token type
        """
        Return cached mask for given token type
        
        Args:
            token_type (Type): The type to get the mask for.

        Returns:
            The cached mask for the given token type.
        """
        return self.base_set.masks[token_type]                   

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
        if not isinstance(n_types, list):
            raise TypeError("n_types must be a list of types")
    
        masks = [self.base_set.masks[i] for i in n_types]
        return torch.logical_or.reduce(masks)

    def get_all_nodes_mask(self):                                   # Returns a mask for all nodes (Exluding empty or deleted rows)
        """Return mask for all non-deleted nodes"""
        return (self.base_set.nodes[:, TF.DELETED] == B.FALSE)

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
        spaces = torch.sum(self.base_set.nodes[:, TF.DELETED] == B.TRUE)    # Find number of spaces -> count of deleted nodes in the tensor
        if spaces == 0:                                                     # If no spaces, expand tensor
            self.expand_tensor()
        if len(self.base_set.IDs) > 0:
            try:
                ID = max(self.base_set.IDs.keys()) + 1                      # assign token a new ID.
            except Exception as e:
                raise(e)
        else:
            ID = 1
        token[TF.ID] = ID
        deleted_nodes = torch.where(self.base_set.nodes[:, TF.DELETED] == B.TRUE)[0] #find first deleted node
        first_deleted = deleted_nodes[0]                                    # Get index of first deleted node
        self.base_set.nodes[first_deleted, :] = token.tensor                # Replace first deleted node with token
        self.base_set.IDs[ID] = first_deleted                               # map: new ID -> index of replaced node
        self.cache_masks()                                                  # recompute masks
        if token.name is None:
            token.name = f"Token {ID}"
        if self.base_set.names is None:
            raise ValueError("Names dictionary is not initialised.")
        self.base_set.names[ID] = token.name
        return Ref_Token(self.base_set.token_set, ID)
    
    def expand_tensor(self):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes expansion factor. Minimum expansion is 5.
        Expands nodes, connections, links and mappings tensors.
        """
        count = self.base_set.expansion_factor * self.base_set.nodes.size(dim=0)
        if count < 5:
            count = 5
        self.expand_tensor_by_count(count)
    
    def expand_tensor_by_count(self, count: int):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes by count.
        Expands nodes, connections, links and mappings tensors.
        """
                                                                        # Update node tensor:
        current_count = self.base_set.nodes.size(dim=0)
        new_count = int(current_count + count)                              # calculate new size
        new_tensor = torch.full((new_count, self.base_set.nodes.size(dim=1)), null)  # null-filled tensor, increased by expansion factor
        new_tensor[current_count:, TF.DELETED] = B.TRUE                     # deleted = true -> all new tokens
        new_tensor[:current_count, :] = self.base_set.nodes                 # copy over old tensor
        if self.base_set.nodes.dtype != torch.float:                       # make sure correct type
            raise TypeError("nodes must be torch.float.")
        self.base_set.nodes = new_tensor

        # Update supporting data structures:
        if self.base_set.links is not None:                              # Links:
            semantic_count = self.base_set.links[self.base_set.token_set].size(dim=1)  # Get number of semantics in link tensor
            new_links = torch.zeros(new_count, semantic_count).float()      # new tensor (new number of tokens) x (number of semantics)
            new_links[:current_count, :] = self.base_set.links[self.base_set.token_set]  # add current links to the tensor
            self.base_set.links[self.base_set.token_set] = new_links        # update links object with float tensor

        try:                                                            # Mappings:         
            self.base_set.mappings = self.base_set.mappings       # If no mappings, skip.
        except:
            pass
        else:                                    
            driver_count = self.base_set.mappings.size(dim=1)               # Get number of tokens in driver
            stack = []
            for field in MappingFields:                                     # Create new tensor for each mapping field
                stack.append(torch.zeros(
                    new_count, 
                    driver_count, 
                    dtype=torch.float)
                    )
            new_adj_matrix: torch.Tensor = torch.stack(stack, dim=-1)       # Stack into adj_matrix tensor
            new_adj_matrix[:current_count, :] = self.base_set.mappings.adj_matrix  # add current weights
            self.base_set.mappings.adj_matrix = new_adj_matrix              # update mappings object with new tensor
        
                                                                        # Connections:
        new_cons = torch.zeros(new_count, new_count, dtype=torch.float)     # new tensor (new num tokens) x (new num tokens)
        new_cons[:current_count, :current_count] = self.base_set.connections  # add current connections
        self.base_set.connections = new_cons                                # update connections tensor, with new tensor

    def del_token(self, ref_tokens: Ref_Token):                     # Delete nodes from tensor   
        """
        Delete tokens from tensor. Pass in a list of Ref_Tokens to delete multiple tokens at once.
        
        Args:
            ref_tokens (Ref_Token): The token(s) to delete. 
        """

        if not isinstance(ref_tokens, list):                            # If input is single ID, turn into iteratable object.
            ref_tokens = [ref_tokens]
        
        self.del_connections(ref_tokens)                                # Delete connections first, as requires ID in self.IDs

        cache_types = [] 
        for ref_token in ref_tokens:
            index = self.base_set.token_op.get_index(ref_token)
            id = ref_token.ID                                           # Delete nodes in nodes tensor:
            cache_types.append(self.base_set.nodes[index, TF.TYPE])     # Keep list of types that have a deleted node to recache specific masks
            self.base_set.nodes[index, TF.ID] = null                   # Set to null ID
            self.base_set.nodes[index, TF.DELETED] = B.TRUE            # Set to deleted
            self.base_set.IDs[id] = null
            self.base_set.IDs.pop(id)
            self.base_set.names[id] = null
            self.base_set.names.pop(id)

        
        cache_types = list(set(cache_types))                            # Remove duplicates if multiple nodes deleted from same type
        self.cache_masks(cache_types)                                   # Re-cache effected types

    def del_connections(self, ref_token: Ref_Token):
        """
        Delete connection from tensor.
        
        Args:
            ref_token (Ref_Token): Refence to token, to delete connections from.
        """
        if isinstance(ref_token, list):
            for token in ref_token:
                self.del_connections(token)
            return
        id = ref_token.ID
        if not isinstance(id, int):
            raise ValueError("ID is not an integer.")
        try:                                                        # Mappings:
            if self.base_set.token_set == Set.DRIVER:
                for field in MappingFields:
                    self.base_set.mappings[field][:, self.base_set.IDs[id]] = 0.0  # Dim 1 is driver nodes
            else:
                for field in MappingFields:
                    self.base_set.mappings[field][self.base_set.IDs[id], :] = 0.0  # Dim 0 is non-driver sets
        except:
            pass

        try:                                                        # Links:
            self.base_set.links[self.base_set.token_set][self.base_set.IDs[int(id)], :] = 0.0  # Dim 0 is set nodes
        except:
            raise KeyError("Key error in del_token, link tensor. ID: ", id, "IDs: ", self.base_set.IDs)
        
        try:                                                        # Connections:
            self.base_set.connections[self.base_set.IDs[id], :] = 0.0   # Remove children
            self.base_set.connections[:, self.base_set.IDs[id]] = 0.0   # Remove parents
        except:
            if len(self.base_set.IDs) == 0:
                raise KeyError("IDs dictionary is empty.")
            else:
                raise KeyError("Key error in del_token, connection tensor. ID: ", id, "IDs: ", self.base_set.IDs)
    
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
        # Get indices of tokens in the analog
        analog_indices = self.base_set.token_op.get_analog_indices(analog)
        
        if len(analog_indices) == 0:
            raise ValueError(f"Analog {analog} not found in the set.")
        
        # Extract tokens for this analog
        analog_tokens = self.base_set.nodes[analog_indices, :].clone()
        
        # Extract connections for this analog (submatrix)
        analog_connections = self.base_set.connections[analog_indices][:, analog_indices].clone()
        
        # Extract links for this analog, if links exist
        analog_links = None
        if self.base_set.links is not None:
            analog_links = self.base_set.links[self.base_set.token_set][analog_indices, :].clone()
        
        # Create name dictionary for tokens in this analog
        analog_name_dict = {}
        for idx in analog_indices:
            token_id = int(self.base_set.nodes[idx, TF.ID].item())
            if token_id in self.base_set.names:
                analog_name_dict[token_id] = self.base_set.names[token_id]
        
        # Create and return the Analog object
        from ...single_nodes import Analog
        return Analog(analog_tokens, analog_connections, analog_links, analog_name_dict)
            
    def add_analog(self, analog: Analog):
        """
        Add an analog to the set.

        Args:
            analog (Analog): The analog to add.
        """
        # 1. copy the analog to avoid modifying the original
        analog = analog.copy()
        
        # 1.1 get list of ids in the analog
        analog_ids = analog.tokens[:, TF.ID].tolist()
        # 1.2 find available slots in the tensor (deleted nodes)
        deleted_indices = torch.where(self.base_set.nodes[:, TF.DELETED] == B.TRUE)[0]
        available_slots = len(deleted_indices)
        needed_slots = analog.tokens.shape[0]
        
        # Only expand if we need more slots than available
        if needed_slots > available_slots:
            expansion_needed = needed_slots - available_slots
            self.expand_tensor_by_count(expansion_needed)
            # Recalculate deleted indices after expansion
            deleted_indices = torch.where(self.base_set.nodes[:, TF.DELETED] == B.TRUE)[0]

        # 2. re-id the analog, starting from the highest id in the set
        if len(self.base_set.IDs) == 0:
            highest_id = 0
        else:
            highest_id = max(self.base_set.IDs.keys())
        # 2.1 create 1xN tensor of ids, starting from highest_id + 1
        new_ids = torch.arange(highest_id + 1, highest_id + 1 + analog.tokens.shape[0])
        # 2.2 update the ids in the analog
        analog.tokens[:, TF.ID] = new_ids
        # 2.3 add names to set
        for i, old_id in enumerate(analog_ids):
            self.base_set.names[new_ids[i].item()] = analog.name_dict[old_id]
        # 2.4 update analog number
        if self.base_set.analogs is not None and len(self.base_set.analogs) > 0:
            new_analog_number = max(self.base_set.analogs.tolist()) + 1
        else:
            new_analog_number = 1
        analog.tokens[:, TF.ANALOG] = new_analog_number
        
        # 3. add tokens to set, using the first available deleted slots
        slot_indices = deleted_indices[:analog.tokens.shape[0]]
        self.base_set.nodes[slot_indices, :] = analog.tokens
        
        # 4. update ID mappings
        for i, new_id in enumerate(new_ids):
            self.base_set.IDs[new_id.item()] = slot_indices[i].item()
        
        # 5. add connections to set
        # Create a mapping from analog token positions to set positions
        analog_to_set_mapping = {i: slot_indices[i].item() for i in range(analog.tokens.shape[0])}
        
        # Update connections for the new tokens
        for i in range(analog.tokens.shape[0]):
            for j in range(analog.tokens.shape[0]):
                set_i = analog_to_set_mapping[i]
                set_j = analog_to_set_mapping[j]
                self.base_set.connections[set_i, set_j] = analog.connections[i, j]
        
        # 6. add links to set, if links exist
        if self.base_set.links is not None and analog.links is not None:
            # Check if dimensions match
            if analog.links.shape[1] != self.base_set.links[self.base_set.token_set].shape[1]:
                raise ValueError(f"Analog links dimension {analog.links.shape[1]} doesn't match set links dimension {self.base_set.links[self.base_set.token_set].shape[1]}")
            
            for i in range(analog.tokens.shape[0]):
                set_i = analog_to_set_mapping[i]
                self.base_set.links[self.base_set.token_set][set_i, :] = analog.links[i, :]
        
        # 7. update masks
        self.cache_masks()

        return new_analog_number

    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        """Update list of analogs in tensor, and their node counts"""
        self.base_set.analogs, self.base_set.analog_counts = torch.unique(self.base_set.nodes[:, TF.ANALOG], return_counts=True)
   
    def print(self, f_types=None):                                  # Here for testing atm
        """
        Print the set.

        Args:
            f_types (list[TF], optional): The features to print.

        Raises:
            ValueError: If nodePrinter is not found.
        """
        try:
            from nodes.utils import nodePrinter
        except:
            print("Error: nodePrinter not found. Nodes.utils.nodePrinter is required to use this function.")
        else:
            try:
                printer = nodePrinter(print_to_console=True)
                printer.print_set(self, feature_types=f_types)
            except Exception as e:
                print("Error: NodePrinter failed to print set.")
                print(e)
    
    def get_count(self):
        """Get the number of nodes in the set."""
        return self.base_set.nodes.shape[0]
    # --------------------------------------------------------------