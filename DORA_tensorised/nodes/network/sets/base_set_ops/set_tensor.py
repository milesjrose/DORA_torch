# nodes/network/sets/base_set_operations/tensor_operations.py
# Tensor operations for Base_Set class

import torch
import logging

from typing import TYPE_CHECKING
from functools import reduce

from ....enums import *
from ...single_nodes import Token, Analog, Ref_Analog, Ref_Token
from ....utils import tensor_ops as tOps
from ...connections import LD, MD

# name -> base_set.tensor_ops
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..base_set import Base_Set

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
        self.base_set: 'Base_Set' = base_set

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
    
    def get_mask(self, token_type: Type, pmode=None):                           # Returns mask for given token type
        """
        Return cached mask for given token type
        
        Args:
            token_type (Type): The type to get the mask for.
            pmode (Mode, optional): The mode to refine the mask for. Defaults to None.

        Returns:
            The cached mask for the given token type.
        """
        # If p type, refine mask for type if included
        if token_type == Type.P:
            if pmode is not None:
                p_mask = self.base_set.masks[int(token_type)]
                if pmode == Mode.CHILD:
                    child_mask = self.base_set.nodes[p_mask, TF.MODE] == Mode.CHILD
                    return tOps.sub_union(p_mask, child_mask)
                elif pmode == Mode.PARENT:
                    parent_mask = self.base_set.nodes[p_mask, TF.MODE] == Mode.PARENT
                    return tOps.sub_union(p_mask, parent_mask)
                elif pmode == Mode.NEUTRAL:
                    neutral_mask = self.base_set.nodes[p_mask, TF.MODE] == Mode.NEUTRAL
                    return tOps.sub_union(p_mask, neutral_mask)
        return self.base_set.masks[int(token_type)]                   

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
        return reduce(torch.logical_or, masks)
    
    def get_count(self, type: Type=None):
        """ Get number of nodes of given type in set.
        If type is None, return number of all nodes in set.

        Args:
            type (Type, optional): The type to get the count for. Defaults to None.

        Returns:
            The number of nodes of the given type in the set.
        """
        if type is None:
            return self.get_all_nodes_mask().sum()
        else:
            return self.get_mask(type).sum()

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
        logger.debug(f"Adding token to {self.base_set.token_set.name}")
        spaces = torch.sum(self.base_set.nodes[:, TF.DELETED] == B.TRUE)    # Find number of spaces -> count of deleted nodes in the tensor
        if spaces == 0:                                                     # If no spaces, expand tensor
            self.expand_tensor()

        if len(self.base_set.IDs) > 0:
            try:
                ID = max(self.base_set.IDs.keys()) + 1                      # assign token a new ID.
            except Exception as e:
                raise(e)
        else:
            ID = 0
        logger.debug(f"Assigned ID: {ID}")
        token[TF.ID] = float(ID)
        deleted_nodes = torch.where(self.base_set.nodes[:, TF.DELETED] == B.TRUE)[0] #find first deleted node
        idx = deleted_nodes[0]                                    # Get index of first deleted node
        self.base_set.nodes[idx, :] = token.tensor                # Replace first deleted node with token
        self.base_set.IDs[ID] = int(idx.item())                   # map: new ID -> index of replaced node
        self.cache_masks()                                                  # recompute masks
        if token.name is None:
            token.name = f"Token_{ID}"
        if self.base_set.names is None:
            raise ValueError("Names dictionary is not initialised.")
        self.base_set.names[ID] = token.name
        ref_token = Ref_Token(self.base_set.token_set, ID)
        ref_token.name = token.name
        logger.info(f"add_token -> {ref_token.set.name}[{idx}]({ID}):\"{ref_token.name}\"")
        logger.debug(f"-> token:\n{Token(tensor=self.base_set.nodes[idx, :]).get_string()}")
        return ref_token
    
    def expand_tensor(self):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes expansion factor. Minimum expansion is 5.
        Expands nodes, connections, links and mappings tensors.
        """
        count = self.base_set.expansion_factor * self.base_set.nodes.size(dim=0)
        if count < 5:
            count = 5
        self.expand_tensor_by_count(count)
    
    def expand_tensor_by_count(self, count: int):                   # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes by count.
        Expands nodes, connections, links and mappings tensors.
        """
                                                                        # Update node tensor:
        current_count = self.base_set.nodes.size(dim=0)
        new_count = int(current_count + count)                              # calculate new size
        logger.debug(f"Expanding {self.base_set.token_set.name} by {count} (x{self.base_set.expansion_factor}): {current_count} -> {new_count}")
        
        # Nodes expansion
        self.expand_nodes_tensor(new_count)
        # Links expansion
        self.base_set.links.expand_links_tensor(new_count, self.base_set.token_set, LD.TK)
        # Mapping expansion:
        if self.base_set.token_set == Set.DRIVER:
            for set in MAPPING_SETS:
                self.base_set.mappings[set].expand_mapping_tensor(new_count, MD.DRI)
        elif self.base_set.token_set in MAPPING_SETS:
            self.base_set.mappings.expand_mapping_tensor(new_count, MD.REC)
        else:
            logger.debug("Mappings not initialised. Not expanding mappings tensor.")

        # Connections expansion
        self.expand_connections_tensor(new_count)
    
    def expand_nodes_tensor(self, new_count: int):
        """
        Expand nodes tensor for given set.
        """
        new_tensor = torch.full((new_count, self.base_set.nodes.size(dim=1)), null)  # null-filled tensor, increased by expansion factor
        new_tensor[:, TF.DELETED] = B.TRUE                     # deleted = true -> all new tokens
        rows, cols = self.base_set.nodes.shape
        new_tensor[:rows, :cols] = self.base_set.nodes         # copy over old tensor
        if self.base_set.nodes.dtype != torch.float:           # make sure correct type
            raise TypeError("nodes must be torch.float.")
        self.base_set.nodes = new_tensor
        logger.debug(f" -> Expanded {self.base_set.token_set.name} nodes tensor: {self.base_set.nodes.shape}")
    
    def expand_connections_tensor(self, new_count: int):
        """
        Expand connections tensor for given set.
        """
        new_cons = torch.zeros(new_count, new_count, dtype=torch.float) # new tensor (new num tokens) x (new num tokens)
        rows, cols = self.base_set.connections.shape
        new_cons[:rows, :cols] = self.base_set.connections              # add current connections
        self.base_set.connections = new_cons                            # update connections tensor, with new tensor


    def del_token_indicies(self, indices: list[int]):
        """
        Delete tokens from tensor.

        Args:
            indices (list[int]): The indices of the tokens to delete.
        """
        self.del_connections_indices(indices) # Delete connections first
        # Get types/ids of deleted tokens
        cache_types = torch.unique(self.base_set.nodes[indices, TF.TYPE]).tolist() 
        IDs = self.base_set.nodes[indices, TF.ID].tolist()
        # Update IDs/names
        for ID in IDs:
            self.base_set.IDs.pop(int(ID))
            self.base_set.names.pop(int(ID))
        # Update features
        self.base_set.nodes[indices, TF.DELETED] = B.TRUE          
        self.base_set.nodes[indices, TF.ID] = null
        self.cache_masks(cache_types)         # Re-cache effected types
    
    def del_connections_indices(self, indices: list[int]):
        """
        Delete connections from tensor.
        
        Args:
            indices (list[int]): The indices of the tokens to delete connections from.
        """
        try:
            if self.base_set.token_set == Set.DRIVER:
                for field in MappingFields:
                    self.base_set.mappings[field][:, indices] = 0.0     # Dim 1 is driver nodes
            else:
                for field in MappingFields:
                    self.base_set.mappings[field][indices, :] = 0.0     # Dim 0 is non-driver sets
        except:
            pass

        try:                                                        # Links:
            self.base_set.links[self.base_set.token_set][indices, :] = 0.0  # Dim 0 is set nodes
        except:
            raise KeyError("Invalid links indices in del_connections_by_indices.", indices)

        try:                                                        # Connections:
            self.base_set.connections[indices, :] = 0.0                 # Remove children
            self.base_set.connections[:, indices] = 0.0                 # Remove parents
        except:
            raise KeyError("Invalid connections indices in del_connections_by_indices.", indices)

    def del_token(self, ref_tokens: Ref_Token):                 # Delete nodes from tensor by reference token   
        """
        Delete tokens from tensor. Pass in a list of Ref_Tokens to delete multiple tokens at once.
        
        Args:
            ref_tokens (Ref_Token): The token(s) to delete. 
        """

        if not isinstance(ref_tokens, list):
            indices = [self.base_set.token_op.get_index(ref_tokens)]
        else:
            indices = self.base_set.token_op.get_indices(ref_tokens)
        
        self.del_token_indicies(indices)

    def del_connections(self, ref_token: Ref_Token):            # Delete connections from tensor by reference token
        """
        Delete connections from tensor.
        
        Args:
            ref_token (Ref_Token): Refence to token, to delete connections from, pass in list to delete multiple.
        """
        if not isinstance(ref_token, list):
            indices = [self.base_set.token_op.get_index(ref_token)]
        else:
            indices = self.base_set.token_op.get_indices(ref_token)
            
        self.del_connections_indices(indices)
    
    def del_mem_tokens(self):                                       # Delete all memory tokens
        """
        Delete any tokens that have set == memory.
        """
        all_nodes_mask = self.get_all_nodes_mask()                  # Get mask of all memory tokens (not deleted)
        mem_mask = (self.base_set.nodes[all_nodes_mask, TF.SET] == Set.MEMORY)
        
        mem_indices = torch.where(mem_mask)[0]                      # Convert to indicies
        self.del_token_indicies(mem_indices)                        # Delete

    def del_analog(self, analog: Ref_Analog):                              # Delete an analog from the set
        """
        Delete an analog from the set.

        Args:
            analog (Ref_Analog): The analog to delete.
        """
        if not isinstance(analog, Ref_Analog):
            raise TypeError("analog must be a Ref_Analog object.")
        if analog.set != self.base_set.token_set:
            raise ValueError(f"Analog {analog.analog_number} is not in the set {self.base_set.token_set}.")
        
        analog_indices = self.base_set.token_op.get_analog_indices(analog)
        self.del_token_indicies(analog_indices)

    def get_analog(self, analog: Ref_Analog):                              # Get an analog from the set
        """
        Get an analog from the set.
        
        Args:
            analog (Ref_Analog): The analog to get.
        
        Returns:
            Analog: The analog object containing tokens, connections, links, and names.
            
        Raises:
            ValueError: If the analog doesn't exist in the set.
        """
        if not isinstance(analog, Ref_Analog):
            raise TypeError("analog must be a Ref_Analog object.")
        if analog.set != self.base_set.token_set:
            raise ValueError(f"Analog {analog.analog_number} is not in the set {self.base_set.token_set}.")
        indicies = self.base_set.token_op.get_analog_indices(analog)    # Get indices of tokens in the analog
        if len(indicies) == 0:                                          # If no tokens in analog, raise error
            raise ValueError(f"Analog {analog.analog_number} not found in the set.")
        
        tokens = self.base_set.nodes[indicies, :].clone()               # Extract tokens for this analog
        cons = self.base_set.connections[indicies][:, indicies].clone()
        
        links = None                                                    # Extract links for this analog, if links exist
        if self.base_set.links is not None:
            token_set = self.base_set.token_set
            links = self.base_set.links[token_set][indicies, :].clone()
        
        name_dict = {}                                                  # Create name dictionary for tokens in this analog
        for idx in indicies:
            token_id = int(self.base_set.nodes[idx, TF.ID].item())
            if token_id in self.base_set.names:
                name_dict[token_id] = self.base_set.names[token_id]

        return Analog(tokens, cons, links, name_dict)
            
    def add_analog(self, analog: Analog):                           # Add an analog to the set
        """
        Add an analog to the set.

        Args:
            analog (Analog): The analog to add.
        
        Returns:
            Ref_Analog: Reference to the analog that was added.
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
            try:
                self.base_set.names[new_ids[i].item()] = analog.name_dict[old_id]
            except KeyError:
                self.base_set.names[new_ids[i].item()] = "None"
                #raise KeyError(f"{old_id} in {analog.name_dict.keys()} // IDs: {analog_ids} // Names: {analog.name_dict}")
            
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

        return Ref_Analog(new_analog_number, self.base_set.token_set)

    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        """Update list of analogs in tensor, and their node counts"""
        self.base_set.analogs, self.base_set.analog_counts = torch.unique(self.base_set.nodes[:, TF.ANALOG], return_counts=True)

    def get_analog_activation_counts_scatter(self):
        """
        Get unique analog numbers, their counts, and total activation sums. 
        Sets .analogs, .analog_counts, .analog_activations
        """
        # Get unique analog numbers and their counts
        self.analog_node_count()
        analogs = self.base_set.analogs
        analog_counts = self.base_set.analog_counts

        # Create a mapping from analog number to index
        analog_to_idx = {analog.item(): idx for idx, analog in enumerate(analogs)}
        
        # Create index tensor for scatter
        analog_indices = torch.zeros(len(self.base_set.nodes), dtype=torch.long)
        for i, analog_num in enumerate(self.base_set.nodes[:, TF.ANALOG]):
            analog_indices[i] = analog_to_idx[analog_num.item()]
        
        # Use scatter_add_ to sum activations for each analog
        valid_mask = self.get_all_nodes_mask()
        analog_activations = torch.zeros(len(analogs))
        analog_activations.scatter_add_(
            0, 
            analog_indices[valid_mask], 
            self.base_set.nodes[valid_mask, TF.ACT]
        )

        # Update set
        self.base_set.analog_activations = analog_activations
        self.base_set.analog_counts = analog_counts
        self.base_set.analogs = analogs
    
    def get_analog_ref_list(self, mask)-> list[Ref_Analog]:
        """
        Get a list of the unique analogs for a given mask
        """
        unique_analog_numbers = torch.unique(self.base_set.nodes[mask, TF.ANALOG])
        ref_analogs = []
        for analog_num in unique_analog_numbers:
            if analog_num.item() != null:
                ref_analogs.append(Ref_Analog(analog_num.item(), self.base_set.token_set))
        return ref_analogs
    
    def get_new_analog_id(self):
        """
        Get a new unique analog id
        """
        # Get new analog id
        self.base_set.tensor_ops.analog_node_count() # update analog list
        max_id = torch.max(self.base_set.analogs)
        return max_id.item() + 1
        

   
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
                printer.print_set(self.base_set, feature_types=f_types)
            except Exception as e:
                print("Error: NodePrinter failed to print set.")
                print(e)
    
    def print_tokens(self, f_types=None):
        """
        Print the tokens in the set.

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
                printer.print_set(self.base_set, feature_types=f_types, print_cons=False)
            except Exception as e:
                print("Error: NodePrinter failed to print set.")
                print(e)
    
    # --------------------------------------------------------------