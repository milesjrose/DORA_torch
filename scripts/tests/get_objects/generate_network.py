#!/usr/bin/env python
# generate_network.py
# Functions for generating and manipulating networks

import torch
import numpy as np
import time
from DORA_tensorised.nodes.enums import *
from DORA_tensorised.nodes.network import Network
from DORA_tensorised.nodes.network.sets import Driver, Recipient, Memory, New_Set, Semantics, Base_Set
from DORA_tensorised.nodes.network.connections import Links, Mappings
from DORA_tensorised.nodes.network.network_params import Params
from DORA_tensorised.nodes.network.single_nodes import Token, Ref_Token

def expand_network_all_sets(network: Network, expansion_factor=2.0):
    for set in Set:
        network = expand_token_set(network, expansion_factor, set)
    return network

def new_token_data(original_size: int, new_size: int, target_set_obj: Base_Set):
    # Create new tensors with expanded size
    new_nodes = torch.zeros(new_size, target_set_obj.nodes.size(1))
    new_connections = torch.zeros(new_size, new_size)
    
    # Copy original data to new tensors
    new_nodes[:original_size] = target_set_obj.nodes
    new_connections[:original_size, :original_size] = target_set_obj.connections
    
    # Update IDs dictionary
    new_IDs = {}
    for old_id, old_idx in target_set_obj.IDs.items():
        new_IDs[old_id] = old_idx
    
    # Create new names dictionary
    new_names = {}
    for old_id, old_name in target_set_obj.names.items():
        new_names[old_id] = old_name
    
    return new_nodes, new_connections, new_IDs, new_names

def fill_IDs(target_set_obj: Base_Set, new_IDs: dict, new_names: dict, new_nodes: torch.Tensor):
    old_IDs = target_set_obj.IDs
    first_new_index = len(old_IDs)
    max_id = max(old_IDs.keys())

    # Fill with new IDs and names, set tensor ID values to new IDs
    for i in range(first_new_index, new_nodes.size(0)):
        id = max_id + i
        new_IDs[id] = i
        new_names[id] = f"token_{i}"
        new_nodes[i, TF.ID] = id
    
    return new_IDs, new_names, new_nodes

def fill_slice(old_tensor: torch.Tensor, new_tensor: torch.Tensor, start_slice_index: int, end_slice_index: int, dim: int):
    """
    Fill a slice of a tensor with a smaller tensor
    """
    if dim == 0:
        new_tensor[start_slice_index:end_slice_index, :] = old_tensor
    elif dim == 1:
        new_tensor[:, start_slice_index:end_slice_index] = old_tensor
    else:
        raise ValueError(f"Invalid dimension: {dim}")
    return new_tensor

def fill(old_tensor: torch.Tensor, new_tensor: torch.Tensor, dim: int):
    """
    Fill a tensor with a smaller tensor along the first dimension
    """
    original_size = old_tensor.size(dim)
    num_duplicates = new_tensor.size(dim) // original_size
    for i in range(num_duplicates - 1):
        slice_start = original_size + (i * original_size)
        slice_end = slice_start + original_size
        fill_slice(old_tensor, new_tensor, slice_start, slice_end, dim)
    return new_tensor

def fill_square(old_tensor: torch.Tensor, new_tensor: torch.Tensor):
    """
    Fill a square tensor with a smaller square tensor
    """
    if old_tensor.size(0) != old_tensor.size(1):
        raise ValueError("Old tensor must be square")
    if new_tensor.size(0) != new_tensor.size(1):
        raise ValueError("New tensor must be square")
    
    original_size = old_tensor.size(0)
    num_duplicates = new_tensor.size(0) // original_size
    for i in range(num_duplicates-1):
        slice_start = original_size + (i * original_size)
        slice_end = slice_start + original_size
        new_tensor[slice_start:slice_end, slice_start:slice_end] = old_tensor
    return new_tensor
        
def new_mapping_driver(original_network: Network, new_nodes: torch.Tensor, driver):
    """
    Create new mappings for the driver set

    Returns:
        dict: A dictionary of new mappings for each set:
            dict(Set -> map_dict)
            map_dict(MappingFields -> torch.Tensor)
    """
    new_mappings = {}
    for set in Set:
        try:
            mapping = original_network.mappings[set]
            if mapping is not None:
                new_tensors = {}
                for map_field in MappingFields:
                    # Expand the tensor in dim=1 to size of new driver
                    old_tensor = mapping.adj_matrix[:, :, map_field]
                    new_tensor = torch.zeros(old_tensor.size(0), new_nodes.size(0), dtype=tensor_type)
                    new_tensor = fill(old_tensor, new_tensor, 1)
                    new_tensors[map_field] = new_tensor
                    #print(f"DRIVER: {set.name}{map_field} tensor shape: {new_tensor.shape}, old tensor shape: {old_tensor.shape}")
                new_map_obj = Mappings(driver, new_tensors)
                original_network.mappings[set] = new_map_obj
        except:
            if set in [Set.MEMORY, Set.RECIPIENT]:
                raise ValueError(f"Mapping for {set} is not found")
        
    return new_mappings

def new_mapping_other(original_network: Network, new_nodes: torch.Tensor, set: Set, driver):
    """
    Create new mappings for a non-driver set

    Returns:
        dict(MappingField -> torch.Tensor): A dictionary of new mappings for given set
    """
    try:
        mapping = original_network.mappings[set]
        if mapping is not None:
            new_tensors = {}
            for map_field in MappingFields:
                # Expand the tensor in dim=0 to size of new nodes
                old_tensor = mapping.adj_matrix[:, :, map_field]
                new_tensor = torch.zeros(new_nodes.size(0), old_tensor.size(1), dtype=tensor_type)
                new_tensor = fill(old_tensor, new_tensor, 0)
                #print(f"OTHER: {set.name}{map_field} tensor shape: {new_tensor.shape}, old tensor shape: {old_tensor.shape}")
                new_tensors[map_field] = new_tensor
            new_map_obj = Mappings(driver, new_tensors)
            original_network.mappings[set] = new_map_obj
            return new_tensors
    except:
        raise ValueError(f"Mapping for {set} is not found")

def expand_token_set(original_network: Network, num_analogs, target_set):
    """
    Efficiently expand an existing network by duplicating nodes in the target set.
    
    This function takes an existing network and expands it by duplicating nodes in the target set
    It updates all necessary tensors, IDs, and connections.
    
    Args:
        original_network (Network): The original network to expand
        num_duplicates (int): How many times to expand the network
        target_set (Set): Which set to expand
        
    Returns:
        Network: A new network with expanded target set
    """
    if original_network[target_set].nodes.shape[0] == 0:
        return original_network

    start_time = time.time()
    
    # Get the target set to expand
    target_set_obj: Base_Set = original_network.sets[target_set]
    
    # Calculate new sizes
    original_count = target_set_obj.nodes.size(0)
    new_count = int(original_count * num_analogs/2.5)
    
    # =======================[ Token Data ]========================
    new_nodes, new_connections, new_IDs, new_names = new_token_data(original_count, new_count, target_set_obj)
    new_nodes = fill(target_set_obj.nodes, new_nodes, 0)                                 # Fill the nodes tensor
    new_IDs, new_names, new_nodes = fill_IDs(target_set_obj, new_IDs, new_names, new_nodes) # Fill the IDs and names
    # =======================[ Connections ]========================
    new_connections = fill_square(target_set_obj.connections, new_connections)              # Fill the connections tensor

    # =======================[ Set ]========================
    # Create new set object based on target set
    object_dict = {
        Set.DRIVER: Driver,
        Set.RECIPIENT: Recipient,
        Set.MEMORY: Memory,
        Set.NEW_SET: New_Set
    }
    new_set_obj = object_dict[target_set](new_nodes, new_connections, new_IDs, new_names)
    
    # ==========================[ Links ]===========================
    # Create new links tensor for the target set
    new_links_tensor = torch.zeros(new_count, original_network.semantics.nodes.size(0))      # New links tensor
    new_links_tensor[:original_count] = target_set_obj.links[target_set]                     # Copy original links
    new_links_tensor = fill(target_set_obj.links[target_set], new_links_tensor, 0)           # Fill the links tensor
    # Create new links object with updated target set links
    new_links = Links(
        original_network.links.sets, 
        original_network.semantics
        )
    # Update the links for the target set
    new_links[target_set] = new_links_tensor
    
    # =========================[ Mappings ]========================
    # Create new mappings if needed
    new_map_obj = None
    # Handle mappings based on which set is being expanded
    if target_set == Set.DRIVER:
        new_mappings = new_mapping_driver(original_network, new_nodes, new_set_obj) # Pass updated driver set object

    elif target_set in [Set.MEMORY, Set.RECIPIENT]:
        new_mappings = new_mapping_other(original_network, new_nodes, target_set, original_network[Set.DRIVER])   # Pass current driver set object
    
    # =======================[ Network ]========================
    # Create new network with expanded set
    original_network.sets[target_set] = new_set_obj
    new_network = Network(
        original_network.sets,
        original_network.semantics,
        original_network.mappings,
        new_links,
        original_network.params
    )
    return new_network

def create_large_network(base_network, no_analogs):
    """
    Create a large network by expanding base network to target size.
    
    Args:
        base_network (Network): The base network to expand
        target_size (int): The target size for the network
        target_set (Set): Which set to expand (default is MEMORY)
        
    Returns:
        Network: A new network with the target size
    """
    # Expand network
    for set in Set:
        current_network = expand_token_set(base_network, no_analogs, set)

    return current_network