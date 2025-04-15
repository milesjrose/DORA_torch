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

def fill_dim_0(old_tensor: torch.Tensor, new_tensor: torch.Tensor):
    """
    Fill a tensor with a smaller tensor along the first dimension
    """
    original_size = old_tensor.size(0)
    num_duplicates = new_tensor.size(0) // original_size
    for i in range(num_duplicates):
        new_tensor[original_size + i * original_size: (original_size + (i + 1) * original_size)] = old_tensor
    return new_tensor

def fill_dim_1(old_tensor: torch.Tensor, new_tensor: torch.Tensor):
    """
    Fill a tensor with a smaller tensor along the second dimension
    """
    original_size = old_tensor.size(1)
    num_duplicates = new_tensor.size(1) // original_size
    for i in range(num_duplicates):
        new_tensor[:, original_size + i * original_size: (original_size + (i + 1) * original_size)] = old_tensor
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
    for i in range(num_duplicates):
        new_tensor[original_size + i * original_size: (original_size + (i + 1) * original_size), 
                   original_size + i * original_size: (original_size + (i + 1) * original_size)] = old_tensor
    return new_tensor
        
def new_mapping_driver(original_network: Network, new_nodes: torch.Tensor):
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
            mapping = original_network.sets[set].mappings
            if mapping is not None:
                new_tensors = {}
                for map_field in MappingFields:
                    # Expand the tensor in dim=1 to size of new driver
                    old_tensor = mapping.adj_matrix[:, :, map_field]
                    new_tensor = torch.zeros(old_tensor.size(0), new_nodes.size(0))
                    new_tensor = fill_dim_1(old_tensor, new_tensor)
                    new_tensors[map_field] = new_tensor
                new_mappings[set] = new_tensors
        except:
            if set in [Set.MEMORY, Set.RECIPIENT]:
                raise ValueError(f"Mapping for {set} is not found")
        
    return new_mappings

def new_mapping_other(original_network: Network, new_nodes: torch.Tensor, set: Set):
    """
    Create new mappings for a non-driver set

    Returns:
        dict(MappingField -> torch.Tensor): A dictionary of new mappings for given set
    """
    try:
        mapping = original_network.sets[set].mappings
        if mapping is not None:
            new_tensors = {}
            for map_field in MappingFields:
                # Expand the tensor in dim=1 to size of new driver
                old_tensor = mapping.adj_matrix[:, :, map_field]
                new_tensor = torch.zeros(new_nodes.size(0), old_tensor.size(1))
                new_tensor = fill_dim_0(old_tensor, new_tensor)
                new_tensors[map_field] = new_tensor
            return new_tensors
    except:
        raise ValueError(f"Mapping for {set} is not found")

def expand_token_set(original_network: Network, num_duplicates, target_set):
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
    start_time = time.time()
    
    # Get the target set to expand
    target_set_obj: Base_Set = original_network.sets[target_set]
    
    # Calculate new sizes
    original_size = target_set_obj.nodes.size(0)
    new_size = int(original_size * num_duplicates)
    
    # Create new tensors with expanded size
    new_nodes, new_connections, new_IDs, new_names = new_token_data(original_size, new_size, target_set_obj)
    
    # Fill the tensors with the original data
    new_nodes = fill_dim_0(target_set_obj.nodes, new_nodes)
    new_connections = fill_square(target_set_obj.connections, new_connections)

    # Fill the IDs and names
    new_IDs, new_names, new_nodes = fill_IDs(target_set_obj, new_IDs, new_names, new_nodes)
    
    # Create new links tensor for the target set
    new_links_tensor = torch.zeros(new_size, original_network.semantics.nodes.size(0))
    new_links_tensor[:original_size] = target_set_obj.links[target_set]
    # Fill the links tensor
    new_links_tensor = fill_dim_0(target_set_obj.links[target_set], new_links_tensor)
    
    # Create new links object with updated target set links
    new_links = Links(
        original_network[Set.DRIVER ].links.driver, 
        original_network[Set.RECIPIENT].links.recipient, 
        original_network[Set.MEMORY].links.memory,
        original_network[Set.NEW_SET].links.new_set, 
        original_network.semantics
        )
    # Update the links for the target set
    new_links[target_set] = new_links_tensor
    
    # Create new mappings if needed
    new_mappings = None
    
    # Handle mappings based on which set is being expanded
    if target_set == Set.DRIVER:
        new_mappings = new_mapping_driver(original_network, new_nodes)
    
    elif target_set in [Set.MEMORY, Set.RECIPIENT]:
        new_mappings = new_mapping_other(original_network, new_nodes, target_set)
    
    # Create new mappings object
    mapping_tensors = {
        MappingFields.CONNETIONS: new_mappings[MappingFields.CONNETIONS],
        MappingFields.WEIGHT: new_mappings[MappingFields.WEIGHT],
        MappingFields.HYPOTHESIS: new_mappings[MappingFields.HYPOTHESIS],
        MappingFields.MAX_HYP: new_mappings[MappingFields.MAX_HYP]
    }
    new_mappings = Mappings(original_network[Set.DRIVER], mapping_tensors)
    
    # Create new set object based on target set
    match target_set:
        case Set.DRIVER:
            new_set_obj = Driver(new_nodes, new_connections, new_links, new_IDs, new_names, original_network.params)
        case Set.RECIPIENT:
            new_set_obj = Recipient(new_nodes, new_connections, new_links, new_mappings, new_IDs, new_names, original_network.params)
        case Set.MEMORY:
            new_set_obj = Memory(new_nodes, new_connections, new_links, new_mappings, new_IDs, new_names, original_network.params)
        case Set.NEW_SET:
            new_set_obj = New_Set(new_nodes, new_connections, new_links, new_IDs, new_names, original_network.params)
        case _:
            raise ValueError(f"Unsupported target set: {target_set}")
    
    # Create new network with expanded set
    original_network.sets[target_set] = new_set_obj
    new_network = Network(
        original_network.sets,
        original_network.semantics,
        new_mappings,
        new_links,
        original_network.params
    )
    if target_set == Set.DRIVER:
        new_network.sets[Set.RECIPIENT].mappings.driver = new_set_obj
        new_network.sets[Set.MEMORY].mappings.driver = new_set_obj
        new_network.sets[Set.NEW_SET].mappings.driver = new_set_obj

    new_network.set_params(original_network.params)
    
    # Update set_mappings if needed
    if target_set in [Set.MEMORY, Set.RECIPIENT]:
        new_network.mappings[target_set] = new_mappings
    
    # Update links in the new network
    new_network.links = new_links
    
    # Update links in the sets
    for set in Set:
        if set == target_set:
            new_network.sets[set].links = new_links
        else:
            new_network.sets[set].links = original_network.sets[set].links
    
    # Update links in semantics
    new_network.semantics.links = new_links
    
    # Ensure all links tensors have the correct dimensions
    # This is crucial to prevent dimension mismatch errors in semantics.update_input_from_set
    for set in Set:
        set_obj = new_network.sets[set]
        set_size = set_obj.nodes.size(0)
        sem_size = new_network.semantics.nodes.size(0)
        
        # Check if the links tensor has the correct dimensions
        if new_links[set].size(0) != set_size:
            # Create a new links tensor with the correct dimensions
            new_set_links = torch.zeros(set_size, sem_size)
            # Copy the original data
            new_set_links[:min(original_network.sets[set].nodes.size(0), set_size), 
                         :min(original_network.semantics.nodes.size(0), sem_size)] = \
                original_network.sets[set].links[set][:min(original_network.sets[set].nodes.size(0), set_size), 
                                                    :min(original_network.semantics.nodes.size(0), sem_size)]
            # Update the links for this set
            new_links[set] = new_set_links
    
    end_time = time.time()
    
    return new_network

def create_large_network(base_network, target_size, target_set=Set.DRIVER):
    """
    Create a large network by repeatedly expanding a base network until it reaches the target size.
    
    Args:
        base_network (Network): The base network to expand
        target_size (int): The target size for the network
        target_set (Set): Which set to expand (default is MEMORY)
        
    Returns:
        Network: A new network with the target size
    """
    current_network = base_network
    current_size = current_network.sets[target_set].nodes.size(0)

    if current_size > 0:
        while current_size < target_size:
            # Calculate expansion factor needed
            expansion_factor = target_size / current_size
            if expansion_factor > 10:  # Limit expansion factor to avoid memory issues
                expansion_factor = 10
            
            # Expand network
            current_network = expand_token_set(current_network, expansion_factor, target_set)
            current_size = current_network.sets[target_set].nodes.size(0)
            
        
    return current_network

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for generating and manipulating networks.")
    print("Import and use the functions in your code:")
    print("  from generate_network import expand_network, duplicate_nodes_in_memory, create_large_network")
