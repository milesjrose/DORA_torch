# nodes/sets/connections/mappings.py
# Mappings between nodes and semantics.
# TODO: Implement add_mappings, updateHypotheses

import torch

from ...enums import *

class Mappings(object):
    """
    A class for storing mappings and hypothesis information.
    """
    def __init__(self, driver, map_fields: dict[MappingFields, torch.Tensor]):
        """
        Initialize the Mappings object.
        Args:
            driver (Driver): driver that mappings map to.
            map_fields (dict[MappingFields, torch.Tensor]): dictionary of tensors for each mapping field.
        
        Raises:
            ValueError: If the tensors are not torch.Tensor. Or have incorrect shape.
        """
        # Check tensors are correct type and shape
        con_shape = map_fields[MappingFields.CONNECTIONS].shape[0] # Used to check all tensors have same shape
        for field in MappingFields:
            # Check tensor is correct type
            if type(map_fields[field]) != torch.Tensor:
                raise ValueError(f"{field} tensor must be torch.Tensor, but is {type(map_fields[field])}.")
            
            # Check tensor is 2D
            if map_fields[field].dim() != 2:
                raise ValueError(f"{field} tensor should be 2D, but has {map_fields[field].dim()} dimensions.")
            
            # Check all tensors have same shape
            shape = map_fields[field].shape[0]
            if shape != con_shape:
                raise ValueError(f"{field} field tensor shape: {shape}, but connections tensor shape: {con_shape}.")
            
            # Check driver nodes match
            driver_count = driver.nodes.shape[0]
            if map_fields[field].shape[1] != driver_count:
                raise ValueError(f"{field} field tensor shape: {map_fields[field].shape}, but driver nodes shape: {driver.nodes.shape}.")
        
        # Stack the tensors along a new dimension based on MappingFields enum
        self.driver = driver
        field_list = []
        for field in MappingFields:
            field_list.append(map_fields[field])                            # Add each tensor to the list, indexed by MappingFields enum
        self.adj_matrix: torch.Tensor = torch.stack(field_list, dim=-1)     # Stack the tensors, last dimension is the field
        """ Stacked tensor of shape (N, D, F):
        - N: number of nodes in this set
        - D: number of nodes in driver set
        - F: number of fields in MappingFields enum
        """
    
    # ===================[ Getters and setters ]====================
    def size(self, dim):
        return self.adj_matrix.size(dim=dim)
    
    def __getitem__(self, mappingField: MappingFields):
        return self.adj_matrix[:, :, mappingField]
    
    def __setitem__(self, mappingField: MappingFields, value):
        self.adj_matrix[:, :, mappingField] = value
    
    # =====================[ Update functions ]======================
    def updateHypotheses(self, hypotheses):
        """
        Update the hypotheses matrix.
        TODO: implement
        """
        pass
    
    def add_mappings(self,  mappings):
        """
        Add mappings to the adjacency matrix.
        TODO: implement
        """
        pass
    # ----------------------------------------------------------------