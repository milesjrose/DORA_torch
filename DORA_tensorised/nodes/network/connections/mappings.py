# nodes/sets/connections/mappings.py
# Mappings between nodes and semantics.
# TODO: Implement add_mappings, updateHypotheses

import torch

from nodes.enums import *

class Mappings(object):
    """
    A class for storing mappings and hypothesis information.
    """
    def __init__(self, driver, connections: torch.Tensor, weights: torch.Tensor, hypotheses: torch.Tensor, max_hyps: torch.Tensor):
        """
        Initialize the Mappings object.
        Args:
            driver (Driver): driver that mappings map to.
            connections (torch.Tensor): adjacency matrix of connections from recipient to driver.
            weights (torch.Tensor): weight matrix for connections from recipient to driver.
            hypotheses (torch.Tensor): hypothesis values matrix for connections from recipient to driver.
            max_hyps (torch.Tensor): max hypothesis values matrix for connections from recipient to driver.
        
        Raises:
            ValueError: If the tensors are not torch.Tensor.
            ValueError: If the tensors do not have the same shape.
        """
        if type(connections) != torch.Tensor or type(weights) != torch.Tensor or type(hypotheses) != torch.Tensor or type(max_hyps) != torch.Tensor:
            raise ValueError("All tensors must be torch.Tensor.")
        if connections.shape != weights.shape or connections.shape != hypotheses.shape or connections.shape != max_hyps.shape:
            raise ValueError("All tensors must have the same shape.")
        if weights.dim() != 2:
            raise ValueError("Tensors should be 2D")
        # Stack the tensors along a new dimension based on MappingFields enum
        self.driver = driver
        self.adj_matrix: torch.Tensor = torch.stack([
            weights,                    # MappingFields.WEIGHT = 0
            hypotheses,                 # MappingFields.HYPOTHESIS = 1
            max_hyps,                   # MappingFields.MAX_HYP = 2
            connections                 # MappingFields.CONNETIONS = 3
        ], dim=-1)

    def size(self, dim):
        return self.adj_matrix.size(dim=dim)
    
    def connections(self):
        """
        Return the connections matrix from the adjacency matrix.
        """
        return self.adj_matrix[:, :, MappingFields.CONNETIONS]

    def weights(self):
        """
        Return the weights matrix from the adjacency matrix.
        """
        return self.adj_matrix[:, :, MappingFields.WEIGHT]
    
    def hypotheses(self):
        """
        Return the hypotheses matrix from the adjacency matrix.
        """
        return self.adj_matrix[:, :, MappingFields.HYPOTHESIS]
    
    def max_hyps(self):
        """
        Return the max hypotheses matrix from the adjacency matrix.
        """
        return self.adj_matrix[:, :, MappingFields.MAX_HYP]
    
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