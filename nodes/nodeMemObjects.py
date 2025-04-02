from .nodeEnums import *
import torch

class Mappings(object):
    """
    A class for storing mappings and hypothesis information.
    """
    def __init__(self, connections: torch.Tensor, weights: torch.Tensor, hypotheses: torch.Tensor, max_hyps: torch.Tensor):
        """
        Initialize the Mappings object.

        Args:
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
        # Stack the tensors along a new dimension based on MappingFields enum
        self.adj_matrix: torch.Tensor = torch.stack([
            weights,                    # MappingFields.WEIGHT = 0
            hypotheses,                 # MappingFields.HYPOTHESIS = 1
            max_hyps,                   # MappingFields.MAX_HYP = 2
            connections                 # MappingFields.CONNETIONS = 3
        ], dim=-1)
    
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

class Links(object):    # Weighted connections between nodes - want groups as well as placeholder.
    """
    A class for representing weighted connections between token sets and semantics.
    """
    def __init__(self, driver_links, recipient_links, memory_links):  # Takes weighted adjacency matrices
        """
        Initialize the Links object.

        Args:
            driver_links (torch.Tensor): A tensor of weighted connections from the driver set to semantics.
            recipient_links (torch.Tensor): A tensor of weighted connections from the recipient set to semantics.
            memory_links (torch.Tensor): A tensor of weighted connections from the memory set to semantics.
        
        Raises:
            TypeError: If the link tensors are not torch.Tensor.
            ValueError: If the number of semantics (columns) in the link tensors are not the same.
        """
        if type(driver_links) != torch.Tensor:
            raise TypeError("Driver links must be torch.Tensor.")
        if type(recipient_links) != torch.Tensor:
            raise TypeError("Recipient links must be torch.Tensor.")
        if type(memory_links) != torch.Tensor:
            raise TypeError("Memory links must be torch.Tensor.")
        if driver_links.size(dim=1) != recipient_links.size(dim=1) or driver_links.size(dim=1) != memory_links.size(dim=1):
            raise ValueError("All link tensors must have the same number of semantics (columns).")
    
        self.driver: torch.Tensor = driver_links
        self.recipient: torch.Tensor = recipient_links
        self.memory: torch.Tensor = memory_links
        self.sets = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory
        }
    
    def add_links(self, set: Set, links):
        """
        Add links to the adjacency matrix.
        TODO: implement
        """
        pass