from .nodeEnums import *
import torch

class Mappings(object):
    """
    A class for storing mappings and hypothesis information.
    """
    def __init__(self, connections, weights, hypotheses, max_hyps):
        """
        Initialize the Mappings object.

        Args:
            connections (torch.Tensor): adjacency matrix of connections from recipient to driver.
            weights (torch.Tensor): weight matrix for connections from recipient to driver.
            hypotheses (torch.Tensor): hypothesis values matrix for connections from recipient to driver.
            max_hyps (torch.Tensor): max hypothesis values matrix for connections from recipient to driver.
        """
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
        """
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