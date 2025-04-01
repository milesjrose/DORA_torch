from .nodeEnums import *
import torch

class Mappings(object): # 3D tensor storing mapping and hypothesis information
    def __init__(self, connections):    # Takes 3D tensor, of stacked 2D adjacency matrices
        # Takes 3D tensor, of stacked 2D adjacency matrices: Recipient -> driver
        self.adj_matrix: torch.Tensor = connections
    
    def connections(self):
        return self.adj_matrix[:, :, MappingFields.CONNETIONS]

    def weights(self):
        return self.adj_matrix[:, :, MappingFields.WEIGHT]
    
    def hypotheses(self):
        return self.adj_matrix[:, :, MappingFields.HYPOTHESIS]
    
    def max_hyps(self):
        return self.adj_matrix[:, :, MappingFields.MAX_HYP]
    
    def updateHypotheses(self, hypotheses):                         # TODO: implement
        pass
    
    def add_mappings(self,  mappings):                              # TODO: implement
        pass

class Links(object):    # Weighted connections between nodes - want groups as well as placeholder.
    def __init__(self, driver_links, recipient_links, memory_links):  # Takes weighted adjacency matrices
        self.driver: torch.Tensor = driver_links
        self.recipient: torch.Tensor = recipient_links
        self.memory: torch.Tensor = memory_links
        self.sets = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory
        }
    
    def add_links(self, set: Set, links):                           # TODO: implement
        pass