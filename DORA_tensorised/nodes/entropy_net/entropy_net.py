# entropyNet
# Basic entropy net for similarity computation

from enum import IntEnum
from math import isclose
import torch

class BF(IntEnum):
    """ features for basic nodes""" 
    ACT = 0
    """act"""
    INPUT = 1
    """input"""
    LAT = 2
    """lateral loss/inhibition"""
    TYPE = 3
    """Type of node (input or output)"""

class Type(IntEnum):
    """ENcode type feature"""
    IN = 0
    """input node"""
    OUT = 1
    """output node"""

class Ext(IntEnum):
    """access extent nodes"""
    SMALL = 0
    """small extent"""
    LARGE = 1
    """large extent"""

class EntropyNet(object):
    def __init__(self):
        self.nodes : torch.Tensor = None
        """ nodes tensor: holds extents and input nodes"""
        self.connections : torch.Tensor = None
        """ connections tensor: Parent (extent) to child (input)"""
        self.settled = None
        self.settled_delta = None
        self.settlede_iters = None
        self.input_mask = None
        self.output_mask = None
        

    def fillin(self, extent1, extent2 ):
        """
        Populate the entropy net with extents and entropy nodes (semantics and pos)
        instantiate as extents as simple nodes.
        """
        extents = {
            Ext.SMALL: min(int(extent1), int(extent2)),
            Ext.LARGE: max(int(extent1), int(extent2)),
        }
        self.num_in = extents[Ext.LARGE]
        self.nodes: torch.Tensor = torch.zeros(self.num_in+2, len(BF))
        self.nodes[Ext.SMALL, BF.TYPE] = Type.OUT
        self.nodes[Ext.LARGE, BF.TYPE] = Type.OUT
        
        # connections are from output nodes (parents) to input nodes (children).
        # shape: [num_outputs, num_inputs]
        self.connections = torch.zeros(2, self.num_in, dtype=torch.float32)
        # connect outputs to inputs
        self.connections[Ext.SMALL, :extents[Ext.SMALL]] = 1.0
        self.connections[Ext.LARGE, :extents[Ext.LARGE]] = 1.0
        
        # create masks for input/output nodes
        self.input_mask = (self.nodes[:, BF.TYPE] == Type.IN)
        self.output_mask = (self.nodes[:, BF.TYPE] == Type.OUT)
    
    def run_entropy_net(self, gamma=0.3, delta=0.1):
        """
        run the network untill settles (i.e only one output node is active for 3 iterations)
        """
        # set activations of all input nodes to 1.0:
        self.nodes[self.input_mask, BF.ACT] = 1.0
        # until the network settles (i.e., only one output node is active for 3 iterations), keep running.
        settled = 0
        iterations = 0
        delta_outputs_previous = torch.tensor(float('inf')) # Initialise, so first check fails
        while settled < 3 and iterations < 300:
            # update the inputs to the output units.
            self.clear_input_higher_nodes()
            self.update_input_higher_nodes()
            self.update_act_higher_nodes(gamma, delta)
            # check for settling. if the delta_outputs has not changed, add 1 to settled, otherwise, clear unsettled. 
            # Delta is calculated over outputs rounded to 3 decimals.
            delta_outputs = self.nodes[Ext.LARGE, BF.ACT] - self.nodes[Ext.SMALL, BF.ACT]
            if isclose(delta_outputs, delta_outputs_previous, abs_tol=0.001):
                settled += 1
            else:
                settled = 0
            delta_outputs_previous = delta_outputs
            iterations += 1
        self.settled_iters = iterations
    
    def update_input_higher_nodes(self):
        """
        update input for output nodes
        """
        input_activations = self.nodes[self.input_mask, BF.ACT]
        total_input_act = input_activations.sum()

        # input is the sum of connected input node activations.
        weighted_inputs = torch.matmul(self.connections, input_activations)
        self.nodes[self.output_mask, BF.INPUT] += weighted_inputs
        
        # divisively normalise
        if total_input_act > 0:
            self.nodes[self.output_mask, BF.INPUT] /= total_input_act
            
        # lateral loss is the sum of all other output node activations
        self.nodes[self.output_mask, BF.LAT] += self.nodes[self.output_mask, BF.ACT].sum() - self.nodes[self.output_mask, BF.ACT]

    def update_act_higher_nodes(self, gamma, delta):
        """
        update act for output nodes
        """
        self.nodes[self.output_mask, BF.INPUT] -= self.nodes[self.output_mask, BF.LAT]
        delta_act = gamma * self.nodes[self.output_mask, BF.INPUT] * (1.1 - self.nodes[self.output_mask, BF.ACT]) - (delta * self.nodes[self.output_mask, BF.ACT])
        self.nodes[self.output_mask, BF.ACT] += delta_act
        # hard limit activation to between 0.0 and 1.0.
        self.nodes[self.output_mask, BF.ACT] = torch.clamp(self.nodes[self.output_mask, BF.ACT], 0.0, 1.0)
    
    def clear_input_higher_nodes(self):
        """
        clear input and lateral loss for output nodes
        """
        self.nodes[self.output_mask, BF.INPUT] = 0.0
        self.nodes[self.output_mask, BF.LAT] = 0.0
    
    def clear_all_higher_nodes(self):
        """
        clear all values for output nodes
        """
        self.nodes[self.output_mask, BF.INPUT] = 0.0
        self.nodes[self.output_mask, BF.ACT] = 0.0
        self.nodes[self.output_mask, BF.LAT] = 0.0
    
    def get_more_less(self):
        """
        get the more and less nodes
        """
        if self.nodes[Ext.LARGE, BF.ACT] == self.nodes[Ext.SMALL, BF.ACT]:
            return None, None
        elif self.nodes[Ext.LARGE, BF.ACT] > self.nodes[Ext.SMALL, BF.ACT]:
            return Ext.LARGE, Ext.SMALL
        else:
            return Ext.SMALL, Ext.LARGE
    
