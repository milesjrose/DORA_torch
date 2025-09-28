# entropyNet
# Basic entropy net for similarity computation

from enum import IntEnum
import torch

"""
-- basic ent node
-- basic link
-- entropy net
"""

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
        self.connections = torch.ones(2, self.num_in, dtype=torch.float32)
        diff = extents[Ext.LARGE] - extents[Ext.SMALL]
        # for smaller extent set the last diff connections to zero
        self.connections[Ext.SMALL, diff:] = 0.0

        self.first_input = 2

    def run_entropy_net(self, gamma=0.3, delta=0.1):
        """
        run the network untill settles (i.e only one output node is active for 3 iterations)
        """
        pass

    def update_act_node(self, gamma, delta):
        """
        update act for basic ent nodes
        """

    
    def update_input_nodes(self):
        """
        update input for basic ent node
        """
        input_act = self.nodes[self.first_input:, BF.ACT].sum()
        # add connection weight * input act
        self.nodes[Ext.SMALL:Ext.Large, BF.INPUT] += torch.matmul( # NOTE THis is probably wrong, im hungover
            self.nodes[Ext.SMALL:Ext.Large, BF.ACT],
            self.connections[Ext.SMALL:Ext.Large, :]
        )
        # divisively normalise

        pass

    def clear_input(self):
        """
        clear inputt
        """
        pass
