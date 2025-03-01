# nodesMemTypes.py 
# Classes for segments of memory, and set-specific tensor operations
import torch
from nodeEnums import *


class TokenTensor(object):
    def __init__(self, floatTensor, boolTensor, connections):
        self.nodes: torch.Tensor = floatTensor
        self.nodesBool: torch.Tensor = boolTensor
        self.masks = self.cache_masks()
        self.indicies = self.cache_masks()
        # Unweighted directed connections between token of same set. Connetion is from parent to child.
        self.connections = connections

    # Compute and cache masks, can specify which to recompute in via list of tokenTypes
    def cache_masks(self, types_to_recompute = None):
        if types_to_recompute == None:                      #  If no type specified, recompute all
            types_to_recompute = [Type.PO, Type.RB, Type.P, Type.GROUP]

        masks = []
        for token_type in [Type.PO, Type.RB, Type.P, Type.GROUP]:
            if token_type in types_to_recompute:
                masks.append(self.computeMasks(token_type)) # Recompute mask
            else:
                masks.append(self.masks[token_type])        # Use cached mask

        self.masks: torch.Tensor = torch.stack(masks, dim=0)
    
    # Compute the mask for a token type
    def compute_mask(self, token_type: Type):
        mask = (self.nodes[:, tf.TYPE] == token_type) 
        return mask
    
    # Returns mask for token of type Type, or full tensor o.w
    def get_mask(self, token_type: Type = None):
        if token_type is None:              
            return self.masks                               # Return stacked mask tensor
        else:
            return self.masks[token_type]                   # Return mask for inputted type
    
    # Return subtensors based on node type:
    def pos(self):
        return self.nodes[self.get_mask(Type.PO)]
    
    def rbs(self):
        return self.nodes[self.get_mask(Type.RB)]
    
    def ps(self):
        return self.nodes[self.get_mask(Type.Ps)]
    
    def groups(self):
        return self.nodes[self.get_mask(Type.GROUP)]
    
    # set mode for all P units
    def get_p_mode(self):
        # set Pmode to 1 (Parent) if input from my RBs below me is greater than input from RBs above me, set Pmode to -1 (child) if input from RBs above me is greater than input from RBs below me, and set Pmode to 0 (neutral) otherwise.
        # get the input from RBs below me (.myRBs) and RBs above me (.myParentRBs).
        # TODO: implement
        return None

    # As tensor is non-extensible, keep some headroom. If the tensor is full create new tensor with ratio 1.1, o.w just add node
    def add_nodes(self, nodes):
        # TODO: allow for adding nodes - tensor non-extensible, so must create new tensor then add node
        return None
    
    # Tensor size is static, so just set the nodes deleted to zero - then recompute masks to exclude it.
    # TODO: When deleted nodes ratio hits threshold, compress the tensor to remove all deleted nodes.
    def del_Nodes(self, nodes):
        # TODO: Impletment
        return None
    
# =====================[ DRIVER FUNCTIONS ]===================== 
class DriverTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):   
        super().__init__(floatTensor, boolTensor, connections)

# ===============[ P UNIT UPDATE INPUT FUNCTIONS ]==============
    def update_input_p_parent(self):
        # P units in parent mode:
        # sources of input:
        # Exitatory: td (my Groups), bu (my RBs).
        # Inhibitory: lateral (other P units in parent mode*3), inhibitor.

        # 1). get masks
        p = self.get_mask(Type.P)                               # Boolean mask for P nodes
        group = self.get_mask(Type.GROUP)                       # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                             # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups
        self.nodes[p, tf.TD_INPUT] += torch.matmul(             # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                         # Masks connections between p[i] and its groups
            self.nodes[group, tf.ACT]                           # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, tf.BU_INPUT] += torch.matmul(             # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                            # Masks connections between p[i] and its rbs
            self.nodes[rb, tf.ACT]                              # Each p node -> sum of act of connected rb nodes
            )  
        
        # Inhibitory input:
        # 4). LATERAL_INPUT: (3 * other parent p nodes in driver), inhibitor
        # 4a). Create tensor mask of parent p nodes, and a tensor to connect p nodes to each other
        parent = (self.nodes(p, tf.MODE) == Mode.PARENT)        # Get sub mask of nodes in p that are in parent mode
        p[p] &= parent                                          # Refine p mask to nodes both in p and in parent mode
        M = sum(p)                                              # Number of parent p nodes
        diag_zeroes = torch.ones((M, M), dtype=torch.float32)   # Create all ones matrix, size of sub-tensor of p nodes in parent mode
        diag_zeroes -= torch.eye(M)                             # Remove ones in diagonal to create adj matrix connection connecting parent ps to all but themself
        # 4b). 3 * other parent p nodes in driver
        self.nodes[p, tf.LATERAL_INPUT] -= torch.mul(3, torch.matmul(
            diag_zeroes,                                        # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.nodes[p, tf.ACT]                               # Each parent p node -> 3*(sum of all other parent p nodes)
        ))

    def update_input_p_child(self):
        # P units in child mode:
        # sources of input:
        # Exitatory: td (my parent RBs and my Groups).
        # Inhibitory: lateral (other P units in child mode, and, if in DORA mode, then other POs not connected to same RB and other PO conected to same RB*3), inhibitor.
        # get my td input from my RBs.

        # 1). get masks
        p = self.get_mask(Type.P)                               # Boolean mask for P nodes
        group = self.get_mask(Type.GROUP)                       # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                             # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        self.nodes[p, tf.TD_INPUT] += torch.matmul(             # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                         # Masks connections between p[i] and its groups
            self.nodes[group, tf.ACT]                           # each p node -> sum of act of connected group nodes
            )
        self.nodes[p, tf.BU_INPUT] += torch.matmul(             # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                            # Masks connections between p[i] and its rbs
            self.nodes[rb, tf.ACT]                              # Each p node -> sum of act of connected rb nodes
            )  
    # ----------------------------------------------------------

# ====================[RECPIPIENT FUNCTIONS]====================
class RecipientTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):
        super().__init__(floatTensor, boolTensor, connections)

    def update_driver_inputs(self):
        # Logic that only applies to driver nodes
        pass

# Weighted connections between nodes
class Links(object):
    # Takes weighted adjacency matrices
    def __init__(self, driverLinks, recipientLinks, semLinks):
        self: torch.Tensor = driverLinks
        self.recipient: torch.Tensor = recipientLinks
        self.semantics: torch.Tensor = semLinks
    
    def add_links(self, set: Set, links):
        # TODO: implement
        return None

# 3D tensor storing mapping and hypothesis information
class Mappings(object):
    # Takes 3D tensor, of stacked 2D adjacency matrices
    def __init__(self, connections):
        self.adj_matrix: torch.Tensor = connections
    
    def weights(self):
        return self.adj_matrix[:, :, MappingFields.WEIGHT]
    
    def hypotheses(self):
        return self.adj_matrix[:, :, MappingFields.HYPOTHESIS]
    
    def max_hyps(self):
        return self.adj_matrix[:, :, MappingFields.MAX_HYP]
    
    def updateHypotheses(self, hypotheses):
        # TODO: implement
        return None
    
    def add_mappings(self,  mappings):
        # TODO: implement
        return None