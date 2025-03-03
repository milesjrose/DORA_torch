# nodesMemTypes.py 
# Classes for segments of memory, and set-specific tensor operations
import torch
from nodeEnums import *
import tensorOps as tOps


class TokenTensor(object):
    def __init__(self, floatTensor, boolTensor, connections):
        self.nodes: torch.Tensor = floatTensor
        self.nodesBool: torch.Tensor = boolTensor
        self.masks = self.cache_masks()
        self.indicies = self.cache_masks()
        # Unweighted directed connections between token of same set. Connetion is from parent to child.
        self.connections = connections

    def cache_masks(self, types_to_recompute = None):   # Compute and cache masks, specify types to recompute via list of tokenTypes
        if types_to_recompute == None:                      #  If no type specified, recompute all
            types_to_recompute = [Type.PO, Type.RB, Type.P, Type.GROUP]

        masks = []
        for token_type in [Type.PO, Type.RB, Type.P, Type.GROUP]:
            if token_type in types_to_recompute:
                masks.append(self.computeMasks(token_type)) # Recompute mask
            else:
                masks.append(self.masks[token_type])        # Use cached mask

        self.masks: torch.Tensor = torch.stack(masks, dim=0)
    
    def compute_mask(self, token_type: Type):       # Compute the mask for a token type
        mask = (self.nodes[:, TF.TYPE] == token_type) 
        return mask
    
    def get_mask(self, token_type: Type = None):    # Returns mask for token of type Type, or full tensor o.w
        if token_type is None:              
            return self.masks                               # Return stacked mask tensor
        else:
            return self.masks[token_type]                   # Return mask for inputted type
    
    def pos(self):              # Return PO subtensor
        return self.nodes[self.get_mask(Type.PO)]
    
    def rbs(self):              # Return RB subtensor
        return self.nodes[self.get_mask(Type.RB)]
    
    def ps(self):               # Return P subtensor
        return self.nodes[self.get_mask(Type.Ps)]
    
    def groups(self):           # Return Group subtensor
        return self.nodes[self.get_mask(Type.GROUP)]
    
    def get_p_mode(self):       # TODO: Set mode for all P units
        # set Pmode to 1 (Parent) if input from my RBs below me is greater than input from RBs above me, set Pmode to -1 (child) if input from RBs above me is greater than input from RBs below me, and set Pmode to 0 (neutral) otherwise.
        # get the input from RBs below me (.myRBs) and RBs above me (.myParentRBs).
        return None

    def add_nodes(self, nodes): # TODO: Add nodes 
        # As tensor is non-extensible, keep some headroom. If the tensor is full create new tensor with ratio 1.1, o.w just add node
        # TODO: allow for adding nodes - tensor non-extensible, so must create new tensor then add node
        return None
    
    def del_Nodes(self, nodes): # TODO: Delete nodes
        # Tensor size is static, so just set the nodes deleted to zero - then recompute masks to exclude it.
        # TODO: When deleted nodes ratio hits threshold, compress the tensor to remove all deleted nodes.
        # TODO: Impletment
        return None
    
    # ======================[ TOKEN FUNCTIONS ]========================
    def initialise_input(self, refresh):        # TODO: initialize inputs to 0, and td_input to refresh.
        pass

    def initialise_act(self, refresh):          # TODO: initialize act to 0.0,  and call initialise_inputs
        pass

    def initialise_state(self):                 # TODO: set  self.retrieved to false, and call initialise_act
        pass
    
    def update_act(self, gamma, delta, HebbBias): # TODO: update act of node
        pass

    def zero_lateral_input(self):               # TODO: set lateral_input to 0 (to allow synchrony at different levels by 0-ing lateral inhibition at that level (e.g., to bind via synchrony, 0 lateral inhibition in POs).
        pass
    
    def update_inhibitor_input(self):           # TODO: update the input to my inhibitor by my current activation.
        pass

    def reset_inhibitor(self):                  # TODO: reset the inhibitor_input and act to 0.0.
        pass
    # --------------------------------------------------------------


class DriverTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):   
        super().__init__(floatTensor, boolTensor, connections)

    # ==============[ DRIVER UPDATE INPUT FUNCTIONS ]===============
    def update_input(self, as_DORA):            # Update all input in driver
        self.update_input_p_parent()
        self.update_input_p_child(as_DORA)
        self.update_input_rb(as_DORA)
        self.update_input_po(as_DORA)

    def update_input_p_parent(self):
        # P units in parent mode:
        # Exitatory: td (my Groups) / bu (my RBs)
        # Inhibitory: lateral (other P units in parent mode*3), inhibitor.
        # 1). get masks
        p = self.get_mask(Type.P)                                   # Boolean mask for P nodes
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.PARENT)   # Boolean mask for Parent P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, TF.BU_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                                # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # Each p node -> sum of act of connected rb nodes
            )  
        
        # Inhibitory input:
        # 4). LATERAL_INPUT: (3 * other parent p nodes in driver), inhibitor
        # 4a). Create tensor mask of parent p nodes, and a tensor to connect p nodes to each other
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting parent ps to all but themselves
        # 4b). 3 * other parent p nodes in driver
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(3, torch.matmul(
            diag_zeroes,                                            # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.nodes[p, TF.ACT]                                   # Each parent p node -> 3*(sum of all other parent p nodes)
        ))

    def update_input_p_child(self, as_DORA):
        # P units in child mode:
        # Exitatory: td (my parent RBs), (if phase_set>1: my groups)
        # Inhibitory: lateral (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 1). get masks
        p = self.get_mask(Type.P)
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.CHILD)    # Boolean mask for Child P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes
        po = self.get_mask(Type.PO)

        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        # 2a). groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        # 2b). parent_rbs
        t_con = torch.transpose(self.connections)                   # transpose, so gives child -> parent connections
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs matrix (sum(p) x 1) of values to add to current input value
            t_con[p, rb],                                           # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # For each p node -> sum of act of connected parent rb nodes
            )
        
        # Inhibitory input:
        # 3). LATERAL_INPUT: (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 3a). other p in child mode
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting child ps to all but themselves
        self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
            diag_zeroes,                                            # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.nodes[p, TF.ACT]                                   # Each child p node -> 3*(sum of all other parent p nodes)
        )
        # 3b). if as_DORA: Objects not connected to same RBs
        if as_DORA:
            obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)# Boolean mask for objects
            ones = torch.ones((sum(p), sum(obj)))                   # tensor connecting every p to every object
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(    
                ones,                                               # connects all p to all object
                self.nodes[obj, TF.ACT]                             # Each  p node -> sum of all object acts
            )
        # 3c). Else: Object acts
        else: 
            # 3ci). Create masks
            obj = self.get_mask(Type.PO)                            # get PO mask
            obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.TRUE) # refine mask for objects only
            # 3cii). Find objects not connected to me               # NOTE: check if p.myRB contains p.myParentRBs !currently assume true!
            ud_con = torch.bitwise_or(self.connections, t_con)      # undirected connections tensor (OR connections with its transpose)
            shared = torch.matmul(ud_con[p, rb], ud_con[rb, obj])   # PxObject tensor, shared[i][j] > 1 if p[i] and object[j] share an RB, 0 o.w
            shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
            non_shared = 1 - shared                                 # non_shared[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
            # 3cii). update input using non shared objects
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
                non_shared,                                         # sum(p)xsum(object) matrix, listing non shared objects for each p
                self.nodes[obj, TF.ACT]                             # sum(objects)x1 matrix, listing act of each object
            )
   
    def update_input_rb(self, as_DORA):                             # TODO: implement
        pass
    
    def update_input_po(self, as_DORA):                             # TODO: implement
        pass

    # --------------------------------------------------------------


class RecipientTensor(TokenTensor):
    def __init__(self, floatTensor, boolTensor, connections):
        super().__init__(floatTensor, boolTensor, connections)

    # ============[ RECIPIENT UPDATE INPUT FUNCTIONS ]==============
    def update_input(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False):            # Update all input in recipient
        self.update_input_p_parent()
        self.update_input_p_child(as_DORA)
        self.update_input_rb(as_DORA)
        self.update_input_po(as_DORA)

    def update_input_p_parent(self):                                # TODO: implement
        pass

    def update_input_p_child(self):                                 # TODO: implement
        pass
    
    def update_input_rb(self, as_DORA, phase_set, lateral_input_level): # TODO: implement
        pass
    
    def update_input_po(self, as_DORA, phase_set, lateral_input_level, ignore_object_semantics=False): # TODO: implement
        pass

    # --------------------------------------------------------------
    
class SemanticTensor(TokenTensor):
    def __init__(self):
        pass


class Links(object):    # Weighted connections between nodes
    def __init__(self, driverLinks, recipientLinks, semLinks):  # Takes weighted adjacency matrices
        self: torch.Tensor = driverLinks
        self.recipient: torch.Tensor = recipientLinks
        self.semantics: torch.Tensor = semLinks
    
    def add_links(self, set: Set, links):                           # TODO: implement
        pass


class Mappings(object): # 3D tensor storing mapping and hypothesis information
    def __init__(self, connections):    # Takes 3D tensor, of stacked 2D adjacency matrices
        # Takes 3D tensor, of stacked 2D adjacency matrices
        self.adj_matrix: torch.Tensor = connections
    
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