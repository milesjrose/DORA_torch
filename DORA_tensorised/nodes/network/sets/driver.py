# nodes/network/sets/driver.py
# Represents the driver set of tokens.
 
import torch

from ...enums import *
from ...utils import tensor_ops as tOps

from ..network_params import Params
from .base_set import Base_Set

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..connections import Mappings

class Driver(Base_Set):
    """
    A class for representing the driver set of tokens.

    Attributes:
        names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        links (torch.Tensor): A Tensor of links from tokens in this set to the semantics.
            NOTE: ^ Not used in driver set. Could remove to save memory? ^
        connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        masks (torch.Tensor): A Tensor of masks for the tokens in this set.
        IDs (dict): A dictionary mapping token IDs to index in the tensor.
        params (Params): An object containing shared parameters.
    """
    def __init__(self, nodes, connections, IDs: dict[int, int], names: dict[int, str] = {}):   
        """
        Initialize the Driver object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        Raises:
            ValueError: If the number of tokens in floatTensor, and connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
            ValueError: If all tokens in floatTensor do not have TF.SET == Set.DRIVER.
            TypeError: If connections is not a torch.Tensor.
        """
        super().__init__(nodes, connections, IDs, names)
        self.token_set = Set.DRIVER
        # TODO: Assign this in init - too lazy to update the builder currently
        self.mappings: 'dict[Set, Mappings]' = {}

        """ Dictionary of mappings for each set. """
        if nodes.size(dim=0) > 0:
            if not torch.all(nodes[:, TF.SET] == Set.DRIVER):
                raise ValueError("All tokens in driver floatTensor must have TF.SET == Set.DRIVER.")
    
    def set_mappings(self, mappings: 'dict[Set, Mappings]'):
        """ Set the mapping dict for driver. """
        self.mappings = mappings
    
    def check_local_inhibitor(self):                                # Return true if any PO.inhibitor_act == 1.0
        """Return true if any PO.inhibitor_act == 1.0"""
        po = self.get_mask(Type.PO)
        return torch.any(self.nodes[po, TF.INHIBITOR_ACT] == 1.0) 

    def check_global_inhibitor(self):                               # Return true if any RB.inhibitor_act == 1.0
        """Return true if any RB.inhibitor_act == 1.0"""
        rb = self.get_mask(Type.RB)
        return torch.any(self.nodes[rb, TF.INHIBITOR_ACT] == 1.0) 
    

    # ==============[ DRIVER UPDATE INPUT FUNCTIONS ]===============

    def update_input(self):                                         # Update all input in driver
        """Update all input in driver"""
        self.update_input_p_parent()
        self.update_input_p_child()
        self.update_input_rb()
        self.update_input_po()

    def update_input_p_parent(self):                                # P units in parent mode - driver
        """Update input in driver for P units in parent mode"""
        # Exitatory: td (my Groups) / bu (my RBs)
        # Inhibitory: lateral (other P units in parent mode*3), inhibitor.
        # 1). get masks
        p = self.get_mask(Type.P)                                   # Boolean mask for P nodes
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.PARENT)   # Boolean mask for Parent P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups
        p_shape = p.shape[0]
        group_shape = group.shape[0]
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p][:, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, TF.BU_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p][:, rb],                                # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # Each p node -> sum of act of connected rb nodes
            )  
        
        # Inhibitory input:
        # 4). LATERAL_INPUT: (3 * other parent p nodes in driver), inhibitor
        # 4a). Create tensor mask of parent p nodes, and a tensor to connect p nodes to each other
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting parent ps to all but themselves
        # 4b). 3 * other parent p nodes in driver
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(
            3, 
            torch.matmul(
                diag_zeroes,                                        # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
                self.nodes[p, TF.ACT]                               # Each parent p node -> 3*(sum of all other parent p nodes)
            )
        )

    def update_input_p_child(self):                                 # P units in child mode  - driver:
        """Update input in driver for P units in child mode"""
        as_DORA = self.params.as_DORA
        # Exitatory: td (my parent RBs), (if phase_set>1: my groups)
        # Inhibitory: lateral (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 1). get masks
        p = self.get_mask(Type.P)
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.CHILD)    # Boolean mask for Child P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes
        po = self.get_mask(Type.PO)
        obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)    # get object mask

        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        # 2a). groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p][:, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        # 2b). parent_rbs
        t_con = torch.transpose(self.connections, 0 , 1)                   # transpose, so gives child -> parent connections
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs matrix (sum(p) x 1) of values to add to current input value
            t_con[p][:, rb],                                           # Masks connections between p[i] and its rbs
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
        # 3b). if as_DORA: Object acts
        if as_DORA:
            ones = torch.ones((sum(p), sum(obj)))                   # tensor connecting every p to every object
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(    
                ones,                                               # connects all p to all object
                self.nodes[obj, TF.ACT]                             # Each  p node -> sum of all object acts
            )
        # 3c). Else: Objects not connected to same RBs
        else: 
            # 3ci). Find objects not connected to me                # NOTE: check if p.myRB contains p.myParentRBs !not true! TODO: fix
            ud_con = torch.bitwise_or(self.connections, t_con)      # undirected connections tensor (OR connections with its transpose)
            shared = torch.matmul(ud_con[p, rb], ud_con[rb, obj])   # PxObject tensor, shared[i][j] > 1 if p[i] and object[j] share an RB, 0 o.w
            shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
            non_shared = 1 - shared                                 # non_shared[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
            # 3cii). update input using non shared objects
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
                non_shared,                                         # sum(p)xsum(object) matrix, listing non shared objects for each p
                self.nodes[obj, TF.ACT]                             # sum(objects)x1 matrix, listing act of each object
            )
   
    def update_input_rb(self):                                      # update RB inputs - driver:
        """Update input in driver for RB units"""
        # Exitatory: td (my parent P), bu (my PO and child P).
        # Inhibitory: lateral (other RBs*3), inhibitor.
        # 1). get masks
        rb = self.get_mask(Type.RB)
        po = self.get_mask(Type.PO)
        p = self.get_mask(Type.P)

        # Exitatory input:
        # 2). TD_INPUT: my_parent_p
        t_con = torch.transpose(self.connections, 0 , 1)                   # Connnections: Parent -> child, take transpose to get list of parents instead
        self.nodes[rb, TF.TD_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            t_con[rb][:, p],                                           # Masks connections between rb[i] and its ps
            self.nodes[p, TF.ACT]                                   # For each rb node -> sum of act of connected p nodes
            )
        # 3). BU_INPUT: my_po, my_child_p                           # NOTE: Old function explicitly took myPred[0].act etc. as there should only be one pred/child/etc. This version sums all connections, so if rb mistakenly connected to multiple of a node type it will not give expected output.
        po_p = torch.bitwise_or(po, p)                              # Get mask of both pos and ps
        self.nodes[rb, TF.BU_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            self.connections[rb][:, po_p],                             # Masks connections between rb[i] and its po and child p nodes
            self.nodes[po_p, TF.ACT]                                # For each rb node -> sum of act of connected po and child p nodes
            )
        
        # Inhibitory input:
        # 4). LATERAL: (other RBs*3), inhibitor*10
        # 4a). (other RBs*3)
        diag_zeroes = tOps.diag_zeros(sum(rb))                      # Connects each rb to every other rb, but not themself
        self.nodes[rb, TF.LATERAL_INPUT] -= torch.mul(
            3, 
            torch.matmul(                                           # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                diag_zeroes,                                        # Masks connections between rb[i] and its po and child p nodes
                self.nodes[rb, TF.ACT]                              # For each rb node -> sum of act of connected po and child p nodes
            )
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[rb, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[rb, TF.LATERAL_INPUT] -= inhib_act         # Update lat input
    
    def update_input_po(self):                                      # update PO inputs - driver:
        """Update input in driver for PO units"""
        as_DORA = self.params.as_DORA
        # Exitatory: td (my RB) * gain (2 for preds, 1 for objects).
        # Inhibitory: lateral (other POs not connected to my RB and Ps in child mode, if in DORA mode, then other PO connected to my RB), inhibitor.
        # 1). get masks
        rb = self.get_mask(Type.RB)
        po = self.get_mask(Type.PO)
        pred_sub = (self.nodes[po, TF.PRED] == B.TRUE)              # predicate sub mask of po nodes
        obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)    # get object mask
        pred = tOps.refine_mask(self.nodes, po, TF.PRED, B.TRUE)    # get predicate mask
        parent_cons = torch.transpose(self.connections, 0 , 1)      # Transpose of connections matrix, so that index by child node (PO) to parent (RB)

        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:2, obj:1)
        cons = parent_cons[po][:, rb].clone()                           # get copy of connections matrix from po to rb (child to parent connection)
        cons[pred_sub] = cons[pred_sub] * 2                         # multipy predicate -> rb connections by 2
        self.nodes[po, TF.TD_INPUT] += torch.matmul(                # matmul outputs martix (sum(po) x 1) of values to add to current input value
            cons,                                                   # Masks connections between po[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # For each po node -> sum of act of connected rb nodes (multiplied by 2 for predicates)
            )
        
        # Inhibitory input:
        # 3). LATERAL: 3 * (if DORA_mode: PO connected to same rb / Else: POs not connected to same RBs)
        # 3a). find other PO connected to same RB     
        shared = torch.matmul(                                      # PxObject tensor, shared[i][j] > 1 if Po[i] and Po[j] share an RB, 0 o.w
            parent_cons[po][:, rb],
            self.connections[rb][:, po]
            ) 
        shared = torch.gt(shared, 0).int()                          # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
        diag_zeroes = tOps.diag_zeros(sum(po))                      # (po x po) matrix: 0 for po[i] -> po[i] connections, 1 o.w
        shared = torch.bitwise_and(shared.int(), diag_zeroes.int()).float() # remove po[i] -> po[i] connections
        # 3ai). if DORA: Other po connected to same rb / else: POs not connected to same RBs
        if as_DORA: # PO connected to same rb
            po_connections = shared                                 # shared PO
        else:       # POs not connected to same RBs
            po_connections = 1 - shared                             # non shared PO: mask[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
        # 3aii). updaet lat input: 3 * (filtered nodes connected in po_connections)
        self.nodes[po, TF.LATERAL_INPUT] -= 3 * torch.matmul(
            po_connections,
            self.nodes[po, TF.ACT]
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[po, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[po, TF.LATERAL_INPUT] -= inhib_act         # Update lat input
    # --------------------------------------------------------------
