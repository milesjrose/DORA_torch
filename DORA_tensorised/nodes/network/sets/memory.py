# nodes/network/sets/memory.py
# Represents the memory set of tokens.

import torch

from nodes.enums import *
from nodes.utils import tensorOps as tOps

from ..network_params import Params
from ..connections import Links

from .base_set import Base_Set

class Memory(Base_Set):
    """
    A class for representing a memory of tokens.
    """
    def __init__(self, floatTensor, connections, links: Links, mappings, IDs: dict[int, int], names: dict[int, str] = {}, params: Params = None):
        super().__init__(floatTensor, connections, links, IDs, names, params)
        self.mappings = mappings
        self.token_set = Set.MEMORY

    # =========[ USES RECIPIENT UPDATE INPUT FUNCTIONS ]===========
    def update_input(self):                # Update all input in recipient
        """
        Update all input in recipient
        """
        self.update_input_p_parent()
        self.update_input_p_child()
        self.update_input_rb()
        self.update_input_po()

    def update_input_p_parent(self):    # P units in parent mode - recipient
        """
        Update input for P units in parent mode
        """
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (my Groups), bu (my RBs), mapping input.
        # Inhibitory: lateral (other P units in parent mode*lat_input_level), inhibitor.
        # 1). get masks
        p = self.get_mask(Type.P)                                   # Boolean mask for P nodes
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.PARENT)   # Boolean mask for Parent P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes
        # Exitatory input:
        # 2). TD_INPUT: my_groups
        if phase_set >= 1:
            self.nodes[p, TF.TD_INPUT] += torch.matmul(             # matmul outputs martix (sum(p) x 1) of values to add to current input value
                self.connections[p, group],                         # Masks connections between p[i] and its groups
                self.nodes[group, TF.ACT]                           # each p node -> sum of act of connected group nodes
                )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, TF.BU_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                                # Masks connections between p[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # Each p node -> sum of act of connected rb nodes
            )  
        # 4). Mapping input
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p) 
        # Inhibitory input:
        # 5). LATERAL_INPUT: (lat_input_level * other parent p nodes in recipient), inhibitor
        # 5a). Tensor to connect p nodes to each other
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting parent ps to all but themselves
        # 5b). 3 * other parent p nodes in driver
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(
            lateral_input_level,
            torch.matmul(
                diag_zeroes,                                        # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
                self.nodes[p, TF.ACT]                               # Each parent p node -> (sum of all other parent p nodes)
            )
        )
        # 5c). Inhibitor
        inhib_input = self.nodes[p, TF.INHIBITOR_ACT]
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(10, inhib_input)
    
    def update_input_p_child(self):     # P Units in child mode - recipient:
        """
        Update input for P units in child mode
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (RBs above me), mapping input, bu (my semantics [currently not implmented]).
        # Inhibitory: lateral (other Ps in child, and, if in DORA mode, other PO objects not connected to my RB, and 3*PO connected to my RB), inhibitor.
        # 1). get masks
        p = self.get_mask(Type.P)
        p = tOps.refine_mask(self.nodes, p, TF.MODE, Mode.CHILD)    # Boolean mask for Child P nodes
        group = self.get_mask(Type.GROUP)                           # Boolean mask for GROUP nodes
        rb = self.get_mask(Type.RB)                                 # Boolean mask for RB nodes
        po = self.get_mask(Type.PO)
        obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)    # get object mask
        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        """ NOTE: Says this should be input in comments, but not in code?
        # 2a). groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        """
        # 2b). parent_rbs
        if phase_set >= 1:
            t_con = torch.transpose(self.connections)               # transpose, so gives child -> parent connections
            self.nodes[p, TF.TD_INPUT] += torch.matmul(             # matmul outputs matrix (sum(p) x 1) of values to add to current input value
                t_con[p, rb],                                       # Masks connections between p[i] and its rbs
                self.nodes[rb, TF.ACT]                              # For each p node -> sum of act of connected parent rb nodes
                )
        # 3). BU_INPUT: Semantics                                     NOTE: not implenmented yet
        # 4). Mapping input
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p) 
        # Inhibitory input:
        # 5). LATERAL_INPUT: (Other p in child mode), (if DORA_mode: POs not connected to same RBs / Else: PO acts)
        # 5a). other p in child mode
        diag_zeroes = tOps.diag_zeros(sum(p))                       # adj matrix connection connecting child ps to all but themselves
        self.nodes[p, TF.LATERAL_INPUT] -= torch.mul(
            lateral_input_level,
            torch.matmul(
                diag_zeroes,                                        # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
                self.nodes[p, TF.ACT]                               # Each parent p node -> (sum of all other parent p nodes)
            )
        )
        # 3b). if not as_DORA: Object acts
        if not as_DORA:
            ones = torch.ones((sum(p), sum(obj)))                   # tensor connecting every p to every object
            self.nodes[p, TF.LATERAL_INPUT] -= torch.matmul(    
                ones,                                               # connects all p to all object
                self.nodes[obj, TF.ACT]                             # Each  p node -> sum of all object acts
            )
        # 3c). Else(asDORA): Objects not connected to same RBs
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
   
    def update_input_rb(self):                                              # RB inputs - recipient
        """
        Update input for RB units
        """
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (my P units), bu (my pred and obj POs, and my child Ps), mapping input.
        # Inhibitory: lateral (other RBs*3), inhbitor.
        # 1). get masks
        rb = self.get_mask(Type.RB)
        po = self.get_mask(Type.PO)
        p = self.get_mask(Type.P)

        # Exitatory input:
        # 2). TD_INPUT: my_parent_p
        if phase_set > 1:
            t_con = torch.transpose(self.connections)               # Connnections: Parent -> child, take transpose to get list of parents instead
            self.nodes[rb, TF.TD_INPUT] += torch.matmul(            # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                t_con[rb, p],                                       # Masks connections between rb[i] and its ps
                self.nodes[p, TF.ACT]                               # For each rb node -> sum of act of connected p nodes
                )
        # 3). BU_INPUT: my_po, my_child_p                           # NOTE: Old function explicitly took myPred[0].act etc. as there should only be one pred/child/etc. This version sums all connections, so if rb mistakenly connected to multiple of a node type it will not give expected output.
        po_p = torch.bitwise_or(po, p)                              # Get mask of both pos and ps
        self.nodes[rb, TF.BU_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            self.connections[rb, po_p],                             # Masks connections between rb[i] and its po and child p nodes
            self.nodes[po_p, TF.ACT]                                # For each rb node -> sum of act of connected po and child p nodes
            )
        # 4). Mapping input
        self.nodes[rb, TF.MAP_INPUT] += self.map_input(rb) 
        # Inhibitory input:
        # 5). LATERAL: (other RBs*lat_input_level), inhibitor*10
        # 5a). (other RBs*lat_input_level)
        diag_zeroes = tOps.diag_zeros(sum(rb))                      # Connects each rb to every other rb, but not themself
        self.nodes[rb, TF.LATERAL_INPUT_INPUT] -= torch.mul(
            lateral_input_level, 
            torch.matmul(                                           # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                diag_zeroes,                                        # Connect rb[i] to every rb except rb[i]
                self.nodes[rb, TF.ACT]                              # For each rb node -> sum of act of other rb nodes
            )
        )
        # 5b). ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[rb, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[rb, TF.LATERAL_INPUT_INPUT] -= inhib_act         # Update lat inhibition
    
    def update_input_po(self):                                      # PO units in - recipient
        """
        Update input for PO units
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        ignore_object_semantics = self.params.ignore_object_semantics
        semantics = self.links.semantics
        # NOTE: Currently inferred nodes not updated so excluded from po mask. Inferred nodes do update other PO nodes - so all_po used for updating lat_input.
        # Exitatory: td (my RBs), bu (my semantics/sem_count[for normalisation]), mapping input.
        # Inhibitory: lateral (PO nodes s.t(asDORA&sameRB or [if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB)]), (as_DORA: child p not connect same RB // not_as_DORA: (if object: child p)), inhibitor
        # Inhibitory: td (if asDORA: not-connected RB nodes)
        # 1). get masks
        rb = self.get_mask(Type.RB)
        all_po = self.get_mask(Type.PO)                             # All POs
        po = tOps.refine_mask(self.nodes, all_po, TF.INFERRED, B.FALSE) # Non inferred POs
        pred_sub = (self.nodes[po, TF.PRED] == B.TRUE)              # predicate sub mask of po nodes
        obj_sub = (self.nodes[po, TF.PRED] == B.FALSE)              # object sub mask of po nodes
        obj = tOps.sub_union(po, obj_sub)                           # objects
        parent_cons = torch.transpose(self.connections)             # Transpose of connections matrix, so that index by child node (PO) to parent (RB)
        child_p = tOps.refine_mask(self.nodes, self.get_mask[Type.P], TF.MODE, Mode.CHILD) # P nodes in child mode
        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:1, obj:1)  NOTE: neither change, so removed checking for type
        if phase_set > 1:
            self.nodes[po, TF.TD_INPUT] += torch.matmul(            # matmul outputs martix (sum(po) x 1) of values to add to current input value
                parent_cons[po, rb],                                # Masks connections between po[i] and its parent rbs
                self.nodes[rb, TF.ACT]                              # For each po node -> sum of act of connected rb nodes 
                )
        # 3). BU_INPUT: my_semantics [normalised by no. semantics po connects to]
        sem_input = torch.matmul(
            self.links[po],
            semantics.nodes[:, SF.ACT]
        )
        self.nodes[po, TF.BU_INPUT] += sem_input / self.nodes[po, TF.SEM_COUNT]
        # 4). Mapping input
        self.nodes[po, TF.MAP_INPUT] += self.map_input(po) 
        # Inhibitory input:
        # 5). LATERAL: PO nodes s.t(asDORA&sameRB or [if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB)])
        # 5a). find other PO connected to same RB
        shared = torch.matmul(                                      # POxAll_PO tensor, shared[i][j] > 1 if po[i] and all_po[j] share an RB, 0 o.w
            parent_cons[po, rb],
            self.connections[rb, all_po]                            # NOTE: connecting from po -> all_po, as non inferred po not updated, but used in updating inferred
            ) 
        shared = torch.gt(shared, 0).int()                          # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
        diag_zeroes = tOps.diag_zeros(sum(all_po))[po]              # (po x all_po) matrix: 0 for po[i] -> all_po[i] connections, 1 o.w NOTE: create (all_po x all_po) diagonal, then mask by [po, :] to get (po x all_po) tensor.
        shared = torch.bitwise_and(shared, diag_zeroes)             # remove po[i] -> po[i] connections
        # 5b). asDORA: sameRB * (2*lateral_input_level) // not_as_DORA (if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB))
        if as_DORA: # 5bi). PO connected to same rb
            po_connections = shared                                 # shared PO
            self.nodes[po, TF.LATERAL_INPUT] -= torch.mul(
                2*lateral_input_level,                              # NOTE: the 2 here is a place-holder for a multiplier for within RB inhibition (right now it is a bit higher than between RB inhibition).
                torch.matmul(
                    po_connections,                                 # po x all_po (connections)
                    self.nodes[all_po, TF.ACT]                      # all_po x  1 (acts)
                )
            )
        else: # 5bii). POs not connected to same RBs
            po_connections = 1 - shared                             # non shared PO: mask[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
            # if ignore_sem: Only connect nodes of same type
            if ignore_object_semantics:
                po_connections[pred_sub][obj_sub] = 0               # Remove pred -> obj connections
                po_connections[obj_sub][pred_sub] = 0               # Remove obj -> pred connections
            self.nodes[po, TF.LATERAL_INPUT] -= torch.matmul(
                po_connections,                                     # po x all_po (connections)
                self.nodes[all_po, TF.ACT]                          # all_po x  1 (acts)
            )
        # 6). LATERAL: (as_DORA: child p not connect same RB // not_as_DORA: (if object: child p))
        if as_DORA: # 6a). as_DORA: child p not connect same RB
            shared = torch.matmul(                                  # POxChild_P tensor, shared[i][j] > 1 if po[i] and child_p[j] share an RB, 0 o.w
            parent_cons[po, rb],
            self.connections[rb, child_p]                              
            ) 
            shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if p[i] and child_p[j] share an RB, 0 o.w
            non_shared = 1 - shared                                 # now maps po to (child_p not connected to same rb)
            self.nodes[po, TF.LATERAL_INPUT] -= torch.mul(
                3,
                torch.matmul(
                    non_shared,                                     # po x child_p
                    self.nodes[child_p, TF.ACT]                     # child_p x 1
                )
            )
        else: # 6b). not_as_DORA: if object: child_p
            child_p_sum = self.nodes[child_p, TF.ACT].sum()         # Get act of all child_p
            delta_input = lateral_input_level * child_p_sum
            self.nodes[obj, TF.LATERAL_INPUT] -= delta_input        # Update just objects
        # 7). TD: non-connected RB
        if as_DORA:
            non_connect_rb = 1 - parent_cons[po, rb]                # PO[i] -> non_connected_rb[j] = -1 // po is child so use parent_cons
            #non_connect_rb = lateral_input_level * non_connect_rb  NOTE: you might want to set multiplyer on other RB inhibition to lateral_input_level
            self.nodes[po, TF.TD_INPUT] += torch.matmul(            # "+=" here as non_connect_rb = -1 for po->rb
                non_connect_rb,
                self.nodes[rb, TF.ACT]
            )
        # 8). LATERAL: ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[po, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[po, TF.LATERAL_INPUT_INPUT] -= inhib_act         # Update lat input
        pass
    # --------------------------------------------------------------
    
    # =================[ MAPPING INPUT FUNCTION ]===================
    def map_input(self, t_mask):        # Return (sum(t_mask) x 1) matrix of mapping_input for tokens in mask
        """
        Calculate mapping input for tokens in mask

        Args:
            t_mask (torch.Tensor): A mask of tokens to calculate mapping input for  
        Returns:
            torch.Tensor: A (sum(t_mask) x 1) matrix of mapping input for tokens in mask
        """
        driver = self.mappings.driver
        pmap_weights = self.mappings.weights()[t_mask] 
        pmap_connections = self.mappings.connections()[t_mask]

        # 1). weight = (3*map_weight*driverToken.act)
        weight = torch.mul(                                         
            3,
            torch.matmul(
                pmap_weights,
                driver.nodes[:, TF.ACT]
            )
        )

        # 2). pmax_map = (self.max_map*driverToken.act)
        act_sum = torch.matmul(                                     
            pmap_connections,
            driver.nodes[:, TF.ACT]
        )
        tmax_map = act_sum * self.nodes[t_mask, TF.MAX_MAP]

        # 3). dmax_map = (driverToken.max_map*driverToken.act)
        dmax_map = torch.mult(                                      
            driver.nodes[:, TF.MAX_MAP],
            driver.nodes[:, TF.ACT]
        )
        dmax_map = torch.mult(
            pmap_connections,
            dmax_map
        )
        # 4). map_input = (3*driver.act*mapping_weight) - max(mapping_weight_driver_unit) - max(own_mapping_weight)
        return (weight - tmax_map - dmax_map)                       
    # --------------------------------------------------------------
