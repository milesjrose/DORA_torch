from ..tokens.tensor.token_tensor import Token_Tensor
from ..network_params import Params
from ...enums import *
from .base_set import Base_Set
import torch
from ...utils import tensor_ops as tOps
from ..tokens.connections.links import Links
from .semantics import Semantics

class Memory(Base_Set):
    """
    A class for representing a memory of tokens.
    """
    def __init__(self, tokens: Token_Tensor, token_set: Set, params: Params):
        super().__init__(tokens, token_set, params)
        
    def update_input(self, semantics: Semantics, links: Links):
        """
        Update all input in the recipient.
        """
        self.update_input_p_parent()
        self.update_input_p_child()
        self.update_input_rb()
        self.update_input_po(semantics, links)
    
    def update_input_p_parent(self):
        """
        Update input for P units in parent mode
        """
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (my Groups), bu (my RBs), mapping input.
        # Inhibitory: lateral (other P units in parent mode*lat_input_level), inhibitor.
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        # 1). get masks
        p = cache.get_arbitrary_mask({TF.TYPE: Type.P, TF.SET: Set.RECIPIENT, TF.MODE: Mode.PARENT})
        if not torch.any(p): return;
        group = cache.get_type_mask(Type.GROUP)  # Boolean mask for GROUP nodes
        rb = cache.get_type_mask(Type.RB)        # Boolean mask for RB nodes
        # Exitatory input:
        # 2). TD_INPUT: my_groups
        if phase_set >= 1:
            nodes[p, TF.TD_INPUT] += torch.matmul(   # matmul outputs martix (sum(p) x 1) of values to add to current input value
                con_tensor[p][:, group].float(),     # Masks connections between p[i] and its groups
                nodes[group, TF.ACT]                 # each p node -> sum of act of connected group nodes
                )
        # 3). BU_INPUT: my_RBs
        nodes[p, TF.BU_INPUT] += torch.matmul(      # matmul outputs martix (sum(p) x 1) of values to add to current input value
            con_tensor[p][:, rb].float(),           # Masks connections between p[i] and its rbs
            nodes[rb, TF.ACT]                       # Each p node -> sum of act of connected rb nodes
            )  
        # 4). Mapping input
        nodes[p, TF.MAP_INPUT] += self.map_input(p) 
        # Inhibitory input:
        # 5). LATERAL_INPUT: (lat_input_level * other parent p nodes in recipient), inhibitor
        # 5a). Tensor to connect p nodes to each other
        diag_zeroes = tOps.diag_zeros(sum(p)).float()  # adj matrix connection connecting parent ps to all but themselves
        # 5b). 3 * other parent p nodes in driver
        nodes[p, TF.LATERAL_INPUT] -= torch.mul(
            lateral_input_level,
            torch.matmul(
                diag_zeroes,                  # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
                nodes[p, TF.ACT]              # Each parent p node -> (sum of all other parent p nodes)
            )
        )
        # 5c). Inhibitor
        inhib_input = nodes[p, TF.INHIBITOR_ACT]
        nodes[p, TF.LATERAL_INPUT] -= torch.mul(10, inhib_input)

    def update_input_p_child(self):     # P Units in child mode - recipient:
        """
        Update input for P units in child mode
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        # Exitatory: td (RBs above me), mapping input, bu (my semantics [currently not implmented]).
        # Inhibitory: lateral (other Ps in child, and, if in DORA mode, other PO objects not connected to my RB, and 3*PO connected to my RB), inhibitor.
        # 1). get masks
        p = cache.get_arbitrary_mask({TF.TYPE: Type.P, TF.SET: Set.RECIPIENT, TF.MODE: Mode.CHILD})
        if not torch.any(p): return;
        rb = cache.get_type_mask(Type.RB)                           # Boolean mask for RB nodes
        po = cache.get_type_mask(Type.PO)
        obj = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.RECIPIENT, TF.PRED: B.FALSE}) # get object mask
        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        """ NOTE: Says this should be input in comments, but not implemented in code.
        # 2a). groups
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        """
        # 2). parent_rbs
        if phase_set >= 1:
            t_con = torch.transpose(con_tensor, 0, 1)  # transpose, so gives child -> parent connections
            nodes[p, TF.TD_INPUT] += torch.matmul(     # matmul outputs matrix (sum(p) x 1) of values to add to current input value
                t_con[p][:, rb].float(),               # Masks connections between p[i] and its rbs
                nodes[rb, TF.ACT]                      # For each p node -> sum of act of connected parent rb nodes
                )
        # 3). BU_INPUT: Semantics                                   NOTE: Not implemented yet
        # 4). Mapping input
        nodes[p, TF.MAP_INPUT] += self.map_input(p) 
        # Inhibitory input:
        # 5). LATERAL_INPUT: (Other child p), (if DORA_mode: POs not connected to same RBs / Else: All Objects)
        # 5a). other p in child mode
        diag_zeroes = tOps.diag_zeros(sum(p)).float()  # PxP, child p -> all other child p
        nodes[p, TF.LATERAL_INPUT] -= torch.mul(
            lateral_input_level,
            torch.matmul(
                diag_zeroes,                           # PxP, child p -> all other child p
                nodes[p, TF.ACT]                       # Px1, act of each p
            )
        )
        # 5b). if not as_DORA: Object acts
        if not as_DORA:
            obj_sum = nodes[obj, TF.ACT].sum()       # sum of all object acts
            ones = torch.ones((sum(p), 1))           # Px1, ones tensor
            sum_tensor = torch.mul(ones, obj_sum)    # Px1, sum of object acts for each p
            nodes[p, TF.LATERAL_INPUT] -= sum_tensor.squeeze(1)  # Update lateral input (squeeze to match shape)
        # 5c). Else(asDORA): POs not connected to same RBs
        else:
            if torch.any(po) and torch.any(rb):
                # 5ci). Find POs not connected to same RBs              NOTE: Should this use my parent RBs?
                shared = torch.matmul(con_tensor[p][:, rb].float(), con_tensor[rb][:, po].float())  # PxPO, shared[i][j] > 1 if p[i], po[j] share RB, 0 o.w
                shared = torch.gt(shared, 0).int()          # shared[i][j] = 1 if p[i], po[j] share RB, 0 o.w
                non_shared = 1 - shared                     # non_shared[i][j] = 0 if p[i], po[j] share RB, 1 o.w
                # 5cii). update input using non shared POs
                nodes[p, TF.LATERAL_INPUT] -= torch.matmul(
                    non_shared.float(),         # PxPO, non shared POs for each p
                    nodes[po, TF.ACT]           # POx1, act of each PO
                )
  
    def update_input_rb(self):                                              # RB inputs - recipient
        """
        Update input for RB units
        """
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (my P units), bu (my pred and obj POs, and my child Ps), mapping input.
        # Inhibitory: lateral (other RBs*3), inhbitor.
        # 1). get masks
        rb = cache.get_arbitrary_mask({TF.TYPE: Type.RB, TF.SET: Set.RECIPIENT})
        if not torch.any(rb): return;
        po = cache.get_type_mask(Type.PO)
        p = cache.get_type_mask(Type.P)

        # Exitatory input:
        # 2). TD_INPUT: my_parent_p
        if phase_set > 1:
            t_con = torch.transpose(con_tensor, 0, 1)               # Connnections: Parent -> child, take transpose to get list of parents instead
            nodes[rb, TF.TD_INPUT] += torch.matmul(            # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                t_con[rb][:, p].float(),                                       # Masks connections between rb[i] and its ps
                nodes[p, TF.ACT]                               # For each rb node -> sum of act of connected p nodes
                )
        # 3). BU_INPUT: my_po, my_child_p                           # NOTE: Old function explicitly took myPred[0].act etc. as there should only be one pred/child/etc. This version sums all connections, so if rb mistakenly connected to multiple of a node type it will not give expected output.
        po_p = torch.bitwise_or(po, p)                              # Get mask of both pos and ps
        nodes[rb, TF.BU_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            con_tensor[rb][:, po_p].float(),                             # Masks connections between rb[i] and its po and child p nodes
            nodes[po_p, TF.ACT]                                # For each rb node -> sum of act of connected po and child p nodes
            )
        # 4). Mapping input
        nodes[rb, TF.MAP_INPUT] += self.map_input(rb) 
        # Inhibitory input:
        # 5). LATERAL: (other RBs*lat_input_level), inhibitor*10
        # 5a). (other RBs*lat_input_level)
        diag_zeroes = tOps.diag_zeros(sum(rb))                      # Connects each rb to every other rb, but not themself
        nodes[rb, TF.LATERAL_INPUT] -= torch.mul(
            lateral_input_level, 
            torch.matmul(                                           # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                diag_zeroes,                                        # Connect rb[i] to every rb except rb[i]
                nodes[rb, TF.ACT]                              # For each rb node -> sum of act of other rb nodes
            )
        )
        # 5b). ihibitior * 10
        inhib_act = torch.mul(10, nodes[rb, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        nodes[rb, TF.LATERAL_INPUT] -= inhib_act       # Update lat inhibition
    
    def update_input_po(self, semantics: Semantics, links: Links):                                      # PO units in - recipient
        """
        Update input for PO units
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        ignore_object_semantics = self.params.ignore_object_semantics
        sem_links = links.adj_matrix
        
        # NOTE: Currently inferred nodes not updated so excluded from po mask. Inferred nodes do update other PO nodes - so all_po used for updating lat_input.
        # Exitatory: td (my RBs), bu (my semantics/sem_count[for normalisation]), mapping input.
        # Inhibitory: lateral (PO nodes s.t(asDORA&sameRB or [if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB)]), (as_DORA: child p not connect same RB // not_as_DORA: (if object: child p)), inhibitor
        # Inhibitory: td (if asDORA: not-connected RB nodes)
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        # 1). get masks
        all_po = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.RECIPIENT})
        if not torch.any(all_po): return;
        po = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.RECIPIENT, TF.INFERRED: B.FALSE}) # non-infered pos
        rb = cache.get_type_mask(Type.RB)
        pred_sub = (nodes[po, TF.PRED] == B.TRUE)              # predicate sub mask of po nodes
        obj_sub = (nodes[po, TF.PRED] == B.FALSE)              # object sub mask of po nodes
        obj = tOps.sub_union(po, obj_sub)                           # objects
        parent_cons = torch.transpose(con_tensor, 0 , 1)             # Transpose of connections matrix, so that index by child node (PO) to parent (RB)
        child_p = cache.get_arbitrary_mask({TF.TYPE: Type.P, TF.SET: Set.RECIPIENT, TF.MODE: Mode.CHILD}) # P nodes in child mode
        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:1, obj:1)  NOTE: neither change, so removed checking for type
        if phase_set > 1:
            nodes[po, TF.TD_INPUT] += torch.matmul(            # matmul outputs martix (sum(po) x 1) of values to add to current input value
                parent_cons[po][:, rb].float(),                                # Masks connections between po[i] and its parent rbs
                nodes[rb, TF.ACT]                              # For each po node -> sum of act of connected rb nodes 
                )
        # 3). BU_INPUT: my_semantics [normalised by no. semantics po connects to]
        sem_input = torch.matmul(
            sem_links[po],
            semantics.nodes[:, SF.ACT]
        )
        # need to get sem count, for po normalisation.
        nodes[po, TF.SEM_COUNT] = links.get_sem_count(torch.where(po)[0])
        # mask by sem_count = zero to avoid division by zero
        has_sem = nodes[:, TF.SEM_COUNT] != 0
        nodes[po&has_sem, TF.BU_INPUT] += sem_input / nodes[po&has_sem, TF.SEM_COUNT]
        # 4). Mapping input
        nodes[po, TF.MAP_INPUT] += self.map_input(po) 
        # Inhibitory input:
        # 5). LATERAL: PO nodes s.t(asDORA&sameRB or [if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB)])
        # 5a). find other PO connected to same RB
        shared = torch.matmul(                                      # POxAll_PO tensor, shared[i][j] > 1 if po[i] and all_po[j] share an RB, 0 o.w
            parent_cons[po][:, rb].float(),
            con_tensor[rb][:, all_po].float()                         # NOTE: connecting from po -> all_po, as non inferred po not updated, but used in updating inferred
            ) 
        shared = torch.gt(shared, 0).int()                          # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
        po_submask = po[all_po]                                     # Needs to be mask size of all_po for use in diag_zeros
        diag_zeroes = tOps.diag_zeros(sum(all_po))[po_submask]      # (po x all_po) matrix: 0 for po[i] -> all_po[i] connections, 1 o.w NOTE: create (all_po x all_po) diagonal, then mask by [po, :] to get (po x all_po) tensor.
        shared = torch.bitwise_and(shared.int(), diag_zeroes.int()) # remove po[i] -> po[i] connections
        # 5b). asDORA: sameRB * (2*lateral_input_level) // not_as_DORA (if ingore_sem: not(sameRB)&same(predOrObj) / else: not(sameRB))
        if as_DORA: # 5bi). PO connected to same rb
            po_connections = shared.float()                         # shared PO
            nodes[po, TF.LATERAL_INPUT] -= torch.mul(
                2*lateral_input_level,                              # NOTE: the 2 here is a place-holder for a multiplier for within RB inhibition (right now it is a bit higher than between RB inhibition).
                torch.matmul(
                    po_connections,                                 # po x all_po (connections)
                    nodes[all_po, TF.ACT]                      # all_po x  1 (acts)
                )
            )
        else: # 5bii). POs not connected to same RBs
            po_connections = (1 - shared).float()                             # non shared PO: mask[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
            # if ignore_sem: Only connect nodes of same type
            if ignore_object_semantics:
                pred_indices = torch.where(pred_sub)[0]
                obj_indices = torch.where(obj_sub)[0]
                if pred_indices.shape[0] > 0 and obj_indices.shape[0] > 0:
                    po_connections[pred_indices[:, None], obj_indices] = 0.0 # Remove pred -> obj connections
                    po_connections[obj_indices[:, None], pred_indices] = 0.0 # Remove obj -> pred connections
            nodes[po, TF.LATERAL_INPUT] -= torch.matmul(
                po_connections,                                     # po x all_po (connections)
                nodes[all_po, TF.ACT]                          # all_po x  1 (acts)
            )
        # 6). LATERAL: (as_DORA: child p not connect same RB // not_as_DORA: (if object: child p))
        if as_DORA: # 6a). as_DORA: child p not connect same RB
            shared = torch.matmul(                                  # POxChild_P tensor, shared[i][j] > 1 if po[i] and child_p[j] share an RB, 0 o.w
                parent_cons[po][:, rb].float(),
                con_tensor[rb][:, child_p].float()                              
            ) 
            shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if p[i] and child_p[j] share an RB, 0 o.w
            non_shared = 1 - shared                                 # now maps po to (child_p not connected to same rb)
            nodes[po, TF.LATERAL_INPUT] -= torch.mul(
                3,
                torch.matmul(
                    non_shared.float(),                             # po x child_p
                    nodes[child_p, TF.ACT]                     # child_p x 1
                )
            )
        else: # 6b). not_as_DORA: if object: child_p
            child_p_sum = nodes[child_p, TF.ACT].sum()         # Get act of all child_p
            delta_input = lateral_input_level * child_p_sum
            nodes[obj, TF.LATERAL_INPUT] -= delta_input        # Update just objects
        # 7). TD: non-connected RB
        if as_DORA:
            non_connect_rb = 1 - parent_cons[po][:, rb].float()             # PO[i] -> non_connected_rb[j] = -1 // po is child so use parent_cons
            #non_connect_rb = lateral_input_level * non_connect_rb  NOTE: you might want to set multiplyer on other RB inhibition to lateral_input_level
            nodes[po, TF.TD_INPUT] += torch.matmul(            # "+=" here as non_connect_rb = -1 for po->rb
                non_connect_rb,
                nodes[rb, TF.ACT]
            )
        # 8). LATERAL: ihibitior * 10
        inhib_act = torch.mul(10, nodes[po, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        nodes[po, TF.LATERAL_INPUT] -= inhib_act               # Update lat input
    
    def map_input(self, t_mask):
        """
        No mapping for memory set, uses recipient map_input function so just return 0.
        """
        return 0