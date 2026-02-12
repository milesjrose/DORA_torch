from numpy import r_
from .base_set import Base_Set
from ...enums import *
import torch 
from ..tokens import Tokens
from ..tokens.connections.links import Links
from .semantics import Semantics
from ..network_params import Params
from ...utils import tensor_ops as tOps
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tokens import Mapping

from logging import getLogger, INFO
log_po = getLogger("REC_PO")
log_po.setLevel(INFO)

class Recipient(Base_Set):
    """
    A class for representing the recipient set of tokens.
    """
    def __init__(self, tokens: Tokens, params: Params):
        """
        Initialise a Recipient object
        Args:
            tokens: Tokens - Tokens object.
            params: Params - The parameters for the recipient set.
            mappings: Mapping - The mappings object for the recipient set.
        """
        super().__init__(tokens, Set.RECIPIENT, params)
        self.mappings: 'Mapping' = self.tokens.mapping
    
    def update_input(self, semantics: Semantics, links: Links):
        """
        Update all input in the recipient.
        """
        self.update_input_p_parent()
        self.update_input_p_child()
        self.update_input_rb()
        self.update_input_po(semantics, links)
    
    def update_input_p_parent(self, mappings: torch.Tensor=None):
        """
        Update input for P units in parent mode
        """
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        # Exitatory: td (my Groups), bu (my RBs), mapping input.
        # Inhibitory: lateral (other P units in parent mode*lat_input_level), inhibitor.
        cache = self.glbl.cache
        con_tensor = self.tokens.connections.tensor
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
        nodes[p, TF.MAP_INPUT] += self.map_input(Type.P, mappings, Mode.PARENT) 
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

    def update_input_p_child(self, mappings: torch.Tensor=None):     # P Units in child mode - recipient:
        """
        Update input for P units in child mode
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        cache = self.glbl.cache
        con_tensor = self.tokens.connections.tensor
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
        nodes[p, TF.MAP_INPUT] += self.map_input(Type.P, mappings, Mode.CHILD) 
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
  
    def update_input_rb(self, mappings: torch.Tensor=None):                                              # RB inputs - recipient
        """
        Update input for RB units
        """
        cache = self.glbl.cache
        con_tensor = self.tokens.connections.tensor
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
        if phase_set >= 1:
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
        nodes[rb, TF.MAP_INPUT] += self.map_input(Type.RB, mappings) 
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
    
    def update_input_po(self, semantics: Semantics, links: Links, mappings: torch.Tensor=None):                                      # PO units in - recipient
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
        con_tensor = self.tokens.connections.tensor
        nodes = self.glbl.tensor
        # 1). get masks
        all_po = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.RECIPIENT})
        if not torch.any(all_po): return;
        po = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.RECIPIENT, TF.INFERRED: B.FALSE}) # non-infered pos
        rb = cache.get_type_mask(Type.RB)
        pred_sub = (nodes[po, TF.PRED] == B.TRUE)              # predicate sub mask of po nodes
        obj_sub = (nodes[po, TF.PRED] == B.FALSE)              # object sub mask of po nodes
        obj = tOps.sub_union(po, obj_sub)                           # objects
        pred = tOps.sub_union(po, pred_sub)                          # predicates
        parent_cons = torch.transpose(con_tensor, 0 , 1)             # Transpose of connections matrix, so that index by child node (PO) to parent (RB)
        child_p = cache.get_arbitrary_mask({TF.TYPE: Type.P, TF.SET: Set.RECIPIENT, TF.MODE: Mode.CHILD}) # P nodes in child mode
        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:1, obj:1)  NOTE: neither change, so removed checking for type
        if phase_set >= 1:
            delta = torch.matmul(            # matmul outputs martix (sum(po) x 1) of values to add to current input value
                parent_cons[po][:, rb].float(),                                # Masks connections between po[i] and its parent rbs
                nodes[rb, TF.ACT]                              # For each po node -> sum of act of connected rb nodes 
                )
            log_po.debug(f"2). td_input: {delta}")
            nodes[po, TF.TD_INPUT] += delta
        # 3). BU_INPUT: my_semantics [normalised by no. semantics po connects to]
        # need to get sem count, for po normalisation.
        nodes[po, TF.SEM_COUNT] = links.get_sem_count(torch.where(po)[0])
        # mask by sem_count = zero to avoid division by zero
        has_sem = nodes[:, TF.SEM_COUNT] != 0
        po_has_sem = po&has_sem
        sem_input = torch.matmul(
            sem_links[po_has_sem],
            semantics.nodes[:, SF.ACT]
        )
        nodes[po_has_sem, TF.BU_INPUT] = sem_input / nodes[po_has_sem, TF.SEM_COUNT]
        # 4). Mapping input
        nodes[po, TF.MAP_INPUT] += self.map_input(Type.PO, mappings)
        # Inhibitory input:
        # 6). LATERAL:   - If ignore_object_semantics: (po.act * lateral_input_level)
        #              a).  T: POs not connected to the same RB, that have same type (pred or obj)
        #              b).  F: POs not connected to the same RB
        #                - If asDORA:
        #              c).  T: POs connected to the same RB (po.act * 2 * lateral_input_level)
        #              c).  T: Child Ps that don't don't have the same parent RB (p.act * 3)
        #              d).  F: Ojbect updated by child p (p.act lateral_input_level)
        #               
        # child p not connect same RB // not_as_DORA: (if object: child p))
        if ignore_object_semantics: 
            # 6a). POs not connected to the same RB, that have same type (pred or obj)
            # 6ai). Preds
            non_shared = self.non_shared(pred, pred, rb, con_tensor, parent_cons)
            delta = torch.mul(
                lateral_input_level,
                torch.matmul(
                    non_shared.float(),
                    nodes[pred, TF.ACT]
                    )
                )
            nodes[pred, TF.LATERAL_INPUT] -= delta
            log_po.debug(f"6ai).po lat_input: preds<-> preds -={delta}")
            # 6aii). Objects
            non_shared = self.non_shared(obj, obj, rb, con_tensor, parent_cons)
            delta = torch.mul(
                lateral_input_level,
                torch.matmul(
                    non_shared.float(),
                    nodes[obj, TF.ACT]
                    )
                )
            nodes[obj, TF.LATERAL_INPUT] -= delta
            log_po.debug(f"6aii).po lat_input: objects<-> objects -={delta}")
        else: 
            # 6b). POs not connected to the same RB
            non_shared = self.non_shared(po, po, rb, con_tensor, parent_cons)
            # -= lateral_input_level * (po.act * non_shared)
            delta = torch.mul(
                lateral_input_level,
                torch.matmul(
                    non_shared.float(),
                    nodes[po, TF.ACT]
                )
            )
            log_po.debug(f"6b).po lat_input: POs not connected to same RB -={delta}")
            nodes[po, TF.LATERAL_INPUT] -= delta
        if as_DORA: 
            # 6c). as_DORA: child p not same parent RB & POs connect same RB
            # 6ci). POs connected to the same RB
            shared = self.shared(po, po, rb, con_tensor, parent_cons)
            # remove self connections
            diag_zeroes = tOps.diag_zeros(sum(po))
            shared = torch.bitwise_and(shared.int(), diag_zeroes.int())
            delta = torch.mul(
                2*lateral_input_level,
                torch.matmul(
                    shared.float(),
                    nodes[po, TF.ACT]
                )
            )
            nodes[po, TF.LATERAL_INPUT] -= delta
            log_po.debug(f"6ci).po lat_input: POs connected to same RB -={delta}")
            # 6cii). child p not same parent RB
            non_shared = self.non_shared(po, child_p, rb, con_tensor, parent_cons)
            delta = torch.mul(
                3,
                torch.matmul(non_shared.float(), nodes[child_p, TF.ACT])
            )
            nodes[po, TF.LATERAL_INPUT] -= delta
            log_po.debug(f"6cii).po lat_input: child p not same parent RB -={delta}")
        else: 
            # 6d). not_as_DORA: if object: child_p
            child_p_sum = nodes[child_p, TF.ACT].sum()         # Get act of all child_p
            delta_input = lateral_input_level * child_p_sum
            nodes[obj, TF.LATERAL_INPUT] -= delta_input        # Update just objects
            log_po.debug(f"6d).po lat_input: objects from child p -= {delta_input}")
        # 7). TD: non-connected RB
        if as_DORA and phase_set >= 1:
            non_connect_rb = 1 - parent_cons[po][:, rb].float()             # PO[i] -> non_connected_rb[j] = -1 // po is child so use parent_cons
            #non_connect_rb = lateral_input_level * non_connect_rb  NOTE: you might want to set multiplyer on other RB inhibition to lateral_input_level
            delta = torch.matmul(         
                non_connect_rb,
                nodes[rb, TF.ACT]
            )
            log_po.debug(f"7). td_input: {delta}")      
            nodes[po, TF.TD_INPUT] -= delta
        # 8). LATERAL: ihibitior * 10
        inhib_act = torch.mul(10, nodes[po, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        nodes[po, TF.LATERAL_INPUT] -= inhib_act               # Update lat input
    
    def non_shared(self, child1_mask, child2_mask, parent_mask, con_tensor, parent_cons):
        """ Returns a child1xchild2 tensor of 1 if child1 and child2 are not both connected to the same parent, 0 o.w """
        non_shared = 1 - self.shared(child1_mask, child2_mask, parent_mask, con_tensor, parent_cons)
        return non_shared
    
    def shared(self, child1_mask, child2_mask, parent_mask, con_tensor, parent_cons):
        """ Returns a child1xchild2 tensor of 1 if child1 and child2 are not both connected to the same parent, 0 o.w """
        c1 = child1_mask
        c2 = child2_mask
        p = parent_mask
        shared = torch.matmul(                                  # c1xc2 tensor, shared[i][j] > 1 if c1[i] and c2[j] share a parent, 0 o.w
                parent_cons[c1][:, p].float(),
                con_tensor[p][:, c2].float()                              
            ) 
        shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if c1[i] and c2[j] share a parent, 0 o.w
        return shared
    
    # =================[ MAPPING INPUT FUNCTION ]===================
    def map_input(self, type: Type, mappings: torch.Tensor, p_mode: Mode = None):                                        # Return (sum(t_mask) x 1) matrix of mapping_input for tokens in mask
        """
        Calculate mapping input for tokens in mask
        NOTE: Not implemented yet.

        Args:
            t_mask (torch.Tensor): Token mask to get mapping input for.
        Returns:
            torch.Tensor: (sum(t_mask) x 1) matrix of mapping input.
        """
        # Recipient mask
        if type is Type.P and p_mode is not None:
            r_t_mask = self.tensor_op.get_arb_mask({TF.TYPE: type, TF.MODE: p_mode})
        else:
            r_t_mask = self.tensor_op.get_mask(type)
        if mappings is None:
            return torch.zeros_like(r_t_mask[r_t_mask], dtype=tensor_type)
        # Driver mask
        d_mask = self.tokens.arb_mask({TF.SET: Set.DRIVER})
        driver = self.tokens.token_tensor.tensor[d_mask]
        # mapping weights and connections
        map_weights = mappings[:, :,MappingFields.WEIGHT][r_t_mask] 
        map_connections = (map_weights > 0).to(tensor_type)

        # 1). weight = (3*map_weight*driverToken.act)
        weight = torch.mul(                                         
            3,
            torch.matmul(
                map_weights,
                driver[:, TF.ACT]
            )
        )

        # 2). max_map = (self.max_map*driverToken.act)
        act_sum = torch.matmul(                                     
            map_connections,
            driver[:, TF.ACT]
        )
        max_map = act_sum * self.lcl[r_t_mask, TF.MAX_MAP]

        # 3). driver_max_map = (driverToken.max_map*driverToken.act)
        driver_max_map_vals = torch.mul(                                      
            driver[:, TF.MAX_MAP],
            driver[:, TF.ACT]
        )
        driver_max_map = torch.matmul(
            map_connections,
            driver_max_map_vals
        )
        # 4). map_input = (3*driver.act*mapping_weight) 
        #                   - max(mapping_weight_driver_unit) 
        #                   - max(own_mapping_weight)
        log_po.debug(f"weight: {weight}")
        log_po.debug(f"max_map: {max_map}")
        log_po.debug(f"driver_max_map: {driver_max_map}")
        map_input = (weight - max_map - driver_max_map)                   
        return map_input                   
    # --------------------------------------------------------------