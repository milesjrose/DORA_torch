from .base_set import Base_Set
from ...enums import *
import torch 
from logging import getLogger
logger = getLogger(__name__)
from ..tokens.tensor.token_tensor import Token_Tensor
from ..network_params import Params
from ...utils import tensor_ops as tOps

class Driver(Base_Set):
    """
    A class for representing the driver set of tokens.
    """
    def __init__(self, tokens: Token_Tensor, params: Params):
        super().__init__(tokens, Set.DRIVER, params)
        """
        Initialize the Driver object.
        Args:
            tokens: Token_Tensor - The tokens in the driver set.
            mappings: Mappings - The mappings object for the driver set.
            params: Params - The parameters for the driver set.
        """
    
    def check_local_inhibitor(self):
        """
        Check if any PO.inhibitor_act == 1.0 in the driver set.
        """
        po = self.get_mask(Type.PO)
        return torch.any(self.lcl[po, TF.INHIBITOR_ACT] == 1.0)
    
    def check_global_inhibitor(self):
        """
        Check if any RB.inhibitor_act == 1.0 in the driver set.
        """
        rb = self.get_mask(Type.RB)
        return torch.any(self.lcl[rb, TF.INHIBITOR_ACT] == 1.0)
    
    # ====================[ UPDATE INPUT FUNCTIONS ]===================
    def update_input(self): 
        """Update all input in driver"""
        self.update_input_p_parent()
        self.update_input_p_child()
        self.update_input_rb()
        self.update_input_po()

    def update_input_p_parent(self):
        """Update input in driver for P units in parent mode"""
        con_tensor = self.glbl.connections.connections
        cache = self.glbl.cache
        nodes = self.glbl.tensor
        # Exitatory: td (my Groups) / bu (my RBs)
        # Inhibitory: lateral (other P units in parent mode*3), inhibitor.
        # 1). get masks
        p = cache.get_arbitrary_mask(                # Boolean mask for Driver Parent P nodes
            {TF.TYPE: Type.P, 
            TF.MODE: Mode.PARENT, 
            TF.SET: Set.DRIVER}
            )
        if not torch.any(p): return;
        group = cache.get_type_mask(Type.GROUP)     # Boolean mask for GROUP nodes
        rb = cache.get_type_mask(Type.RB)           # Boolean mask for RB nodes

        # Exitatory input:
        # 2). TD_INPUT: my_groups
        nodes[p, TF.TD_INPUT] += torch.matmul(      # matmul outputs martix (sum(p) x 1) of values to add to current input value
            con_tensor[p][:, group].float(),         # Masks connections between p[i] and its groups
            nodes[group, TF.ACT]                    # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        nodes[p, TF.BU_INPUT] += torch.matmul(      # matmul outputs martix (sum(p) x 1) of values to add to current input value
            con_tensor[p][:, rb].float(),            # Masks connections between p[i] and its rbs
            nodes[rb, TF.ACT]                       # Each p node -> sum of act of connected rb nodes
            )  
        
        # Inhibitory input:
        # 4). LATERAL_INPUT: (3 * other parent p nodes in driver), inhibitor
        # 4a). Create tensor mask of parent p nodes, and a tensor to connect p nodes to each other
        local_p = self.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.PARENT})
        diag_zeroes = tOps.diag_zeros(sum(local_p)) # adj matrix connection connecting parent ps to all but themselves
        # 4b). 3 * other parent p nodes in driver
        change = torch.mul(
            3,
            torch.matmul(
                diag_zeroes,                        # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
                self.lcl[local_p, TF.ACT]           # Each parent p node -> 3*(sum of all other parent p nodes)
            )
        )
        self.lcl[local_p, TF.LATERAL_INPUT] -= change

    def update_input_p_child(self):
        """Update input in driver for P units in child mode"""
        cache = self.glbl.cache
        nodes = self.glbl.tensor
        con_tensor = self.glbl.connections.connections
        as_DORA = self.params.as_DORA
        # Exitatory: td (my parent RBs), (if phase_set>1: my groups)
        # Inhibitory: lateral (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 1). get masks
        p = cache.get_arbitrary_mask({              # Boolean mask for Child P nodes
            TF.TYPE: Type.P,
            TF.MODE: Mode.CHILD,
            TF.SET: Set.DRIVER
        })
        if not torch.any(p): return;
        group = cache.get_type_mask(Type.GROUP)     # Boolean mask for GROUP nodes
        rb = cache.get_type_mask(Type.RB)           # Boolean mask for RB nodes
        obj = cache.get_arbitrary_mask({
            TF.TYPE: Type.PO,
            TF.PRED: B.FALSE
        })

        # Exitatory input:
        # 2). TD_INPUT: my_groups and my_parent_RBs
        # 2a). groups
        nodes[p, TF.TD_INPUT] += torch.matmul(    # matmul outputs martix (sum(p) x 1) of values to add to current input value
            con_tensor[p][:, group].float(),                # Masks connections between p[i] and its groups
            nodes[group, TF.ACT]                  # For each p node -> sum of act of connected group nodes
            )
        # 2b). parent_rbs
        t_con = torch.transpose(con_tensor, 0 , 1)   # transpose, so gives child -> parent connections
        nodes[p, TF.TD_INPUT] += torch.matmul(       # matmul outputs matrix (sum(p) x 1) of values to add to current input value
            t_con[p][:, rb].float(),                           # Masks connections between p[i] and its rbs
            nodes[rb, TF.ACT]                        # For each p node -> sum of act of connected parent rb nodes
            )
        
        # Inhibitory input: NOTE: Inhibitory input only comes from local nodes.
        # 3). LATERAL_INPUT: (Other p in child mode), (if DORA_mode: PO acts / Else: POs not connected to same RBs)
        # 3a). other p in child mode 
        local_p = self.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
        diag_zeroes = tOps.diag_zeros(sum(local_p))  # adj matrix connection connecting child ps to all but themselves
        change = torch.matmul(
            diag_zeroes,                             # Tensor size sum(p)xsum(p), to ignore p[i] -> p[i] connections
            self.lcl[local_p, TF.ACT]                # Each child p node -> 3*(sum of all other parent p nodes)
        )
        self.lcl[local_p, TF.LATERAL_INPUT] -= change
        # 3b). if as_DORA: Object acts
        local_obj = self.tnop.get_arb_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE})
        if as_DORA:
            ones = torch.ones((sum(local_p), sum(local_obj))) # tensor connecting every p to every object
            change = torch.matmul(    
                ones,                                  # connects all p to all object
                self.lcl[local_obj, TF.ACT]               # Each  p node -> sum of all object acts
            )
            self.lcl[local_p, TF.LATERAL_INPUT] -= change
        # 3c). Else: Objects not connected to same RBs
        else:
            # NOTE: We are using only P and Objects in the driver set, but they can share RBs with other sets I think? Maybe only possible to have a parent in another set?
            set_obj = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE, TF.SET: Set.DRIVER})
            # 3ci). Find objects not connected to me TODO: Check if using ud_con is required - if rb is always parent, can remove the ud_con.
            ud_con = torch.bitwise_or(con_tensor, t_con) # undirected connections tensor (OR connections with its transpose)
            if torch.any(set_obj) and torch.any(rb):
                shared = torch.matmul(con_tensor[p, rb].float(), ud_con[rb, set_obj].float()) # PxObject tensor, shared[i][j] > 1 if p[i] and object[j] share an RB, 0 o.w
                shared = torch.gt(shared, 0).int()         # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
                non_shared = 1 - shared                    # non_shared[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
                # 3cii). update input using non shared objects
                obj_acts = nodes[set_obj, TF.ACT]
                # Ensure dimensions are correct for matmul
                if non_shared.dim() == 0:
                    # Single P and single object case
                    result = non_shared * obj_acts
                elif obj_acts.dim() == 0:
                    # Single object case
                    result = non_shared * obj_acts
                else:
                    # Normal case: (num_p x num_obj) @ (num_obj,) -> (num_p,)
                    result = torch.matmul(non_shared, obj_acts)
                
                # Ensure result is 1D for indexing
                if result.dim() == 0:
                    result = result.unsqueeze(0)
                nodes[p, TF.LATERAL_INPUT] -= result
   
    def update_input_rb(self):                                      # update RB inputs - driver:
        """Update input in driver for RB units"""
        # Exitatory: td (my parent P), bu (my PO and child P).
        # Inhibitory: lateral (other RBs*3), inhibitor.
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        # 1). get masks
        rb = cache.get_arbitrary_mask({TF.TYPE: Type.RB, TF.SET: Set.DRIVER})
        po = cache.get_type_mask(Type.PO)
        p = cache.get_type_mask(Type.P)

        # Exitatory input:
        if not torch.any(rb): return;
        # 2). TD_INPUT: my_parent_p
        t_con = torch.transpose(con_tensor, 0 , 1)   # Connnections: Parent -> child, take transpose to get list of parents instead
        nodes[rb, TF.TD_INPUT] += torch.matmul(      # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            t_con[rb][:, p].float(),                 # Masks connections between rb[i] and its ps
            nodes[p, TF.ACT]                         # For each rb node -> sum of act of connected p nodes
            )
        # 3). BU_INPUT: my_po, my_child_p           # NOTE: Old function explicitly took myPred[0].act etc. as there should only be one pred/child/etc. This version sums all connections, so if rb mistakenly connected to multiple of a node type it will not give expected output.
        po_p = torch.bitwise_or(po, p)              # Get mask of both pos and ps
        nodes[rb, TF.BU_INPUT] += torch.matmul(     # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            con_tensor[rb][:, po_p].float(),        # Masks connections between rb[i] and its po and child p nodes
            nodes[po_p, TF.ACT]                     # For each rb node -> sum of act of connected po and child p nodes
            )
        
        # Inhibitory input: NOTE: Inhibitory input only comes from local nodes.
        # 4). LATERAL: (other RBs*3), inhibitor*10
        # 4a). (other RBs*3)
        diag_zeroes = tOps.diag_zeros(sum(rb))      # Connects each rb to every other rb, but not themself
        nodes[rb, TF.LATERAL_INPUT] -= torch.mul(
            3, 
            torch.matmul(                 # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                diag_zeroes,              # Masks connections between rb[i] and its po and child p nodes
                nodes[rb, TF.ACT]         # For each rb node -> sum of act of connected po and child p nodes
            )
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, nodes[rb, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        nodes[rb, TF.LATERAL_INPUT] -= inhib_act               # Update lat input
    
    def update_input_po(self): 
        """Update input in driver for PO units"""
        as_DORA = self.params.as_DORA
        # Exitatory: td (my RB) * gain (2 for preds, 1 for objects).
        # Inhibitory: lateral (other POs not connected to my RB and Ps in child mode, if in DORA mode, then other PO connected to my RB), inhibitor.
        cache = self.glbl.cache
        con_tensor = self.glbl.connections.connections
        nodes = self.glbl.tensor
        # 1). get masks
        po = cache.get_arbitrary_mask({TF.TYPE: Type.PO, TF.SET: Set.DRIVER})
        if not torch.any(po): return;
        rb = cache.get_type_mask(Type.RB)
        pred_sub = (nodes[po, TF.PRED] == B.TRUE)        # predicate sub mask of po nodes
        parent_cons = torch.transpose(con_tensor, 0 , 1) # Transpose of connections matrix, so that index by child node (PO) to parent (RB)

        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:2, obj:1)
        cons = parent_cons[po][:, rb].clone().float()   # get copy of connections matrix from po to rb (child to parent connection)
        cons[pred_sub] = cons[pred_sub] * 2             # multipy predicate -> rb connections by 2
        nodes[po, TF.TD_INPUT] += torch.matmul(         # matmul outputs martix (sum(po) x 1) of values to add to current input value
            cons,                                       # Masks connections between po[i] and its rbs
            nodes[rb, TF.ACT]                           # For each po node -> sum of act of connected rb nodes (multiplied by 2 for predicates)
            )
        
        # Inhibitory input:
        # 3). LATERAL: 3 * (if DORA_mode: PO connected to same rb / Else: POs not connected to same RBs)
        # 3a). find other PO connected to same RB     
        shared = torch.matmul(                     # PxObject tensor, shared[i][j] > 1 if Po[i] and Po[j] share an RB, 0 o.w
            parent_cons[po][:, rb].float(),
            con_tensor[rb][:, po].float()
            ) 
        shared = torch.gt(shared, 0).int()         # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
        diag_zeroes = tOps.diag_zeros(sum(po))     # (po x po) matrix: 0 for po[i] -> po[i] connections, 1 o.w
        shared = torch.bitwise_and(shared.int(), diag_zeroes.int()).float() # remove po[i] -> po[i] connections
        # 3ai). if DORA: Other po connected to same rb / else: POs not connected to same RBs
        if as_DORA: # PO connected to same rb
            po_connections = shared                # shared PO
        else:       # POs not connected to same RBs
            po_connections = 1 - shared            # non shared PO: mask[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
        # 3aii). updaet lat input: 3 * (filtered nodes connected in po_connections)
        nodes[po, TF.LATERAL_INPUT] -= 3 * torch.matmul(
            po_connections.float(),
            nodes[po, TF.ACT]
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, nodes[po, TF.INHIBITOR_ACT])  # Get inhibitor act * 10
        nodes[po, TF.LATERAL_INPUT] -= inhib_act                # Update lat input
    # --------------------------------------------------------------