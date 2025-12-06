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
   