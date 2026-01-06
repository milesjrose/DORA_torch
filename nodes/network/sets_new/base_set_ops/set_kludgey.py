import torch
from ....enums import *
from ...single_nodes import Pairs

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base_set import Base_Set

class KludgeyOperations:
    """
    Kludgey operations for the Base_Set class. !NOTE: Not implemented for new structure yet.
    """
    def __init__(self, base_set: 'Base_Set'):
        self.bs: 'Base_Set' = base_set
        """Base set object"""

    def get_pred_rb_no_ps(self, pairs: Pairs) -> Pairs:
        """
        Get all pairs of preds that are connected to RBs that are not connected to any P
        - only neeed one non_p rb to be counted as valid for this.
        """
        cache = self.bs.glbl.cache
        con_tensor = self.bs.glbl.connections.connections
        # get masks # NOTE: only connections to tokens in the same set are considered.
        rb = cache.get_set_type_mask(self.bs.tk_set, Type.RB)
        p = cache.get_set_type_mask(self.bs.tk_set, Type.P)
        pred = cache.get_arbitrary_mask(
            {TF.SET: self.bs.tk_set, 
            TF.TYPE: Type.PO, 
            TF.PRED: B.TRUE}
            )
        # get indices
        rb_indices = torch.where(rb)[0]
        p_indices = torch.where(p)[0]
        pred_indices = torch.where(pred)[0]
        
        # connections from rb -> p
        connections_to_p = con_tensor[rb_indices, :][:, p_indices] # TODO: check if correct direction
        rb_no_p_connections = torch.sum(connections_to_p, dim=1) == 0
        # convert to full-size mask
        rb_no_p_mask = torch.zeros_like(rb, dtype=torch.bool)
        rb_no_p_mask[rb_indices[rb_no_p_connections]] = True

        # connections from pred -> rb_no_p
        rb_no_p_indices = torch.where(rb_no_p_mask)[0]
        connections_to_rb_no_p = con_tensor[pred_indices, :][:, rb_no_p_indices]
        preds_connected_to_rb_no_p = torch.sum(connections_to_rb_no_p, dim=1) > 0
        # convert to full-size mask
        pred_rb_no_p_mask = torch.zeros_like(pred, dtype=torch.bool)
        pred_rb_no_p_mask[pred_indices[preds_connected_to_rb_no_p]] = True
        
        # get pairs
        row = pred_rb_no_p_mask.unsqueeze(1)
        col = pred_rb_no_p_mask.unsqueeze(0)
        pred_rb_no_p_pairs = torch.bitwise_and(row, col)
        # remove duplicates (below the diagonal + the diagonal)
        pred_rb_no_p_pairs = torch.triu(pred_rb_no_p_pairs, diagonal=1)
        
        # create list of pair indices in pairs obj
        pair_indices = torch.where(pred_rb_no_p_pairs)
        for i, j in zip(pair_indices[0], pair_indices[1]):
            pairs.add(i.item(), j.item())
        return pairs
        
    
    def get_pred_rb_shared_p(self, pairs: Pairs) -> Pairs:
        """
        Get all pairs of preds that are connected to the same P.
        (i.e preds connected to RBs that are connected to the same P)
        """
        cache = self.bs.glbl.cache
        con_tensor = self.bs.glbl.connections.connections
        # get masks for different token types NOTE: only connections to tokens in the same set are considered.
        rb = cache.get_set_type_mask(self.bs.tk_set, Type.RB)
        p = cache.get_set_type_mask(self.bs.tk_set, Type.P)
        pred = cache.get_arbitrary_mask(
            {TF.SET: self.bs.tk_set, 
            TF.TYPE: Type.PO, 
            TF.PRED: B.TRUE}
            )
        # convert to indices
        rb_indices = torch.where(rb)[0]
        p_indices = torch.where(p)[0]
        pred_indices = torch.where(pred)[0]
        
        # get rb s.t rb -> p
        rb_to_p_connections = con_tensor[rb_indices, :][:, p_indices]
        rb_with_p_mask_local = torch.sum(rb_to_p_connections, dim=1) > 0
        rb_with_p_indices = rb_indices[rb_with_p_mask_local]
        if len(rb_with_p_indices) == 0:
            return pairs
        # get specific connections from rb_with_p -> p
        rb_with_p_to_p_connections = con_tensor[rb_with_p_indices, :][:, p_indices]

        # matrix of rb_with_p, with true if they share a p, false o.w
        num_rbs_with_p = len(rb_with_p_indices)
        rb_shared_p = torch.zeros((num_rbs_with_p, num_rbs_with_p), dtype=torch.bool)
        for i in range(num_rbs_with_p):
            for j in range(i + 1, num_rbs_with_p):
                shared_p = torch.bitwise_and(rb_with_p_to_p_connections[i].bool(), rb_with_p_to_p_connections[j].bool()).sum() > 0
                rb_shared_p[i, j] = shared_p

        # get specific connections from pred -> rb_with_p
        pred_to_rb_connections = con_tensor[pred_indices, :][:, rb_with_p_indices]
        num_preds = len(pred_indices)
        if num_preds == 0:
            return pairs

        # create matrix of preds, with true if they share a p, false o.w 
        # NOTE: doesn't seem even vaguely efficient, but should be a low number of tokens involved afaik.
        #       should probs try improve this, but don't seem worth it for now.
        pred_shared_p = torch.zeros((num_preds, num_preds), dtype=torch.bool)
        for i in range(num_preds):
            for j in range(i + 1, num_preds):
                pred_i_rbs = pred_to_rb_connections[i]
                pred_j_rbs = pred_to_rb_connections[j]
                
                shared = False
                for rb_i_idx in range(num_rbs_with_p):
                    if pred_i_rbs[rb_i_idx]:
                        for rb_j_idx in range(num_rbs_with_p):
                            if pred_j_rbs[rb_j_idx] and rb_shared_p[rb_i_idx, rb_j_idx]:
                                shared = True
                                break
                        if shared:
                            break
                
                pred_shared_p[i, j] = shared
        
        # create list of pair indices in pairs obj
        pair_indices_local = torch.where(pred_shared_p)
        for i, j in zip(pair_indices_local[0], pair_indices_local[1]):
            global_i = pred_indices[i].item()
            global_j = pred_indices[j].item()
            pairs.add(global_i, global_j)
            
        return pairs
        