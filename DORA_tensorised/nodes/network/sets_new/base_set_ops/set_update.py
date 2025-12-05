import torch

from ....enums import *
from logging import getLogger
from typing import TYPE_CHECKING

from ....utils import tensor_ops as tOps

if TYPE_CHECKING:
    from ..base_set import Base_Set

logger = getLogger(__name__)

class UpdateOperations:
    """
    Update operations for the Base_Set class.
    """
    def __init__(self, base_set: 'Base_Set'):
        """
        Initialize the UpdateOperations object.
        Args:
            base_set: The Base_Set object.
        """
        self.base_set: 'Base_Set' = base_set
        """
        Reference to the Base_Set object.
        """

    # ====================[ TOKEN FUNCTIONS ]=======================
    def init_float(self, n_type: list[Type], features: list[TF], value: float = 0.0) -> None:
        """
        Initialise the given features to 0.0

        Args:
            n_type (list[Type]): The types of nodes to initialise.
            features (list[TF]): The features to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            for feature in features:
                self.base_set.lcl[type_mask, feature] = value
    
    def init_input(self, n_type: list[Type], refresh: float) -> None:
        """
        Initialise the input of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
            refresh (float): The value to set the td_input to.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.base_set.lcl[type_mask, TF.TD_INPUT] = refresh
            features = [TF.BU_INPUT,TF.LATERAL_INPUT,TF.MAP_INPUT,TF.NET_INPUT]
            self.init_float(n_type, features)
        
    def init_act(self, n_type: list[Type]) -> None:
        """
        Initialise the act of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.init_input(n_type, 0.0)
            self.init_float(n_type, [TF.ACT])
    
    def init_state(self, n_type: list[Type]) -> None:
        """
        Initialise the state of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.init_act(n_type)
            self.init_float(n_type, [TF.RETRIEVED])
    
    def update_act(self) -> None:
        """
        Update the act of the tokens.
        """
        if self.base_set.get_count() == 0: return;
        net_input_types = [
            TF.TD_INPUT,
            TF.BU_INPUT,
            TF.LATERAL_INPUT
        ]
        gamma = self.base_set.params.gamma
        delta = self.base_set.params.delta
        HebbBias = self.base_set.params.HebbBias
        net_input = self.base_set.lcl[:, net_input_types].sum(dim=1, keepdim=True)  # sum non mapping inputs
        net_input += (self.base_set.lcl[:, TF.MAP_INPUT] * HebbBias).unsqueeze(1)   # Add biased mapping input, reshape to match
        acts = self.base_set.lcl[:, TF.ACT]                                         # Get node acts
        delta_act = gamma * net_input.squeeze(1) * (1.1 - acts) - (delta * acts)    # Find change in act for each node
        self.base_set.lcl[:, TF.ACT] = acts + delta_act                             # Update acts (assign back to tensor)
        self.base_set.lcl[(self.base_set.lcl[:, TF.ACT] > 1.0), TF.ACT] = 1.0       # Limit activation to 1.0 or below
        self.base_set.lcl[(self.base_set.lcl[:, TF.ACT] < 0.0), TF.ACT] = 0.0       # Limit activation to 0.0 or above

    def zero_laternal_input(self, n_type: list[Type]) -> None:
        """
        Zero the lateral input of the tokens.
        Args:
            n_type (list[Type]): The types of nodes to zero the lateral input of.
        """
        if len(n_type) > 0:
            self.init_float(n_type, [TF.LATERAL_INPUT])
    
    def update_inhibitor_input(self, n_type: list[Type]) -> None:
        """
        Update the inhibitor input of the tokens.
        Args:
            n_type (list[Type]): The types of nodes to update the inhibitor input of.
        """
        mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(mask):
            self.base_set.lcl[mask, TF.INHIBITOR_INPUT] += self.base_set.lcl[mask, TF.ACT]

    def reset_inhibitor(self, n_type: list[Type]) -> None:
        """
        Reset the inhibitor input and act of the tokens.
        """
        if len(n_type) > 0:
            self.init_float(n_type, [TF.INHIBITOR_INPUT], 0.0)
            self.init_float(n_type, [TF.INHIBITOR_ACT], 0.0)
    
    def update_inhibitor_act(self, n_type: list[Type]) -> None:
        """
        Update the inhibitor act of the tokens.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            input = self.base_set.lcl[type_mask, TF.INHIBITOR_INPUT]
            threshold = self.base_set.lcl[type_mask, TF.INHIBITOR_THRESHOLD]
            nodes_to_update = (input >= threshold)                          # if inhib_input >= inhib_threshold
            # turn into full local mask
            update_mask = torch.zeros_like(type_mask, dtype=torch.bool)
            update_mask[type_mask] = nodes_to_update
            #update
            self.base_set.lcl[update_mask, TF.INHIBITOR_ACT] = 1.0          # then set to 1
    # ---------------------------------------------------------------
        
    # ===========================[ P FUNCTIONS ]==========================
    def p_initialise_mode(self) -> None:
        """
        Initialise the mode of the P tokens.
        """
        p = self.base_set.tensor_op.get_mask(Type.P)
        if torch.any(p):
            self.base_set.lcl[p, TF.MODE] = Mode.NEUTRAL

    def p_get_mode(self) -> None:
        """Set mode for all P units"""
        tk_tensor = self.base_set.glbl.tensor
        con_tensor = self.base_set.glbl.connections.connections
        # Pmode = Parent: child RB act> parent RB act / Child: parent RB act > child RB act / Neutral: o.w
        
        # Get local mask of p, and global indices
        p = self.base_set.tensor_op.get_mask(Type.P)
        if not torch.any(p): return;
        p_indices = torch.where(p)[0]
        global_p = self.base_set.lcl.to_global(p_indices)
        # Get local mask of rb, and global indices
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        if not torch.any(rb): return;
        rb_indices = torch.where(rb)[0]
        global_rb = self.base_set.lcl.to_global(rb_indices)

        # use global indices to get child and parent inputs
        # Need to use advanced indexing to get submatrix: con_tensor[global_p, global_rb] with broadcasting
        # Use [:, None] to add dimension for broadcasting: [3] -> [3, 1] so [3, 1] x [3] -> [3, 3]
        con_submatrix = con_tensor[global_p[:, None], global_rb].float()  # Shape: [num_p, num_rb]
        tk_act = tk_tensor[global_rb, TF.ACT]  # Shape: [num_rb]
        child_input = torch.matmul(                                 # Px1 matrix: sum of child rb for each p
            con_submatrix,
            tk_act
        )
        
        # For parent input, use transpose of connections
        con_t_submatrix = torch.t(con_tensor)[global_p[:, None], global_rb].float()  # Shape: [num_p, num_rb]
        parent_input = torch.matmul(                                # Px1 matrix: sum of parent rb for each p
            con_t_submatrix,
            tk_act
        )
        
        # Get local masks of p, by mode
        input_diff = parent_input - child_input                     # (input_diff > 0) <-> (parents > childs)
        logger.debug(f"input_diff_shape: {input_diff.shape}")
        logger.debug(f"p_shape: {p.shape}")
        # Create boolean masks for each mode
        child_p = tOps.sub_union(p, (input_diff > 0.0))       # (input_diff > 0) -> (parents > childs) -> (p mode = child)
        parent_p = tOps.sub_union(p, (input_diff < 0.0))      # (input_diff < 0) -> (parents < childs) -> (p mode = parent) 
        neutral_p = tOps.sub_union(p, (input_diff == 0.0))    # input_diff == 0 -> p mode = neutral
        
        # Set mode values:
        self.base_set.lcl[child_p, TF.MODE] = Mode.CHILD                   
        self.base_set.lcl[parent_p, TF.MODE] = Mode.PARENT
        self.base_set.lcl[neutral_p, TF.MODE] = Mode.NEUTRAL
    # ---------------------------------------------------------------

    # =======================[ PO FUNCTIONS ]======================== # TODO: Can move out of tensor to save memory, as shared values.
    def po_get_weight_length(self) -> None:                           # Sum value of links with weight > 0.1 for all PO nodes
        """Sum value of links with weight > 0.1 for all PO nodes - Used for semNormalisation"""
        raise NotImplementedError("Need to implement links first")
        if self.base_set.links is None:
            raise ValueError("Links is not initialised, po_get_weight_length.")
        
        po = self.base_set.tensor_op.get_mask(Type.PO)                # mask links with PO
        if not torch.any(po): return;
        mask = self.base_set.links[self.base_set.token_set][po] > 0.1 # Create sub mask for links with weight > 0.1
        weights = (self.base_set.links[self.base_set.token_set][po] * mask).sum(dim=1, keepdim = False)  # Sum links > 0.1
        self.base_set.lcl[po, TF.SEM_COUNT] = weights               # Set semNormalisation
            
    def po_get_max_semantic_weight(self) -> None:                     # Get max link weight for all PO nodes
        """Get max link weight for all PO nodes - Used for semNormalisation"""
        raise NotImplementedError("Need to implement links first")
        if self.base_set.links is None:
            raise ValueError("Links is not initialised, po_get_max_semantic_weight.")
        
        po = self.base_set.tensor_op.get_mask(Type.PO)
        if not torch.any(po): return;
        max_values, _ = torch.max(self.base_set.links[self.base_set.token_set][po], dim=1, keepdim=False)  # (max_values, _) unpacks tuple returned by torch.max
        self.base_set.nodes[po, TF.MAX_SEM_WEIGHT] = max_values         # Set max
    # ---------------------------------------------------------------