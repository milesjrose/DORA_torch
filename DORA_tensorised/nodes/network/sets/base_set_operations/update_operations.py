# nodes/network/sets/base_set_operations/update_operations.py
# Update operations for Base_Set class
import torch 

from ....enums import *
from ....utils import tensor_ops as tOps

class UpdateOperations:
    """
    Update operations for the Base_Set class.
    Handles update operations.
    """
    def __init__(self, base_set):
        """
        Initialize UpdateOperations with reference to Base_Set.
        
        Args:
            base_set: Reference to the Base_Set object
        """
        self.base_set = base_set

    # ====================[ TOKEN FUNCTIONS ]=======================
    def initialise_float(self, n_type: list[Type], features: list[TF]): # Initialise given features
        """
        Initialise given features
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            features (list[TF]): The features to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)   # Get mask of nodes to update
        init_subt = self.base_set.nodes[type_mask, features]            # Get subtensor of features to intialise
        self.base_set.nodes[type_mask, features] = torch.zeros_like(init_subt)  # Set features to 0
    
    def initialise_input(self, n_type: list[Type], refresh: float):     # Initialize inputs to 0, and td_input to refresh.
        """ 
        Initialize inputs to 0, and td_input to refresh
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            refresh (float): The value to set the td_input to.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        self.base_set.nodes[type_mask, TF.TD_INPUT] = refresh           # Set td_input to refresh
        features = [TF.BU_INPUT,TF.LATERAL_INPUT,TF.MAP_INPUT,TF.NET_INPUT]
        self.initialise_float(n_type, features)                         # Set types to 0.0

    def initialise_act(self, n_type: list[Type]):                       # Initialize act to 0.0,  and call initialise_inputs
        """Initialize act to 0.0,  and call initialise_inputs
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        self.initialise_input(n_type, 0.0)
        self.initialise_float(n_type, [TF.ACT])

    def initialise_state(self, n_type: list[Type]):                     # Set self.retrieved to false, and call initialise_act
        """Set self.retrieved to false, and call initialise_act
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        self.initialise_act(n_type)
        self.initialise_float(n_type, [TF.RETRIEVED])                       
        
    def update_act(self):                                               # Update act of nodes
        """Update act of nodes. Based on params.gamma, params.delta, and params.HebbBias."""
        net_input_types = [
            TF.TD_INPUT,
            TF.BU_INPUT,
            TF.LATERAL_INPUT
        ]
        gamma = self.base_set.params.gamma
        delta = self.base_set.params.delta
        HebbBias = self.base_set.params.HebbBias
        net_input = self.base_set.nodes[:, net_input_types].sum(dim=1, keepdim=True) # sum non mapping inputs
        net_input += self.base_set.nodes[:, TF.MAP_INPUT] * HebbBias        # Add biased mapping input
        acts = self.base_set.nodes[:, TF.ACT]                               # Get node acts
        delta_act = gamma * net_input * (1.1 - acts) - (delta * acts)       # Find change in act for each node
        acts += delta_act                                                   # Update acts
        
        self.base_set.nodes[(self.base_set.nodes[:, TF.ACT] > 1.0), TF.ACT] = 1.0  # Limit activation to 1.0 or below
        self.base_set.nodes[(self.base_set.nodes[:, TF.ACT] < 0.0), TF.ACT] = 0.0  # Limit activation to 0.0 or above                                      # update act

    def zero_lateral_input(self, n_type: list[Type]):                   # Set lateral_input to 0 
        """
        Set lateral_input to 0;
        to allow synchrony at different levels by 0-ing lateral inhibition at that level 
        (e.g., to bind via synchrony, 0 lateral inhibition in POs).
        
        Args:
            n_type (list[Type]): The types of nodes to set lateral_input to 0.
        """
        self.initialise_float(n_type, [TF.LATERAL_INPUT])
    
    def update_inhibitor_input(self, n_type: list[Type]):               # Update inputs to inhibitors by current activation for nodes of type n_type
        """
        Update inputs to inhibitors by current activation for nodes of type n_type
        
        Args:
            n_type (list[Type]): The types of nodes to update inhibitor inputs.
        """
        mask = self.base_set.tensor_op.get_combined_mask(n_type)
        self.base_set.nodes[mask, TF.INHIBITOR_INPUT] += self.base_set.nodes[mask, TF.ACT]

    def reset_inhibitor(self, n_type: list[Type]):                      # Reset the inhibitor input and act to 0.0 for given type
        """
        Reset the inhibitor input and act to 0.0 for given type
        
        Args:
            n_type (list[Type]): The types of nodes to reset inhibitor inputs and acts.
        """
        mask = self.base_set.tensor_op.get_combined_mask(n_type)
        self.base_set.nodes[mask, TF.INHIBITOR_INPUT] = 0.0
        self.base_set.nodes[mask, TF.INHIBITOR_ACT] = 0.0
    
    def update_inhibitor_act(self, n_type: list[Type]):                 # Update the inhibitor act for given type
        """
        Update the inhibitor act for given type
        
        Args:
            n_type (list[Type]): The types of nodes to update inhibitor acts.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        input = self.base_set.nodes[type_mask, TF.INHIBITOR_INPUT]
        threshold = self.base_set.nodes[type_mask, TF.INHIBITOR_THRESHOLD]
        nodes_to_update = (input >= threshold)                      # if inhib_input >= inhib_threshold
        self.base_set.nodes[nodes_to_update, TF.INHIBITOR_ACT] = 1.0  # then set to 1
    # --------------------------------------------------------------

    # =======================[ P FUNCTIONS ]========================
    def p_initialise_mode(self):                                        # Initialize all p.mode back to neutral.
        """Initialize mode to neutral for all P units."""
        p = self.base_set.tensor_op.get_mask(Type.P)
        self.base_set.nodes[p, TF.MODE] = Mode.NEUTRAL

    def p_get_mode(self):                                               # Set mode for all P units
        """Set mode for all P units"""
        # Pmode = Parent: child RB act> parent RB act / Child: parent RB act > child RB act / Neutral: o.w
        p = self.base_set.tensor_op.get_mask(Type.P)
        rb = self.base_set.tensor_op.get_mask(Type.RB)
        child_input = torch.matmul(                                 # Px1 matrix: sum of child rb for each p
            self.base_set.connections[p, rb],
            self.base_set.nodes[rb, TF.ACT]
        )
        parent_input = torch.matmul(                               # Px1 matrix: sum of parent rb for each p
            torch.transpose(self.base_set.connections)[p, rb],
            self.base_set.nodes[rb, TF.ACT]
        )
        # Get global masks of p, by mode
        input_diff = parent_input - child_input                     # (input_diff > 0) <-> (parents > childs)
        child_p = tOps.sub_union(p, (input_diff[:, 0] > 0.0))       # (input_diff > 0) -> (parents > childs) -> (p mode = child)
        parent_p = tOps.sub_union(p, (input_diff[:, 0] < 0.0))      # (input_diff < 0) -> (parents < childs) -> (p mode = parent) 
        neutral_p = tOps.sub_union(p, (input_diff[:, 0] == 0.0))    # input_diff == 0 -> p mode = neutral
        # Set mode values:
        self.base_set.nodes[child_p, TF.MODE] = Mode.CHILD                   
        self.base_set.nodes[parent_p, TF.MODE] = Mode.PARENT
        self.base_set.nodes[neutral_p, TF.MODE] = Mode.NEUTRAL
    # ---------------------------------------------------------------

    # =======================[ PO FUNCTIONS ]======================== # TODO: Can move out of tensor to save memory, as shared values.
    def po_get_weight_length(self):                                     # Sum value of links with weight > 0.1 for all PO nodes
        """Sum value of links with weight > 0.1 for all PO nodes - Used for semNormalisation"""
        if self.base_set.links is None:
            raise ValueError("Links is not initialised, po_get_weight_length.")
        
        po = self.base_set.tensor_op.get_mask(Type.PO)                  # mask links with PO
        mask = self.base_set.links[self.base_set.token_set][po] > 0.1   # Create sub mask for links with weight > 0.1
        weights = (self.base_set.links[self.base_set.token_set][po] * mask).sum(dim=1, keepdim = True)  # Sum links > 0.1
        self.base_set.nodes[po, TF.SEM_COUNT] = weights                 # Set semNormalisation
            
    def po_get_max_semantic_weight(self):                               # Get max link weight for all PO nodes
        """Get max link weight for all PO nodes - Used for semNormalisation"""
        if self.base_set.links is None:
            raise ValueError("Links is not initialised, po_get_max_semantic_weight.")
        
        po = self.base_set.tensor_op.get_mask(Type.PO)
        max_values, _ = torch.max(self.base_set.links[self.base_set.token_set][po], dim=1, keepdim=True)  # (max_values, _) unpacks tuple returned by torch.max
        self.base_set.nodes[po, TF.MAX_SEM_WEIGHT] = max_values         # Set max

    # ---------------------------------------------------------------