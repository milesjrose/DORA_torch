# nodes/network/sets/base_set.py
# Base class for all set classes.

import torch

from nodes.enums import *
from nodes.utils import tensorOps as tOps

from ..connections import Links, Mappings
from ..network_params import Params
from ..single_nodes import Token, Ref_Token

class Base_Set(object):
    """
    A class for holding a tensor of tokens, and performing general tensor operations.

    Attributes:
        - names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        - nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        - analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        - analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        - links (Links): A shared Links object containing interset links from tokens to semantics.
        - connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        - masks (torch.Tensor): A Tensor of masks for the tokens in this set.
        - IDs (dict): A dictionary mapping token IDs to index in the tensor.
        - params (Params): An object containing shared parameters.
        - token_set (Set): This set's enum, used to access links and mappings for this set in shared mem objects.
    """
    def __init__(self, floatTensor, connections, links: Links, IDs: dict[int, int],names: dict[int, str] = {}, params: Params = None):
        """
        Initialize the TokenTensor object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            links (Links): A shared Links object containing interset links from tokens to semantics.
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
            params (Params, optional): An object containing shared parameters. Defaults to None.
        Raises:
            TypeError: If links, connections, or floatTensor are not torch.Tensor.
            ValueError: If the number of tokens in floatTensor, links, and connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
        """
        # Validate input
        if type(links) != Links:
            raise TypeError("Links must be Links object.")
        if type(connections) != torch.Tensor:
            raise TypeError("Connections must be torch.Tensor.")
        if type(floatTensor) != torch.Tensor:
            raise TypeError("floatTensor must be torch.Tensor.")
        
        if floatTensor.size(dim=0) != connections.size(dim=0):
            raise ValueError("floatTensor and connections must have same number of tokens.")
        if floatTensor.size(dim=1) != len(TF):
            raise ValueError("floatTensor must have number of features listed in TF enum.")
        
        self.names = names
        "Dict ID -> Name"
        self.nodes: torch.Tensor = floatTensor
        "NxTF Tensor: Tokens"
        self.cache_masks()
        self.analogs = None
        "Ax1 Tensor: Analogs in the set"
        self.analog_counts = None
        "Ax1 Tensor: Node count for each analog in self.analogs"
        self.analog_node_count()
        self.links = links
        """Links object for the set.
            contains links.set[self.set]:  NxS tensor of connctions from Token to Semnatic
        """
        self.connections = connections
        "NxN tensor: Connections from parent to child"
        self.IDs = IDs
        "Dict ID -> index"
        self.params = params
        "Params object, holding parameters for tensor functions"
        self.expansion_factor = 1.1
        """Factor used in expanding tensor. 
            E.g: expansion_factor = 1.1 -> 10 percent increase in size on expansion
        """
        self.token_set = None
        """Set: This sets set type"""

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get_feature(self, ID, feature):                             # Get feature of single node
        """
        Get a feature for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to get the feature for.
            feature (TF): The feature to get.

        Returns:
            The feature for the token with the given ID.
        """
        try:
            return self.nodes[self.IDs[ID], feature]
        except:
            raise ValueError("Invalid ID or feature.")

    def set_feature(self, ID, feature, value):                      # Set feature of single node
        """
        Set a feature for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.
        """
        if type(feature) != TF:
            raise TypeError("Feature must be a TF enum.")
        try:
            self.nodes[self.IDs[ID], feature] = float(value)
        except:
            raise ValueError("Invalid ID or feature.")

    def get_name(self, ID):                                         # Get name of node by ID
        """
        Get the name for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to get the name for.
        """
        return self.names[ID]

    def set_name(self, ID, name):                                   # Set name of node by ID
        """
        Set the name for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to set the name for.
            name (str): The name to set the token to.
        """
        self.names[ID] = name
    
    def get_ID_from_name(self, name):                               # Get ID of node with given name
        """
        Get the ID for a token with a given name.
        
        Args:
            name (str): The name of the token to get the ID for.
        """
        try:
            return self.IDs.keys()[self.IDs.values().index(name)]
        except:
            raise ValueError("Invalid name.")
    
    def get_IDs_from_type(self, Type):                              # Get list of IDs for all nodes of type
        """
        Get the IDs of the tokens of a given type.
        """
        return self.get_IDs_by_mask(self.get_mask(Type))
    
    def get_IDs_by_mask(self, mask):                                # Get list of IDs for nodes in mask
        """
        Get the IDs of the tokens of a given mask.
        """
        return self.nodes[mask, TF.ID]
    
    def get_index(self, ID):                                        # Get index in tensor of node with given ID
        """
        Get index in tesor of node with ID.
        """
        return self.IDs[ID]
    # --------------------------------------------------------------

    # ====================[ TENSOR FUNCTIONS ]======================
    def cache_masks(self, types_to_recompute = None):               # Compute and cach masks for given types
        """Compute and cache masks, specify types to recompute via list of tokenTypes"""
        if types_to_recompute == None:                              #  If no type specified, recompute all
            types_to_recompute = [Type.PO, Type.RB, Type.P, Type.GROUP]

        masks = []
        for token_type in [Type.PO, Type.RB, Type.P, Type.GROUP]:
            if token_type in types_to_recompute:
                masks.append(self.compute_mask(token_type))         # Recompute mask
            else:
                masks.append(self.masks[token_type])                # Use cached mask

        self.masks: torch.Tensor = torch.stack(masks, dim=0)
    
    def compute_mask(self, token_type: Type):                       # Compute the mask for a token type
        """Compute the mask for a token type"""
        mask = (self.nodes[:, TF.TYPE] == token_type) & (self.nodes[:, TF.DELETED] == B.FALSE)
        return mask
    
    def get_mask(self, token_type: Type):                           # Returns mask for given token type
        """Return mask for given token type"""
        return self.masks[token_type]                   

    def get_combined_mask(self, n_types: list[Type]):               # Returns combined mask of give types
        """Return combined mask of given types"""
        masks = [self.masks[i] for i in n_types]
        return torch.logical_or.reduce(masks)

    def get_all_nodes_mask(self):                                   # Returns a mask for all nodes (Exluding empty or deleted rows)
        """Return mask for all non-deleted nodes"""
        return (self.nodes[:, TF.DELETED] == B.FALSE)

    def add_token(self, token: Token):                              # Add a token to the tensor
        """
        Add a token to the tensor. If tensor is full, expand it first.

        Returns:
            Ref_Token: Reference to the token that was added.
        """
        spaces = torch.sum(self.nodes[:, TF.DELETED] == B.TRUE)             # Find number of spaces -> count of deleted nodes in the tensor
        if spaces == 0:                                                     # If no spaces, expand tensor
            self.expand_tensor()
        if len(self.IDs) > 0:
            self.print()
            try:
                ID = max(self.IDs.keys()) + 1                                   # assign token a new ID.
            except Exception as e:
                raise(e)
        else:
            ID = 1
        token.tensor[TF.ID] = ID
        deleted_nodes = torch.where(self.nodes[:, TF.DELETED] == B.TRUE)[0] #find first deleted node
        first_deleted = deleted_nodes[0]
        self.nodes[first_deleted, :] = token.tensor                         # add token to tensor
        self.IDs[ID] = first_deleted                                        # update IDs
        self.cache_masks()                                                  # recompute masks
        return Ref_Token(self.token_set, ID)
    
    def expand_tensor(self):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes expansion factor. Minimum expansion is 5.
        Expands nodes, connections, links and mappings tensors.
        """
                                                                        # Update node tensor:
        current_count = self.nodes.size(dim=0)
        new_count = int(current_count * self.expansion_factor)              # calculate new size
        if new_count < 5:                                                   # minimum expansion is 5
            new_count = 5
        new_tensor = torch.zeros(new_count, self.nodes.size(dim=1))         # create new tensor with expansion factor
        new_tensor[current_count:, TF.DELETED] = B.TRUE                     # set deleted to true for all new tokens
        new_tensor[:current_count, :] = self.nodes                          # copy over old tensor
        self.nodes = new_tensor                                             # update tensor
                                                                        # Update supporting data structures:
                                                                        # Links:
        semantic_count = self.links[self.token_set].size(dim=1)             # Get number of semantics in link tensor
        new_links = torch.zeros(new_count, semantic_count)                  # new tensor (new number of tokens) x (number of semantics)
        new_links[:current_count, :] = self.links[self.token_set]           # add current links to the tensor
        self.links[self.token_set] = new_links                              # update links object with new tensor

        if self.token_set not in [Set.DRIVER, Set.NEW_SET]:             # Mappings (if not driver): TODO: See if new set needs mappings?
            self.mappings: Mappings = self.mappings
            driver_count = self.mappings.size(dim=1)                        # Get number of tokens in driver
            stack = []
            for field in MappingFields:                                     # Create new tensor for each mapping field
                stack.append(torch.zeros(new_count, driver_count))
            new_adj_matrix: torch.Tensor = torch.stack(stack, dim=-1)       # Stack into adj_matrix tensor
            new_adj_matrix[:current_count, :] = self.mappings.adj_matrix    # add current weights
            self.mappings.adj_matrix = new_adj_matrix                       # update mappings object with new tensor
                                                                        # Connections:
        new_connections = torch.zeros(new_count, new_count)                 # new tensor (new num tokens) x (new num tokens)
        new_connections[:current_count, :current_count] = self.connections  # add current connections
        self.connections = new_connections                                  # update connections tensor, with new tensor

    def del_nodes(self, ref_tokens: list[Ref_Token]):                                        # Delete nodes from tensor     TODO: Remove connections, links, mappings etc.
        """
        Delete nodes from tensor
        
        Args:
            ref_tokens (list[Ref_Token]): The tokens to delete.
        """
        if not isinstance(ref_tokens, list):                                        # If input is single ID, turn into iteratable object.
            ref_tokens = [ref_tokens]
        
        cache_types = [] 
        for ref_token in ref_tokens:     
            id = ref_token.ID                                               # Delete nodes in nodes tensor:
            cache_types.append(self.nodes[self.IDs[id], TF.TYPE])           # Keep list of types that have a deleted node to recache specific masks
            self.nodes[self.IDs[id], TF.DELETED] = B.TRUE                   # Mark as deleted
            self.IDs.pop(id)
            self.names.pop(id)
            
            self.links[self.token_set][self.IDs[id], :] = 0.0
            self.connections[self.IDs[id], :] = 0.0
            self.connections[:, self.IDs[id]] = 0.0

        cache_types = list(set(cache_types))                            # Remove duplicates if multiple nodes deleted from same type
        self.cache_masks(cache_types)                                   # Re-cache effected types

    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        """Update list of analogs in tensor, and their node counts"""
        self.analogs, self.analog_counts = torch.unique(self.nodes[:, TF.ANALOG], return_counts=True)
   
    def print(self, f_types=None):                                  # Here for testing atm
        """
        Print the set.

        Args:
            f_types (list[TF], optional): The features to print.

        Raises:
            ValueError: If nodePrinter is not found.
        """
        
        try:
            from nodes.utils import nodePrinter
            printer = nodePrinter(print_to_console=True)
            printer.print_set(self, feature_types=f_types)
        except:
            print("Error: nodePrinter not found. Nodes.utils.nodePrinter is required to use this function.")
    # --------------------------------------------------------------

    # ====================[ TOKEN FUNCTIONS ]=======================
    def initialise_float(self, n_type: list[Type], features: list[TF]): # Initialise given features
        """
        Initialise given features
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            features (list[TF]): The features to initialise.
        """
        type_mask = self.get_combined_mask(n_type)                      # Get mask of nodes to update
        init_subt = self.nodes[type_mask, features]                     # Get subtensor of features to intialise
        self.nodes[type_mask, features] = torch.zeros_like(init_subt)   # Set features to 0
    
    def initialise_input(self, n_type: list[Type], refresh: float):     # Initialize inputs to 0, and td_input to refresh.
        """ 
        Initialize inputs to 0, and td_input to refresh
        
        Args:
            n_type (list[Type]): The types of nodes to initialise.
            refresh (float): The value to set the td_input to.
        """
        type_mask = self.get_combined_mask(n_type)
        self.nodes[type_mask, TF.TD_INPUT] = refresh                    # Set td_input to refresh
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
        gamma = self.params.gamma
        delta = self.params.delta
        HebbBias = self.params.HebbBias
        net_input = self.nodes[:, net_input_types].sum(dim=1, keepdim=True) # sum non mapping inputs
        net_input += self.nodes[:, TF.MAP_INPUT] * HebbBias                 # Add biased mapping input
        acts = self.nodes[:, TF.ACT]                                        # Get node acts
        delta_act = gamma * net_input * (1.1 - acts) - (delta * acts)       # Find change in act for each node
        acts += delta_act                                                   # Update acts
        
        self.nodes[(self.nodes[:, TF.ACT] > 1.0), TF.ACT] = 1.0             # Limit activation to 1.0 or below
        self.nodes[(self.nodes[:, TF.ACT] < 0.0), TF.ACT] = 0.0             # Limit activation to 0.0 or above                                      # update act

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
        mask = self.get_combined_mask(n_type)
        self.nodes[mask, TF.INHIBITOR_INPUT] += self.nodes[mask, TF.ACT]

    def reset_inhibitor(self, n_type: list[Type]):                      # Reset the inhibitor input and act to 0.0 for given type
        """
        Reset the inhibitor input and act to 0.0 for given type
        
        Args:
            n_type (list[Type]): The types of nodes to reset inhibitor inputs and acts.
        """
        mask = self.get_combined_mask(n_type)
        self.nodes[mask, TF.INHIBITOR_INPUT] = 0.0
        self.nodes[mask, TF.INHIBITOR_ACT] = 0.0
    
    def update_inhibitor_act(self, n_type: list[Type]):                 # Update the inhibitor act for given type
        """
        Update the inhibitor act for given type
        
        Args:
            n_type (list[Type]): The types of nodes to update inhibitor acts.
        """
        type_mask = self.get_combined_mask(n_type)
        input = self.nodes[type_mask, TF.INHIBITOR_INPUT]
        threshold = self.nodes[type_mask, TF.INHIBITOR_THRESHOLD]
        nodes_to_update = (input >= threshold)                      # if inhib_input >= inhib_threshold
        self.nodes[nodes_to_update, TF.INHIBITOR_ACT] = 1.0         # then set to 1
    # --------------------------------------------------------------

    # =======================[ P FUNCTIONS ]========================
    def p_initialise_mode(self):                                        # Initialize all p.mode back to neutral.
        """Initialize mode to neutral for all P units."""
        p = self.get_mask(Type.P)
        self.nodes[p, TF.MODE] = Mode.NEUTRAL

    def p_get_mode(self):                                               # Set mode for all P units
        """Set mode for all P units"""
        # Pmode = Parent: child RB act> parent RB act / Child: parent RB act > child RB act / Neutral: o.w
        p = self.get_mask(Type.P)
        rb = self.get_mask(Type.RB)
        child_input = torch.matmul(                                 # Px1 matrix: sum of child rb for each p
            self.connections[p, rb],
            self.nodes[rb, TF.ACT]
        )
        parent_input = torch.matmult(                               # Px1 matrix: sum of parent rb for each p
            torch.transpose(self.connections)[p, rb],
            self.nodes[rb]
        )
        # Get global masks of p, by mode
        input_diff = parent_input - child_input                     # (input_diff > 0) <-> (parents > childs)
        child_p = tOps.sub_union(p, (input_diff[:, 0] > 0.0))       # (input_diff > 0) -> (parents > childs) -> (p mode = child)
        parent_p = tOps.sub_union(p, (input_diff[:, 0] < 0.0))      # (input_diff < 0) -> (parents < childs) -> (p mode = parent) 
        neutral_p = tOps.sub_union(p, (input_diff[:, 0] == 0.0))    # input_diff == 0 -> p mode = neutral
        # Set mode values:
        self.nodes[child_p, TF.MODE] = Mode.CHILD                   
        self.nodes[parent_p, TF.MODE] = Mode.PARENT
        self.nodes[neutral_p, TF.MODE] = Mode.NEUTRAL
    # ---------------------------------------------------------------

    # =======================[ PO FUNCTIONS ]========================
    def po_get_weight_length(self):                                     # Sum value of links with weight > 0.1 for all PO nodes
        """Set sem count feature for all PO nodes"""
        po = self.get_mask(Type.PO)                                     # mask links with PO
        mask = self.links[po] > 0.1                                     # Create sub mask for links with weight > 0.1
        weights = (self.links[po] * mask).sum(dim=1, keepdim = True)    # Sum links > 0.1
        self.nodes[po, TF.SEM_COUNT] = weights                          # Set semNormalisation

    def po_get_max_semantic_weight(self):                               # Get max link weight for all PO nodes
        """Set max link weight feature for all PO nodes"""
        po = self.get_mask(Type.PO)
        max_values, _ = torch.max(self.links[po], dim=1, keepdim=True)  # (max_values, _) unpacks tuple returned by torch.max
        self.nodes[po, TF.MAX_SEM_WEIGHT] = max_values                  # Set max
    # ---------------------------------------------------------------
