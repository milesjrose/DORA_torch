# nodesMemTypes.py 
# Classes for segments of memory, and set-specific tensor operations
import torch
from .nodeMemObjects import *
from .nodeEnums import *
from . import tensorOps as tOps
from .nodeParameters import NodeParameters
# note : ignore higher order semantics for now - breaks compression.

class Tokens(object):
    """
    A class for holding a tensor of tokens, and performing general tensor operations.

    Attributes:
        names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        links (Links): A shared Links object containing interset links from tokens to semantics.
        mappings (Mappings): A shared Mappings object containing interset mappings from tokens to driver.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        masks (torch.Tensor): A Tensor of masks for the tokens in this set.
        IDs (dict): A dictionary mapping token IDs to index in the tensor.
        params (NodeParameters): An object containing shared parameters.
        token_set (Set): This set's enum, used to access links and mappings for this set in shared mem objects.
    """
    def __init__(self, floatTensor, connections, links: Links, mappings: Mappings, IDs: dict[int, int],names: dict[int, str] = {}, params: NodeParameters = None):
        """
        Initialize the TokenTensor object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            links (Links): A shared Links object containing interset links from tokens to semantics.
            mappings (Mappings): A shared Mappings object containing interset mappings from tokens to driver.
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
            params (NodeParameters, optional): An object containing shared parameters. Defaults to None.
        Raises:
            TypeError: If links, connections, or floatTensor are not torch.Tensor.
            ValueError: If the number of tokens in floatTensor, links, and connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
        """
        # Validate input
        if type(links) != Links:
            raise TypeError("Links must be Links object.")
        if type(mappings) != Mappings:
            raise TypeError("Mappings must be Mappings object.")
        if type(connections) != torch.Tensor:
            raise TypeError("Connections must be torch.Tensor.")
        if type(floatTensor) != torch.Tensor:
            raise TypeError("floatTensor must be torch.Tensor.")
        
        if floatTensor.size(dim=0) != links.size(dim=0):
            raise ValueError("floatTensor and links must have same number of tokens.")
        if floatTensor.size(dim=0) != connections.size(dim=0):
            raise ValueError("floatTensor and connections must have same number of tokens.")
        if floatTensor.size(dim=1) != len(TF):
            raise ValueError("floatTensor must have number of features listed in TF enum.")
        
        self.names = names
        self.nodes: torch.Tensor = floatTensor
        self.cache_masks()
        self.analogs = None
        self.analog_counts = None
        self.analog_node_count()
        self.links = links
        self.connections = connections
        self.IDs = IDs
        self.params = params
        self.expansion_factor = 1.1
        self.token_set = None

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get(self, ID, feature):
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

    def set(self, ID, feature, value):
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

    def get_name(self, ID):
        """
        Get the name for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to get the name for.
        """
        return self.names[ID]

    def set_name(self, ID, name):
        """
        Set the name for a token with a given ID.
        
        Args:
            ID (int): The ID of the token to set the name for.
            name (str): The name to set the token to.
        """
        self.names[ID] = name
    
    def get_ID(self, name):
        """
        Get the ID for a token with a given name.
        
        Args:
            name (str): The name of the token to get the ID for.
        """
        try:
            return self.IDs.keys()[self.IDs.values().index(name)]
        except:
            raise ValueError("Invalid name.")
    # --------------------------------------------------------------

    # ====================[ TENSOR FUNCTIONS ]======================
    def cache_masks(self, types_to_recompute = None):
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

    def add_token(self, token: New_Token):                          # Add a token to the tensor
        """
        Add a token to the tensor. If tensor is full, expand it first.

        Returns:
            ID (int): The ID of the added token.
        """
        spaces = torch.sum(self.nodes[:, TF.DELETED] == B.TRUE)             # Find number of spaces -> count of deleted nodes in the tensor
        if spaces == 0:                                                     # If no spaces, expand tensor
            self.expand_tensor()
        ID = self.IDs.keys()[-1] + 1                                        # assign token a new ID.
        token.tensor[TF.ID] = ID
        deleted_nodes = torch.where(self.nodes[:, TF.DELETED] == B.TRUE)[0] #find first deleted node
        first_deleted = deleted_nodes[0]
        self.nodes[first_deleted, :] = token.tensor                         # add token to tensor
        self.IDs[ID] = first_deleted                                        # update IDs
        self.cache_masks()                                                  # recompute masks
        return ID
    
    def expand_tensor(self):                                        # Expand nodes, links, mappings, connnections tensors by self.expansion_factor
        """
        Expand tensor by classes expansion factor. Minimum expansion is 5.
        Expands nodes, connections, links and mappings tensors.
        """
        current_count = self.nodes.size(dim=0)
        new_count = int(current_count * self.expansion_factor)              # calculate new size
        if new_count < 5:                                                   # minimum expansion is 5
            new_count = 5
        new_tensor = torch.zeros(new_count, self.nodes.size(dim=1))         # create new tensor with expansion factor
        new_tensor[:, TF.DELETED] = B.TRUE                                  # set all new tokens to deleted
        new_tensor[:self.nodes.size(dim=0), :] = self.nodes                 # copy over old tensor
        self.nodes = new_tensor                                             # update tensor

        new_links = torch.zeros(new_count, self.links.size(dim=1))          # update supporting data structures:
        new_links[:self.links.size(dim=0), :] = self.links
        self.links.sets[self.token_set] = new_links

        new_mappings = torch.zeros(new_count, self.mappings.size(dim=1))
        new_mappings[:self.mappings.size(dim=0), :] = self.mappings
        self.mappings.sets[self.token_set] = new_mappings

        new_connections = torch.zeros(new_count, new_count)
        new_connections[:self.connections.size(dim=0), :self.connections.size(dim=1)] = self.connections
        self.connections = new_connections

    def del_nodes(self, ID):                                        # Delete nodes from tensor     TODO: compress tensor if deleted nodes ratio hits threshold
        """
        Delete nodes from tensor
        
        Args:
            ID (int or list[int]): The ID(s) of the node(s) to delete.
        """
        if not isinstance(ID, list):
            ID = [ID]
        
        cache_types = [] 
        for id in ID:
            cache_types.append(self.nodes[self.IDs[id], TF.TYPE])
            self.nodes[self.IDs[id], TF.DELETED] = B.TRUE

        cache_types = list(set(cache_types))
        self.cache_masks(cache_types)

    def analog_node_count(self):                                    # Updates list of analogs in tensor, and their node counts
        """Update list of analogs in tensor, and their node counts"""
        self.analogs, self.analog_counts = torch.unique(self.nodes[:, TF.ANALOG], return_counts=True)
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
        
    def update_act(self):  # Update act of nodes
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

class Driver(Tokens):
    """
    A class for representing the driver set of tokens.

    Attributes:
        names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        links (torch.Tensor): A Tensor of links from tokens in this set to the semantics.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        masks (torch.Tensor): A Tensor of masks for the tokens in this set.
        IDs (dict): A dictionary mapping token IDs to index in the tensor.
        params (NodeParameters): An object containing shared parameters.
    """
    def __init__(self, floatTensor, connections, links, IDs: dict[int, int], names: dict[int, str] = {}, params: NodeParameters = None):   
        """
        Initialize the Driver object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            links (torch.Tensor): A Tensor of links between driver tokens and the semantics.
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
            params (NodeParameters, optional): An object containing shared parameters. Defaults to None.
        Raises:
            ValueError: If the number of tokens in floatTensor, links, and connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
            ValueError: If all tokens in floatTensor do not have TF.SET == Set.DRIVER.
            TypeError: If links is not a torch.Tensor.
        """
        super().__init__(floatTensor, connections, links, IDs, names, params)
        self.token_set = Set.DRIVER
        if floatTensor.size(dim=0) > 0:
            if not torch.all(floatTensor[:, TF.SET] == Set.DRIVER):
                raise ValueError("All tokens in driver floatTensor must have TF.SET == Set.DRIVER.")
    
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
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # each p node -> sum of act of connected group nodes
            )
        # 3). BU_INPUT: my_RBs
        self.nodes[p, TF.BU_INPUT] += torch.matmul(                 # matmul outputs martix (sum(p) x 1) of values to add to current input value
            self.connections[p, rb],                                # Masks connections between p[i] and its rbs
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
            self.connections[p, group],                             # Masks connections between p[i] and its groups
            self.nodes[group, TF.ACT]                               # For each p node -> sum of act of connected group nodes
            )
        # 2b). parent_rbs
        t_con = torch.transpose(self.connections)                   # transpose, so gives child -> parent connections
        self.nodes[p, TF.TD_INPUT] += torch.matmul(                 # matmul outputs matrix (sum(p) x 1) of values to add to current input value
            t_con[p, rb],                                           # Masks connections between p[i] and its rbs
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
        t_con = torch.transpose(self.connections)                   # Connnections: Parent -> child, take transpose to get list of parents instead
        self.nodes[rb, TF.TD_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            t_con[rb, p],                                           # Masks connections between rb[i] and its ps
            self.nodes[p, TF.ACT]                                   # For each rb node -> sum of act of connected p nodes
            )
        # 3). BU_INPUT: my_po, my_child_p                           # NOTE: Old function explicitly took myPred[0].act etc. as there should only be one pred/child/etc. This version sums all connections, so if rb mistakenly connected to multiple of a node type it will not give expected output.
        po_p = torch.bitwise_or(po, p)                              # Get mask of both pos and ps
        self.nodes[rb, TF.BU_INPUT] += torch.matmul(                # matmul outputs martix (sum(rb) x 1) of values to add to current input value
            self.connections[rb, po_p],                             # Masks connections between rb[i] and its po and child p nodes
            self.nodes[po_p, TF.ACT]                                # For each rb node -> sum of act of connected po and child p nodes
            )
        
        # Inhibitory input:
        # 4). LATERAL: (other RBs*3), inhibitor*10
        # 4a). (other RBs*3)
        diag_zeroes = tOps.diag_zeros(sum(rb))                      # Connects each rb to every other rb, but not themself
        self.nodes[rb, TF.LATERAL_INPUT_INPUT] -= torch.mul(
            3, 
            torch.matmul(                                           # matmul outputs martix (sum(rb) x 1) of values to add to current input value
                diag_zeroes,                                        # Masks connections between rb[i] and its po and child p nodes
                self.nodes[rb, TF.ACT]                              # For each rb node -> sum of act of connected po and child p nodes
            )
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[rb, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[rb, TF.LATERAL_INPUT_INPUT] -= inhib_act         # Update lat input
    
    def update_input_po(self):                                      # update PO inputs - driver:
        """Update input in driver for PO units"""
        as_DORA = self.params.as_DORA
        # Exitatory: td (my RB) * gain (2 for preds, 1 for objects).
        # Inhibitory: lateral (other POs not connected to my RB and Ps in child mode, if in DORA mode, then other PO connected to my RB), inhibitor.
        # 1). get masks
        rb = self.get_mask(Type.RB)
        po = self.get_mask(Type.PO)
        pred_sub = (self.nodes[po, TF.PRED] == B.TRUE)                # predicate sub mask of po nodes
        obj = tOps.refine_mask(self.nodes, po, TF.PRED, B.FALSE)    # get object mask
        pred = tOps.refine_mask(self.nodes, po, TF.PRED, B.TRUE)    # get predicate mask
        parent_cons = torch.transpose(self.connections)             # Transpose of connections matrix, so that index by child node (PO) to parent (RB)

        # Exitatory input:
        # 2). TD_INPUT: my_rb * gain(pred:2, obj:1)
        cons = parent_cons[po, rb].copy()                           # get copy of connections matrix from po to rb (child to parent connection)
        cons[pred_sub] = cons[pred_sub] * 2                         # multipy predicate -> rb connections by 2
        self.nodes[po, TF.TD_INPUT] += torch.matmul(                # matmul outputs martix (sum(po) x 1) of values to add to current input value
            cons,                                                   # Masks connections between po[i] and its rbs
            self.nodes[rb, TF.ACT]                                  # For each po node -> sum of act of connected rb nodes (multiplied by 2 for predicates)
            )
        
        # Inhibitory input:
        # 3). LATERAL: 3 * (if DORA_mode: PO connected to same rb / Else: POs not connected to same RBs)
        # 3a). find other PO connected to same RB     
        shared = torch.matmul(                                      # PxObject tensor, shared[i][j] > 1 if Po[i] and Po[j] share an RB, 0 o.w
            parent_cons[po, rb],
            self.connections[rb, po]
            ) 
        shared = torch.gt(shared, 0).int()                          # now shared[i][j] = 1 if p[i] and object[j] share an RB, 0 o.w
        diag_zeroes = tOps.diag_zeros(sum(po))                      # (po x po) matrix: 0 for po[i] -> po[i] connections, 1 o.w
        shared = torch.bitwise_and(shared, diag_zeroes)             # remove po[i] -> po[i] connections
        # 3ai). if DORA: Other po connected to same rb / else: POs not connected to same RBs
        if as_DORA: # PO connected to same rb
            po_connections = shared                                 # shared PO
        else:       # POs not connected to same RBs
            po_connections = 1 - shared                             # non shared PO: mask[i][j] = 0 if p[i] and object[j] share an RB, 1 o.w
        # 3aii). updaet lat input: 3 * (filtered nodes connected in po_connections)
        self.nodes[po, TF.LATERAL_INPUT] -= torch.mult(
            3,
            torch.matmul(
                po_connections,
                self.nodes[po, TF.ACT]
            )
        )
        # 4b). ihibitior * 10
        inhib_act = torch.mul(10, self.nodes[po, TF.INHIBITOR_ACT]) # Get inhibitor act * 10
        self.nodes[po, TF.LATERAL_INPUT_INPUT] -= inhib_act         # Update lat input
    # --------------------------------------------------------------

class Recipient(Tokens):
    """
    A class for representing the recipient set of tokens.
    
    Attributes:
        names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
        nodes (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
        analogs (torch.Tensor): An Ax1 tensor listing all analogs in the tensor.
        analog_counts (torch.Tensor): An Ax1 tensor listing the number of tokens per analog
        links (Links): A shared Links object containing interset links from tokens to semantics.
        mappings (Mappings): A shared Mappings object containing interset mappings from tokens to driver.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for tokens in this set.
        IDs (dict): A dictionary mapping token IDs to index in the tensor.
        params (NodeParameters): An object containing shared parameters.
    """
    def __init__(self, floatTensor, connections, links: Links, mappings: Mappings, IDs: dict[int, int], names= None, params: NodeParameters = None):
        """
        Initialize the Recipient object.

        Args:
            floatTensor (torch.Tensor): An NxTokenFeatures tensor of floats representing the tokens.
            connections (torch.Tensor): An NxN tensor of connections between the tokens.
            links (Links): A shared Links object
            mappings (Mappings): A shared Mappings object
            IDs (dict): A dictionary mapping token IDs to index in the tensor.
            names (dict, optional): A dictionary mapping token IDs to token names. Defaults to None.
            params (NodeParameters, optional): An object containing shared parameters. Defaults to None.
        Raises:
            ValueError: If the number of tokens in floatTensor, links, and connections do not match.
            ValueError: If the number of features in floatTensor does not match the number of features in TF enum.
            ValueError: If all tokens in floatTensor do not have TF.SET == Set.RECIPIENT.
        """
        super().__init__(floatTensor, connections, links, mappings, IDs, names, params)
        self.token_set = Set.RECIPIENT
        if floatTensor.size(dim=0) > 0:
            if not torch.all(floatTensor[:, TF.SET] == Set.RECIPIENT):
                raise ValueError("All tokens in recipient floatTensor must have TF.SET == Set.RECIPIENT.")

    # ============[ RECIPIENT UPDATE INPUT FUNCTIONS ]==============
    def update_input(self, semantics, mappings, driver):                # Update all input in recipient
        """
        Update all input in recipient
        
        Args:
            semantics (Semantics): A Semantics object
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        """
        self.update_input_p_parent(mappings, driver)
        self.update_input_p_child(mappings, driver)
        self.update_input_rb(mappings, driver)
        self.update_input_po(semantics, mappings, driver)

    def update_input_p_parent(self, mappings: Mappings, driver: Driver):    # P units in parent mode - recipient
        """
        Update input for P units in parent mode
        
        Args:
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
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
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p, mappings, driver) 
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
    
    def update_input_p_child(self, mappings: Mappings, driver: Driver):     # P Units in child mode - recipient:
        """
        Update input for P units in child mode

        Args:
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
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
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p, mappings, driver) 
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
   
    def update_input_rb(self, mappings: Mappings, driver: Driver):          # RB inputs - recipient
        """
        Update input for RB units

        Args:
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
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
        self.nodes[rb, TF.MAP_INPUT] += self.map_input(rb, mappings, driver) 
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
    
    def update_input_po(self, semantics, mappings, driver):                 # PO units in - recipient
        """
        Update input for PO units

        Args:
            semantics (Semantics): A Semantics object
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        """
        as_DORA = self.params.as_DORA
        phase_set = self.params.phase_set
        lateral_input_level = self.params.lateral_input_level
        ignore_object_semantics = self.params.ignore_object_semantics
        
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
        self.nodes[po, TF.MAP_INPUT] += self.map_input(po, mappings, driver) 
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
    def map_input(self, t_mask, mappings: Mappings, driver: Driver):        # Return (sum(t_mask) x 1) matrix of mapping_input for tokens in mask
        """
        Calculate mapping input for tokens in mask

        Args:
            t_mask (torch.Tensor): A mask of tokens to calculate mapping input for
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        
        Returns:
            torch.Tensor: A (sum(t_mask) x 1) matrix of mapping input for tokens in mask
        """
        pmap_weights = mappings.weights()[t_mask] 
        pmap_connections = mappings.connections()[t_mask]

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

class New_Set(Tokens):
    """
    A class for representing a new set of tokens.
    """
    def __init__(self, floatTensor, connections, links, IDs: dict[int, int], names: dict[int, str] = {}, params: NodeParameters = None):
        super().__init__(floatTensor, connections, links, IDs, names, params)
        self.token_set = Set.NEW_SET
    
    def update_input_type(self, n_type: Type):
        """
        Update the input for a given type of token.
        """
        

class Memory(Tokens):
    """
    A class for representing a memory of tokens.
    """
    def __init__(self, floatTensor, connections, links, IDs: dict[int, int], names: dict[int, str] = {}, params: NodeParameters = None):
        super().__init__(floatTensor, connections, links, IDs, names, params)
        self.token_set = Set.MEMORY

    # =========[ USES RECIPIENT UPDATE INPUT FUNCTIONS ]===========
    def update_input(self, as_DORA, phase_set, lateral_input_level, semantics, mappings, driver, ignore_object_semantics=False): # Update all input in recipient
        """
        Update all input in recipient
        
        Args:
            as_DORA (bool): Whether to use DORA mode
            phase_set (int): The current phase set
            lateral_input_level (float): The level of lateral input
            semantics (Semantics): A Semantics object
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
            ignore_object_semantics (bool): Whether to ignore object semantics
        """
        self.update_input_p_parent(phase_set, lateral_input_level, mappings, driver)
        self.update_input_p_child(as_DORA, phase_set, lateral_input_level, mappings, driver)
        self.update_input_rb(phase_set, lateral_input_level, mappings, driver)
        self.update_input_po(as_DORA, phase_set, lateral_input_level, semantics, mappings, driver, ignore_object_semantics)

    def update_input_p_parent(self, phase_set, lateral_input_level, mappings: Mappings, driver: Driver):  # P units in parent mode - recipient
        """
        Update input for P units in parent mode
        
        Args:
            phase_set (int): The current phase set
            lateral_input_level (float): The level of lateral input
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        """
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
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p, mappings, driver) 
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
    
    def update_input_p_child(self, as_DORA, phase_set, lateral_input_level, mappings: Mappings, driver: Driver): # P Units in child mode - recipient:
        """
        Update input for P units in child mode

        Args:
            as_DORA (bool): Whether to use DORA mode
            phase_set (int): The current phase set
            lateral_input_level (float): The level of lateral input
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        """
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
        self.nodes[p, TF.MAP_INPUT] += self.map_input(p, mappings, driver) 
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
   
    def update_input_rb(self, phase_set, lateral_input_level, mappings: Mappings, driver: Driver): # RB inputs - recipient
        """
        Update input for RB units

        Args:
            phase_set (int): The current phase set
            lateral_input_level (float): The level of lateral input
            mappings (Mappings): A Mappings object
            driver (Driver): A Driver object
        """
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
        self.nodes[rb, TF.MAP_INPUT] += self.map_input(rb, mappings, driver) 
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
    
    def update_input_po(self, as_DORA, phase_set, lateral_input_level, semantics, mappings, driver, ignore_object_semantics=False): # PO units in - recipient
        """
        Update input for PO units

        Args:
            as_DORA (bool): Whether to use DORA mode
            phase_set (int): The current phase set
            lateral_input_level (float): The level of lateral input
        """
        
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
        self.nodes[po, TF.MAP_INPUT] += self.map_input(po, mappings, driver) 
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

class Semantics(object):
    """
    A class for representing semantics nodes.

    Attributes:
        IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
        names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
        nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
        links (Links): A Links object containing links from token sets to semantics.
        params (NodeParameters): An object containing shared parameters. Defaults to None.
    """
    def __init__(self, nodes, connections, links: Links, IDs: dict[int, int], names= None, params: NodeParameters = None):
        """
        Initialise a Semantics object

        Args:
            nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
            connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
            links (Links): A Links object containing links from token sets to semantics.
            IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
            names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
            params (NodeParameters, optional): An object containing shared parameters. Defaults to None.
        Raises:
            ValueError: If the number of semantics in nodes, connections, and links do not match.
            ValueError: If the number of features in nodes does not match the number of features in SF enum.
            ValueError: If the number of semantics in links tensors do not match the number of semantics in nodes.
        """
        if nodes.size(dim=0) != connections.size(dim=0):
            raise ValueError("nodes and connections must have the same number of semantics.")
        if nodes.size(dim=1) != len(SF):
            raise ValueError("nodes must have number of features listed in SF enum.")
        if links.driver.size(dim=1) != nodes.size(dim=0):
            raise ValueError("links tensors must have same number of semantics as semantics.nodes")
        self.names = names 
        self.nodes: torch.Tensor = nodes
        self.connections: torch.Tensor = connections
        self.all_links = links
        self.IDs = IDs
        self.params = params

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get(self, ID, feature):
        """
        Get a feature for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to get the feature for.
            feature (TF): The feature to get.

        Returns:
            The feature for the semantic with the given ID.
        """
        try:
            return self.nodes[self.IDs[ID], feature]
        except:
            raise ValueError("Invalid ID or feature.")

    def set(self, ID, feature, value):
        """
        Set a feature for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.
        """
        if type(feature) != TF:
            raise TypeError("Feature must be a TF enum.")
        try:
            self.nodes[self.IDs[ID], feature] = float(value)
        except:
            raise ValueError("Invalid ID or feature.")

    def get_name(self, ID):
        """
        Get the name for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to get the name for.
        """
        return self.names[ID]

    def set_name(self, ID, name):
        """
        Set the name for a semantic with a given ID.
        
        Args:
            ID (int): The ID of the semantic to set the name for.
            name (str): The name to set the semantic to.
        """
        self.names[ID] = name
    
    def get_ID(self, name):
        """
        Get the ID for a semantic with a given name.
        
        Args:
            name (str): The name of the semantic to get the ID for.
        """
        try:
            return self.IDs.keys()[self.IDs.values().index(name)]
        except:
            raise ValueError("Invalid name.")
    # --------------------------------------------------------------

    # ===================[ SEMANTIC FUNCTIONS ]=====================
    def intitialse_sem(self):                                       # Set act and input to 0 TODO: Check how used
        """Initialise the semantics """
        self.nodes[:, SF.ACT] = 0.0
        self.nodes[:, SF.INPUT] = 0.0

    def initialise_input(self, refresh):                            # Set nodes to refresh value TODO: Check how used
        """Initialise the input of the semantics """
        self.nodes[:, SF.INPUT] = refresh

    def set_max_input(self, max_input):                             # TODO: Check how used
        """Set the max input of the semantics """
        self.nodes[:, SF.MAX_INPUT] = max_input

    def update_act(self):                                           # Update act of all sems
        """Update the acts of the semantics """
        sem_mask = self.nodes[:, SF.MAX_INPUT] > 0                  # Get sem where max_input > 0
        input = self.nodes[sem_mask, SF.INPUT]
        max_input = self.nodes[sem_mask, SF.MAX_INPUT]
        self.nodes[sem_mask, SF.ACT] = input / max_input            # - Set act of sem to input/max_input
        sem_mask = self.nodes[:, SF.MAX_INPUT] == 0                 # Get sem where max_input == 0       
        self.nodes[sem_mask, SF.ACT] = 0.0                          #  -  Set act of sem to 0
    
    def update_input(self, driver, recipient, memory = None, ignore_obj=False, ignore_mem=False):
        """Update the input of the semantics """
        self.update_input_from_set(driver, Set.DRIVER, ignore_obj)
        self.update_input_from_set(recipient, Set.RECIPIENT, ignore_obj)
        if not ignore_mem:
            self.update_input_from_set(memory, Set.MEMORY, ignore_obj)

    def update_input_from_set(self, tensor: Tokens, set: Set, ignore_obj=False):
        """Update the input of the semantics from a set of tokens """
        if ignore_obj:
            po_mask = tOps.refine_mask(po_mask, tensor.get_mask(Type.PO), TF.PRED, B.TRUE) # Get mask of POs non object POs
        else:
            po_mask = tensor.get_mask(Type.PO)
        #group_mask = tensor.get_mask(Type.GROUP)
        #token_mask = torch.bitwise_or(po_mask, group_mask)         # In case groups used in future

        links: torch.Tensor = self.all_links[set]
        connected_nodes = (links[:, po_mask] != 0).any(dim=1)       # Get mask of nodes linked to a sem
        connected_sem = (links != 0).any(dim=0)                     # Get mask of sems linked to a node

        sem_input = torch.matmul(                                   # Get sum of act * link_weight for all connected nodes and sems
            links[connected_sem, connected_nodes],                  # connected_sem x connected_nodes matrix of link weights
            tensor.nodes[connected_nodes, TF.ACT]                   # connected_nodes x 1 matrix of node acts
        )
        self.nodes[connected_sem, SF.INPUT] += sem_input            # Update input of connected sems
    # --------------------------------------------------------------