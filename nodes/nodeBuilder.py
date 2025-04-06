import numpy as np
import torch
from .nodeEnums import *
from .nodes import Nodes
from .nodeTensors import *
from .nodeMemObjects import Links, Mappings

# ===========[ INTERMIDIATE DATA STRUCTURES ]===========
class Inter_Node(object):
    """
    An intermediate class for representing a node in the network.

    Attributes:
        name (str): The name of the node.
        features (list): A list of features for the node.
        ID (int): The ID of the node.
    """
    def __init__(self, name):
        """
        Initialise the node.

        Args:
            name (str): The name of the node.
        """
        self.name = name
        self.features = None
        self.ID = None
    
    def set(self, feature, value: float):
        """
        Set the feature of the node.

        Args:
            feature (str): The feature to set.
            value (float): The value to set the feature to.
        """
        self.features[feature] = value
    
    def set_ID(self, ID):
        """
        Set the ID of the node.

        Args:
            ID (int): The ID to set the node to.
        """
        self.ID = ID
        self.features[TF.ID] = ID

class Inter_Semantics(Inter_Node):
    """
    An intermediate class for representing a semantic node.

    Attributes:
        name (str): The name of the semantic.
        features (list): A list of features for the semantic, indexed by SF.
        ID (int): The ID of the semantic.
    """
    def __init__(self, name):
        """
        Initialise the semantic with name, and default features.

        Args:
            name (str): The name of the semantic.
        """
        super().__init__(name)
        self.features = [None] * len(SF)
        self.initialise_defaults()
    
    def initialise_defaults(self):      # TODO: Check defaults
        """
        Initialise the default features for the semantic.
        """
        self.features[SF.TYPE] = Type.SEMANTIC
        self.features[SF.ONT_STATUS] = 0
        self.features[SF.AMOUNT] = 0
        self.features[SF.INPUT] = 0
        self.features[SF.MAX_INPUT] = 0
        self.features[SF.ACT] = 0
    
    def floatate_features(self):
        """
        Convert semantic features to floats, required for tensorisation.
        """
        for feature in SF:
            if self.features[feature] is not None:
                self.features[feature] = float(self.features[feature])
            else:
                self.features[feature] = 0.0

class Inter_Token(Inter_Node):
    """
    An intermediate class for representing a token node.

    Attributes:
        name (str): The name of the token.
        features (list): A list of features for the token, indexed by TF.
        ID (int): The ID of the token.
        children (list): A list of children of the token, for use in building the connections matrix.
    """
    def __init__(self, name, set: Set, analog: int):
        """
        Initialise the token with name, set, and analog, and default features.

        Args:
            name (str): The name of the token.
            set (Set): The set of the token.
            analog (int): The analog of the token.
        """
        super().__init__(name)
        self.features = [None] * len(TF)
        self.initialise_defaults()
        self.set(TF.SET, set)
        self.set(TF.ANALOG, analog)
        self.children = []
    
    def initialise_defaults(self):   # TODO: Check defaults
        """
        Initialise the default features for the token.
        """
        self.features[TF.ID] = None
        self.features[TF.TYPE] = None
        self.features[TF.SET] = None
        self.features[TF.ANALOG] = None
        self.features[TF.MAX_MAP_UNIT] = 0
        self.features[TF.MADE_UNIT] = 0
        self.features[TF.MAKER_UNIT] = 0
        self.features[TF.INHIBITOR_THRESHOLD] = 0
        self.features[TF.GROUP_LAYER] = 0
        self.features[TF.MODE] = 0
        self.features[TF.TIMES_FIRED] = 0
        self.features[TF.SEM_COUNT] = 0
        self.features[TF.ACT] = 0
        self.features[TF.MAX_ACT] = 0
        self.features[TF.INHIBITOR_INPUT] = 0
        self.features[TF.INHIBITOR_ACT] = 0
        self.features[TF.MAX_MAP] = 0
        self.features[TF.NET_INPUT] = 0
        self.features[TF.MAX_SEM_WEIGHT] = 0
        self.features[TF.INFERRED] = False
        self.features[TF.RETRIEVED] = False
        self.features[TF.COPY_FOR_DR] = False
        self.features[TF.COPIED_DR_INDEX] = 0
        self.features[TF.SIM_MADE] = False
        self.features[TF.DELETED] = False
        self.features[TF.PRED] = False
    
    def floatate_features(self):
        """
        Convert token features to floats, required for tensorisation.
        """
        for feature in TF:
            if self.features[feature] is not None:
                self.features[feature] = float(self.features[feature])
            else:
                self.features[feature] = 0.0

class Inter_Prop(Inter_Token):
    """
    An intermediate class for representing a Prop node.

    Attributes:
        name (str): The name of the Prop.
        features (list): A list of features for the Prop, indexed by TF.
        ID (int): The ID of the Prop.
        children (list): A list of children of the Prop, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog):
        """
        Initialise the Prop with name, set and analog, and default Pfeatures.

        Args:
            name (str): The name of the Prop.
            set (Set): The set of the Prop.
            analog (int): The analog of the Prop.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.P)

class Inter_RB(Inter_Token):
    """
    An intermediate class for representing a RB node.

    Attributes:
        name (str): The name of the RB.
        features (list): A list of features for the RB, indexed by TF.
        ID (int): The ID of the RB.
        children (list): A list of children of the RB, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog):
        """
        Initialise the RB with name, set and analog, and default RB features.

        Args:
            name (str): The name of the RB.
            set (Set): The set of the RB.
            analog (int): The analog of the RB.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.RB)

class Inter_PO(Inter_Token):
    """
    An intermediate class for representing a PO node.

    Attributes:
        name (str): The name of the PO.
        features (list): A list of features for the PO, indexed by TF.
        ID (int): The ID of the PO.
        children (list): A list of children of the PO, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog, is_pred: bool):
        """
        Initialise the PO with name, set, analog, and is_pred, and default PO features.

        Args:
            name (str): The name of the PO.
            set (Set): The set of the PO.
            analog (int): The analog of the PO.
            is_pred (bool): Whether the PO is a predicate.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.PO)
        self.set(TF.PRED, is_pred)

class Token_set(object):
    """
    An intermediate class for representing a set of tokens.

    Attributes:
        set (Set): The set of the tokens.
        tokens (dict): A dictionary of tokens, mapping type to list of tokens.
        name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
        id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
        num_tokens (int): The number of tokens in the set.
        connections (np.ndarray): A matrix of connections between tokens.
        links (np.ndarray): A matrix of links between tokens and semantics.
    """
    def __init__(self, set: Set, tokens: dict[Type, list[Inter_Token]], name_dict: dict[str, Inter_Token], id_dict: dict[int, Inter_Token]):
        """
        Initialise the token set with set, tokens, name_dict, and id_dict.

        Args:
            set (Set): The set of the tokens.
            tokens (dict): A dictionary of tokens, mapping type to list of tokens.
            name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
            id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
        """
        self.set = set
        self.tokens = tokens
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_tokens = sum([len(self.tokens[type]) for type in Type])
        self.connections = np.zeros((self.num_tokens, self.num_tokens))
        self.links = np.zeros((self.num_tokens, self.num_tokens))
    
    def get_token(self, name):
        """
        Get a token from the token set by name.

        Args:
            name (str): The name of the token.
        """
        return self.name_dict[name]

    def get_token_by_id(self, ID):
        """
        Get a token from the token set by ID.

        Args:
            ID (int): The ID of the token.
        """
        return self.id_dict[ID] 
    
    def get_token_tensor(self):
        """
        Get the token tensor for the token set.
        """
        token_tensor = torch.zeros((self.num_tokens, len(TF)))
        for type in Type:
            for token in self.tokens[type]:
                token.floatate_features()
                token_tensor[token.ID] = torch.tensor(token.features)
        return token_tensor
    
    def tensorise(self):
        """
        Tensorise the token set, creating a tensor of tokens, and tensors of connections and links to semantics.
        """
        self.token_tensor = self.get_token_tensor()
        self.connections_tensor = torch.tensor(self.connections)
        self.links_tensor = torch.tensor(self.links)

class Sem_set(object):
    """
    An intermediate class for representing a set of semantics.

    Attributes:
        sems (list): A list of semantics.
        name_dict (dict): A dictionary of semantics, mapping name to semantic in sems.
        id_dict (dict): A dictionary of semantics, mapping ID to semantic in sems.
        num_sems (int): The number of semantics in the set.
        connections (np.ndarray): A matrix of connections between semantics.
    """
    def __init__(self, sems: list[Inter_Semantics], name_dict: dict[str, Inter_Semantics], id_dict: dict[int, Inter_Semantics]):
        """
        Initialise the semantic set with sems, name_dict, and id_dict.

        Args:
            sems (list): A list of semantics.
            name_dict (dict): A dictionary of semantics, mapping name to semantic in sems.
            id_dict (dict): A dictionary of semantics, mapping ID to semantic in sems.
        """
        self.sems = sems
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_sems = len(sems)
        self.connections = np.zeros((self.num_sems, self.num_sems))
    
    def get_sem(self, name):
        """
        Get a semantic from the semantic set by name.

        Args:
            name (str): The name of the semantic.
        """
        return self.name_dict[name]
    
    def get_sem_by_id(self, ID):
        """
        Get a semantic from the semantic set by ID.

        Args:
            ID (int): The ID of the semantic.
        """
        return self.id_dict[ID]

    def tensorise(self):
        """
        Tensorise the semantic set, creating a tensor of semantics, and a tensor of connections between semantics.
        """
        self.node_tensor = torch.zeros((self.num_sems, len(SF)))
        self.connections_tensor = torch.tensor(self.connections)
        for sem in self.sems:
            sem.floatate_features()
            self.node_tensor[sem.ID] = torch.tensor(sem.features)
# ------------------------------------------------------


# ===================[ BUILD CLASSES ]==================
class Build_set(object):                                # Builds the nodes for a given set
    """
    A class for building the nodes for a given set.

    Attributes:
        symProps (list): A list of symProps relating to the set.
        tokens (dict): A dictionary of tokens, mapping type to list of tokens.
        set (Set): The set to be built.
        name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
        id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
    """
    def __init__(self, symProps: list[dict], set: Set):
        """
        Initialise the build set with symProps, set, and empty dictionaries for tokens, name_dict, and id_dict.

        Args:
            symProps (list): A list of symProps relating to the set.
            set (Set): The set to be built.
        """
        self.symProps = symProps
        self.tokens = {}
        self.set = set
        self.name_dict = {}
        self.id_dict = {}
        self.names = []
        self.token_set = None
    
    def build_set(self):    #  Returns a new token_set
        """
        Returns a new token_set object
        """
        self.get_nodes()       # Get the nodes from the symProps
        self.id_tokens()       # Create basic tokens
        self.token_set = Token_set(self.set, self.tokens, self.name_dict, self.id_dict)   # Return new token_set
        return self.token_set


    def get_nodes(self):         # Gets lists of unique tokens by type
        """
        Gets lists of unique tokens by type.
        """
        for type in Type:                    # Initialise empty lists for each type
            self.tokens[type] = []
        
        non_exist = "non_exist"
        for prop in self.symProps:
            prop_name = prop['name']
            analog = prop['analog']
            if prop_name != non_exist:
                self.create_token(prop_name, Inter_Prop, analog)
            for rb in prop['RBs']:
                pred_name = rb['pred_name']
                obj_name = rb['object_name']
                self.create_token(pred_name + "_" + obj_name, Inter_RB, analog)
                if pred_name != non_exist:
                    self.create_token(pred_name, Inter_PO, analog, True)
                if obj_name != non_exist:
                    self.create_token(obj_name, Inter_PO, analog, False)


    def id_tokens(self):   # Give each token an ID
        """
        Assign each token an ID, unique for the set.
        """
        self.names = list(set(self.names))
        i = 0
        for name in self.names:
            node_obj =  self.name_dict[name]
            node_obj.set_ID(i)
            self.id_dict[i] = name
            i += 1

    def create_token(self, name, token_class, analog, is_pred = None): # Create a token and add it to the name/dict
        """
        Create a token object and add it to the name/dict.

        Args:
            name (str): The name of the token.
            token_class (Token): The class of the token.
            analog (int): The analog of the token.
            is_pred (bool): Whether the token is a predicate.
        """
        if name not in self.names:
            if is_pred is not None:
                obj = token_class(name, self.set, analog, is_pred)
            else:
                obj = token_class(name, self.set, analog)
            self.tokens[obj.features[TF.TYPE]].append(obj)
            self.name_dict[name] = obj
            self.names.append(name)


class Build_sems(object):                               # Builds the semantic objects
    """
    A class for building the semantic objects.

    Attributes:
        sems (list): A list of semantic names.
        nodes (list): A list of semantic objects.
        name_dict (dict): A dictionary of semantic objects, mapping name to semantic object in nodes.
        id_dict (dict): A dictionary of semantic objects, mapping ID to semantic object in nodes.
        num_sems (int): The number of semantics, iterated from 0 when assigning IDs.
        symProps (list): A list of symProps relating to the set.
    """
    def __init__(self, symProps: list[dict]):
        """
        Initialise the build sems with symProps, and empty dictionaries for nodes, name_dict, and id_dict.

        Args:
            symProps (list): A list of symProps relating to the set.
        """
        self.sems = []
        self.nodes = []
        self.name_dict = {}
        self.id_dict = {}
        self.num_sems = 0
        self.symProps = symProps
    
    def build_sems(self):
        """
        Create the sem_set object.
        """
        self.get_sems(self.symProps)
        self.nodulate()
        self.sem_set = Sem_set(self.nodes, self.name_dict, self.id_dict)
        return self.sem_set

    def nodulate(self):                                 # Turn each unique semantic into a semantic object:
        """
        Turn each unique semantic into a semantic object (node) with a unique ID.
        """
        self.num_sems = 0
        for sem in self.sems:
            new_sem = Inter_Semantics(sem)
            new_sem.set_ID(self.num_sems)
            self.nodes.append(new_sem)
            self.id_dict[self.num_sems] = sem
            self.name_dict[sem] = new_sem
            self.num_sems += 1
    
    def get_sems(self, symProps: list[dict]):           # Depth first search to get list of all nodes in the symProps:
        """
        Get the list of all semantic names in the symProps.
        """
        for prop in symProps:
            for rb in prop['RBs']:
                if rb['pred_sem'] is not []:
                    for sem in rb['pred_sem']:
                        self.sems.append(sem)
                if rb['object_sem'] is not []:
                    for sem in rb['object_sem']:
                        self.sems.append(sem)

        self.sems = list(set(self.sems))
        self.num_sems = len(self.sems)


class Build_children(object):                           # Builds the children of each token
    """
    A class for building the list of children for each token.

    Attributes:
        set (Set): The set of the tokens.
        tokens (Token_set): The token set.
        sems (Sem_set): The semantic set.
        symProps (list): A list of symProps relating to the set.
    """
    def __init__(self, set: Set, tokens: Token_set, sems: Sem_set, symProps: list[dict]):
        """
        Initialise the build children with set, tokens, sems, and symProps.

        Args:
            set (Set): The set of the tokens.
            tokens (Token_set): The token set object.
            sems (Sem_set): The semantic set object.
            symProps (list): A list of symProps relating to the set.
        """
        self.set = set
        self.token_set = tokens
        self.sems = sems
        self.symProps = symProps

    def get_children(self):                                 # Add child nodes to each token object
        """
        Recursively add child nodes IDs to each token objects children list.
        """
        for prop in self.symProps:
            self.get_prop_children(prop)
        for type in Type:                                   # Remove any duplicates
            for token in self.token_set.tokens[type]:
                token.children = list(set(token.children))
    
    def get_prop_children(self, prop: dict):                # Add children to the prop object, recursively call lower get child functions
        """
        Step one in recursively adding child nodes IDs to each token objects children list.
        """
        prop_obj = self.get_object(prop['name'])            # Get the prop object
        for rb in prop['RBs']:
            child = self.get_rb_children(rb)                # Find rbs children, and return rb ID
            if prop_obj is not None:
                prop_obj.children.append(child)             # Append child ID to prop object

    def get_rb_children(self, rb: dict):                    # Add children to the rb object, recursively call lower get child functions
        """
        Step two in recursively adding child nodes IDs to each token objects children list.
        """
        pred_name = rb['pred_name']
        obj_name = rb['object_name']
        rb_obj = self.get_object(pred_name + "_" + obj_name)
        child_pred = self.get_po_children(pred_name, rb['pred_sem'])      # Get children of pred, return pred ID
        child_obj = self.get_po_children(obj_name, rb['object_sem'])   # Get children of obj, return obj ID
        rb_obj.children.append(child_pred)
        rb_obj.children.append(child_obj)
        return rb_obj.ID

    def get_po_children(self, name, sems: list):       # Add children to the po object, return po ID
        """
        Step three in recursively adding child nodes IDs to each token objects children list.
        """
        po_obj = self.get_object(name)
        for sem in sems:
            sem_obj = self.sems.get_sem(sem)
            po_obj.children.append(sem_obj.ID)
        return po_obj.ID
    
    def get_object(self, name):                             # Returns token object, if it exists.  O.w returns None
        """
        Return token object if it exists. Else return None

        Returns:
            token (Token): The token object.
            None: If the token does not exist.
        """
        non_exist = "non_exist"
        non_exist_rb = "non_exist_non_exist"
        obj = None
        if name != non_exist and name != non_exist_rb:
            obj = self.token_set.get_token(name)
        return obj  


class Build_connections(object):                        # Builds links and connections for each set
    """
    A class for building the links and connections for each set.    

    Attributes:
        token_sets (dict): A dictionary of token sets, mapping set to token set.
        sems (Sem_set): The semantic set object.
    """
    def __init__(self, token_sets: dict[Set, Token_set], sems: Sem_set):
        """
        Initialise the build connections with token_sets and sems.

        Args:
            token_sets (dict): A dictionary of token sets, mapping set to token_set object.
            sems (Sem_set): The semantic set object.
        """
        self.token_sets = token_sets
        self.sems = sems

    def build_connections_links(self):
        """
        Build the connections and links for each set.
        """
        for set in Set:
            token_set = self.token_sets[set]
            token_set.connections = self.build_set_connections(token_set)
            token_set.links = self.build_set_links(token_set)

    def build_set_connections(self, token_set: Token_set):  # Returns matrix of all connections for a given set
        """
        Build the connections matrix for a given set.
        
        Returns:
            connections (np.ndarray): The NxN connections matrix for the set.
        """
        num_tks = token_set.num_tokens
        connections = np.zeros((num_tks, num_tks))          # len tokens x len tokens matrix for connections.
        for type in Type:
            if type != Type.PO:
                for node in token_set.tokens[type]:
                    for child in node.children:
                        connections[node.ID][child] = 1
        return connections
    
    def build_set_links(self, token_set: Token_set):        # Returns matrix of all po -> sem links for a given set
        """
        Build the links matrix for a given set.

        Returns:
            links (np.ndarray): The NxM links matrix for the set.
        """
        num_tks = token_set.num_tokens
        num_sems = self.sems.num_sems
        links = np.zeros((num_tks, num_sems))               # Len tokens x len sems matrix for links.
        for po in token_set.tokens[Type.PO]:
            for child in po.children:
                links[po.ID][child] = 1
        return links


class nodeBuilder(object):                            # Builds tensors for each set, memory, and semantic objects. Finally build the nodes object.
    """
    A class for building the nodes object.

    Attributes:
        symProps (list): A list of symProps.
        file_path (str): The path to the sym file.
        token_sets (dict): A dictionary of token sets, mapping set to token set object.
        mappings (dict): A dictionary of mapping object, mappings sets to mapping object.
        set_map (dict): A dictionary of set mappings, mapping set name to set. Used for reading set from symProps file.
    """
    def __init__(self, symProps: list[dict] = None, file_path: str = None):
        """
        Initialise the nodeBuilder with symProps and file_path.

        Args:
            symProps (list): A list of symProps.
            file_path (str): The path to the sym file.
        """
        self.symProps = symProps
        self.file_path = file_path
        self.token_sets = {}
        self.mappings = {}
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }

    def build_nodes(self, DORA_mode= True):          # Build the nodes object
        """
        Build the nodes object.

        Args:
            DORA_mode (bool): Whether to use DORA mode.
        
        Returns:
            nodes (Nodes): The nodes object.
        
        Raises:
            ValueError: If no symProps or file_path set.
        """
        if self.file_path is not None:
            self.get_symProps_from_file()
        if self.symProps is not None:
            self.build_set_tensors()
            self.build_node_tensors()
            self.nodes = Nodes(self.driver_tensor, self.recipient_tensor, self.memory_tensor, self.new_set_tensor, self.semantics_tensor, self.mappings, DORA_mode)
            return self.nodes
        else:
            raise ValueError("No symProps or file_path provided")

    def build_set_tensors(self):    # Build sem_set, token_sets
        """
        Build the sem_set and token_sets.
        """
        props = {}

        # 1). Build the sems
        build_sems = Build_sems(self.symProps)
        self.sems = build_sems.build_sems()

        # 2). Initiliase empty lists for each set
        for set in Set:                                 
            props[set] = []

        # 3). Add the prop to the correct set
        for prop in self.symProps:
            props[self.set_map[prop["set"]]].append(prop)    
        
        # 4). Build the basic token sets
        for set in Set:                                
            build_set = Build_set(props[set], set)
            self.token_sets[set] = build_set.build_set()
        
        # 5). Build the children lists of IDs
        for set in Set:
            build_children = Build_children(set, self.token_sets[set], self.sems, props[set])
            build_children.get_children()
        
        # 6). Build the connections and links matrices                              
        build_connections = Build_connections(self.token_sets, self.sems)
        build_connections.build_connections_links()
        
        # 7). Tensorise the sets
        for set in Set:
            self.token_sets[set].tensorise()
        self.sems.tensorise()

    def build_link_object(self):    # Build the mem objects
        """
        Build the mem objects. (links, mappings)
        """
        # Create links object
        driver_links = self.token_sets[Set.DRIVER].links_tensor
        recipient_links = self.token_sets[Set.RECIPIENT].links_tensor
        memory_links = self.token_sets[Set.MEMORY].links_tensor
        new_set_links = self.token_sets[Set.NEW_SET].links_tensor
        self.links = Links(driver_links, recipient_links, memory_links, new_set_links, None)
        return self.links
    
    def build_map_object(self,set):
        # Create mapping tensors
        map_cons = torch.zeros(self.token_sets[set].num_tokens, self.token_sets[Set.DRIVER].num_tokens)
        map_weights = torch.zeros_like(map_cons)
        map_hyp = torch.zeros_like(map_cons)
        map_max_hyp = torch.zeros_like(map_cons)
                # Create mappings object
        mappings = Mappings(self.driver_tensor, map_cons, map_weights, map_hyp, map_max_hyp)
        self.mappings[set] = mappings
        return mappings
        

    def build_node_tensors(self):   # Build the node tensor objects
        """
        Build the per set tensor objects. (driver, recipient, memory, new_set, semantics)
        """
        self.build_link_object()
        self.semantics_tensor: Semantics = Semantics(self.sems.node_tensor, self.sems.connections_tensor, self.links, self.sems.id_dict)
        self.links.semantics = self.semantics_tensor

        driver_set = self.token_sets[Set.DRIVER]
        self.driver_tensor = Driver(driver_set.token_tensor, driver_set.connections_tensor, self.links, driver_set.id_dict)
        
        recipient_set = self.token_sets[Set.RECIPIENT]
        recipient_maps = self.build_map_object(Set.RECIPIENT)
        self.recipient_tensor = Recipient(recipient_set.token_tensor, recipient_set.connections_tensor, self.links, recipient_maps, recipient_set.id_dict)

        memory_set = self.token_sets[Set.MEMORY]
        mem_maps = self.build_map_object(Set.MEMORY)
        self.memory_tensor = Memory(memory_set.token_tensor, memory_set.connections_tensor, self.links, mem_maps, memory_set.id_dict)

        new_set = self.token_sets[Set.NEW_SET]
        self.new_set_tensor = New_Set(new_set.token_tensor, new_set.connections_tensor, self.links, new_set.id_dict)
    
    def get_symProps_from_file(self):
        """
        Read the symProps from the file into a list of dicts.
        """
        import json
        file = open(self.file_path, "r")    
        if file:
            simType = ""
            di = {"simType": simType}  # porting from Python2 to Python3
            file.seek(0)  # to get to the beginning of the file.
            exec(file.readline(), di)  # porting from Python2 to Python3
            if di["simType"] == "sym_file":  # porting from Python2 to Python3
                symstring = ""
                for line in file:
                    symstring += line
                do_run = True
                symProps = []  # porting from Python2 to Python3
                di = {"symProps": symProps}  # porting from Python2 to Python3
                exec(symstring, di)  # porting from Python2 to Python3
                self.symProps = di["symProps"]  # porting from Python2 to Python3
            # now load the parameter file, if there is one.
            # if self.parameterFile:
            #     parameter_string = ''
            #     for line in self.parameterFile:
            #         parameter_string += line
            #     try:
            #         exec (parameter_string)
            #     except:
            #         print ('\nYour loaded paramter file is wonky. \nI am going to run anyway, but with preset parameters.')
            #         keep_running = input('Would you like to continue? (Y) or any key to exit>')
            #         if keep_running.upper() != 'Y':
            #             do_run = False
            elif di["simType"] == "json_sym":  # porting from Python2 to Python3
                # you've loaded a json generated sym file, which means that it's in json format, and thus must be loaded via the json.load() routine.
                # load the second line of the sym file via json.load().
                symProps = json.loads(file.readline())
                self.symProps = symProps
            else:
                print(
                    "\nThe sym file you have loaded is formatted incorrectly. \nPlease check your sym file and try again."
                )
                input("Enter any key to return to the MainMenu>")
# ------------------------------------------------------