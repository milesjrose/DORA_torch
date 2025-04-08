# nodes/builder/intermediate_types.py
# Intermediate types for the builder.

import torch
import numpy as np

from nodes.enums import *


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
