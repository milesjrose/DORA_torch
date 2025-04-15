# nodes/builder/nodes/token.py
# Intermediate class for representing a token node.

from ...enums import *

from .node import Inter_Node

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

