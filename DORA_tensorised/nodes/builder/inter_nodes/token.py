# nodes/builder/nodes/token.py
# Intermediate class for representing a token node.

from ...enums import *

from .node import Inter_Node
from nodes.network.single_nodes.token import get_default_features

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
    
    def initialise_defaults(self):
        """
        Initialise the default features for the token.
        """
        default_features = get_default_features()
        for feature in default_features:
            self.features[feature] = float(default_features[feature])
    
    def floatate_features(self):
        """
        Convert token features to floats, required for tensorisation.
        """
        for feature in TF:
            if self.features[feature] is not None:
                self.features[feature] = float(self.features[feature])
            else:
                self.features[feature] = 0.0

