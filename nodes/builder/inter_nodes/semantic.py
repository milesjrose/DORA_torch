# nodes/builder/nodes/semantic.py
# Intermediate class for representing a semantic node.

from ...enums import *

from .node import Inter_Node

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
        self.features[SF.ONT] = 0
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