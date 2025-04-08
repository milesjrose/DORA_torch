# nodes/sets/node_representations/new_nodes.py
# Classes for representing new nodes, to make it easier to add to set tensors.

import torch

from nodes.enums import *

class New_Token(object):
    """
    A class for representing a single token.

    Attributes:
        tensor (torch.Tensor): A tensor of features for the token.
    """
    def __init__(self, type: Type, features: dict[TF, float]):
        """
        Initialize the New_Token object.

        Args:
            type (Type): The type of the token.
            features (dict[TF, float]): A dictionary of features for the token.
        """
        self.tensor = torch.zeros(len(TF))
        self.tensor[TF.TYPE] = type
        match type:
            case Type.P:
                self.tensor[TF.INHIBITOR_THRESHOLD] = 440
            case Type.RB:
                self.tensor[TF.INHIBITOR_THRESHOLD] = 220
            case Type.PO:
                self.tensor[TF.INHIBITOR_THRESHOLD] = 110
                if TF.PRED not in features:
                    raise ValueError("TF.PRED must be included for PO tokens.")
        for feature in features:
            self.tensor[feature] = features[feature]

class New_Semantic(object):
    """
    A class for representing a single semantic.
    
    Attributes:
        tensor (torch.Tensor): A tensor of features for the semantic.
    """
    def __init__(self, name: str, features: dict[SF, float]):
        """
        Initialize the New_Semantic object.

        Args:
            features (dict[SF, float]): A dictionary of features for the semantic.
        """
        self.name = name
        self.tensor = torch.zeros(len(SF))
        self.tensor[SF.TYPE] = Type.SEMANTIC
        for feature in features:
            self.tensor[feature] = features[feature]
