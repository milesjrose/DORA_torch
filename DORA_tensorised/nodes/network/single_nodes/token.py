# nodes/network/single_nodes/token.py
# Represents a single token.

import torch

from nodes.enums import *

class Token(object):
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

class Ref_Token(object):
    """
    A class for referencing a single token, to make it easier to port old code.
    Only holds set and ID, to find in tensors.

    Attributes:
        set (Set): The set of the node.
        ID (int): The ID of the node.
    """
    def __init__(self, set: Set, ID: int):
        """
        Initialize the Ref_Token object.

        Args:
            set (Set): The set of the node.
            ID (int): The ID of the node.
        """
        self.set = set
        self.ID = ID
