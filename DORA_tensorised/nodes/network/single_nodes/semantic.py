# nodes/network/single_nodes/semantic.py
# Classes for representing new nodes, to make it easier to add to set tensors.

import torch

from nodes.enums import *

class Semantic(object):
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

class Ref_Semantic(object):
    """
    A class for referencing a single semantic, to make it easier to port old code.
    Only holds ID, to find in semantics tensor.

    Attributes:
        ID (int): The ID of the semantic.
    """
    def __init__(self, ID: int):
        self.ID = ID