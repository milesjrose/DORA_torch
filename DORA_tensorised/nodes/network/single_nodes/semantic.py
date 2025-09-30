# nodes/network/single_nodes/semantic.py
# Classes for representing new nodes, to make it easier to add to set tensors.

import torch

from ...enums import *

class Semantic(object):
    """
    A class for representing a single semantic.
    
    Attributes:
        tensor (torch.Tensor): A tensor of features for the semantic.
    """
    def __init__(self, name: str, features: dict[SF, float] = {}):
        """
        Initialize the New_Semantic object.

        Args:
            features (dict[SF, float]): A dictionary of features for the semantic.
        """
        self.name = name
        self.tensor = torch.zeros(len(SF))
        self.tensor[SF.TYPE] = Type.SEMANTIC
        self.tensor[SF.DIM] = null
        self.tensor[SF.AMOUNT] = null
        self.tensor[SF.ONT] = null
        for feature in features:
            self.tensor[feature] = features[feature]
    
    def __getitem__(self, key: SF):
        return self.tensor[key]
    
    def __setitem__(self, key: SF, value: float):
        self.tensor[key] = value
    
    def get_string(self):
        max_feature_length = max(len(feature.name) for feature in SF)
        string = ""
        for feature in SF:
            feature_value = (self.tensor[feature].item()) if self.tensor[feature].item() != null else null
            string += f"{feature.name:<{max_feature_length}} : "
            string += f"{feature_value}\n"
        return string
        

class Ref_Semantic(object):
    """
    A class for referencing a single semantic, to make it easier to port old code.
    Only holds ID, to find in semantics tensor.

    Attributes:
        ID (int): The ID of the semantic.
        name (str, optional): The name of the semantic. Defaults to None.
    """
    def __init__(self, ID: int, name: str = None):
        self.ID = ID
        self.name = name
    
    def print(self):
        print("ID:", self.ID, "Name:", self.name)
