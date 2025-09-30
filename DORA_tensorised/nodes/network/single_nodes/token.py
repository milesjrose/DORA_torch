# nodes/network/single_nodes/token.py
# Represents a single token.

import torch

from ...enums import *

class Token(object):
    """
    A class for representing a single token.

    Attributes:
        tensor (torch.Tensor): A tensor of features for the token.
    """
    def __init__(self, type: Type = None, features: dict[TF, float] = {}, set: Set = None, tensor: torch.Tensor = None, name:str = None):
        """
        Initialize the New_Token object, with either type,set,features, or a tensor

        Args:
            type (Type): The type of the token.
            features (dict[TF, float]): A dictionary of features for the token.
        """
        self.name = name
        self.tensor = torch.zeros(len(TF))
        # set default features
        if tensor is not None:
            # Checks for tensor size
            if tensor.size(dim=0) != len(TF):
                raise ValueError("Tensor must have number of features listed in TF enum.")
            self.tensor = tensor
        else:
            # Checks
            if type is not None:
                if type not in Type:
                    raise ValueError("Type must be a valid type.")
                features[TF.TYPE] = type
            elif TF.TYPE not in features:
                    raise ValueError("TF.TYPE must be included for tokens.")

            if set is not None:
                if set not in Set:
                    raise ValueError("Set must be a valid set.")
                features[TF.SET] = set
            elif TF.SET not in features:
                raise ValueError("TF.SET must be included for tokens.")

            # set default features
            default_features = get_default_features()
            for feature in default_features:
                self.tensor[feature] = default_features[feature]
            match type:
                case Type.P:
                    self.tensor[TF.INHIBITOR_THRESHOLD] = 440
                case Type.RB:
                    self.tensor[TF.INHIBITOR_THRESHOLD] = 220
                case Type.PO:
                    self.tensor[TF.INHIBITOR_THRESHOLD] = 110
                    if TF.PRED not in features:
                        raise ValueError("TF.PRED must be included for PO tokens.")
            
            # set assigned features
            for feature in features:
                self.tensor[feature] = features[feature]
    
    def __getitem__(self, key: TF):
        return self.tensor[key]

    def __setitem__(self, key: TF, value: float):
        self.tensor[key] = value
    
    def get_string(self):
        max_feature_length = max(len(feature.name) for feature in TF)
        max_value_length = max(len(str(self.tensor[feature])) for feature in TF)
        string = ""
        for feature in TF:
            feature_value = self.tensor[feature]
            try:
                feature_value = TF_type(feature)(feature_value)
            except:
                pass
            string += f"{feature.name:<{max_feature_length}} : {feature_value:<{max_value_length}}"
            if TF_type(feature) in [Type, Set, Mode, OntStatus, B]:
                string += f" ({feature_value.name})\n"
            elif TF_type(feature) == int():
                string += f" ({int(feature_value)})\n"
            else:
                string += f"\n"
        return string

def get_default_features():
    """
    Get the default features for a token.
    """
    default_features = {
        TF.ID: null,
        TF.TYPE: null,
        TF.SET: null,
        TF.ANALOG: null,
        TF.MAX_MAP_UNIT: 0,
        TF.MADE_UNIT: null,
        TF.MADE_SET: null,
        TF.MAKER_UNIT: null,
        TF.MAKER_SET: null,
        TF.INHIBITOR_THRESHOLD: 0,
        TF.GROUP_LAYER: 0,
        TF.MODE: 0,
        TF.TIMES_FIRED: 0,
        TF.SEM_COUNT: 0,
        TF.ACT: 0,
        TF.MAX_ACT: 0,
        TF.INHIBITOR_INPUT: 0,
        TF.INHIBITOR_ACT: 0,
        TF.MAX_MAP: 0,
        TF.TD_INPUT: 0,
        TF.BU_INPUT: 0,
        TF.LATERAL_INPUT: 0,
        TF.MAP_INPUT: 0,
        TF.NET_INPUT: 0,
        TF.MAX_SEM_WEIGHT: 0,
        TF.INFERRED: False,
        TF.RETRIEVED: False,
        TF.COPY_FOR_DR: False,
        TF.COPIED_DR_INDEX: 0,
        TF.SIM_MADE: False,
        TF.DELETED: False,
        TF.PRED: False,
    }
    if len(default_features) != len(TF):
        raise ValueError("Default features length does not match TF length.")
    return default_features

class Ref_Token(object):
    """
    A class for referencing a single token, to make it easier to port old code.
    Only holds set and ID, to find in tensors.

    Attributes:
        set (Set): The set of the node.
        ID (int): The ID of the node.
    """
    def __init__(self, set: Set, ID: int, name: str = None):
        """
        Initialize the Ref_Token object.

        Args:
            set (Set): The set of the node.
            ID (int): The ID of the node.
            name (str, optional): The name of the node. Defaults to None.
        """
        self.set: Set = set
        self.ID = ID
        self.name = name
    
    def print(self):
        print("Set:", self.set.name, "ID:", self.ID, "Name:", self.name)
