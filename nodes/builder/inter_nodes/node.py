# nodes/builder/nodes/base_node.py
# Base node class for all nodes in the network.
from ...enums import *

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
        self.features[feature] = float(value)
    
    def set_ID(self, ID):
        """
        Set the ID of the node.

        Args:
            ID (int): The ID to set the node to.
        """
        self.ID = int(ID)
        self.features[TF.ID] = float(ID)