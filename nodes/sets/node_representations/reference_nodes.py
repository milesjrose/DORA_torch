# nodes/sets/node_representations/reference_nodes.py
# Classes for referencing a single node, in any set - to make it easier to port old code.

from nodes.enums import *

class Ref_Node(object):
    """
    A class for referencing a single node, to make it easier to port old code.
    Only holds set and ID, to find in tensors.

    Attributes:
        set (Set): The set of the node.
        ID (int): The ID of the node.
    """
    def __init__(self, set: Set, ID: int):
        """
        Initialize the Ref_Node object.

        Args:
            set (Set): The set of the node.
            ID (int): The ID of the node.
        """
        self.set = set
        self.ID = ID

class Ref_Semantic(object):
    """
    A class for referencing a single semantic, to make it easier to port old code.
    Only holds ID, to find in semantic tensor.

    Attributes:
        ID (int): The ID of the semantic.
    """
    def __init__(self, ID: int):
        self.ID = ID
