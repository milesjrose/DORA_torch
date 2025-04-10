# nodes/builder/nodes/rb_token.py
# Intermediate class for representing a RB token node.

from nodes.enums import *

from .token import Inter_Token

class Inter_RB(Inter_Token):
    """
    An intermediate class for representing a RB node.

    Attributes:
        name (str): The name of the RB.
        features (list): A list of features for the RB, indexed by TF.
        ID (int): The ID of the RB.
        children (list): A list of children of the RB, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog):
        """
        Initialise the RB with name, set and analog, and default RB features.

        Args:
            name (str): The name of the RB.
            set (Set): The set of the RB.
            analog (int): The analog of the RB.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.RB)