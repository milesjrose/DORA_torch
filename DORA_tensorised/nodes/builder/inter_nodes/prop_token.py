# nodes/builder/nodes/prop_token.py
# Intermediate class for representing a Prop token node.

from nodes.enums import *

from .token import Inter_Token

class Inter_Prop(Inter_Token):
    """
    An intermediate class for representing a Prop node.

    Attributes:
        name (str): The name of the Prop.
        features (list): A list of features for the Prop, indexed by TF.
        ID (int): The ID of the Prop.
        children (list): A list of children of the Prop, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog):
        """
        Initialise the Prop with name, set and analog, and default Pfeatures.

        Args:
            name (str): The name of the Prop.
            set (Set): The set of the Prop.
            analog (int): The analog of the Prop.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.P)