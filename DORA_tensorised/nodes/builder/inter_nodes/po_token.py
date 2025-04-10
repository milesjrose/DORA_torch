# nodes/builder/nodes/po_token.py
# Intermediate class for representing a PO node.

from nodes.enums import *

from .token import Inter_Token

class Inter_PO(Inter_Token):
    """
    An intermediate class for representing a PO node.

    Attributes:
        name (str): The name of the PO.
        features (list): A list of features for the PO, indexed by TF.
        ID (int): The ID of the PO.
        children (list): A list of children of the PO, for use in building the connections matrix.
    """
    def __init__(self, name, set, analog, is_pred: bool):
        """
        Initialise the PO with name, set, analog, and is_pred, and default PO features.

        Args:
            name (str): The name of the PO.
            set (Set): The set of the PO.
            analog (int): The analog of the PO.
            is_pred (bool): Whether the PO is a predicate.
        """
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.PO)
        self.set(TF.PRED, is_pred)