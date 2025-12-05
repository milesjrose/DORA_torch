import torch
from ....enums import *
from logging import getLogger
logger = getLogger(__name__)

class UpdateOperations:
    """
    Update operations for the Base_Set class.
    """
    def __init__(self, base_set):
        self.base_set = base_set

    def init_float(self, n_type: list[Type], features: list[TF]):
        """
        Initialise the given features to 0.0

        Args:
            n_type (list[Type]): The types of nodes to initialise.
            features (list[TF]): The features to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            for feature in features:
                self.base_set.lcl[type_mask, feature] = 0.0
    
    def init_input(self, n_type: list[Type], refresh: float):
        """
        Initialise the input of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
            refresh (float): The value to set the td_input to.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.base_set.lcl[type_mask, TF.TD_INPUT] = refresh
            features = [TF.BU_INPUT,TF.LATERAL_INPUT,TF.MAP_INPUT,TF.NET_INPUT]
            self.init_float(n_type, features)
        
    def init_act(self, n_type: list[Type]):
        """
        Initialise the act of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.init_input(n_type, 0.0)
            self.init_float(n_type, [TF.ACT])
    
    def init_state(self, n_type: list[Type]):
        """
        Initialise the state of the tokens.

        Args:
            n_type (list[Type]): The types of nodes to initialise.
        """
        type_mask = self.base_set.tensor_op.get_combined_mask(n_type)
        if torch.any(type_mask):
            self.init_act(n_type)
            self.init_float(n_type, [TF.RETRIEVED])
    
    def update_act(self):
        """
        Update the act of the tokens.!NOTE: Not implemented for new structure yet.
        """
        pass