import torch
from ....enums import *

class AnalogOperations:
    """
    Class to perform operations on analogs. !NOTE: Need to move over references from set_token.py to here.
    """
    def __init__(self, base_set):
        self.base_set = base_set

    def get_analog_indices(self, analog: int) -> torch.Tensor:
        """
        Get the indices of the tokens in an analog.
        """
        return torch.where(self.base_set.lcl[:, TF.ANALOG] == analog)[0]
    
    def get_analogs_where(self, feature: TF, value: float) -> torch.Tensor:
        """
        Get any analogs that contain a token with a given feature and value.
        """
        matching_tokens = self.base_set.lcl[:, feature] == value
        if not torch.any(matching_tokens):
            return []
        
        matching_analog_id = self.base_set.lcl[matching_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(matching_analog_id)
        return unique_analog_ids
    
    def get_analogs_where_not(self, feature:TF, value: float) -> torch.Tensor:
        """
        Get any analogs containing a token that does not have a given feature value.
        """
        non_matching_tokens = self.base_set.lcl[:, feature] != value
        if not torch.any(non_matching_tokens):
            return []
        
        non_matching_analog_id = self.base_set.lcl[non_matching_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(non_matching_analog_id)
        return unique_analog_ids
    
    def get_analogs_active(self) -> torch.Tensor:
        """
        Get all analogs that have at least one active token.
        """
        active_tokens = self.base_set.lcl[:, TF.ACT] > 0.0
        if not torch.any(active_tokens):
            return []
        
        active_analog_id = self.base_set.lcl[active_tokens, TF.ANALOG]
        unique_analog_ids = torch.unique(active_analog_id)
        return unique_analog_ids
    
    def get_analog_ref_list(self, mask)-> torch.Tensor:
        """
        Get a list of the unique analogs for a given mask
        """
        unique_analog_numbers = torch.unique(self.base_set.lcl[mask, TF.ANALOG])
        return unique_analog_numbers