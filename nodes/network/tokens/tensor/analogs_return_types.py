import torch
from ....enums import *
from enum import IntEnum
from logging import getLogger
logger = getLogger(__name__)

class AnalogInfoCols(IntEnum):
    """
    Enum for the columns of the analogs information tensor.
    """
    NUM = 0
    """ The analog number."""
    SET = 1
    """ The set of the analog."""
    COUNT = 2
    """ The count of tokens in the analog."""
    ACT = 3
    """ The activation of the analog."""

class AnalogInfo:
    """
    Return type for all analogs information.
    """
    def __init__(self, data: torch.Tensor):
        self.data = data
        self.Cols = AnalogInfoCols
    
    def get_analog_info(self, analog_number: int) -> torch.Tensor|None:
        """
        Get the info for a given analog number.

        Args:
            analog_number: int - The number of the analog to get information for.
        Returns:
            torch.Tensor|None - The information for the given analog number, or None if the analog number is not found.
        """
        analog_idx = self.data[:, self.Cols.NUM] == analog_number
        if not analog_idx.any():
            logger.debug(f"Analog {analog_number} not found in analogs information.")
            return None
        return self.data[analog_idx, :].squeeze()
    
    def get_set_info(self, set: Set) -> torch.Tensor|None:
        """
        Get the info for a given set.
        """
        set_mask = self.data[:, self.Cols.SET] == set
        if not set_mask.any():
            logger.debug(f"Set {set} not found in analogs information.")
            return None
        data = self.data[set_mask, :]
        return AnalogInfo(data)