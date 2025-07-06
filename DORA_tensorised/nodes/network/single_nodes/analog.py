# nodes/network/single_nodes/analog.py
# Representation of an analog

import torch
from ...enums import *

class Analog:
    """
    Holds the data for an analog.
    - token tensor: [n_tokens, len_tf]
    - connections tensor: [n_tokens, n_tokens]
    - links tensor: [n_tokens, n_semantics]
    - name_dict: {id: name} for tokens in the analog

    Args:
        tokens (torch.Tensor): The tokens in the analog.
        connections (torch.Tensor): The connections in the analog.
        links (torch.Tensor): The links in the analog.
        name_dict (dict[int, str]): A dictionary of the names of the tokens in the analog.

    """

    def __init__(self, tokens: torch.Tensor, connections: torch.Tensor, links: torch.Tensor, name_dict: dict[int, str]):
        self.tokens = tokens
        self.connections = connections
        self.links = links
        self.name_dict = name_dict
        self.analog_number = self.tokens[0, TF.ANALOG].item()

    def copy(self):
        """
        Create a deep copy of this analog.
        
        Returns:
            Analog: A new Analog object with copied data.
        """
        return Analog(
            tokens=self.tokens.clone(),
            connections=self.connections.clone(),
            links=self.links.clone(),
            name_dict=self.name_dict.copy()
        )

