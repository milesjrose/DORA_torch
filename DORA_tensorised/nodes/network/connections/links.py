# nodes/sets/connections/links.py
# Weighted connections between nodes and semantics.
# TODO: Implement add_links

import torch
import logging
logger = logging.getLogger(__name__)

from ...enums import *
from ..single_nodes import Ref_Token

class Links(object): 
    """
    A class for representing weighted connections between token sets and semantics.
    Links tensor: [Token,Semantics]
    Holds dict of tensors, mapping set to tensor.
    """
    def __init__(self, links: dict[Set, torch.Tensor], semantics):  # Takes weighted adjacency matrices
        """
        Initialize the Links object.

        Args:
            links (dict[Set, torch.Tensor]): A dictionary of weighted connections from each set to semantics.
            semantics (Semantics): The semantics that links connect to.
        
        Raises:
            TypeError: If the link tensors are not torch.Tensor (torch.float)
            ValueError: If the number of semantics (columns) in the link tensors are not the same.
        """
        # check input
        sem_count = semantics.nodes.size(dim=0)
        for set in Set:
                tensor = links[set]
                # Check type
                if type(tensor) != torch.Tensor:
                    raise TypeError(f"Link for {set} must be torch.Tensor.")
                # Check float
                if tensor.dtype != torch.float:
                    raise TypeError(f"Link for {set} must be torch.float.")
                # Check correct number of semantics
                if tensor.size(dim=1) != sem_count:
                    raise ValueError(f"{set} links tensor has {tensor.size(dim=1)} semantics, but semantics has {sem_count}.")
        # initialise
        self.semantics = semantics
        self.sets = links
        self.params = None 
        self.network = None

    def set_network(self, network):
        """set the network for the links object"""
        self.network = network
    
    def set_params(self, params):
        """
        Set the parameters for the links.
        """
        self.params = params
    
    def update_link(self, set, token_index, semantic_index, weight):
        """
        Update the link between a token and a semantic.
        """
        self.sets[set][token_index, semantic_index] = weight
    
    def update_link_weights(self, ref_token, mask=None):
        """
        Update the link weight for given token
        """
        if mask is None:
            # No mask, do for all semantics
            mask = torch.ones(self.sets[ref_token.set].size(dim=1), dtype=torch.bool)
        # Get token index
        token_index = self.network.get_index(ref_token)
        # sem acts
        sem_acts = self.semantics.nodes[mask, SF.ACT]
        # link weights
        link_weights = self.sets[ref_token.set][token_index, mask]
        # Update link weights
        self.sets[ref_token.set][token_index, mask] = 1 * (sem_acts - link_weights) * self.params.gamma

    def __getitem__(self, key):                                     # Allows for links[set], instead of links.sets[set]
        return self.sets[key]

    def __setitem__(self, key, value):                              # Allows for links[set] = tensor, instead of links.sets[set] = tensor
        self.sets[key] = value

    def del_small_link(self, threshold: float):
        """
        Delete links below threshold.
        """
        # set any values in links tensor below threshold to 0.0
        for set in Set:
            self.sets[set] = torch.where(self.sets[set] < threshold, 0.0, self.sets[set])
    
    def round_big_link(self, threshold: float):
        """
        Round links above threshold to 1.0.
        """
        # set any values in links tensor above threshold to 1.0
        for set in Set:
            self.sets[set] = torch.where(self.sets[set] > threshold, 1.0, self.sets[set])

    def calibrate_weights(self):
        """Update weights for most strongly connected semantics for driver POs (set to 1.0)"""
        # For each po, get the max weight link, then set that to 1.0
        po_mask = self.network.driver().get_mask(Type.PO)
        links = self.sets[Set.DRIVER]
        strongest_links = torch.max(links[po_mask], dim=1).indices
        po_idxs = po_mask.nonzero().squeeze(1)
        # TODO: vectorise
        for po, index in enumerate(strongest_links):
            self.sets[Set.DRIVER][po_idxs[po-1], index] = 1.0

    def swap_driver_recipient(self):
        """Swap the driver and recipient links"""
        self.sets[Set.DRIVER], self.sets[Set.RECIPIENT] = self.sets[Set.RECIPIENT], self.sets[Set.DRIVER]

    def get_max_linked_sem(self, ref_tk: Ref_Token):
        """
        Get the semantic with the highest link weight to the token.
        """
        idx_tk = self.network.get_index(ref_tk)
        return self.get_max_linked_sem_idx(ref_tk.set, idx_tk)
    
    def get_max_linked_sem_idx(self, set:Set, idx_tk: int):
        """
        See get_max_lined_sem, Set = RECIPIENT, idx_tx: token index.
        """
        semantic_index = torch.max(self.sets[set][idx_tk, :], dim=0).indices
        ref_sem = self.network.semantics.get_reference(index=semantic_index)
        return ref_sem
    
    def connect_comparitive(self, set: Set, idx_tk: int, comp_type: SDM):
        """
        Connect token to the semantic, with weight of 1.
        """
        ref_comp = self.network.semantics.sdms[comp_type]
        if ref_comp is None:
            raise ValueError("Comps not initialised")
        idx_comp = self.network.get_index(ref_comp)
        self.sets[set][idx_tk, idx_comp] = 1.0