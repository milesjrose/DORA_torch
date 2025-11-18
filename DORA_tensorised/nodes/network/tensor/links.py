from ...enums import *
from enum import IntEnum
import torch 
from logging import getLogger
logger = getLogger(__name__)

class LD(IntEnum):
    """
    Enum to access link dimension, i.e links[set].shape[LD.SEMANTICS]
    """
    TK = 0
    """ Token dimension """
    SEM = 1
    """ Semantics dimension """

class Links:
    """
    A class for holding the links between tokens and semantics.
    """
    def __init__(self, links: torch.Tensor):
        """
        Initialise the Links object.
        Args:
            links: torch.Tensor - The tensor of links.
        """
        self.adj_matrix: torch.Tensor = links
        """ Tensor of links between tokens and semantics """
    
    def size(self, dim):
        return self.adj_matrix.size(dim=dim)
    
    def round_big_link(self, threshold: float):
        """
        Round links above threshold to 1.0
        Args:
            threshold: float - The threshold to round links above.
        """
        self.adj_matrix = torch.where(self.adj_matrix > threshold, 1.0, self.adj_matrix)
        logger.debug(f"-> Rounded links above threshold to 1.0")
    
    def del_small_link(self, threshold: float):
        """
        Delete links below threshold.
        Args:
            threshold: float - The threshold to delete links below.
        """
        # set any values in links tensor below threshold to 0.0
        self.adj_matrix = torch.where(self.adj_matrix < threshold, 0.0, self.adj_matrix)
        logger.debug(f"-> Deleted links below threshold")
    
    def update_link(self, token_index: int, semantic_index: int, weight: float):
        """
        Update the link between a token and a semantic.
        Args:
            token_index: int - The index of the token.
            semantic_index: int - The index of the semantic.
            weight: float - The weight of the link.
        """
        self.adj_matrix[token_index, semantic_index] = weight
    
    def calibrate_weights(self, driver_po_idxs: torch.Tensor):
        """
        Update weights for most strongly connected semantics for driver POs (set to 1.0)
        
        Args:
            driver_po_idxs: torch.Tensor - 1D tensor of token indices for driver POs
        """
        # Get the indices of the strongest links (max along semantic dimension)
        strongest_links = torch.max(self.adj_matrix[driver_po_idxs], dim=LD.SEM).indices
        # Set all max links to 1.0
        self.adj_matrix[driver_po_idxs, strongest_links] = 1.0
        logger.debug(f"-> Calibrated weights for {len(driver_po_idxs)} driver POs")
    
    def get_max_linked_sem(self, tk_idx: int):
        """
        Get the semantic with the highest link weight to the token.

        Args:
            tk_idx: int - The index of the token.
        Returns:
            int - The index of the semantic with the highest link weight.
        """
        return torch.max(self.adj_matrix[tk_idx], dim=LD.SEM).indices

    def expand(self, new_size: int, dimension: LD):
        """
        Expand the links tensor for a given dimension
        Args:
            new_size: int - The new size of the tensor.
            dimension: LD - The dimension to expand along.
        """
        try:
            # Get old dimensions
            old_num_token = self.adj_matrix.size(dim=LD.TK)
            old_num_sem = self.adj_matrix.size(dim=LD.SEM)
            # Get new dimensions
            new_num_token = new_size if dimension == LD.TK else old_num_token
            new_num_sem = new_size if dimension == LD.SEM else old_num_sem
            # Create new tensor
            self.adj_matrix = torch.zeros(new_num_token, new_num_sem)
            # Copy over old links
            self.adj_matrix[:old_num_token, :old_num_sem] = self.adj_matrix
            logger.info(f"-> Expanded links tensor: {old_num_token}x{old_num_sem} -> {new_num_token}x{new_num_sem}")
        except Exception as e:
            logger.error(f"-> Error expanding links tensor: {e}")
            raise e
    
    def get_sem_count(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get the number of semantics connected to the tokens at the given indicies.

        Args:
            indices: torch.Tensor - 1D tensor of token indices.
        Returns:
            torch.Tensor - 1D tensor of the number of semantics connected to the tokens at the given indices.
        """
        return self.adj_matrix[indices, :].sum(dim=LD.SEM)
