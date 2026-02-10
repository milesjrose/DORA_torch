# nodes/network/sets/semantics.py
# Represents the semantics set of tokens.

import torch
import logging
logger = logging.getLogger("set")

from ...enums import *
from ...utils import tensor_ops as tOps

from ..single_nodes import Ref_Semantic
from ..tokens.connections import Links, LD
from ..network_params import Params
from ..single_nodes import Semantic

from .base_set import Base_Set

class Semantics(object):
    """
    A class for representing semantics nodes.

    Attributes:
        IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
        names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
        nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
        connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
        links (Links): A Links object containing links from token sets to semantics.
        params (Params): An object containing shared parameters. Defaults to None.
    """
    def __init__(self, nodes, connections, IDs: dict[int, int], names: dict[int, str] = None):
        """
        Initialise a Semantics object

        Args:
            nodes (torch.Tensor): An NxSemanticFeatures tensor of floats representing the semantics.
            connections (torch.Tensor): An NxN tensor of connections from parent to child for semantics in this set.
            IDs (dict): A dictionary mapping semantic IDs to index in the tensor.
            names (dict, optional): A dictionary mapping semantic IDs to semantic names. Defaults to None.
        Raises:
            ValueError: If the number of semantics in nodes, connections, and links do not match.
            ValueError: If the number of features in nodes does not match the number of features in SF enum.
        """
        if nodes.size(dim=0) != connections.size(dim=0):
            raise ValueError("nodes and connections must have the same number of semantics.")
        if nodes.size(dim=1) != len(SF):
            raise ValueError("nodes must have number of features listed in SF enum.")
        if names is not None:
            if not isinstance(names, dict):
                raise ValueError(f"names must be a dictionary, not {type(names)}.")
            if not all(isinstance(name, str) for name in names.values()):
                # get types that are not strings
                non_strings = [type(name) for name in names.values() if type(name) != str]
                raise ValueError(f"names must be a dictionary of strings, not {non_strings}.")
        self.names = names 
        """Map ID to name string"""
        self.nodes: torch.Tensor = nodes
        """Semantic nodes tensor"""
        self.connections: torch.Tensor = connections
        """Same-set connections for semantics"""
        self.links: Links = None
        """Links between tokens and semantics (Shape: [Tokens, Semantics])"""
        self.IDs = IDs
        """Map ID to index in tensor"""
        self.dimensions = {}
        """Map dimension feature to dimension: TODO: add to builder/save/load"""
        self.params = None
        """Shared parameters"""
        self.expansion_factor = 1.1
        """Factor to expand when adding sem to full tensor"""
        self.sdms = {
            SDM.MORE: None,
            SDM.LESS: None,
            SDM.SAME: None,
            SDM.DIFF: None,
        }
        """ Map SDM to ref_semantic"""
        self.sdm_dims = {
            SDM.MORE: None,
            SDM.LESS: None,
            SDM.SAME: None,
            SDM.DIFF: None,
        }
        """ Map SDM to dimension key"""
    
    def add_dim(self, dimension: str) -> int:
        """Add a dimension to the dimensions dictionary"""
        new_dim_key = max(self.dimensions.keys()) + 1 if self.dimensions else 1
        self.dimensions[new_dim_key] = dimension
        return new_dim_key
    
    def get_dim(self, idx: int) -> int:
        """Get the dimension of a semantic"""
        dim_key = int(self.get(idx, SF.DIM))
        return dim_key
    
    def get_dim_name(self, dim_key: int) -> str:
        """Get the name of a dimension"""
        return self.dimensions.get(dim_key, None)
    
    def set_dim_name(self, dim_key: int, name: str):
        """Set the name of a dimension"""
        self.dimensions[dim_key] = name
    
    def get_dim_key(self, dimension: str) -> int:
        """Get the key of a dimension"""
        try:
            return list(self.dimensions.keys())[list(self.dimensions.values()).index(dimension)]
        except ValueError:
            return None
    
    def set_dim(self, idx: int, dimension: str):
        """
        Set the dimension of a semantic
        NOTE: Inefficient: Use sems.set(sem, SF.DIMENSION, encoded_dim_key) if possible
        """
        dim_key = self.add_dim(dimension) if dimension not in self.dimensions.values() else list(self.dimensions.keys())[list(self.dimensions.values()).index(dimension)]
        self.set(idx, SF.DIM, dim_key)

    def init_sdm(self):
        """Initialise the comparative semantics"""
        # Check if more, less, same already exist in class, then check if they are in the semantics tensor. 
        # If neither, then create the semantic and set attribute
        logger.debug("Initialising comparative semantics")
        for sdm in SDM:
            if self.sdms[sdm] is None and sdm.name not in self.names.values():
                self.sdm_dims[sdm] = self.add_dim(sdm.name)
                sdm_sem = Semantic(sdm.name, {SF.TYPE: Type.SEMANTIC, SF.DIM: self.sdm_dims[sdm], SF.ONT: OntStatus.SDM})
                self.sdms[sdm] = self.add_semantic(sdm_sem)
    
    def get_sdm_indices(self, include_diff: bool = False) -> torch.Tensor:
        """Get the indices of the SDM/comparative semantics"""
        if None in self.sdm_dims.values():
            raise ValueError("SDM dimensions not initialised")
        if include_diff:
            sdm_dims = torch.tensor(list(self.sdm_dims.values()))
        else:
            sdm_dims = torch.tensor([self.sdm_dims[SDM.MORE], self.sdm_dims[SDM.LESS], self.sdm_dims[SDM.SAME]])
        indices = torch.isin(self.nodes[:, SF.DIM], sdm_dims).nonzero()
        logger.debug(f"DIMS: {self.nodes[:, SF.DIM]}")
        return indices
    
    def check_sdm_init(self) -> bool:
        """Check if all SDM/comparative semantics are initialised"""
        for sdm in SDM:
            if self.sdms[sdm] is None:
                return False
        return True
            
    def add_semantic(self, semantic: Semantic) -> int:
        """
        Add a semantic to the semantics tensor.

        Args:
            semantic (Semantic): The semantic to add.
        """
        logger.debug(f"Add sem: {semantic.name}")
        deleted_mask = self.nodes[:, SF.DELETED] == B.TRUE          # find all deleted semantics in nodes tensor
        if not deleted_mask.any():                                  # if no deleted semantics, expand tensor
            self.expand_tensor()
        empty_rows = torch.where(self.nodes[:, SF.DELETED] == B.TRUE)[0]                   # find all empty rows in nodes tensor
        mt_idx = int(empty_rows[0].item())                                # find first empty row
        self.nodes[mt_idx, :] = semantic.tensor                  # add semantic to empty row
        new_id = max(self.IDs.keys()) + 1 if self.IDs else 1     # get new id
        self.IDs[new_id] = mt_idx                                # add id to IDs
        if semantic.name is None:
            semantic.name = f"Semantic {new_id}"
        self.names[new_id] = semantic.name                       # add name to names
        self.nodes[mt_idx, SF.ID] = new_id                       # set node id feature
        return mt_idx
    
    def expand_tensor(self):
        """
        Expand the nodes, connections, and links tensors by the expansion factor.
        """
        current_size = self.nodes.size(dim=SD.NODES)
        new_size = max(int(current_size * self.expansion_factor), current_size + 5)  # ensure we actually expand
        logger.debug(f"Expand: {current_size} -> {new_size}")
        new_nodes = torch.zeros(new_size, len(SF))                  # create new nodes tensor
        new_nodes[current_size:, SF.DELETED] = B.TRUE               # set all deleted to 1 for all new nodes
        new_nodes[:current_size, :] = self.nodes                    # copy over old nodes
        self.nodes = new_nodes                                      # update nodes

        new_cons = torch.zeros(new_size, new_size)                  # create new connections tensor
        new_cons[:current_size, :current_size] = self.connections   # copy over old connections
        self.connections = new_cons                                 # update connections

        if self.links is not None:
            self.links.expand_to(new_size, LD.SEM)
        else:
            logger.debug("Links not initialised. Not expanding links tensor.")
    
    def check_links_size(self):
        """ Check the size of the links tensor is the same as the number of semantics, and expand if not."""
        if self.links is not None:
            link_size = self.links.size(LD.SEM)
            sem_size = self.nodes.size(dim=0)
            if link_size < sem_size:
                self.links.expand_to(self.nodes.size(dim=0), LD.SEM)
            elif link_size > sem_size:
                logger.critical(f"Links tensor size is greater than semantics tensor size, no way to handle this atm.")

    def del_semantic(self, idx: int):                                     # Delete a semantic from the semantics tensor.
        """
        Delete a semantic from the semantics tensor.
        """ 
        logger.debug(f"Deleting semantic {idx}")
        self.nodes[idx, SF.DELETED] = B.TRUE
        self.names.pop(idx)
        
        if self.links is not None:
            self.links.tensor[:, idx] = 0.0

        self.connections[idx, :] = 0.0
        self.connections[:, idx] = 0.0

    def get_count(self):
        """Get the number of semantics in the semantics tensor."""
        return (self.nodes[:, SF.DELETED]==B.FALSE).sum()
    
    def get_active_mask(self, thresh: float = 0.01) -> torch.Tensor:
        """Get the mask of active semantics"""
        return self.nodes[:, SF.ACT] > thresh
    

    # ==================[ LINKS ]=====================
    # NOTE: Maybe move thes somewhere else? These are links functions, but I don't want to add a 
    #       semantics reference to links as this breaks the encapsulation of links. 
    #       For now this seems the most sensible place I think, maybe move to a links operations class
    #       in the network operations module at some point?

    def connect_comparitive(self, idx_tk: int, comp_type: SDM):
        """
        Connect token to the comparative semantic, with weight of 1.

        Args:
            idx_tk: int - The global index of the token to connect.
            comp_type: SDM - The type of comparative semantic to connect.
        """
        idx_comp = self.sdms.get(comp_type, None)
        if idx_comp is None:
            raise ValueError("Comps not initialised")
        self.links[idx_tk, idx_comp] = 1.0
    
    def _to_int(self, idx: int|torch.Tensor|list[int]) -> int:
        """
        Convert an index to an integer.
        """
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        elif isinstance(idx, list):
            idx = torch.tensor(idx)
        return int(idx)

    def update_link_weights(self, idx_tk: int, mask: torch.Tensor = None):
        """
        Update the weights of the links between a token and its semantics.

        link_weight += 1 * (sem_act - link_weight) * gamma

        Args:
            idx_tk: int - The global index of the token to update the link weights for.
            mask: torch.Tensor - The mask of semantics to update the link weights for.
        """
        idx_tk = self._to_int(idx_tk)
        if mask is None:
            sem_acts = self.nodes[:, SF.ACT]
            link_weights = self.links[idx_tk, :]
            self.links[idx_tk, :] += 1 * (sem_acts - link_weights) * self.params.gamma
        else:
            sem_acts = self.nodes[mask, SF.ACT]
            link_weights = self.links[idx_tk, mask]
            self.links[idx_tk, mask] += 1 * (sem_acts - link_weights) * self.params.gamma

    # ===============[ INDIVIDUAL TOKEN FUNCTIONS ]=================   
    def get(self, idx: int, feature):
        """
        Get a feature for a semantic with a given ID.
        
        Args:
            idx: int - The index of the semantic to get the feature for.
            feature (TF): The feature to get.

        Returns:
            The feature for the semantic.
        """

        try:
            return self.nodes[idx, feature]
        except Exception as e:
            logger.critical(f"Error getting feature {feature} for semantic {idx}: {e}")
            raise e
    
    def getc(self, idx: int, feature: SF) -> any:
        """
        Get a type casted feature for a semantic with given index
    
        Args:
            idx: int - The index of the semantic to get the feature for.
            feature (SF): The feature to get.

        Returns:
            The type casted feature for the semantic.
        """
        val = self.get(idx, feature).item()
        if val == null:
            return None
        return SF_type(feature)(val)

    def set(self, idx: int, feature, value):
        """
        Set a feature for a semantic with a given ID.
        
        Args:
            idx: int - The index of the semantic to set the feature for.
            feature (TF): The feature to set.
            value (float): The value to set the feature to.

        Raises:
            TypeError: If the feature is not a TF enum.
            ValueError: If the ID or feature is invalid.
        """

        try:
            self.nodes[idx, feature] = float(value)
        except:
            raise ValueError("Invalid semantic or feature.")
    
    def get_single_semantic(self, idx: int, copy=True):
        """
        Get a single semantic from the semantics tensor.

        - If copy is set to False, changes to the returned semantic will affect the semantic set tensor.

        Args:
            ref_semantic (Ref_Semantic): The reference semantic.
            copy (bool, optional): Whether to use a copy of the semantic sub-tensor. Defaults to True.

        Returns:
            A Semantic object.
        
        Raises:
            ValueError: If the reference semantic is invalid.
        """
        tensor = self.nodes[idx, :]
        sem = Semantic(self.names[idx], {SF.TYPE: Type.SEMANTIC})
        if copy:
            sem.tensor = tensor.clone()
        else:
            sem.tensor = tensor
        return sem
    
    def get_name(self, idx: int) -> str:
        """Get the name of a semantic"""
        try:
            return self.names[idx]
        except:
            logger.critical(f"Semantic name not found for index {idx}, names: {self.names}, IDs: {self.IDs}, shape: {self.nodes.shape}")
            raise ValueError(f"Semantic name not found for index {idx}")
            
    
    def set_name(self, idx: int, name: str):
        """Set the name of a semantic"""
        self.names[idx] = name

    # --------------------------------------------------------------

    # ===================[ SEMANTIC FUNCTIONS ]=====================
    def init_sem(self):                                             # Set act and input to 0 TODO: Check how used
        """Initialise the semantics """
        self.nodes[:, SF.ACT] = 0.0
        self.nodes[:, SF.INPUT] = 0.0

    def init_input(self, refresh):                                  # Set nodes to refresh value TODO: Check how used
        """Initialise the input of the semantics """
        self.nodes[:, SF.INPUT] = refresh

    def set_max_input(self, max_input):                             # set max input of all semantics
        """Set the max input of the semantics """
        self.nodes[:, SF.MAX_INPUT] = max_input
    
    def get_max_input(self):                                        # Get the max input in semantics
        """Get the maximum input in semantics """
        return self.nodes[:, SF.INPUT].max()

    def update_act(self):                                           # Update act of all sems
        """Update the acts of the semantics """
        sem_mask = self.nodes[:, SF.MAX_INPUT] > 0                  # Get sem where max_input > 0
        input = self.nodes[sem_mask, SF.INPUT]
        max_input = self.nodes[sem_mask, SF.MAX_INPUT]
        self.nodes[sem_mask, SF.ACT] = input / max_input            # - Set act of sem to input/max_input
        sem_mask = self.nodes[:, SF.MAX_INPUT] == 0                 # Get sem where max_input == 0       
        self.nodes[sem_mask, SF.ACT] = 0.0                          #  -  Set act of sem to 0
    
    def update_input(self, driver, recipient, memory = None, ignore_obj=False, retrieval_license=False):
        """
        Update the input of the semantics
        - Note, if memory is not provided, equivalent to "ignore_mem = True"
        
        Args:
            driver (Base_Set): The driver set.
            recipient (Base_Set): The recipient set.
            memory (Base_Set, optional): The memory set. Defaults to None.
            ignore_obj (bool, optional): Whether to ignore the object set. Defaults to False.

        Raises:
            ValueError: If ignore_obj is set to False and no memory is provided.
        """
        if not retrieval_license:
            self.init_input(0.0)
        else:
            logger.critical("Not sure when this is used, so i'm going to throw an error for now :/")
            raise NotImplementedError("Retrieval license not implemented yet.")
        self.update_input_from_set(driver, Set.DRIVER, ignore_obj)
        self.update_input_from_set(recipient, Set.RECIPIENT, ignore_obj)
        if memory is not None:
            self.update_input_from_set(memory, Set.MEMORY, ignore_obj)

    def update_input_from_set(self, tensor: Base_Set, set: Set, ignore_obj=False):
        """Update the input of the semantics from a set of tokens """
        if self.links is None:
            raise ValueError("Links not initialised. Should be set when network is created.")
        
        # Get mask of POs
        if ignore_obj:
            po_mask = tensor.tensor_op.get_arb_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE})
        else:
            po_mask = tensor.tensor_op.get_arb_mask({TF.TYPE: Type.PO})
        #group_mask = tensor.get_mask(Type.GROUP)
        #token_mask = torch.bitwise_or(po_mask, group_mask)             # In case groups used in future

        # Update based on linked tokens
        links: torch.Tensor = self.links[tensor.lcl._indices]           # only looking at links to the given set
        connected_nodes_sub = (links[po_mask, :] != 0).any(dim=1)       # Mask all PO that have a link to a sem
        connected_nodes = tOps.sub_union(po_mask, connected_nodes_sub)  # Resize mask to full tensor size
        connected_sem = (links[po_mask, :] != 0).any(dim=0)             # Mask all sems that have a link to a PO

        links_cons = torch.transpose(links[connected_nodes][:, connected_sem], 0, 1)

        sem_input = torch.matmul(                                       # Get sum of act * link_weight for all connected nodes and sems
            links_cons,                                                 # connected_sem x connected_nodes matrix of link weights
            tensor.lcl[connected_nodes, TF.ACT]                         # connected_nodes x 1 matrix of node acts
        )
        self.nodes[connected_sem, SF.INPUT] += sem_input                # Update input of connected sems
    # --------------------------------------------------------------

