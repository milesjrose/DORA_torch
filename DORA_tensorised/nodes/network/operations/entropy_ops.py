# nodes/network/operations/entropy_ops.py
# Entropy operations for Network class

from pickle import TRUE
from ...enums import *
from ..single_nodes import Ref_Token
from ...utils import tensor_ops as tOps
from typing import TYPE_CHECKING
import torch
from random import sample
if TYPE_CHECKING:
    from ..network import Network

class EntropyOperations:
    """
    Entropy operations for the Network class.
    Handles entropy and magnitude comparison operations.
    """
    
    def __init__(self, network):
        """
        Initialize EntropyOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network

    # ---------------------[ TODO: IMPLEMENT ]----------------------------
    
    def en_based_mag_checks(self, po1: Ref_Token, po2: Ref_Token):
        """
        Check if POs code the same dimension, or connected to SDM semantics - 
        for deciding whether to include in magnitude comparison.

        Returns:
            high_dim: highest intersecting dimension
            num_sdm_above: Num pos with con to SDM sems above threshold
            num_sdm_below: Num pos with con to SDM sems below threshold
        
        TODO: TEST
        """
        # 1). do they code for intersecting dimentsions
        # -> get instersecting dimension, by getting set of dimensions encoded by each pos semantics
        idxs = {
            po1: self.network.get_index(po1),
            po2: self.network.get_index(po2),
        }
        dims = {}
        po_sems = {}
        for ref_po in [po1, po2]:
            # get semantics that have weight>0.9
            po_set = ref_po.set
            po_sems = self.network.links[po_set][idxs[ref_po], :] > 0.9
            # then get the dimensions of these sems
            indices = po_sems.nonzero()
            dims[ref_po] = set()
            for index in indices:
                re = self.network.semantics.get_reference(index=index)
                dimension = self.network.semantics.get_dim(re)
                dims[ref_po].add(dimension)
        # then take the intersection of the two sets
        intersecting_dims = dims[po1].intersection(dims[po2])
            
        # 2). do either PO have connections to SDM (comparative) semantics above threshold (=.9)
        #   or do both connect to any SDM sems below thresh (=.9)
        sdm_indicies = self.network.semantics.get_sdm_indices()
        # Get slice of connections to SDM tensors, with T/F if they above threshold
        sdms = {po1 : {},po2 : {}} # map PO to dict of SDM above/below threshold
        for po in [po1, po2]:
            po_set = po.set
            con_sdm = self.network.links[po_set][[idxs[po], sdm_indicies]]
            con_sdm_above = con_sdm > 0.9
            con_sdm_below = 0.0 < con_sdm < 0.9
            sdms[po]["above"] = con_sdm_above.any()
            sdms[po]["below"] = con_sdm_below.any()

        num_sdm_above = po1["above"] + po2["above"]
        num_sdm_below = po1["below"] + po2["below"]

        # 3). find the dimension of highest overlap 
        # (sum the semantics coding dimensions value that connect to both POs)
        # -> for each intersecting dim, sum value if on_status = value
        # -> if the highest weights are equal, add to list and randomly select one
        high_dim = []
        high_dim_weight = 0.0
        for dim in intersecting_dims:
            dim_weight = {}
            for po in [po1, po2]:
                sems: torch.Tensor = self.network.semantics.nodes[po_sems[po],:]
                # dim_weight = list of connected sem where dim=dim and on_status = value
                ont_sems = sems[:, SF.DIM] == dim and sems[:, SF.ONT] == OntStatus.VALUE
                # dim_weight = list of semantics where value == max weight
                links: torch.Tensor = self.network.links[po.set]
                max_weight = max(links[ont_sems, SF.AMOUNT])            # Get the max weight of the ont_dims semantics
                max_sems = links[idxs[po], :] == max_weight             # Get the semantics whose link weight is equal to the max weight
                # combine
                dim_sems = ont_sems & max_sems
                if dim_sems.any():
                    dim_weight[po] = self.network.links[idxs[po],dim_sems].sum()
                else:
                    dim_weight[po] = None
            # if both POs have dim_weight, sum weights
            if dim_weight[po1] is not None and dim_weight[po2] is not None:
                curr_weight = dim_weight[po1] + dim_weight[po2]
                if curr_weight > high_dim_weight:
                    high_dim = [dim]
                    high_dim_weight = curr_weight
                elif curr_weight == high_dim_weight and curr_weight > 0:
                    high_dim.append(dim)
        # if multiple equal high dims, randomly select one
        if len(high_dim) > 1:
            high_dim = sample(high_dim, 1)
        # Finally return the intersect dim and information about 
        # mag (SDM) sem present (above/below threshold)
        return (
            high_dim,
            num_sdm_above,
            num_sdm_below,
        )
    
    def check_and_run_ent_ops_within(
        self,
        po1: Ref_Token,
        po2: Ref_Token,
        intersect_dim: list[int],
        num_sdm_above: int,
        num_sdm_below: int,
        extend_SDML:bool,
        pred_only:bool,
        pred_present:bool,
        mag_decimal_precision:int
    ):
        """
        Check whether to run entropy based magnitude comparision (within), 
        and run if appropriate
        TODO: TEST
        """
        is_pred = self.network.node_ops.get_value(po1, TF.PRED) == B.TRUE
        if (is_pred):
            if (
                num_sdm_above == 0
                and num_sdm_below < 2
                and len(intersect_dim) > 0
            ):
                self.basic_en_based_mag_comparison(po1, po2, intersect_dim, mag_decimal_precision)
            elif (
                extend_SDML
                and num_sdm_above < 2
                and (num_sdm_above > 0 or num_sdm_below > 0)
            ):
                self.basic_en_based_mag_refinement(po1, po2)
        else: # is object
            if (
                len(intersect_dim) > 0
                and not pred_only
                and not pred_present
                and num_sdm_above == 0
            ):
                self.basic_en_based_mag_comparison(po1, po2, intersect_dim, mag_decimal_precision)
        return
    
    def basic_en_based_mag_comparison(self):
        """
        Basic magnitude comparison.
        """
        # Implementation using network.sets, network.semantics
        pass
    
    def basic_en_based_mag_refinement(self):
        """
        Basic magnitude refinement.
        """
        # Implementation using network.sets, network.semantics
        pass
    
    def ent_magnitude_more_less_same(self):
        """
        Magnitude comparison logic.
        """
        # Implementation using network.sets, network.semantics
        pass
    
    def attach_mag_semantics(self):
        """
        Attach magnitude semantics.
        """
        # Implementation using network.sets, network.semantics
        pass
    
    def update_mag_semantics(self):
        """
        Update magnitude semantics.
        """
        # Implementation using network.sets, network.semantics
        pass
    
    def ent_overall_same_diff(self):
        """
        Overall same/different calculation.
        """
        # Implementation using network.sets, network.semantics
        pass 