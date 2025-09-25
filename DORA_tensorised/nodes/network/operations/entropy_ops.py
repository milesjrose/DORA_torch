# nodes/network/operations/entropy_ops.py
# Entropy operations for Network class

from logging import getLogger

from ...enums import *
from ..single_nodes import Ref_Token
from ...utils import tensor_ops as tOps
from typing import TYPE_CHECKING
import torch
from random import sample
if TYPE_CHECKING:
    from ..network import Network

logger = getLogger(__name__)

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
            # NOTE: idk if this is inefficient, but it should work for now
            po_sems = self.network.links[po_set][idxs[ref_po], :] > 0.9
            po_sems = tOps.sub_union(po_sems, self.network.semantics.nodes[po_sems, SF.DIM] != null)
            po_sems = tOps.sub_union(po_sems, self.network.semantics.nodes[po_sems, SF.ONT] == OntStatus.VALUE)
            dims[ref_po] = set(self.network.semantics.nodes[po_sems, SF.DIM].unique())
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
    

    def basic_en_based_mag_comparison(
        self,
        po1: Ref_Token,
        po2: Ref_Token,
        intersect_dim: list[int],
        mag_decimal_precision:int = 0
    ):
        """
        Basic magnitude comparison.
        """
        sems: torch.Tensor = self.network.semantics.nodes
        sem_link = {}
        other_sem_link = {}
        extent = {}
        idxs = {
            po1: self.network.get_index(po1),
            po2: self.network.get_index(po2),
        }
        dim = intersect_dim[0]
        for po in [po1, po2]:
            # 1). find the semantic links connecting to the absolute dimensional value.
            sem_link[po] = self.find_links_to_abs_dim(po, dim, OntStatus.VALUE)
            # 2). if the dimension is numeric, then get the average value of all 
            # dimensionnal values in the sem_links and assign these to 
            # extent1 and extend2 respectively
            idx = sem_link[po].nonzero()[0]
            is_numeric = bool((sems[idx, SF.AMOUNT] != null).item())
            if is_numeric:
                extent[po] = sems[sem_link[po], SF.AMOUNT].mean()
            else:
                po_sem = self.network.semantics.get_reference(index=idxs[po])
                extent[po] = self.network.semantics.get_name(po_sem)  # NOTE: This could be wrong, not sure if I need to store the value for catagorical values in a seperate hashmap. 
        # 3). compute ent_magnitudeMoreLessSame()
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(extent[po1], extent[po2], mag_decimal_precision)
        for po in [po1, po2]:
            # 4). Find any other dimensional semantics with high weights can be reduced by the entropy process.
            po_links = self.network.links[po.set][idxs[po], :]
            other_dim_sems = ~torch.isin(sems[po_links, SF.DIM], [dim, null])
            other_sem_link[po] = tOps.sub_union(sem_link[po], other_dim_sems)
            sem_link[po] = sem_link[po] | other_sem_link[po]
        # 5). Connect the two POs to the appropriate relative magnitude semantics 
        # (based on the invariant patterns detected just above).
        if more == extent[po2]:
            self.network.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        else:
            self.network.attach_mag_semantics(same_flag, po2, po1, sem_link, idxs)


    def basic_en_based_mag_refinement(self, po1: Ref_Token, po2: Ref_Token):
        """
        Basic magnitude refinement:

        if there are magnitude semantics present, and there are some matching dimensions, 
        then activate the appropriate magnitude semantics and matching dimensions, and adjust
        weights as appropriate (i.e., turn on the appropriate magnitude semantics for each PO, 
        and adjust weight accordingly).
        TODO: TEST
        """
        mag_decimal_precision = 1
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
            # NOTE: idk if this is inefficient, but it should work for now
            po_sems = self.network.links[po_set][idxs[ref_po], :] > 0.9
            po_sems = tOps.sub_union(po_sems, self.network.semantics.nodes[po_sems, SF.DIM] != null)
            po_sems = tOps.sub_union(po_sems, self.network.semantics.nodes[po_sems, SF.ONT] == OntStatus.STATE)
            dims[ref_po] = set(self.network.semantics.nodes[po_sems, SF.DIM].unique())
        # then take the intersection of the two sets
        intersecting_dims = dims[po1].intersection(dims[po2])
        # 2). if there is a single matching dimension, then find value on that dimension for each
        # object and update magnitude semantic weights; elif there are multiple matching dimensions,
        # find the matching dimension that each PO is most strongly connected to, and update magnitude
        # semantic weights.
        if len(intersecting_dims) == 1:
            self.single_dim_refinement(idxs, po1, po2, intersecting_dims[0], mag_decimal_precision)
        else:
            self.multi_dim_refinement(idxs, po1, po2, mag_decimal_precision)


    def single_dim_refinement(self, idxs, po1: Ref_Token, po2: Ref_Token, dim: int, mag_decimal_precision:int = 1):
        """
        there is a single matching dimension, then find value on that dimension for each
        object and update magnitude semantic weights
        """
        # 2a). single matching dimension
        # find the semantic links connecting to the absolute dimensional value.
        sem_link = {}
        po_dim_val = {}
        for po in [po1, po2]:
            sem_link[po] = self.find_links_to_abs_dim(idxs[po], dim, OntStatus.STATE)
            # find value on that dimension for each link, then update magnitude semantic weights
            # -> first get index of an object with the same rb
            rbs = self.network.sets[po.set].get_mask(Type.RB)
            idx_rb = ((self.network.sets[po.set].connections[rbs, idxs[po]] == B.TRUE).nonzero())[0].item()
            objs = self.network.sets[po.set].tensor_op.get_mask(Type.PO)
            objs = objs & self.network.sets[po.set].nodes[:, TF.PRED] == B.FALSE
            idx_obj = ((self.network.sets[po.set].connections[idx_rb, objs] == B.TRUE).nonzero())[0].item()
            # -> then find the value on the dimension for the object
            dim_sems = self.find_links_to_abs_dim(idx_obj, dim, OntStatus.VALUE)
            if dim_sems.any():
                po_dim_val[po] = self.network.semantics.nodes[dim_sems, SF.AMOUNT][0].item()
            else:
                po_dim_val[po] = 0.0
        # 3a). compute ent_magnitudeMoreLessSame()
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(po_dim_val[po1], po_dim_val[po2], mag_decimal_precision)
        # 4a). connect the two POs to the appropriate relative magnitude semantics
        #     (based on the invariant patterns detected just above) 
        if more == po_dim_val[po2]:
            self.network.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        else:
            self.network.attach_mag_semantics(same_flag, po2, po1, sem_link, idxs)


    def multi_dim_refinement(self, idxs, po1: Ref_Token, po2: Ref_Token, mag_decimal_precision:int = 1):
        """
        there are multiple matching dimensions,
        find the matching dimension that each PO is most strongly connected to, and update magnitude
        semantic weights.
        """
        # 2b). find the matching dimension that each PO is most strongly connected to, 
        # and update magnitude semantic weights.
        max_weight = {}
        # 2bi). find max dim for po1
        sems = self.network.semantics.nodes
        links = self.network.links[po1.set][idxs[po1], :] > 0.9
        numeric_sems = links & sems[:, SF.AMOUNT] != null
        # Get the numeric dimension sems with the highest weight
        max_weight[po1], max_idx = torch.max(self.network.links[po1.set][idxs[po1], numeric_sems], dim=0)
        if not max_weight[po1].any():
            return # po1 has no max
        # get max_dim
        max_weight[po1] = max_weight[0].item()
        max_idx = max_idx[0].item()
        max_dim = self.network.semantics.nodes[max_idx, SF.DIM].item()
        # 2bii). find max dim weight for po2
        links = self.network.links[po2.set][idxs[po2], :] > 0.9
        numeric_sems = links & sems[:, SF.AMOUNT] != null
        dim_sems = numeric_sems & sems[:, SF.DIM] == max_dim
        max_weight[po2], max_idx = torch.max(self.network.links[po2.set][idxs[po2], dim_sems], dim=0)
        if not max_weight[po2].any():
            max_weight[po2] = 0.0
        else:
            max_weight[po2] = max_weight[po2].item()
        # 3). Use current max_dim and values to compute ent_magnitudeMoreLessSame().
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(max_weight[po1], max_weight[po2], mag_decimal_precision)
        # 4). find the semantic links connecting to the absolute dimensional value.
        sem_link = {}
        for po in [po1, po2]:
            sem_link[po] = self.find_links_to_abs_dim(po, max_dim, OntStatus.STATE)
        # 5). connect the two POs to the appropriate relative magnitude semantics
        #     (based on the invariant patterns detected just above) 
        if more == max_weight[po2]:
            self.network.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        else:
            self.network.attach_mag_semantics(same_flag, po2, po1, sem_link, idxs)

    
    def ent_magnitude_more_less_same(self):
        """
        !!!! Not implemented yet. !!!!
        Magnitude comparison logic.
        alculates more/less/same from two codes of extent based on entropy and competion.
        """
        # NOTE: Have not implented entorpy net from dataTypes yet.
        #       Going to leave for now, as that seems like effort.
        # TODO: Implement entropy net from dataTypes.
        pass

    
    def attach_mag_semantics(self, same_flag: bool, po1: Ref_Token, po2: Ref_Token, sem_links: dict[Ref_Token, torch.Tensor], idxs: dict[Ref_Token, int]):
        """
        Attach magnitude semantics.
        TODO: TEST
        """
        # NOTE: I have left this bit in, but added a logger to see if it actually ever happens.
        #       If it doesn't, can just remove at some point. If it does, remove the logger.
        sdm_sems = self.network.semantics.get_sdm_indices()
        for po in [po1, po2]:
            if self.network.links[po.set][idxs[po], sdm_sems].any():
                logger.warning(f"ATTACH MAG SEMANTICS: SDM sems aleady found for {po.set}[{idxs[po]}] -> remove me :]")
                return
        # if not same, then po1 = more and po2 = less
        sdms = {
            po1: SDM.SAME if same_flag else SDM.MORE,
            po2: SDM.SAME if same_flag else SDM.LESS,
        }
        for po in [po1, po2]:
            # Connect the sems
            self.network.links.connect_comparitive(po.set, idxs[po], sdms[po])
            # reduce weights to absolute value semantics to 0.5 
            # (as this process constitutes a comparison).
            self.network.links[po.set][idxs[po], sem_links[po]] /= 2

    def update_mag_semantics(self, same_flag: bool, po1: Ref_Token, po2: Ref_Token, sem_links: dict[Ref_Token, torch.Tensor], idxs: dict[Ref_Token, int]):
        """
        Function to update the connections to magnitude semantics during the basic_en_based_mag_refinement() function.
        """
        sdms = {
            po1: SDM.SAME if same_flag else SDM.MORE,
            po2: SDM.SAME if same_flag else SDM.LESS,
        }
        for po in [po1, po2]:
            # Connect the sems
            # half other sem weights
            self.network.links[po.set][idxs[po], ~sem_links[po]] /= 2 
            # set sdm weight to 1.0
            self.network.links.connect_comparitive(po.set, idxs[po], sdms[po]) 
            # set the sem_links weights to 1.0
            self.network.links[po.set][idxs[po], sem_links[po]] = 1.0
    

    def ent_overall_same_diff(self):
        """
        Overall same/different calculation.
        """
        # Implementation using network.sets, network.semantics
        pass 

# ---------------------[ Helpers ]----------------------------

    def find_links_to_abs_dim(self, idx: int, dim: int, ont_status: OntStatus):
        """
        Find the semantic links connecting to the absolute dimensional value.
        """
        sems: torch.Tensor = self.network.semantics.nodes
        links_mask = {}
        sem_link = {}
            # 1). find the semantic links connecting to the absolute dimensional value.
        links_mask = self.network.links[idx.set][idx, :] > 0.0 # Get connected semantics
        dim_sems = sems[links_mask, SF.DIM] == dim and sems[links_mask, SF.ONT] == ont_status
        sem_link = tOps.sub_union(links_mask, dim_sems)
        return sem_link