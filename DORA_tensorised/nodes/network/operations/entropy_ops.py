# nodes/network/operations/entropy_ops.py
# Entropy operations for Network class

from logging import getLogger

from ...enums import *
from ..single_nodes import Ref_Token
from ...utils import tensor_ops as tOps
from typing import TYPE_CHECKING
import torch
from random import sample
from ...entropy_net.entropy_net import EntropyNet, Ext
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
    
    def en_based_mag_checks(self, po1: Ref_Token, po2: Ref_Token):
        """
        Check if POs code the same dimension, or connected to SDM semantics - 
        for deciding whether to include in magnitude comparison.

        Returns:
            high_dim: highest intersecting dimension
            num_sdm_above: Num pos with con to SDM sems above threshold
            num_sdm_below: Num pos with con to SDM sems below threshold
        """
        logger.debug(f"EN BASED MAG CHECKS: {po1.set.name}[{self.network.get_index(po1)}] and {po2.set.name}[{self.network.get_index(po2)}]")
        # 1). do they code for intersecting dimentsions
        # -> get instersecting dimension, by getting set of dimensions encoded by each pos semantics
        idxs = {
            po1: self.network.get_index(po1),
            po2: self.network.get_index(po2),
        }
        dims = {}
        for ref_po in [po1, po2]:
            # get semantics that have weight>0.9
            po_set = ref_po.set
            # NOTE: idk if this is inefficient, but it should work for now
            
            po_links_mask = self.network.links[po_set][idxs[ref_po], :] > 0.9
            all_sems = self.network.semantics.nodes
            dim_mask = all_sems[:, SF.DIM] != null
            ont_mask = all_sems[:, SF.ONT] == OntStatus.VALUE
            po_sems = po_links_mask & dim_mask & ont_mask
            logger.debug(f"-> {po_sems.sum().item()} po_sems found")
            if po_sems.any():
                dims[ref_po] = set(self.network.semantics.nodes[po_sems, SF.DIM].unique().tolist())
            else:
                dims[ref_po] = set()
        # then take the intersection of the two sets
        intersecting_dims = dims.get(po1).intersection(dims.get(po2))
        logger.debug(f"-> {len(intersecting_dims)} intersecting dims found")
            
        # 2). do either PO have connections to SDM (comparative) semantics above threshold (=.9)
        #   or do both connect to any SDM sems below thresh (=.9)
        sdm_indicies = self.network.semantics.get_sdm_indices()
        if sdm_indicies.shape[0] != len(SDM):
            logger.critical(f"SDM indices do not match the number of SDM enums")
        # Get slice of connections to SDM tensors, with T/F if they above threshold
        sdms = {po1 : {},po2 : {}} # map PO to dict of SDM above/below threshold
        for po in [po1, po2]:
            po_set = po.set
            con_sdm = self.network.links[po_set][idxs[po], sdm_indicies]
            sdms[po]["above"]= (con_sdm > 0.9).any().item()
            sdms[po]["below"] = ((0.0 < con_sdm) & (con_sdm < 0.9)).any().item()

        num_sdm_above = int(sdms[po1]["above"]) + int(sdms[po2]["above"])
        num_sdm_below = int(sdms[po1]["below"]) + int(sdms[po2]["below"])
        logger.debug(f"-> {num_sdm_above} num_sdm_above and {num_sdm_below} num_sdm_below")

        # 3). find the dimension of highest overlap 
        # (sum the semantics coding dimensions value that connect to both POs)
        # -> for each intersecting dim, sum value if on_status = value
        # -> if the highest weights are equal, add to list and randomly select one
        high_dim = []
        high_dim_weight = 0.0
        for dim in intersecting_dims:
            dim_weight = {}
            for po in [po1, po2]:
                # Find semantics connected to `po` on `dim` with `OntStatus.VALUE`
                all_sems = self.network.semantics.nodes
                sem_mask = (all_sems[:, SF.DIM] == dim) & \
                           (all_sems[:, SF.ONT] == OntStatus.VALUE)
                
                # Get links from `po` to these semantics
                link_weights = self.network.links[po.set][idxs[po], sem_mask]

                # if there are links, then get the the max weight
                if link_weights.any():
                    dim_weight[po] = torch.max(link_weights).item()
                else:
                    dim_weight[po] = 0.0
            logger.debug(f"-> [DIM:{dim}] {dim_weight[po1]} dim_weight[po1] and {dim_weight[po2]} dim_weight[po2]")

            # if both POs have dim_weight, sum weights
            curr_weight = dim_weight[po1] + dim_weight[po2]
            if curr_weight > high_dim_weight:
                high_dim = [dim]
                high_dim_weight = curr_weight
            elif curr_weight == high_dim_weight and curr_weight > 0:
                high_dim.append(dim)
        logger.debug(f"-> {high_dim} high_dim")

        # if multiple equal high dims, randomly select one
        if len(high_dim) > 1:
            logger.debug(f"-> randomly selecting one")
            high_dim = sample(high_dim, 1)
            logger.debug(f"-> {high_dim} high_dim")
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
        """
        logger.debug(f"CHECK AND RUN ENT OPS WITHIN: {po1.set}[{self.network.get_index(po1)}] and {po2.set}[{self.network.get_index(po2)}]")
        is_pred = self.network.node_ops.get_value(po1, TF.PRED) == B.TRUE
        if (is_pred):
            if (
                num_sdm_above == 0
                and num_sdm_below < 2
                and len(intersect_dim) > 0
            ):
                return self.basic_en_based_mag_comparison(po1, po2, intersect_dim, mag_decimal_precision)
            elif (
                extend_SDML
                and num_sdm_above < 2
                and (num_sdm_above > 0 or num_sdm_below > 0)
            ):
                return self.basic_en_based_mag_refinement(po1, po2)
        else: # is object
            if (
                len(intersect_dim) > 0
                and not pred_only
                and not pred_present
                and num_sdm_above == 0
            ):
                return self.basic_en_based_mag_comparison(po1, po2, intersect_dim, mag_decimal_precision)
    

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
        logger.debug(f"BASIC EN BASED MAG COMPARISON: {po1.set.name}[{self.network.get_index(po1)}] and {po2.set.name}[{self.network.get_index(po2)}]")
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
            sem_link[po] = self.find_links_to_abs_dim(po.set, idxs[po], dim, OntStatus.VALUE)
            # 2). if the dimension is numeric, then get the average value of all 
            # dimensionnal values in the sem_links and assign these to 
            # extent1 and extend2 respectively
            if not sem_link[po].any():
                logger.critical(f"-> No links to absolute dimensional value found for {po.set.name}[{idxs[po]}], dim: {dim}, ont_status: {OntStatus.VALUE}")
            idx = sem_link[po].nonzero()[0]
            is_numeric = bool((sems[idx, SF.AMOUNT] != null).item())
            assert is_numeric, f"-> Dimension {dim} is non-numeric for {po.set.name}[{idxs[po]}]" # NOTE: This should always be true, but gonna leave in for now.
            extent[po] = sems[sem_link[po], SF.AMOUNT].mean().item()
        # 3). compute ent_magnitudeMoreLessSame() NOTE: What if the extent is non-numeric?
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(float(extent.get(po1)), float(extent.get(po2)), mag_decimal_precision)
        for po in [po1, po2]:
            # 4). Find any other dimensional semantics with high weights can be reduced by the entropy process.
            po_links_mask = self.network.links[po.set][idxs[po], :] > 0.0
            other_dim_sems = ~torch.isin(sems[po_links_mask, SF.DIM], torch.tensor([dim, null], dtype=torch.float))
            other_sem_link[po] = tOps.sub_union(sem_link[po], other_dim_sems)
            sem_link[po] = sem_link[po] | other_sem_link[po]
        # 5). Connect the two POs to the appropriate relative magnitude semantics 
        # (based on the invariant patterns detected just above).
        if same_flag:
            self.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        elif more == extent[po1]:
            self.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        else: # more == extent[po2]
            self.attach_mag_semantics(same_flag, po2, po1, sem_link, idxs)


    def basic_en_based_mag_refinement(self, po1: Ref_Token, po2: Ref_Token):
        """
        Basic magnitude refinement:

        if there are magnitude semantics present, and there are some matching dimensions, 
        then activate the appropriate magnitude semantics and matching dimensions, and adjust
        weights as appropriate (i.e., turn on the appropriate magnitude semantics for each PO, 
        and adjust weight accordingly).
        """
        logger.debug(f"BASIC EN BASED MAG REFINEMENT: {po1.set.name}[{self.network.get_index(po1)}] and {po2.set.name}[{self.network.get_index(po2)}]")
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
            po_sems_mask = self.network.links[po_set][idxs[ref_po], :] > 0.9
            dim_mask = self.network.semantics.nodes[:, SF.DIM] != null
            state_mask = self.network.semantics.nodes[:, SF.ONT] == OntStatus.STATE
            final_mask = po_sems_mask & dim_mask & state_mask
            
            if final_mask.any():
                dims[ref_po] = set(self.network.semantics.nodes[final_mask, SF.DIM].unique().tolist())
            else:
                dims[ref_po] = set()

        # then take the intersection of the two sets
        intersecting_dims = dims.get(po1, set()).intersection(dims.get(po2, set()))
        # 2). if there is a single matching dimension, then find value on that dimension for each
        # object and update magnitude semantic weights; elif there are multiple matching dimensions,
        # find the matching dimension that each PO is most strongly connected to, and update magnitude
        # semantic weights.
        if len(intersecting_dims) == 1:
            self.single_dim_refinement(idxs, po1, po2, list(intersecting_dims)[0], mag_decimal_precision)
        elif len(intersecting_dims) > 1:
            self.multi_dim_refinement(idxs, po1, po2, mag_decimal_precision)


    def single_dim_refinement(self, idxs, po1: Ref_Token, po2: Ref_Token, dim: int, mag_decimal_precision:int = 1):
        """
        there is a single matching dimension, then find value on that dimension for each
        object and update magnitude semantic weights
        """
        logger.debug(f"SINGLE DIM REFINEMENT: {po1.set}[{self.network.get_index(po1)}] and {po2.set}[{self.network.get_index(po2)}]")
        # 2a). single matching dimension
        # find the semantic links connecting to the absolute dimensional value.
        sem_link = {}
        po_dim_val = {}
        for po in [po1, po2]:
            sem_link[po] = self.find_links_to_abs_dim(po.set, idxs[po], dim, OntStatus.STATE)
            # find value on that dimension for each link, then update magnitude semantic weights
            # -> first get index of an object with the same rb
            rbs = self.network.sets[po.set].get_mask(Type.RB)
            parent_mask = self.network.sets[po.set].connections[:, idxs[po]].bool()
            idx_rb = ((parent_mask & rbs)).nonzero()[0].item()
            objs = self.network.sets[po.set].tensor_op.get_mask(Type.PO)
            objs = objs & (self.network.sets[po.set].nodes[:, TF.PRED] == B.FALSE)
            child_mask = self.network.sets[po.set].connections[idx_rb, :].bool()
            idx_obj = ((child_mask & objs)).nonzero()[0].item()
            # -> then find the value on the dimension for the object
            dim_sems = self.find_links_to_abs_dim(po.set, idx_obj, dim, OntStatus.VALUE)
            if dim_sems.any():
                po_dim_val[po] = self.network.semantics.nodes[dim_sems, SF.AMOUNT][0].item()
            else:
                po_dim_val[po] = 0.0
        # 3a). compute ent_magnitudeMoreLessSame()
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(float(po_dim_val[po1]), float(po_dim_val[po2]), mag_decimal_precision)
        # 4a). connect the two POs to the appropriate relative magnitude semantics
        #     (based on the invariant patterns detected just above)
        if same_flag:
            self.update_mag_semantics(True, po1, po2, sem_link, idxs)
        elif more == po_dim_val[po1]:
            self.update_mag_semantics(False, po1, po2, sem_link, idxs)
        else:
            self.update_mag_semantics(False, po2, po1, sem_link, idxs)


    def multi_dim_refinement(self, idxs, po1: Ref_Token, po2: Ref_Token, mag_decimal_precision:int = 1):
        """
        there are multiple matching dimensions,
        find the matching dimension that each PO is most strongly connected to, and update magnitude
        semantic weights.
        """
        logger.debug(f"MULTI DIM REFINEMENT: {po1.set}[{self.network.get_index(po1)}] and {po2.set}[{self.network.get_index(po2)}]")
        # 2b). find the matching dimension that each PO is most strongly connected to, 
        # and update magnitude semantic weights.
        max_weight = {}
        # 2bi). find max dim for po1
        sems = self.network.semantics.nodes
        links = self.network.links[po1.set][idxs[po1], :] > 0.9
        numeric_sems = links & (sems[:, SF.AMOUNT] != null)
        # Get the numeric dimension sems with the highest weight
        if not numeric_sems.any():
            return # No numeric semantics to compare

        max_weight_tensor, max_idx_tensor = torch.max(self.network.links[po1.set][idxs[po1], numeric_sems], dim=0)
        if not max_weight_tensor.any():
            return  # po1 has no max

        # get max_dim
        max_weight[po1] = max_weight_tensor.item()
        numeric_indices = numeric_sems.nonzero().flatten()
        max_idx = numeric_indices[max_idx_tensor.item()].item()
        max_dim = self.network.semantics.nodes[max_idx, SF.DIM].item()
        
        # 2bii). find max dim weight for po2
        links = self.network.links[po2.set][idxs[po2], :] > 0.9
        numeric_sems = links & (sems[:, SF.AMOUNT] != null)
        dim_sems = numeric_sems & (sems[:, SF.DIM] == max_dim)
        if not dim_sems.any():
            max_weight[po2] = 0.0
        else:
            max_weight_tensor, _ = torch.max(self.network.links[po2.set][idxs[po2], dim_sems], dim=0)
            if not max_weight_tensor.any():
                max_weight[po2] = 0.0
            else:
                max_weight[po2] = max_weight_tensor.item()
        # 3). Use current max_dim and values to compute ent_magnitudeMoreLessSame().
        more, less, same_flag, iterations = self.ent_magnitude_more_less_same(float(max_weight[po1]), float(max_weight[po2]), mag_decimal_precision)
        # 4). find the semantic links connecting to the absolute dimensional value.
        sem_link = {}
        for po in [po1, po2]:
            sem_link[po] = self.find_links_to_abs_dim(po.set, idxs[po], max_dim, OntStatus.STATE)
        # 5). connect the two POs to the appropriate relative magnitude semantics
        #     (based on the invariant patterns detected just above)
        if same_flag:
            self.attach_mag_semantics(same_flag, po1, po2, sem_link, idxs)
        elif more == max_weight[po1]:
            self.attach_mag_semantics(False, po1, po2, sem_link, idxs)
        else:
            self.attach_mag_semantics(False, po2, po1, sem_link, idxs)

    def ent_magnitude_more_less_same(self, extent1: float, extent2: float, mag_decimal_precision:int = 0):
        """
        Magnitude comparison logic.
        alculates more/less/same from two codes of extent based on entropy and competion.
        """
        logger.debug(f"ENT MAGNITUDE MORE LESS SAME: ext1: {extent1}, ext2: {extent2}")
        extent1_rounded = round(extent1 * (pow(100, mag_decimal_precision))) + 1
        extent2_rounded = round(extent2 * (pow(100, mag_decimal_precision))) + 1
        logger.debug(f"-> extent1_rounded: {extent1_rounded}, extent2_rounded: {extent2_rounded}")
        entropyNet = EntropyNet()
        entropyNet.fillin(extent1_rounded, extent2_rounded)
        entropyNet.run_entropy_net(0.3, 0.1)
        logger.debug(f"-> entropyNet.settled_iters: {entropyNet.settled_iters}")
        nodes = {
            Ext.SMALL: extent1 if extent1_rounded < extent2_rounded else extent2,
            Ext.LARGE: extent2 if extent1_rounded < extent2_rounded else extent1,
        }
        more_ext, less_ext = entropyNet.get_more_less()
        more = nodes[more_ext] if more_ext is not None else None
        less = nodes[less_ext] if less_ext is not None else None
        same_flag = more_ext is None
        logger.debug(f"-> more: {more}, less: {less}, same_flag: {same_flag}")
        return more, less, same_flag, entropyNet.settled_iters

    
    def attach_mag_semantics(self, same_flag: bool, po1: Ref_Token, po2: Ref_Token, sem_links: dict[Ref_Token, torch.Tensor], idxs: dict[Ref_Token, int]):
        """
        Attach magnitude semantics.
        """
        logger.debug(f"ATTACH MAG SEMANTICS: {po1.set.name}[{self.network.get_index(po1)}] and {po2.set.name}[{self.network.get_index(po2)}]")
        # NOTE: I have left this bit in, but added a logger to see if it actually ever happens.
        #       If it doesn't, can just remove at some point. If it does, remove the logger.
        sdm_sems = self.network.semantics.get_sdm_indices()
        for po in [po1, po2]:
            if (self.network.links[po.set][idxs[po], sdm_sems]>0.0).sum().item() > 0:
                logger.debug(len(sdm_sems))
                logger.debug(f"-> {self.network.links[po.set][idxs[po], sdm_sems]}")
                logger.debug(f"!!!ATTACH MAG SEMANTICS!!!: SDM sems aleady found for {po.set.name}[{idxs[po]}] -> remove me :]")
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
        logger.debug(f"UPDATE MAG SEMANTICS: {po1.set}[{self.network.get_index(po1)}] and {po2.set}[{self.network.get_index(po2)}]")
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
        Function to calculate over-all same/diff from entropy across all semantics.
        """
        logger.debug(f"ENT OVERALL SAME DIFF")
        # check semantics and calulate a similarity score
        # as a ratio of unshared to total features.
        sems = self.network.semantics.nodes
        act_tensor = sems[:, SF.ACT].clone()
        error_tensor = torch.zeros_like(act_tensor)
        error_mask = act_tensor > 0.1
        error_tensor[error_mask] = 1.0
        act_tensor[~error_mask] = 0.0
        # calculate the error by subtracting act_tensor from error_tensor.
        error_tensor -= act_tensor
        # sum the error and act tensors.
        sum_diff = error_tensor.sum()
        sum_act = act_tensor.sum()
        # make sure that you're not dividing by 0 (which can happen if you've tried to
        #  compute entropy for empty objects).
        if sum_act.item() > 0:
            difference_ratio = (sum_diff / sum_act).item()
        else:
            difference_ratio = 0.0
        return difference_ratio

# ---------------------[ Helpers ]----------------------------

    def find_links_to_abs_dim(self, set: Set, idx: int, dim: int, ont_status: OntStatus):
        """
        Find the semantic links connecting to the absolute dimensional value.
        """
        logger.debug(f"FIND LINKS TO ABS DIM: {set.name}[{idx}], dim: {dim}, ont_status: {ont_status.name}")
        sems: torch.Tensor = self.network.semantics.nodes
        # 1). find the semantic links connecting to the absolute dimensional value.
        links_mask = self.network.links[set][idx, :] > 0.0 # Get connected semantics
        dim_sems = (sems[:, SF.DIM] == dim) & (sems[:, SF.ONT] == ont_status)
        sem_link= links_mask & dim_sems
        logger.debug(f"-> {sem_link.sum().item()}/{sems.shape[0]} sem_link")
        return sem_link

class en_based_mag_checks_results:
    """
    Results of en_based_mag_checks.
    """
    def __init__(self, po1: Ref_Token, po2: Ref_Token, high_dim: int, num_sdm_above: int, num_sdm_below: int):
        self.po1 = po1
        self.po2 = po2
        self.high_dim = high_dim
        self.num_sdm_above = num_sdm_above
        self.num_sdm_below = num_sdm_below
