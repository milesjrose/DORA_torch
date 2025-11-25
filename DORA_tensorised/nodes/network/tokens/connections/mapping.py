import torch
from enum import IntEnum
from ....enums import *
from ....utils import tensor_ops as tOps
from logging import getLogger
from ..tensor_view import TensorView
logger = getLogger(__name__)

class MD(IntEnum):
    """
    Enum to access mapping dimension, i.e mappings.shape[MD.DRIVER]
    """
    REC = 0
    """ Recipient dimension """
    DRI = 1
    """ Driver dimension """

class Mapping:
    """
    A class to hold the mapping tensor (recipient -> driver).
    """
    def __init__(self, adj_matrix: torch.Tensor):
        """
        Initialise the Mapping object.
        """
        # mapping tensor shape: [recipient_nodes, driver_nodes, fields]
        # mapping tensor contains:
        # - weight: float - The actual mapping weight.
        # - hypothesis: float - Hypothesis used to update the weight.
        # - max_hyp: float - Maximum hypothesis of the token.
        assert len(adj_matrix.shape) == 3, "Mapping tensor must have 3 dimensions"
        assert adj_matrix.shape[2] == len(MappingFields), f"Mapping tensor must have {len(MappingFields)} fields in the third dimension"
        self.adj_matrix: torch.Tensor = adj_matrix
    
    # ===================[ Getters and setters ]====================
    def size(self, dim):
        return self.adj_matrix.size(dim=dim)
    
    def __getitem__(self, mappingField: MappingFields):
        return self.adj_matrix[:, :, mappingField]
    
    def __setitem__(self, mappingField: MappingFields, value):
        self.adj_matrix[:, :, mappingField] = value
    
    def get_max_map(self):
        """
        For each token, find the token with the highest connection weight.

        NOTE: Not sure on usage on these values currently. 
        For now they are returned, and then used by higher classes to assign to the set tensors.

        Returns:
            max_recipient (torch.Tensor): Object containing index and weight of driver token with highest weight for recipient token.
            max_driver (torch.Tensor): Object containing index and weight of recipient token with highest weight for driver token.
        """
        # max_connection[i] = max(connection[i,:]))
        max_recipient: torch.return_types.max = self[MappingFields.WEIGHT].max(dim=MD.DRI)
        max_driver: torch.return_types.max = self[MappingFields.WEIGHT].max(dim=MD.REC)
        return max_recipient, max_driver
    
    # =====================[ Reset functions ]======================
    def reset_hypotheses(self):
        """
        Reset the hypotheses/max hypotheses to 0.
        """
        logger.debug(f"-> Resetting hypotheses")
        self[MappingFields.HYPOTHESIS] = 0.0
        self[MappingFields.MAX_HYP] = 0.0
    
    def reset_mapping_units(self):
        """
        Reset the hypotheses and connections.
        """
        logger.debug(f"-> Resetting mapping units")
        self.reset_hypotheses()
        self[MappingFields.WEIGHT] = 0.0
    
    def reset_mappings(self):
        """
        Reset the hypotheses, connections, and max map.
        """
        logger.debug(f"-> Resetting mappings")
        self.reset_mapping_units()
        self.driver.token_ops.set_features_all(TF.MAX_MAP, 0.0)
        self.recipient.token_ops.set_features_all(TF.MAX_MAP, 0.0)
        self[MappingFields.MAX_HYP] = 0.0
    
    # =====================[ Update functions ]======================
    def update_weight(self, eta: float):
        """
        Update the weight of the mapping.
        Args:
            eta: float - The learning rate.
        """
        logger.debug(f"-> Updating weight: eta={eta}")
        # 1). Divisively normalise all mapping hypotheses:
        self.update_max_hyp()
        # Mask out max hyp=0 to avoid division by zero
        mask = self[MappingFields.MAX_HYP] > 0
        self[MappingFields.HYPOTHESIS][mask] /= self[MappingFields.MAX_HYP][mask]
        # 2). Subtractively normalise each hypothesis:
        self[MappingFields.MAX_HYP] = tOps.efficient_local_max_excluding_self(self[MappingFields.HYPOTHESIS])
        self[MappingFields.HYPOTHESIS] -= self[MappingFields.MAX_HYP]
        # 3). Update the weights matrix, clamped to between 0 and 1.
        self[MappingFields.WEIGHT] = torch.clamp(
            eta * (1.1 - self[MappingFields.WEIGHT]) * self[MappingFields.HYPOTHESIS], 
            0, 1)
    
    def update_max_hyp(self):
        """
        For each hypothesis, find the maximum hypothesis of either unit involved in that hypothesis.
        """
        logger.debug(f"-> Updating max hypothesis")
        # max_hypothesis[i,j] = max(max(hypothesis[i,:]), max(hypothesis[:,j]))
        max_recipient = self[MappingFields.HYPOTHESIS].max(dim=MD.DRI).values
        max_driver = self[MappingFields.HYPOTHESIS].max(dim=MD.REC).values
        max_values = tOps.max_broadcast(max_recipient, max_driver)

        self[MappingFields.MAX_HYP] = max_values
    
    def update_hypothesis(self, driver_mask, recipient_mask):
        """
        Update the hypothesis matrix, for nodes in given masks.
        """
        raise NotImplementedError("update_hypothesis not ported to new tensor structure yet")
        # Hypothesis = hypothesis + (driver_token.act * recipient_token.act)
        driver_acts = self.driver.nodes[:, TF.ACT]
        map_from_acts = self.recipient.nodes[:, TF.ACT]
        
        # Create outer product of activations for all combinations
        activation_product = torch.outer(map_from_acts, driver_acts)
        
        # Apply masks to only update the relevant node combinations
        mask_2d = torch.outer(recipient_mask, driver_mask)
        self[MappingFields.HYPOTHESIS] += activation_product * mask_2d
    
    def update_hypotheses(self, driver_mask, recipient_mask):
        """
        Update the hypotheses matrix.
        NOTE: Seems very inefficient
        TODO: Implement a more efficient method
        """
        raise NotImplementedError("update_hypotheses not ported to new tensor structure yet")
        # Need to check that the type of p/po nodes match.
        # Can do this by refining masks to type, then updating the these masks first. So only matching node types will be included
        r_p = self.recipient.get_mask(Type.P)
        d_p = self.driver.get_mask(Type.P)

        r_po = self.recipient.get_mask(Type.PO)
        d_po = self.driver.get_mask(Type.PO)

        print("r_p: ", r_p)
        print("d_p: ", d_p)

        # Update child p
        r_pc = tOps.refine_mask(self.recipient.nodes, r_p, TF.MODE, Mode.CHILD)
        d_pc = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.CHILD)
        self.update_hypothesis(d_pc, r_pc)
        print("r_pc: ", r_pc)
        print("d_pc: ", d_pc)

        # Update parent p
        r_pp = tOps.refine_mask(self.recipient.nodes, r_p, TF.MODE, Mode.PARENT)
        d_pp = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.PARENT)
        self.update_hypothesis(d_pp, r_pp)

        # Update neutral p
        r_pn = tOps.refine_mask(self.recipient.nodes, r_p, TF.MODE, Mode.NEUTRAL)
        d_pn = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.NEUTRAL)
        self.update_hypothesis(d_pn, r_pn)

        # Update Pred
        r_pr = tOps.refine_mask(self.recipient.nodes, r_po, TF.PRED, B.TRUE)
        d_pr = tOps.refine_mask(self.driver.nodes, d_po, TF.PRED, B.TRUE)
        self.update_hypothesis(d_pr, r_pr)

        # Update Obj
        r_ob = tOps.refine_mask(self.recipient.nodes, r_po, TF.PRED, B.FALSE)
        d_ob = tOps.refine_mask(self.driver.nodes, d_po, TF.PRED, B.FALSE)
        self.update_hypothesis(d_ob, r_ob)

        # Update other
        r_other = r_p & ~(r_pc | r_pp | r_pn | r_pr | r_ob)
        d_other = d_p & ~(d_pc | d_pp | d_pn | d_pr | d_ob)
        self.update_hypothesis(d_other, r_other)
    
    # =====================[ Other functions ]======================
    def get_view(self, indices: torch.Tensor):
        """
        Get a view of the mapping tensor for the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to create a view of.
        Returns:
            TensorView - A view-like object that maps operations back to the original tensor.
        """
        return TensorView(self.adj_matrix, indices)

    def expand(self, new_count: int, dimension: MD):
        """
        Expand mapping tensor for given set. If driver, expand along dim=1, else along dim=0.
        """
        try:
            # Get old tensor dimensions
            old_driver_count = self.adj_matrix.size(dim=MD.DRI)
            old_recipient_count = self.adj_matrix.size(dim=MD.REC)
            # Get new tensor dimensions
            new_driver_count = new_count if dimension == MD.DRI else old_driver_count
            new_recipient_count = new_count if dimension == MD.REC else old_recipient_count
            # Create new tensor for each mapping field, and stack into new adj_matrix
            stack = []
            for field in MappingFields:          
                stack.append(torch.zeros(
                    new_recipient_count, 
                    new_driver_count, 
                    dtype=torch.float)
                    )
            new_adj_matrix: torch.Tensor = torch.stack(stack, dim=-1)
            rows, cols, _ = self.adj_matrix.shape    

            # Copy weights and update object                   
            new_adj_matrix[:rows, :cols, :] = self.adj_matrix  
            self.adj_matrix = new_adj_matrix 
            logger.debug(f"-> Expanded mapping tensor: {old_recipient_count}x{old_driver_count} -> {new_recipient_count}x{new_driver_count}")
        except Exception as e:
            logger.error(f"-> Error expanding {self.recipient.token_set.name} mappings tensor {old_recipient_count}x{old_driver_count} -> {new_recipient_count}x{new_driver_count}")
            raise e
    
    def swap_driver_recipient(self):
        """
        Called when swapping the driver and recipient in the network.
        NOTE: clear mappings? Currently just transposing the mapping tensor
        """
        stack = []
        for field in MappingFields:
            stack.append(self[field].t())
        #stack
        self.adj_matrix = torch.stack(stack, dim=-1)
    
    def delete_connections(self, indices: torch.Tensor):
        """
        Delete connections to/from the given indices.
        """
        self.adj_matrix[:, indices, :] = 0.0
        self.adj_matrix[indices, :, :] = 0.0
