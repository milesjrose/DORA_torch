# nodes/sets/connections/mappings.py
# Mappings between nodes and semantics.

import torch
import logging
logger = logging.getLogger(__name__)

from ...enums import *
from enum import IntEnum
from ...utils import tensor_ops as tOps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sets import Driver, Recipient

class MD(IntEnum):
    """
    Enum to access mapping dimension, i.e mappings.shape[MD.DRIVER]
    """
    REC = 0
    """ Recipient dimension """
    DRI = 1
    """ Driver dimension """

class Mappings(object):
    """
    A class for storing mappings and hypothesis information.
    """
    def __init__(self, driver, map_fields: dict[MappingFields, torch.Tensor]):
        """
        Initialize the Mappings object.
        Args:
            driver (Driver): driver that mappings map to.
            map_fields (dict[MappingFields, torch.Tensor]): dictionary of tensors for each mapping field.
        
        Raises:
            ValueError: If the tensors are not torch.Tensor. Or have incorrect shape.
        """
        # TODO: Combine connections and weight tensors into one tensor.
        # Check tensors are correct type and shape
        con_shape = map_fields[MappingFields.CONNECTIONS].shape[0] # Used to check all tensors have same shape
        for field in MappingFields:
            # Check tensor is correct type
            if type(map_fields[field]) != torch.Tensor:
                raise ValueError(f"{field} tensor must be torch.Tensor, but is {type(map_fields[field])}.")
            
            # Check tensor is 2D
            if map_fields[field].dim() != 2:
                raise ValueError(f"{field} tensor should be 2D, but has {map_fields[field].dim()} dimensions.")
            
            # Check all tensors have same shape
            shape = map_fields[field].shape[0]
            if shape != con_shape:
                raise ValueError(f"{field} field tensor shape: {shape}, but connections tensor shape: {con_shape}.")
            
            # Check driver nodes match
            driver_count = driver.nodes.shape[0]
            if map_fields[field].shape[1] != driver_count:
                raise ValueError(f"{field} field tensor shape: {map_fields[field].shape}, but driver nodes shape: {driver.nodes.shape}.")
        
        # Stack the tensors along a new dimension based on MappingFields enum
        self.driver: 'Driver' = driver
        self.recipient: 'Recipient' = None # Set by network
        field_list = []
        for field in MappingFields:
            field_list.append(map_fields[field])                            # Add each tensor to the list, indexed by MappingFields enum
        self.adj_matrix: torch.Tensor = torch.stack(field_list, dim=-1)     # Stack the tensors, last dimension is the field
        """ Stacked tensor of shape (N, D, F):
        - N: number of nodes in this set
        - D: number of nodes in driver set
        - F: number of fields in MappingFields enum
        """
    
    # ===================[ Getters and setters ]====================
    def size(self, dim):
        return self.adj_matrix.size(dim=dim)
    
    def __getitem__(self, mappingField: MappingFields):
        return self.adj_matrix[:, :, mappingField]
    
    def __setitem__(self, mappingField: MappingFields, value):
        self.adj_matrix[:, :, mappingField] = value
    
    def set_map_from(self, map_from):
        """
        Set the map_from attribute.
        Args:
            map_from (Set): The set that the mappings map from (e.g recipient, memory, new_set).
        """
        self.recipient: 'Recipient' = map_from

    # =====================[ Update functions ]======================
    def update_hypotheses(self):
        """!
        Update the hypotheses matrix.
        NOTE: Seems very inefficient
        TODO: Implement a more efficient method
        """
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

    def update_hypothesis(self, driver_mask, map_from_mask):
        """!
        Update the hypothesis matrix, for nodes in given masks.
        """
        # Hypothesis = hypothesis + (driver_token.act * recipient_token.act)
        driver_acts = self.driver.nodes[:, TF.ACT]
        map_from_acts = self.recipient.nodes[:, TF.ACT]
        
        # Create outer product of activations for all combinations
        activation_product = torch.outer(map_from_acts, driver_acts)
        
        # Apply masks to only update the relevant node combinations
        mask_2d = torch.outer(map_from_mask, driver_mask)
        self[MappingFields.HYPOTHESIS] += activation_product * mask_2d

    def reset_hypotheses(self):
        """!
        Reset the hypotheses/max hypotheses to 0.
        """
        self[MappingFields.HYPOTHESIS] = 0.0
        self[MappingFields.MAX_HYP] = 0.0
    
    def reset_mapping_units(self):
        """!
        Reset the hypotheses and connections.
        """
        self.reset_hypotheses()
        self[MappingFields.WEIGHT] = 0.0
    
    def reset_mappings(self):
        """!
        Reset the hypotheses, connections, and max map.
        """
        self.reset_mapping_units()
        self.driver.token_ops.set_features_all(TF.MAX_MAP, 0.0)
        self.recipient.token_ops.set_features_all(TF.MAX_MAP, 0.0)
        self[MappingFields.MAX_HYP] = 0.0
    
    def update_connections(self, eta):
        """! update_weight
        Update the weight matrix.
        Args:
            eta (float): Learning rate.
        """
        # 1). Divisively normalise all mapping hypotheses:
        self.get_max_hypothesis()
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

    def get_max_hypothesis(self):
        """ ! update_max_hyp
        For each hypothesis, find the maximum hypothesis of either unit involved in that hypothesis.
        """
        # max_hypothesis[i,j] = max(max(hypothesis[i,:]), max(hypothesis[:,j]))

        max_recipient = self[MappingFields.HYPOTHESIS].max(dim=1).values
        max_driver = self[MappingFields.HYPOTHESIS].max(dim=0).values
        max_values = tOps.max_broadcast(max_recipient, max_driver)

        self[MappingFields.MAX_HYP] = max_values
    
    def get_max_map(self):
        """!
        For each token, find the token with the highest connection weight.

        NOTE: Not sure on usage on these values currently. 
        For now they are returned, and then used by higher classes to assign to the set tensors.

        Returns:
            max_recipient (torch.Tensor): Object containing index and weight of driver token with highest weight for recipient token.
            max_driver (torch.Tensor): Object containing index and weight of recipient token with highest weight for driver token.
        """
        # max_connection[i] = max(connection[i,:]))
        max_recipient: torch.return_types.max = self[MappingFields.WEIGHT].max(dim=1)
        max_driver: torch.return_types.max = self[MappingFields.WEIGHT].max(dim=0)
        return max_recipient, max_driver
    
    def swap_driver_recipient(self):
        """!
        swap the driver and recipient
        NOTE:clear mappings? Currently just transposing the mapping tensor
        """
        stack = []
        for field in MappingFields:
            stack.append(self[field].t())
        #stack
        self.adj_matrix = torch.stack(stack, dim=-1)
    
    # ----------------------------------------------------------------

    def expand_mapping_tensor(self, new_count: int, dimension: MD):
        """!
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
            logger.debug(f"-> Expanded {self.recipient.token_set.name} mappings tensor: {new_adj_matrix.shape}")
        except Exception as e:
            logger.error(f"-> Error expanding {self.recipient.token_set.name} mappings tensor {old_recipient_count}x{old_driver_count} -> {new_recipient_count}x{new_driver_count}")
            raise e

    def print(self, mapping_field: MappingFields = MappingFields.WEIGHT, d_mask=None, r_mask=None):                                  # Here for testing atm
        """
        Print the mappings.

        Args:
            f_types (list[TF], optional): The features to print.

        Raises:
            ValueError: If nodePrinter is not found.
        """
        try:
            from nodes.utils import nodePrinter
        except:
            print("Error: nodePrinter not found. Nodes.utils.nodePrinter is required to use this function.")
        else:
            try:
                p_tensor = self[mapping_field]
                printer = nodePrinter(print_to_console=True)
                if d_mask is not None and r_mask is not None:
                    p_tensor = p_tensor[r_mask][:, d_mask]
                elif d_mask is not None:
                    p_tensor = p_tensor[:, d_mask]
                elif r_mask is not None:
                    p_tensor = p_tensor[r_mask]
                if mapping_field == MappingFields.CONNECTIONS:
                    printer.print_con_tensor(p_tensor)
                else:
                    match mapping_field:
                        case MappingFields.MAX_HYP:
                            printer.print_weight_tensor(p_tensor, headers=["Max Hypothesis:"])
                        case MappingFields.HYPOTHESIS:
                            printer.print_weight_tensor(p_tensor, headers=["Hypothesis:"])
                        case MappingFields.WEIGHT:
                            printer.print_weight_tensor(p_tensor, headers=["Weight:"])
                        case _:
                            raise ValueError(f"Invalid mapping field: {mapping_field}")

            except Exception as e:
                print("Error: NodePrinter failed to print set.")
                print(e)