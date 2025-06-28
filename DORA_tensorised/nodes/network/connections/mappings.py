# nodes/sets/connections/mappings.py
# Mappings between nodes and semantics.
# TODO: Implement add_mappings, updateHypotheses

import torch

from ...enums import *
from ...utils import tensor_ops as tOps

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
        self.driver = driver
        self.map_from = None # Set to recipient/memory/new_set by network
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
        self.map_from = map_from

    # =====================[ Update functions ]======================
    def update_hypotheses(self):
        """
        Update the hypotheses matrix.
        NOTE: Seems very inefficient
        TODO: Implement a more efficient method, and add tests.
        """
        # Need to check that the type of p/po nodes match.
        # Can do this by refining masks to type, then updating the these masks first. So only matching node types will be included
        r_p = self.map_from.get_mask(Type.P)
        d_p = self.driver.get_mask(Type.P)

        r_po = self.map_from.get_mask(Type.PO)
        d_po = self.driver.get_mask(Type.PO)

        print("r_po.shape: ", r_po.shape)
        print("recipient shape: ", self.map_from.nodes.shape)
        print("d_po.shape: ", d_po.shape)
        print("driver shape: ", self.driver.nodes.shape)

        # Update child p
        r_pc = tOps.refine_mask(self.map_from.nodes, r_p, TF.MODE, Mode.CHILD)
        d_pc = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.CHILD)
        self.update_hypothesis(d_pc, r_pc)

        # Update parent p
        r_pp = tOps.refine_mask(self.map_from.nodes, r_p, TF.MODE, Mode.PARENT)
        d_pp = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.PARENT)
        self.update_hypothesis(d_pp, r_pp)

        # Update neutral p
        r_pn = tOps.refine_mask(self.map_from.nodes, r_p, TF.MODE, Mode.NEUTRAL)
        d_pn = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.NEUTRAL)
        self.update_hypothesis(d_pn, r_pn)

        # Update Pred
        r_pr = tOps.refine_mask(self.map_from.nodes, r_po, TF.PRED, B.TRUE)
        d_pr = tOps.refine_mask(self.driver.nodes, d_po, TF.PRED, B.TRUE)
        self.update_hypothesis(d_pr, r_pr)

        # Update Obj
        r_ob = tOps.refine_mask(self.map_from.nodes, r_po, TF.PRED, B.FALSE)
        d_ob = tOps.refine_mask(self.driver.nodes, d_po, TF.PRED, B.FALSE)
        self.update_hypothesis(d_ob, r_ob)

        # Update other
        r_other = r_p & ~(r_pc | r_pp | r_pn | r_pr | r_ob)
        d_other = d_p & ~(d_pc | d_pp | d_pn | d_pr | d_ob)
        self.update_hypothesis(d_other, r_other)

    def update_hypothesis(self, driver_mask, map_from_mask):
        """
        Update the hypothesis matrix, for nodes in given masks.
        """
        # Hypothesis = hypothesis + (driver_token.act * recipient_token.act)
        driver_acts = self.driver.nodes[:, TF.ACT]
        map_from_acts = self.map_from.nodes[:, TF.ACT]
        
        # Create outer product of activations for all combinations
        activation_product = torch.outer(map_from_acts, driver_acts)
        
        # Apply masks to only update the relevant node combinations
        mask_2d = torch.outer(map_from_mask, driver_mask)
        self[MappingFields.HYPOTHESIS] += activation_product * mask_2d

    def reset_hypotheses(self):
        """
        Reset the hypotheses/max hypotheses to 0.
        """
        self[MappingFields.HYPOTHESIS] = 0.0
        self[MappingFields.MAX_HYPOTHESIS] = 0.0

    def add_mappings(self,  mappings):
        """
        [Not implemented] Add mappings to the adjacency matrix.
        TODO: implement
        """
        pass

    # ----------------------------------------------------------------

    def print(self, f_types=None):                                  # Here for testing atm
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
                printer = nodePrinter(print_to_console=True)
                printer.print_con_tensor(self.adj_matrix)
            except Exception as e:
                print("Error: NodePrinter failed to print set.")
                print(e)