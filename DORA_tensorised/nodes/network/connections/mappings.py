# nodes/sets/connections/mappings.py
# Mappings between nodes and semantics.
# TODO: Implement add_mappings, updateHypotheses

import torch

from ..sets import Recipient

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
    
    # =====================[ Update functions ]======================
    def updateHypotheses(self, recipient: Recipient):
        """
        Update the hypotheses matrix.
        NOTE: Seems very inefficient
        TODO: Implement a more efficient method, and add tests.
        """
        Recipient: recipient = recipient
        # Need to check that the type of p/po nodes match.
        # Can do this by refining masks to type, then updating the these masks first. So only matching node types will be included
        r_p = recipient.get_mask(Type.P)
        d_p = self.driver.get_mask(Type.P)

        r_po = recipient.get_mask(Type.PO)
        d_po = self.driver.get_mask(Type.PO)

        # Update child p
        r_pc = tOps.refine_mask(recipient.nodes, r_p, TF.MODE, Mode.CHILD)
        d_pc = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.CHILD)
        self.updateHypothesis(r_pc, d_pc)

        # Update parent p
        r_pp = tOps.refine_mask(recipient.nodes, r_p, TF.MODE, Mode.PARENT)
        d_pp = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.PARENT)
        self.updateHypothesis(r_pp, d_pp)

        # Update neutral p
        r_pn = tOps.refine_mask(recipient.nodes, r_p, TF.MODE, Mode.NEUTRAL)
        d_pn = tOps.refine_mask(self.driver.nodes, d_p, TF.MODE, Mode.NEUTRAL)
        self.updateHypothesis(r_pn, d_pn)

        # Update Pred
        r_pr = tOps.refine_mask(recipient.nodes, r_po, TF.MODE, Mode.PRED)
        d_pr = tOps.refine_mask(self.driver.nodes, d_po, TF.MODE, Mode.PRED)
        self.updateHypothesis(r_pr, d_pr)

        # Update Obj
        r_ob = tOps.refine_mask(recipient.nodes, r_po, TF.MODE, Mode.OBJ)
        d_ob = tOps.refine_mask(self.driver.nodes, d_po, TF.MODE, Mode.OBJ)
        self.updateHypothesis(r_ob, d_ob)

        # Update other
        r_other = r_p - (r_pc + r_pp + r_pn + r_pr + r_ob)
        d_other = d_p - (d_pc + d_pp + d_pn + d_pr + d_ob)
        self.updateHypothesis(r_other, d_other)

    def updateHypothesis(self, driver_mask, recipient_mask):
        """
        Update the hypothesis matrix, for nodes in given masks.
        NOTE: Also infefficient as only one mapping connection per node, but uses matrix multiplication on NxM matrix.
        """
        # Hypothesis = hypothesis + (driver_token.act * recipient_token.act)
        self[MappingFields.HYPOTHESIS] += torch.matmul(
            self.driver.nodes[driver_mask],
            self.recipient.nodes[recipient_mask]
        )

    def add_mappings(self,  mappings):
        """
        [Not implemented] Add mappings to the adjacency matrix.
        TODO: implement
        """
        pass

    # ----------------------------------------------------------------