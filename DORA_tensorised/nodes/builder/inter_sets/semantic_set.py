# nodes/builder/sets/semantic_set.py
# Semantic set class for the builder.

import torch
import numpy as np

from nodes.enums import SF

from ..inter_nodes import Inter_Semantics


class Sem_set(object):
    """
    An intermediate class for representing a set of semantics.

    Attributes:
        sems (list): A list of semantics.
        name_dict (dict): A dictionary of semantics, mapping name to semantic in sems.
        id_dict (dict): A dictionary of semantics, mapping ID to semantic in sems.
        num_sems (int): The number of semantics in the set.
        connections (np.ndarray): A matrix of connections between semantics.
    """
    def __init__(self, sems: list[Inter_Semantics], name_dict: dict[str, Inter_Semantics], id_dict: dict[int, Inter_Semantics]):
        """
        Initialise the semantic set with sems, name_dict, and id_dict.

        Args:
            sems (list): A list of semantics.
            name_dict (dict): A dictionary of semantics, mapping name to semantic in sems.
            id_dict (dict): A dictionary of semantics, mapping ID to semantic in sems.
        """
        self.sems = sems
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_sems = len(sems)
        self.connections = np.zeros((self.num_sems, self.num_sems))
    
    def get_sem(self, name):
        """
        Get a semantic from the semantic set by name.

        Args:
            name (str): The name of the semantic.
        """
        return self.name_dict[name]
    
    def get_sem_by_id(self, ID):
        """
        Get a semantic from the semantic set by ID.

        Args:
            ID (int): The ID of the semantic.
        """
        return self.id_dict[ID]

    def tensorise(self):
        """
        Tensorise the semantic set, creating a tensor of semantics, and a tensor of connections between semantics.
        """
        self.node_tensor = torch.zeros((self.num_sems, len(SF)))
        self.connections_tensor = torch.tensor(self.connections)
        for sem in self.sems:
            sem.floatate_features()
            self.node_tensor[sem.ID] = torch.tensor(sem.features)
