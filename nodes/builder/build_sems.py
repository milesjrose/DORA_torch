from .intermediate_types import *

class Build_sems(object):                               # Builds the semantic objects
    """
    A class for building the semantic objects.

    Attributes:
        sems (list): A list of semantic names.
        nodes (list): A list of semantic objects.
        name_dict (dict): A dictionary of semantic objects, mapping name to semantic object in nodes.
        id_dict (dict): A dictionary of semantic objects, mapping ID to semantic object in nodes.
        num_sems (int): The number of semantics, iterated from 0 when assigning IDs.
        symProps (list): A list of symProps relating to the set.
    """
    def __init__(self, symProps: list[dict]):
        """
        Initialise the build sems with symProps, and empty dictionaries for nodes, name_dict, and id_dict.

        Args:
            symProps (list): A list of symProps relating to the set.
        """
        self.sems = []
        self.nodes = []
        self.name_dict = {}
        self.id_dict = {}
        self.num_sems = 0
        self.symProps = symProps
    
    def build_sems(self):
        """
        Create the sem_set object.
        """
        self.get_sems(self.symProps)
        self.nodulate()
        self.sem_set = Sem_set(self.nodes, self.name_dict, self.id_dict)
        return self.sem_set

    def nodulate(self):                                 # Turn each unique semantic into a semantic object:
        """
        Turn each unique semantic into a semantic object (node) with a unique ID.
        """
        self.num_sems = 0
        for sem in self.sems:
            new_sem = Inter_Semantics(sem)
            new_sem.set_ID(self.num_sems)
            self.nodes.append(new_sem)
            self.id_dict[self.num_sems] = sem
            self.name_dict[sem] = new_sem
            self.num_sems += 1
    
    def get_sems(self, symProps: list[dict]):           # Depth first search to get list of all nodes in the symProps:
        """
        Get the list of all semantic names in the symProps.
        """
        for prop in symProps:
            for rb in prop['RBs']:
                if rb['pred_sem'] is not []:
                    for sem in rb['pred_sem']:
                        self.sems.append(sem)
                if rb['object_sem'] is not []:
                    for sem in rb['object_sem']:
                        self.sems.append(sem)

        self.sems = list(set(self.sems))
        self.num_sems = len(self.sems)
