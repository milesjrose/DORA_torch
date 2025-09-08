# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.
import logging
logger = logging.getLogger(__name__)

from ..enums import *

from .sets import Driver, Recipient, Memory, New_Set, Semantics, Base_Set
from .connections import Mappings, Links
from .network_params import Params
from .single_nodes import Token, Semantic
from .single_nodes import Ref_Token, Ref_Semantic
from .operations import TensorOperations, UpdateOperations, MappingOperations, FiringOperations, AnalogOperations, EntropyOperations, NodeOperations, InhibitorOperations
from .routines import Routines
import torch

class Network(object):
    """
    A class for holding set objects and operations.
    """
    def __init__(self, dict_sets: dict[Set, Base_Set], semantics: Semantics, mappings: dict[int, Mappings], links: Links, params: Params = None):
        """
        Initialize the Network object. Checks types; sets inter-set connections and params.

        Args:
            dict_sets (dict[Set, Base_Set]): The dictionary of set objects.
            semantics (Semantics): The semantics object.
            mappings (dict[int, Mappings]): The mappings objects.
            links (Links): The links object.
            params (Params): The parameters object.
        """
        # Check types
        if not isinstance(dict_sets, dict):
            raise ValueError("dict_sets must be a dictionary.")
        if not isinstance(semantics, Semantics):
            raise ValueError("semantics must be a Semantics object.")
        if not isinstance(mappings, dict):
            raise ValueError("mappings must be a dictionary.")
        if not isinstance(links, Links):
            raise ValueError("links must be a Links object.")
        if not isinstance(params, Params):
            raise ValueError("params must be a Params object.")

        # set objects
        self.semantics: Semantics = semantics
        """ Semantics object for the network. """
        self.sets: dict[Set, Base_Set] = dict_sets
        """ Dictionary of set objects for the network. """
        self.params: Params = params
        """ Parameters object for the network. """


        # add links, params, and mappings to each set
        for set in Set:
            try:
                self.sets[set].links = links
            except:
                raise ValueError(f"Error setting links for {set}")
            try:
                self.sets[set].params = params
            except:
                raise ValueError(f"Error setting params for {set}")
            try:
                self.sets[set].mappings = mappings[set]
            except:
                if set in MAPPING_SETS:
                    raise ValueError(f"Error setting mappings for {set}")

        self.semantics.params = params
        self.semantics.links = links
        # inter-set connections
        # TODO: Only need mapping tensor for recipient set -> remove others.
        self.mappings: dict[Set, Mappings] = mappings
            # Set the map_from attribute for each mapping
        self.mappings[Set.RECIPIENT].set_map_from(self.sets[Set.RECIPIENT]) 
        self.mappings[Set.MEMORY].set_map_from(self.sets[Set.MEMORY])
        # Set the mapping object for driver // TODO: Assign this in driver object init, need to update builder
        self.sets[Set.DRIVER].set_mappings(self.mappings)
        """ Dictionary of mappings for each set. """
        self.links: Links = links
        """ Links object for the network. """
        self.links.set_params(params) # Set params for links

        # inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0
        
        # routines
        self.routines: Routines = Routines(self)
        """ Routines object for the network. """

        

        self.tensor_ops = TensorOperations(self)
        self.update_ops = UpdateOperations(self)
        self.mapping_ops = MappingOperations(self)
        self.firing_ops = FiringOperations(self)
        self.analog_ops = AnalogOperations(self)
        self.entropy_ops = EntropyOperations(self)
        self.node_ops = NodeOperations(self)
        self.inhibitor_ops = InhibitorOperations(self)

        self._promoted_components = [
            self.tensor_ops, 
            self.update_ops, 
            self.mapping_ops, 
            self.firing_ops, 
            self.analog_ops, 
            self.entropy_ops, 
            self.node_ops, 
            self.inhibitor_ops
            ]
    
    def __getattr__(self, name):
        # Only search through the designated "promoted" components
        for component in self._promoted_components:
            if hasattr(component, name):
                return getattr(component, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


    def set_params(self, params: Params):                                   # Set the params for sets
        """
        Set the parameters for the network.
        """
        self.params = params
        for set in Set:
            self.sets[set].params = params
        self.semantics.params = params
    
    def __getitem__(self, key: Set):
        """
        Get the set object for the given set.
        """
        return self.sets[key]
    
    def get_count(self, semantics = True):
        """Get the number of nodes in the network."""
        count = 0
        for set in Set:
            count += self.sets[set].get_count()
        if semantics:
            count += self.semantics.get_count()
        return count
    # ========================[ PROPERTIES ]==============================
    @property
    def tensor(self) -> 'TensorOperations':
        """
        Memory management operations for the Network class.
        Handles copying, clearing, and managing memory sets.
        """
        return self.tensor_ops
    
    @property
    def update(self) -> 'UpdateOperations':
        """
        Update operations for the Network class.
        Handles input and activation updates across sets.
        """
        return self.update_ops
    
    @property
    def mapping(self) -> 'MappingOperations':
        """
        Mapping operations for the Network class.
        Handles mapping hypotheses, connections, and related functionality.
        """
        return self.mapping_ops
    
    @property
    def firing(self) -> 'FiringOperations':
        """
        Firing operations for the Network class.
        Handles firing order management.
        """
        return self.firing_ops
    
    @property
    def analog(self) -> 'AnalogOperations':
        """
        Analog operations for the Network class.
        Handles analog management and related functionality.
        """
        return self.analog_ops
    
    @property
    def entropy(self) -> 'EntropyOperations':
        """
        entropy operations object. Functions:
        - NOT IMPLEMENTED
        """
        return self.entropy_ops
    
    @property
    def node(self) -> 'NodeOperations':
        """
        Node operations for the Network class.
        Handles node management.
        """
        return self.node_ops
    
    @property
    def inhibitor(self) -> 'InhibitorOperations':
        """
        Inhibitor operations for the Network class.
        Handles inhibitor management.
        """
        return self.inhibitor_ops
    
    
    # ======================[ SET ACCESS FUNCTIONS ]======================
    def driver(self) -> 'Driver':
        """
        Get the driver set object.
        """
        return self.sets[Set.DRIVER]
    
    def recipient(self) -> 'Recipient':
        """
        Get the recipient set object.
        """
        return self.sets[Set.RECIPIENT]
    
    def memory(self) -> 'Memory':
        """
        Get the memory set object.
        """
        return self.sets[Set.MEMORY]
    
    def new_set(self) -> 'New_Set':
        """
        Get the new_set set object.
        """
        return self.sets[Set.NEW_SET]
    
    def semantics(self) -> 'Semantics':
        """
        Get the semantics set object.
        """
        return self.semantics
    
# ======================[ OTHER FUNCTIONS / TODO: Move to operations] ======================

    def set_name(self, reference: Ref_Token, name: str):
        """
        Set the name for a referenced token.

        Args:
            reference (Ref_Token): A reference to the token to set the name for.
            name (str): The name to set the token to.
        """
        self.sets[reference.set].token_op.set_name(reference, name)
    
    def get_index(self, reference: Ref_Token) -> int:
        """
        Get the index for a referenced token.
        """
        return self.sets[reference.set].token_op.get_index(reference)
    
    def initialise_made_unit(self):
        """
        Initialise the made unit for all tokens.
        TODO: Update tensors to be null by default for these values.
        currently some routines will not work unless this is run.
        """
        for set in Set:
            self.sets[set].nodes[:, TF.MADE_UNIT] = null
    
    
    def get_max_map_value(self, reference: Ref_Token, map_set: Set = None) -> float:
        """
        Get the maximum mapping weight for a referenced token.

        Args:
            reference (Ref_Token): The reference to the token to get the maximum map for.
            map_set (Set, optional): The set to get max_map to. Only required if reference is from driver.

        Returns:
            float: The maximum mapping weight for the referenced token.
        """
        logger.debug(f"Get max map value for {reference.set.name}[{reference.ID}]")
        if reference.set == Set.DRIVER: 
            if map_set == None:
                raise ValueError("Map set must be provided if reference is from driver.")
            else:
                index = self.get_index(reference)
                try:
                    max_val, max_index = torch.max(self.mappings[map_set][MappingFields.WEIGHT][:, index], dim=0)
                except Exception as e:
                    logger.error(f"Can't get max map value for {reference.set.name}[{index}] (ID={reference.ID})  in {map_set.name} map tens: {self.mappings[map_set][MappingFields.WEIGHT].shape} // slice: {self.mappings[map_set][MappingFields.WEIGHT][:, index].shape}")
                    raise(e)
        elif reference.set in MAPPING_SETS:
            map_set = reference.set
            index = self.get_index(reference)
            try:        
                max_val, max_index = torch.max(self.mappings[map_set][MappingFields.WEIGHT][index, :], dim=0)
            except:
                logger.error(f"Can't get max map value for {reference.set.name}[{index}] (ID={reference.ID}) weight tensor shape: {self.mappings[map_set][MappingFields.WEIGHT].shape}")
                raise ValueError(f"Invalid mapping set: {map_set.name}.{map_set.ID}")
        else:
            raise ValueError(f"Invalid reference: {reference.set.name}.{reference.ID}")
        
        logger.info(f"Max map val({reference.set.name}[{reference.ID}]:mapping.{map_set.name}) = {max_val.item()}")
        
        return max_val.item()
    
    def get_ref_string(self, reference: Ref_Token):
        """
        Get a string representation of a reference token.
        """
        if reference is None:
            return "(None)"
        try:
            index = self.get_index(reference)
        except Exception as e:
            logger.critical(f"Cant find index for {reference.set.name}[{reference.ID}]")
            return f"{reference.set.name}[N/A]({reference.ID})"
        return f"{reference.set.name}[{index}]({reference.ID})"