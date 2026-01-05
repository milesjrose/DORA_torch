# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.
import logging
logger = logging.getLogger(__name__)

from ..enums import *

from .network_params import Params, load_from_json
from .operations import TensorOperations, UpdateOperations, MappingOperations, FiringOperations, AnalogOperations, EntropyOperations, NodeOperations, InhibitorOperations
from .routines import Routines
import torch

# new imports
from .tokens import Tokens, Mapping, Links, Token_Tensor
from .sets_new import Driver, Recipient, Memory, New_Set, Semantics, Base_Set

class Network(object):
    """
    A class for holding set objects and operations.
    """
    def __init__(self, tokens: Tokens, semantics: Semantics, mappings: Mapping, links: Links, params: Params = None):
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
        if not isinstance(semantics, Semantics):
            raise ValueError("semantics must be a Semantics object.")
        if not isinstance(mappings, Mapping):
            raise ValueError("mappings must be a Mapping object.")
        if not isinstance(links, Links):
            raise ValueError("links must be a Links object.")
        if not isinstance(params, Params):
            raise ValueError("params must be a Params object.")
        # set objects
        self.tokens: Tokens = tokens
        """ Tokens object for the network. """
        self.token_tensor: Token_Tensor = tokens.token_tensor
        """ Token tensor object for the network. """
        self.semantics: Semantics = semantics
        """ Semantics object for the network. """
        self.params: Params = params
        """ Parameters object for the network. """
        self.mappings: Mapping = mappings
        """ Mappings object for the network. """
        self.links: Links = links
        """ Links object for the network. 
            - Links[set] gives set's links to semantics
            - Link tensor shape: [nodes, semantics]
        """
        self.sets: dict[Set, Base_Set] = {
            Set.DRIVER: Driver(self.token_tensor, self.params, self.mappings),
            Set.RECIPIENT: Recipient(self.token_tensor, self.params, self.mappings),
            Set.MEMORY: Memory(self.token_tensor, self.params),
            Set.NEW_SET: New_Set(self.token_tensor, self.params)
        }
        """ Dictionary of set objects for the network. """

        # Setup sets and semantics
        self.setup_sets_and_semantics()

        # Initialise inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0
        
        # Operations and routines
        self.routines: Routines = Routines(self)
        self.tensor_ops: TensorOperations = TensorOperations(self)
        self.update_ops: UpdateOperations = UpdateOperations(self)
        self.mapping_ops: MappingOperations = MappingOperations(self)
        self.firing_ops: FiringOperations = FiringOperations(self)
        self.analog_ops: AnalogOperations = AnalogOperations(self)
        self.entropy_ops: EntropyOperations = EntropyOperations(self)
        self.node_ops: NodeOperations = NodeOperations(self)
        self.inhibitor_ops: InhibitorOperations = InhibitorOperations(self)
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
        # Check if _promoted_components exists to avoid recursion during initialization
        if not hasattr(self, '_promoted_components'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        for component in self._promoted_components:
            if hasattr(component, name):
                return getattr(component, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def cache_analogs(self):
        """
        Recache the analogs in the network.
        """
        self.tokens.token_tensor.cache.cache_analogs()
    
    def cache_sets(self):
        """
        Recache the tokens in the network.
        """
        self.tokens.token_tensor.cache.cache_sets()
    
    def recache(self):
        """
        Recache the tokens and analogs in the network.
        """
        self.cache_sets()
        self.cache_analogs()

    def set_params(self, params: Params):                                   # Set the params for sets
        """
        Set the parameters for the network.
        """
        self.params = params
        for set in Set:
            self.sets[set].params = params
        self.semantics.params = params
        self.links.set_params(params) # Set params for links
    
    def setup_sets_and_semantics(self):
        """Setup the sets and semantics for the network."""
        # Setup mapping 
        self.mappings.set_driver(self.sets[Set.DRIVER])
        self.mappings.set_recipient(self.sets[Set.RECIPIENT])
        # Add links and params to each set
        for set in Set:
            try:
                self.sets[set].links = self.links
            except:
                raise ValueError(f"Error setting links for {set}")
        # Add links to semantics
        self.semantics.links = self.links
        self.set_params(self.params)
        # Set network for links
        # TODO move the get_index call out of the links object
        self.links.set_network(self)
        # Initialise SDMs
        self.semantics.init_sdm()

    def load_json_params(self, file_path: str):
        """
        Load parameters from a JSON file.
        """
        self.params = load_from_json(file_path)
        self.set_params(self.params)

    def __getitem__(self, key: Set):
        """
        Get the set object for the given set.
        """
        return self.sets[key]
    
    def get_count(self, semantics = True):
        """Get the number of nodes in the network."""
        self.token_tensor.get_count()

    def clear(self, limited=False):
        """
        Clear the network:

        limited:
        - made_units
        - inferences
        - new_set

        full: limited+
        - mappings
        - driver
        - recipient
        """
        # clear made_units, inferences, new_set
        self.tensor_ops.reset_maker_made_units()
        self.tensor_ops.reset_inferences()
        self.tensor_ops.clear_set(Set.NEW_SET)
        if not limited:
            # mappings, driver, recipient
            self.tensor_ops.clear_all_sets()

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
    
# ======================[ OTHER FUNCTIONS / TODO: Move to operations] ======================

    def set_name(self, idx: int, name: str):
        """
        Set the name for a token at the given index.

        Args:
            idx (int): The index of the token to set the name for.
            name (str): The name to set the token to.
        """
        self.tokens.set_name(idx, name)
    
    def get_max_map_value(self, idx: int) -> float:
        """
        Get the maximum mapping weight for a token at the given index.

        Args:
            idx (int): The index of the token to get the maximum map for.
        Returns:
            float: The maximum mapping weight for the token at the given index.
        """
        logger.debug(f"Get max map value for {self.get_ref_string(idx)}")
        tk_set = self.to_type(self.token_tensor.get_feature(idx, TF.SET), TF.SET)
        # Mapping tensor is local to driver/recipient
        local_idx_tensor = self.sets[tk_set].lcl.to_local(torch.tensor([idx]))
        local_idx = local_idx_tensor[0].item()
        return self.mappings.get_single_max_map(local_idx, tk_set)
    
    def get_ref_string(self, idx: int):
        """
        Get a string representation of a reference token.
        """
        return self.token_tensor.get_ref_string(idx)
    
    def to_local(self, idxs):
        """
        Convert global idx(s) to local idx(s)
        """
        if isinstance(idxs, int):
            tk_set = self.token_tensor.get_feature(idxs, TF.SET)
        else:
            tk_sets = (self.token_tensor.get_features(idxs, TF.SET)).unique()
            if tk_sets.size(0) == 1:
                tk_set = tk_sets[0]
            else:
                raise ValueError(f"Multiple sets found for indices: {idxs}")
        return self.sets[tk_set].lcl.to_global(idxs)
    
    def to_global(self, idxs):
        """
        Convert local idx(s) to global idx(s)
        """
        if isinstance(idxs, int):
            tk_set = self.token_tensor.get_feature(idxs, TF.SET)
        else:
            tk_sets: torch.Tensor = (self.token_tensor.get_features(idxs, TF.SET)).unique()
            if tk_sets.size(0) == 1:
                tk_set = tk_sets[0]
            else:
                raise ValueError(f"Multiple sets found for indices: {idxs}")
        return self.sets[tk_set].lcl.to_global(idxs)
    
    def to_type(self, value, feature: TF):
        """
        Convert a value to the type of the feature.
        Args:
            value: torch.Tensor, float, int - The value to convert.
            feature: TF - The feature to convert to.
        Returns:
            TF_type(feature): The converted value.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        else:
            return TF_type(feature)(value)
        return TF_type(feature)(value)
