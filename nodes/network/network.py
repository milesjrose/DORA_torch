# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.
import logging
logger = logging.getLogger("net")

from ..enums import *

from .network_params import Params, load_from_json
from .operations import TensorOperations, UpdateOperations, MappingOperations, FiringOperations, AnalogOperations, EntropyOperations, NodeOperations, InhibitorOperations
from .routines import Routines
import torch

# new imports
from .tokens import Tokens, Mapping, Links, Token_Tensor
from .sets import Driver, Recipient, Memory, New_Set, Semantics, Base_Set

class Network(object):
    """
    A class for holding set objects and operations.
    """
    def __init__(self, tokens: Tokens, semantics: Semantics, params: Params = None):
        """
        Initialize the Network object. Checks types; sets inter-set connections and params.

        Args:
            dict_sets (dict[Set, Base_Set]): The dictionary of set objects.
            semantics (Semantics): The semantics object.
            mappings (dict[int, Mappings]): The mappings objects.
            links (Links): The links object.
            params (Params): The parameters object.
        """
        logger.info(f"> Initialising Network object")
        # Check types
        if not isinstance(semantics, Semantics):
            raise ValueError("semantics must be a Semantics object.")
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
        self.mappings: Mapping = tokens.mapping
        """ Mappings object for the network. """
        self.links: Links = tokens.links
        """ Links object for the network. 
            - Links[set] gives set's links to semantics
            - Link tensor shape: [nodes, semantics]
        """
        self.sets: dict[Set, Base_Set] = {
            Set.DRIVER: Driver(self.tokens, self.params),
            Set.RECIPIENT: Recipient(self.tokens, self.params),
            Set.MEMORY: Memory(self.tokens, self.params),
            Set.NEW_SET: New_Set(self.tokens, self.params)
        }
        """ Dictionary of set objects for the network. """

        # Setup sets and semantics
        self.setup_sets_and_semantics()

        # Cache sets and analogs
        self.recache()

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
        
        logger.info(f"> Network initialised:\n Tensor shapes:\n    Tokens: {self.token_tensor.tensor.shape[0]}x{self.token_tensor.tensor.shape[1]}\n    Semantics: {self.semantics.nodes.shape[0]}x{self.semantics.nodes.shape[1]}\n    Connections: {self.tokens.connections.tensor.shape[0]}x{self.tokens.connections.tensor.shape[1]}\n    Links: {self.links.adj_matrix.shape[0]}x{self.links.adj_matrix.shape[1]}\n    Mapping: {self.mappings.adj_matrix.shape[0]}x{self.mappings.adj_matrix.shape[1]}x{self.mappings.adj_matrix.shape[2]}\n Set counts:\n    Driver: {self.driver().get_count()}\n    Recipient: {self.recipient().get_count()}\n    Memory: {self.memory().get_count()}\n    New Set: {self.new_set().get_count()}\n    Tokens: {self.token_tensor.get_count()}\n    Semantics: {self.semantics.get_count()}")
        #self.print_token_tensor()

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
        self.recache()
        # NOTE: Maybe wasteful, but figure it out later. Just want it to work for now.
        #self.tokens.token_tensor.cache.cache_analogs()
    
    def cache_sets(self):
        """
        Recache the tokens in the network.
        """
        self.recache()
        # NOTE: Maybe wasteful, but figure it out later. Just want it to work for now.
        #self.tokens.token_tensor.cache.cache_sets()
    
    def update_views(self):
        """
        Update the views for the network.
        """
        for set in Set:
            self.sets[set].update_view()
        logger.debug(f"Updated views: driver:{self.driver().lcl.shape[0]}, recipient:{self.recipient().lcl.shape[0]}, memory:{self.memory().lcl.shape[0]}, new_set:{self.new_set().lcl.shape[0]}")

    def recache(self):
        """
        Recache the tokens, analogs, and set views for the network.
        """
        self.tokens.recache()
        self.semantics.check_links_size()
        self.update_views()

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
    
    def get_name(self, idx: int) -> str:
        """
        Get the name for a token at the given index.
        Args:
            idx: int - The index of the token to get the name of.
        Returns:
            str - The name of the token.
        """
        return self.token_tensor.get_name(idx)
    
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
    
    def to_local(self, idxs) -> torch.Tensor:
        """
        Convert global idx(s) to local idx(s)

        Args:
            idxs: int, list[int], torch.Tensor - The global indices to convert to local indices.
        Returns:
            torch.Tensor - The local indices.
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
    
    def to_global(self, idxs, tk_set: Set = None) -> torch.Tensor:
        """
        Convert local idx(s) to global idx(s)
        
        Args:
            idxs: int, list[int], torch.Tensor - The local indices to convert to global indices.
            tk_set: Set - The local set to convert from.
        Returns:
            torch.Tensor - The global indices.
        """
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

# ======================[ INFO FUNCTIONS ]======================

    def print_summary(self):
        """
        Print a summary of the network.
        """
        print(f"Network summary:")
        print(f"  Tokens: {self.token_tensor.get_count()}")
        print(f"  Semantics: {self.semantics.get_count()}")
        print(f"  Driver: {self.driver().get_count()}")
        print(f"  Recipient: {self.recipient().get_count()}")
        print(f"  Memory: {self.memory().get_count()}")
        print(f"  New Set: {self.new_set().get_count()}")

    def print_token_tensor(self, cols_per_table: int = 8, show_deleted: bool = False, indices: torch.Tensor = None, use_names: bool = False, features: list[TF] = None):
        """ 
        Print token tensor to console.
        
        Args:
            cols_per_table (int): Number of feature columns per table. Default 8.
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print. If None, prints all (non-deleted) tokens.
            use_names (bool): Whether to use token names instead of indices. Default False.
            features (list[TF]): Optional specific features to print. If None, prints all features.
        """
        from nodes.utils import Printer
        p = Printer()
        p.print_token_tensor(token_tensor=self.token_tensor, cols_per_table=cols_per_table, show_deleted=show_deleted, indices=indices, use_names=use_names, features=features)
    
    def print_connections(self, show_deleted: bool = False, indices: torch.Tensor = None, use_names: bool = False, connected_char: str = "●", empty_char: str = "·"):
        """ 
        Print connections to console as a matrix table.
        
        Args:
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print. If None, prints all (non-deleted) tokens.
            use_names (bool): Whether to use token names instead of indices. Default False.
            connected_char (str): Character to show for connections. Default "●".
            empty_char (str): Character to show for no connection. Default "·".
        """
        from nodes.utils import Printer
        p = Printer()
        p.print_connections(self.tokens, show_deleted=show_deleted, indices=indices, use_names=use_names, connected_char=connected_char, empty_char=empty_char)
    
    def print_connections_list(self, show_deleted: bool = False, indices: torch.Tensor = None, use_names: bool = False, list_only_connected: bool = True):
        """ 
        Print connections as a list of parent -> child relationships.
        More readable for sparse connection matrices.
        
        Args:
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print. If None, prints all (non-deleted) tokens.
            use_names (bool): Whether to use token names instead of indices. Default False.
            list_only_connected (bool): Whether to only list connected tokens. Default True.
        """
        from nodes.utils import Printer
        p = Printer()
        p.print_connections_list(self.tokens, show_deleted=show_deleted, indices=indices, use_names=use_names, list_only_connected=list_only_connected)
    
    def print_links(self, token_names: dict[int, str] = None, semantic_names: dict[int, str] = None, token_indices: torch.Tensor = None, semantic_indices: torch.Tensor = None, min_weight: float = 0.0, show_weights: bool = True):
        """ 
        Print links to console as a matrix showing token-to-semantic connections.
        
        Args:
            token_names (dict[int, str]): Optional dict mapping token index to name. If None, uses network token names.
            semantic_names (dict[int, str]): Optional dict mapping semantic index to name. If None, uses network semantic names.
            token_indices (torch.Tensor): Optional specific token indices to show. If None, shows all tokens.
            semantic_indices (torch.Tensor): Optional specific semantic indices to show. If None, shows all semantics with at least one link.
            min_weight (float): Minimum weight to display. Links below this are shown as empty. Default 0.0.
            show_weights (bool): If True, show weight values. If False, show "●" for linked. Default True.
        """
        from nodes.utils import Printer
        p = Printer()
        # Use network names if not provided
        if token_names is None:
            token_names = self.token_tensor.names
        if semantic_names is None:
            semantic_names = self.semantics.names if hasattr(self.semantics, 'names') else None
        p.print_links(self.links, token_names=token_names, semantic_names=semantic_names, token_indices=token_indices, semantic_indices=semantic_indices, min_weight=min_weight, show_weights=show_weights)
    
    def print_links_list(self, token_names: dict[int, str] = None, semantic_names: dict[int, str] = None, token_indices: torch.Tensor = None, min_weight: float = 0.0, show_weights: bool = True):
        """ 
        Print links as a list showing each token's linked semantics.
        More readable for sparse matrices.
        
        Args:
            token_names (dict[int, str]): Optional dict mapping token index to name. If None, uses network token names.
            semantic_names (dict[int, str]): Optional dict mapping semantic index to name. If None, uses network semantic names.
            token_indices (torch.Tensor): Optional specific token indices to show. If None, shows all tokens with at least one link.
            min_weight (float): Minimum weight to display. Default 0.0.
            show_weights (bool): If True, show weight values with semantics. Default True.
        """
        from nodes.utils import Printer
        p = Printer()
        # Use network names if not provided
        if token_names is None:
            token_names = self.token_tensor.names
        if semantic_names is None:
            semantic_names = self.semantics.names if hasattr(self.semantics, 'names') else None
        p.print_links_list(self.links, token_names=token_names, semantic_names=semantic_names, token_indices=token_indices, min_weight=min_weight, show_weights=show_weights)
    
    def print_mappings(self, driver_names: dict[int, str] = None, recipient_names: dict[int, str] = None, driver_indices: torch.Tensor = None, recipient_indices: torch.Tensor = None, field: MappingFields = MappingFields.WEIGHT, min_value: float = 0.0, show_values: bool = True):
        """ 
        Print mappings to console as a matrix showing recipient-to-driver mappings.
        
        Args:
            driver_names (dict[int, str]): Optional dict mapping driver index to name. If None, uses network token names for driver set.
            recipient_names (dict[int, str]): Optional dict mapping recipient index to name. If None, uses network token names for recipient set.
            driver_indices (torch.Tensor): Optional specific driver indices to show. If None, shows all drivers with at least one mapping.
            recipient_indices (torch.Tensor): Optional specific recipient indices to show. If None, shows all recipients with at least one mapping.
            field (MappingFields): Which field to display. Default WEIGHT.
            min_value (float): Minimum value to display. Values below this shown as empty. Default 0.0.
            show_values (bool): If True, show values. If False, show "●" for non-zero. Default True.
        """
        from nodes.utils import Printer
        p = Printer()
        # Use network names if not provided
        if driver_names is None:
            driver_names = self.token_tensor.names
        if recipient_names is None:
            recipient_names = self.token_tensor.names
        p.print_mappings(self.mappings, driver_names=driver_names, recipient_names=recipient_names, driver_indices=driver_indices, recipient_indices=recipient_indices, field=field, min_value=min_value, show_values=show_values)
    
    def print_mappings_list(self, driver_names: dict[int, str] = None, recipient_names: dict[int, str] = None, recipient_indices: torch.Tensor = None, field: MappingFields = MappingFields.WEIGHT, min_value: float = 0.0, show_values: bool = True):
        """ 
        Print mappings as a list showing each recipient's mapped drivers.
        More readable for sparse matrices.
        
        Args:
            driver_names (dict[int, str]): Optional dict mapping driver index to name. If None, uses network token names for driver set.
            recipient_names (dict[int, str]): Optional dict mapping recipient index to name. If None, uses network token names for recipient set.
            recipient_indices (torch.Tensor): Optional specific recipient indices to show. If None, shows all recipients with at least one mapping.
            field (MappingFields): Which field to display. Default WEIGHT.
            min_value (float): Minimum value to display. Default 0.0.
            show_values (bool): If True, show values with drivers. Default True.
        """
        from nodes.utils import Printer
        p = Printer()
        # Use network names if not provided
        if driver_names is None:
            driver_names = self.token_tensor.names
        if recipient_names is None:
            recipient_names = self.token_tensor.names
        p.print_mappings_list(self.mappings, driver_names=driver_names, recipient_names=recipient_names, recipient_indices=recipient_indices, field=field, min_value=min_value, show_values=show_values)