# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.

from ..enums import *

from .sets import Driver, Recipient, Memory, New_Set, Semantics, Base_Set
from .connections import Mappings, Links
from .network_params import Params
from .single_nodes import Token, Semantic
from .single_nodes import Ref_Token, Ref_Semantic
from .operations import MemoryOperations, UpdateOperations, MappingOperations, RetrievalOperations, FiringOperations, AnalogOperations, EntropyOperations, NodeOperations, InhibitorOperations

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
        self.mappings: dict[Set, Mappings] = mappings
            # Set the map_from attribute for each mapping
        self.mappings[Set.RECIPIENT].set_map_from(self.sets[Set.RECIPIENT]) 
        self.mappings[Set.MEMORY].set_map_from(self.sets[Set.MEMORY])
        """ Dictionary of mappings for each set. """
        self.links: Links = links
        """ Links object for the network. """

        # inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0

        # operations
        self.memory = MemoryOperations(self)
        """ Memory operations object for the network. """
        self.update = UpdateOperations(self)
        """ Update operations object for the network. """
        self.mapping = MappingOperations(self)
        """ Mapping operations object for the network. """
        self.retrieval = RetrievalOperations(self)
        """ Retrieval operations object for the network. """
        self.firing = FiringOperations(self)
        """ Firing operations object for the network. """
        self.analog = AnalogOperations(self)
        """ Analog operations object for the network. """
        self.entropy = EntropyOperations(self)
        """ Entropy operations object for the network. """
        self.node = NodeOperations(self)
        """ Node operations object for the network. """
        self.inhibitor = InhibitorOperations(self)
        """ Inhibitor operations object for the network. """

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
    
    # ======================[ ACT FUNCTIONS ]============================
    def initialise_act(self):                                               # Initialise acts in active memory/semantics
        """
        Initialise the acts in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        self.update.initialise_act()
    
    def update_acts(self, set: Set):                                        # Update acts in given token set    
        """
        Update the acts in the given set.

        Args:
            set (Set): The set to update acts in.
        """
        self.update.acts(set)
    
    def update_acts_sem(self):                                              # Update acts in semantics
        """
        Update the acts in the semantics.
        """
        self.update.acts_sem()

    def update_acts_am(self):                                               # Update acts in active memory/semantics
        """
        Update the acts in the active memory.
        (driver, recipient, new_set, semantics)
        """
        self.update.acts_am()

    # =======================[ INPUT FUNCTIONS ]=========================
    def initialise_input(self):                                             # Initialise inputs in active memory/semantics
        """
        Initialise the inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        self.update.initialise_input()
    
    def update_inputs(self, set: Set):                                      # Update inputs in given token set
        """
        Update the inputs in the given token set.

        Args:
            set (Set): The set to update inputs in.
        """
        self.update.inputs(set)
    
    def update_inputs_sem(self):                                            # Update inputs in semantics               
        """
        Update the inputs in the semantics.
        """
        self.update.inputs_sem()

    def update_inputs_am(self):                                             # Update inputs in active memory
        """
        Update the inputs in the active memory.
        (driver, recipient, new_set, semantics)
        """
        self.update.inputs_am()

    # =====================[ INHIBITOR FUNCTIONS ]=======================
    def update_inhibitors(self):                                            # Update inputs and acts of inhibitors
        """
        Update the inputs and acts of the driver and recipient inhibitors.
        (driver, recipient).
        - Only updates act of PO if in DORA mode.
        - Only updates act of RB in driver.
        """
        self.inhibitor.update()

    def reset_inhibitors(self):                                             # Reset inhibitors (for RB and PO units) NOTE: Check if required to set for memory and new_set
        """
        Reset the inhibitors (for RB and PO units).
        (driver, recipient, new_set, memory)
        """
        self.inhibitor.reset()

    def check_local_inhibitor(self):                                        # Check local inhibition
        """Check local inhibitor activation."""
        self.inhibitor.check_local()
    
    def fire_local_inhibitor(self):                                         # Fire local inhibitor
        """Fire the local inhibitor."""
        self.inhibitor.fire_local()
    
    def check_global_inhibitor(self):                                       # Check global inhibition
        """Check global inhibitor activation."""
        self.inhibitor.check_global()
        
    def fire_global_inhibitor(self):                                        # Fire global inhibitor
        """Fire the global inhibitor."""
        self.inhibitor.fire_global()

    # ========================[ NODE FUNCTIONS ]==========================
    def get_pmode(self):                                                    # Get p_mode in driver and recipient
        """
        Get parent mode of P units in driver and recipient. Used in time steps activations.
        (driver, recipient)
        """
        self.node.get_pmode()
    
    def initialise_p_mode(self, set: Set = Set.RECIPIENT):                  # Initialise p_mode in the given set
        """
        Initialise p_mode in the given set.
        (default: recipient)

        Args:
            set (Set, optional): The set to initialise p_mode in. (Defaults to recipient)
        """
        self.node.initialise_p_mode(set)
    
    def get_weight_lengths(self):                                           # get weight lenghts in active memory
        """
        Get weight lengths of PO units. Used in run initialisation.
        (driver, recipient, memory, new_set)
        """
        self.node.get_weight_lengths()
    
    def add_token(self, token: Token):                                      # Add a token to the given set
        """
        Add a token to the network.
        - Added to the set specified in the token.

        Args:
            token (network.Token): The token to add.
        
        Returns:
            network.Ref_Token: A reference to the token.
        
        Raises:
            ValueError: If the token set feature is not a valid set.
        """
        self.node.add_token(token)

    def del_token(self, ref_token: Ref_Token):                              # Delete a token
        """
        Delete a referenced token from the network.

        Args:
            ref_token (network.Ref_Token): A reference to the token to delete.
        """
        self.node.del_token(ref_token)

    def add_semantic(self, semantic: Semantic):                             # Add a semantic
        """
        Add a semantic to the given set.

        Args:
            semantic (network.Semantic): The semantic to add.
        
        Raises:
            ValueError: If provided semantic is not semantic type.
        """
        self.node.add_semantic(semantic)

    def del_semantic(self, ref_semantic: Ref_Semantic):                     # Delete a semantic
        """
        Delete a semantic from the semantics.
        
        Args:
            ref_semantic (network.Ref_Semantic): A reference to the semantic to delete.

        Raises:
            ValueError: If ref_semantic is not provided.
        """
        self.node.del_semantic(ref_semantic)

    def set_sem_max_input(self):                                            # Set the maximum input for the semantics
        """
        Set the maximum input for the semantics.
        """
        self.node.set_sem_max_input()
    
    def get_value(self, reference, feature):                                # Get the value of a feature for a referenced token or semantic
        """
        Get the value of a feature for a referenced token or semantic.

        Args:
            reference (Ref_Token or Ref_Semantic): A reference to the token or semantic to get the value of.
            feature (TF or SF): The feature to get the value of.

        Returns:
            float: The value of the feature.

        Raises:
            ValueError: If the reference is not a token or semantic. Or feature type and reference type mismatch.
        """
        self.node.get_value(reference, feature)
    
    def set_value(self, reference, feature, value):                         # Set the value of a feature for a referenced token or semantic
        """
        Set the value of a feature for a referenced token or semantic.

        Args:
            reference (Ref_Token or Ref_Semantic): A reference to the token or semantic to set the value of.
            feature (TF or SF): The feature to set the value of.
            value (float or Enum): The value to set the feature to.

        Raises:
            ValueError: If the reference is not a token or semantic. Or feature type and reference type mismatch.
        """
        self.node.set_value(reference, feature, value)
    
    # ----------------------------------------------------------------------
    def print_set(self, set: Set, feature_types: list[TF] = None):
        """
        Print the given set.

        Args:
            set (Set): The set to print.
            feature_types (list[TF], optional): List features to print, otherwise default features are printed.
        """
        self.utility.print_set(set, feature_types)
