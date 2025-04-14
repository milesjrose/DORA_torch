# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.

from nodes.enums import *

from .sets import Driver, Recipient, Memory, New_Set, Semantics, Base_Set
from .connections import Mappings
from .network_params import Params
from .single_nodes import Token, Semantic
from .single_nodes import Ref_Token, Ref_Semantic

class Network(object):
    """
    A class for holding set objects and operations.
    """
    def __init__(self, driver: Driver, recipient: Recipient, memory: Memory, new_set: New_Set, semantics: Semantics, set_mappings: dict[int, Mappings], params: Params = None):
        """
        Initialize the Network object.

        Args:
            driver (Driver): The driver object.
            recipient (Recipient): The recipient object.
            LTM (Tokens): The long-term memory object.
            new_set (Tokens): The new set object.
            semantics (Semantics): The semantics object.
            set_mappings (Mappings): The mappings object.
            params (Params): The parameters object.
        """
        # node tensors
        self.driver: Driver = driver
        self.recipient: Recipient = recipient
        self.semantics: Semantics = semantics
        self.memory: Memory = memory
        self.new_set: New_Set = new_set
        self.sets: dict[Set, Base_Set] = {
            Set.DRIVER: self.driver,
            Set.RECIPIENT: self.recipient,
            Set.MEMORY: self.memory,
            Set.NEW_SET: self.new_set
        }
        self.params = params
        
        # inter-set connections
        self.set_mappings: Mappings = set_mappings

        # inhibitors
        self.local_inhibitor = 0.0
        self.global_inhibitor = 0.0

        if self.params is not None:
            self.set_params(self.params)
            self.DORA_mode = self.params.DORA_mode
        else:
            self.DORA_mode = True

    def set_params(self, params: Params):                                   # Set the params for sets
        """
        Set the parameters for the network.
        """
        self.params = params
        self.driver.params = params
        self.recipient.params = params
        self.memory.params = params
        self.new_set.params = params
        self.semantics.params = params
    
    # ======================[ ACT FUNCTIONS ]============================
    def initialise_act(self):                                               # Initialise acts in active memory/semantics
        """
        Initialise the acts in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.sets[set].initialise_act()

        self.semantics.intitialise_sem()
    
    def update_acts(self, set: Set):                                        # Update acts in given token set    
        """
        Update the acts in the given set.

        Args:
            set (Set): The set to update acts in.
        """
        self.sets[set].update_act()
    
    def update_acts_sem(self):                                              # Update acts in semantics
        """
        Update the acts in the semantics.
        """
        self.semantics.update_act()

    def update_acts_am(self):                                               # Update acts in active memory/semantics
        """
        Update the acts in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.update_acts(set)
        
        self.update_acts_sem()

    # =======================[ INPUT FUNCTIONS ]=========================
    def initialise_input(self):                                             # Initialise inputs in active memory/semantics
        """
        Initialise the inputs in the active memory/semantics.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.sets[set].initialise_act()
        
        self.semantics.initialise_sem()
    
    def update_inputs(self, set: Set):                                      # Update inputs in given token set
        """
        Update the inputs in the given token set.

        Args:
            set (Set): The set to update inputs in.
        """
        self.sets[set].update_input()
    
    def update_inputs_sem(self):                                            # Update inputs in semantics               
        """
        Update the inputs in the semantics.
        """
        self.semantics.update_input()

    def update_inputs_am(self):                                             # Update inputs in active memory
        """
        Update the inputs in the active memory.
        (driver, recipient, new_set, semantics)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.NEW_SET]
        for set in sets:
            self.update_inputs(set)
        
        self.update_inputs_sem()

    # =====================[ INHIBITOR FUNCTIONS ]=======================
    def update_inhibitors(self):                                            # Update inputs and acts of inhibitors
        """
        Update the inputs and acts of the driver and recipient inhibitors.
        (driver, recipient).
        - Only updates act of PO if in DORA mode.
        - Only updates act of RB in driver.
        """
        #update input driver/recipient RBs, POs
        self.driver.update_inhibitor_input([Type.RB, Type.PO])
        self.recipient.update_inhibitor_input([Type.RB, Type.PO])
        #update driver rb, PO inhibitor act only if in DORA mode
        if self.DORA_mode:
            self.driver.update_inhibitor_act([Type.RB, Type.PO])
            self.recipient.update_inhibitor_act([Type.PO])
        else:
            self.driver.update_inhibitor_act([Type.RB])

    def reset_inhibitors(self):                                             # Reset inhibitors (for RB and PO units) NOTE: Check if required to set for memory and new_set
        """
        Reset the inhibitors (for RB and PO units).
        (driver, recipient, new_set, memory)
        """
        self.driver.reset_inhibitor([Type.RB, Type.PO])
        self.recipient.reset_inhibitor([Type.RB, Type.PO])
        self.memory.reset_inhibitor([Type.RB, Type.PO])
        self.new_set.reset_inhibitor([Type.RB, Type.PO])

    def check_local_inhibitor(self):                                        # Check local inhibition
        """Check local inhibitor activation."""
        if self.driver.check_local_inhibitor():
            self.local_inhibitor = 1.0
    
    def fire_local_inhibitor(self):                                         # Fire local inhibitor
        """Fire the local inhibitor."""
        self.driver.initialise_act(Type.PO)
        self.recipient.initialise_act(Type.PO)
        self.semantics.initialiseSem()
    
    def check_global_inhibitor(self):                                       # Check global inhibition
        """Check global inhibitor activation."""
        if self.driver.check_global_inhibitor():
            self.global_inhibitor = 1.0
        
    def fire_global_inhibitor(self):                                        # Fire global inhibitor
        """Fire the global inhibitor."""
        self.driver.initialise_act([Type.PO, Type.RB, Type.P])
        self.recipient.initialise_act([Type.PO, Type.RB, Type.P])
        self.memory.initialise_act([Type.PO, Type.RB, Type.P])
        self.semantics.initialise_sem()

    # ========================[ NODE FUNCTIONS ]==========================
    def get_pmode(self):                                                    # Get p_mode in driver and recipient
        """
        Get parent mode of P units in driver and recipient. Used in time steps activations.
        (driver, recipient)
        """
        self.driver.p_get_mode()
        self.recipient.p_get_mode()
    
    def initialise_p_mode(self, set: Set = Set.RECIPIENT):                  # Initialise p_mode in the given set
        """
        Initialise p_mode in the given set.
        (default: recipient)

        Args:
            set (Set, optional): The set to initialise p_mode in. (Defaults to recipient)
        """
        self.sets[set].initialise_p_mode()
    
    def get_weight_lengths(self):                                           # get weight lenghts in active memory
        """
        Get weight lengths of PO units. Used in run initialisation.
        (driver, recipient, memory, new_set)
        """
        sets = [Set.DRIVER, Set.RECIPIENT, Set.MEMORY, Set.NEW_SET]
        for set in sets:
            self.sets[set].po_get_weight_length()
    
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
        add_set = int(token.tensor[TF.SET])
        if add_set not in [set.value for set in Set]:
            raise ValueError("Invalid set in token feature.")
        
        reference = self.sets[add_set].add_token(token)
        return reference

    def del_token(self, ref_token: Ref_Token):                              # Delete a token
        """
        Delete a referenced token from the network.

        Args:
            ref_token (network.Ref_Token): A reference to the token to delete.
        """
        self.sets[ref_token.set].del_token(ref_token)

    def add_semantic(self, semantic: Semantic):                             # Add a semantic
        """
        Add a semantic to the given set.

        Args:
            semantic (network.Semantic): The semantic to add.
        
        Raises:
            ValueError: If provided semantic is not semantic type.
        """
        if semantic.tensor[SF.TYPE] != Type.SEMANTIC:
            raise ValueError("Cannot add non-semantic to semantic set.")
        
        self.semantics.add_semantic(semantic)

    def del_semantic(self, ref_semantic: Ref_Semantic):                     # Delete a semantic
        """
        Delete a semantic from the semantics.
        
        Args:
            ref_semantic (network.Ref_Semantic): A reference to the semantic to delete.

        Raises:
            ValueError: If ref_semantic is not provided.
        """
        self.semantics.del_semantic(ref_semantic)

    def set_sem_max_input(self):                                            # Set the maximum input for the semantics
        """
        Set the maximum input for the semantics.
        """
        max_input = self.semantics.get_max_input()
        self.semantics.set_max_input(max_input)
    
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
        if isinstance(reference, Ref_Token):
            if isinstance(feature, TF):
                return self.sets[reference.set].get_feature(reference, feature)
            else:
                raise ValueError("Referenced a token, but feature is not a token feature.")
        elif isinstance(reference, Ref_Semantic):
            if isinstance(feature, SF):
                return self.semantics.get_feature(reference, feature)
            else:
                raise ValueError("Referenced a semantic, but feature is not a semantic feature.")
        else:
            raise ValueError("Invalid reference type.")
    
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
        if isinstance(reference, Ref_Token):
            if isinstance(feature, TF):
                self.sets[reference.set].set_feature(reference, feature, value)
            else:
                raise ValueError("Referenced a token, but feature is not a token feature.")
        elif isinstance(reference, Ref_Semantic):
            if isinstance(feature, SF):
                self.semantics.set_feature(reference, feature, value)
            else:
                raise ValueError("Referenced a semantic, but feature is not a semantic feature.")
        else:
            raise ValueError("Invalid reference type.")
    
    # ----------------------------------------------------------------------
    def print_set(self, set: Set, feature_types: list[TF] = None):
        """
        Print the given set.

        Args:
            set (Set): The set to print.
            feature_types (list[TF], optional): List features to print, otherwise default features are printed.
        """
        self.sets[set].print(feature_types)
