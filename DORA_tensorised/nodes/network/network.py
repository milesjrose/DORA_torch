# nodes/network/network.py
# Class for holding network sets, and accessing operations on them.

from nodes.enums import *

from .sets import Driver, Recipient, Memory, New_Set, Semantics
from .connections import Mappings
from .network_params import Params
from .single_nodes import Token, Semantic
from .single_nodes import Ref_Token, Ref_Semantic

class Network(object):
    """
    A class for holding set objects and operations.
    """
    def __init__(self, driver: Driver, recipient: Recipient, LTM: Memory, new_set: New_Set, semantics: Semantics, set_mappings: dict[int, Mappings], params: Params = None):
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
        self.memory: Memory = LTM
        self.new_set: New_Set = new_set
        self.sets = {
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
    
    def add_token(self, set: Set, token: Token):                            # Add a token to the given set
        """
        Add a token to the given set.
        """
        if token.tensor[TF.SET] != set:
            raise ValueError("Token set does not match set type.")
        
        self.sets[set].add_token(token)

    def del_token(self, set: Set = None, ID = None, ref_token:Ref_Token = None):  # Delete a token
        """
        Delete a token from the given set.

        - Either (set and ID) or (ref_token) must be provided.

        Args:
            set (Set, optional): The set to delete the token from.
            ID (int, optional): The ID of the token to delete.
            ref_token (Ref_Token, optional): A reference to the token to delete.
        """
        if ref_token is not None:
            ID = ref_token.ID
            set = ref_token.set
        elif (set is None) or (ID is None):
            raise ValueError("Either set and ID or ref_token must be provided.")
        
        self.sets[set].del_token(ID)

    def add_semantic(self, semantic: Semantic):                             # Add a semantic
        """
        Add a semantic to the given set.
        """
        if semantic.tensor[SF.TYPE] != Type.SEMANTIC:
            raise ValueError("Cannot add non-semantic to semantic set.")
        
        self.semantics.add_semantic(semantic)

    def del_semantic(self, ID = None, ref_sem:Ref_Semantic = None):         # Delete a semantic
        """
        Delete a semantic from the semantics.
        """
        if ref_sem is not None:
            ID = ref_sem.ID
        elif ID is None:
            raise ValueError("Either ID or ref_sem must be provided.")
        
        self.semantics.del_semantic(ID)

    def set_sem_max_input(self):                                            # Set the maximum input for the semantics
        """
        Set the maximum input for the semantics.
        """
        max_input = self.semantics.get_max_input()
        self.semantics.set_max_input(max_input)
