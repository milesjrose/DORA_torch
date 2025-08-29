# nodes/network/operations/requirement_ops.py
# Requirement operations for Network class

from ...enums import *

class RequirementOperations:
    """
    Requirement operations for the Network class.
    Handles requirement checking for various operations.
    """

    # TODO: Move requirements.py to here? Can't remember why it was in a seperate file
    
    def __init__(self, network):
        """
        Initialize RequirementOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network = network

    def predication_requirements(self):
        """
        Check predication requirements.
        """
        return self.network.requirements.predication()
    
    def rel_form_requirements(self):
        """
        Check relation formation requirements.
        """
        return self.network.requirements.rel_form()
    
    def schema_requirements(self):
        """
        Check schema requirements.
        """
        return self.network.requirements.schema()
    
    def rel_gen_requirements(self):
        """
        Check relation generation requirements.
        """
        return self.network.requirements.rel_gen()