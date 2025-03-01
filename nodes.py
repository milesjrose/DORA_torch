# nodes.py 
# Class for holding memTypes, and inter-set tensor operations
import torch
from nodeEnums import *
from nodesMemTypes import *
from nodesMemTypes import tf as tf

class Nodes(object):
    def __init__(self, driver: DriverTensor, recipient: RecipientTensor, semantics: TokenTensor, LTM: TokenTensor, links: Links, mappings: Mappings):
        # node tensors
        self.driver: DriverTensor = driver
        self.recipient: RecipientTensor = recipient
        self.semantics: TokenTensor = semantics
        self.LTM: TokenTensor = LTM
        
        # inter-set connections
        self.links: Links = links
        self.mappings: Mappings = mappings
    

