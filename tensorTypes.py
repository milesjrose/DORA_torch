import torch
from enum import Enum

# Enum to access token values stored in the token tensor
class tf(Enum):
    # Token general values
    SET = 0                     # Token set type (driver, recipient, etc.)
    MYANALOG = 1                # Analog this token belongs to
    ACT = 2                     # Activation value
    MAX_ACT = 3                 # Maximum activation recorded
    
    # Inhibitor-related values
    INHIBITOR_INPUT = 4         # Input to the inhibitory mechanism
    INHIBITOR_ACT = 5           # Activation level of inhibitor

    # Mapping and connectivity values
    MAX_MAP_UNIT = 6            # The unit this token maps to most strongly
    MAX_MAP = 7                 # Maximum mapping connection strength

    # Input signals
    TD_INPUT = 8                # Top-down input value
    BU_INPUT = 9                # Bottom-up input value
    LATERAL_INPUT = 10          # Lateral inhibition input
    MAP_INPUT = 11              # Mapping input from other tokens
    NET_INPUT = 12              # Net input after processing

    # Memory and inference tracking
    MY_MAKER_UNIT = 13          # Unit that inferred this token
    INFERRED = 14               # Boolean flag: was this token inferred?
    RETRIEVED = 15              # Boolean flag: has this token been retrieved?
    COPY_FOR_DR = 16            # Boolean flag: is this a copy for Driver/Recipient memory?
    COPIED_DR_INDEX = 17        # Index of the copied unit in memory
    SIM_MADE = 18               # Boolean flag: was this created during simulation?

    # Deletion tracking
    IS_DELETED = 19             # Boolean flag: has this token been marked as deleted?

    # Shared values (common across all types)
    MY_TYPE = 20                # Token type identifier (Group, PUnit, etc.)
    INHIBITOR_THRESHOLD = 21    # Threshold for inhibition activation

    # Group-specific values
    MY_GROUP_LAYER = 22         # Group layer level

    # PUnit-specific values
    MODE = 23                   # PUnit mode (parent/child/neutral)

    # RBUnit-specific values
    TIMES_FIRED = 24            # Count of how many times the RB has fired

    # POUnit-specific values
    PRED_OR_OBJ = 25            # Identifier for predicate or object unit
    SEM_NORM = 26               #
    MAX_SEM_WEIGHT = 27

    # Semantic-specific values
    AMOUNT = 28              # Amount of activation/contribution
    ONT_STATUS = 29           # Ontological status
    MY_INPUT = 30             # Input value to the semantic unit
    MAX_SEM_INPUT = 31        # Maximum semantic input recorded

class TokenTensor(object):
    def __init__(self, mem):
        self.dimensions = len(tf)               # Number of variables to store per token
        self.nodeCount = 0                      # Number of nodes in tensor
        self.tokens = None                      # Token tensor
        self.setEdited = False                  # Flag to track if set masks need to be recomputed

        # If provided memorySet object, use it to populate token tensor
        if mem != None:
            self.populateTensor(mem)

    #### probs dont bother, just build from currentsym - like in buildNetwork.py
    def populateTensor(self, mem):
        # get number of nodes to be added to tensor
        self.nodeCount = 0
        for tokenType in [mem.Groups, mem.Ps, mem.RBs, mem.POs, mem.semantics]:
            self.nodeCount += len(tokenType)
        
        # tensor non extendable, so give some headroom for nodes to be recruited in future
        tensorSize = int(self.nodeCount * 1.1)
        self.tokens = torch.zeros(size=(tensorSize, self.dimensions))

    def formatToken(self, token):
        tk = [torch.nan] * self.dimensions           # create empty token with enough features

        # Fill default values
        tk[tf.SET.value] = token.my_set              # self.set = my_set
        tk[tf.MYANALOG.value] = token.myanalog       # self.myanalog = myanalog
        tk[tf.ACT.value] = 0.0                       # self.act = 0.0
        tk[tf.MAX_ACT.value] = 0.0                   # self.max_act = 0.0
        tk[tf.INHIBITOR_INPUT.value] = 0.0           # self.inhibitor_input = 0.0
        tk[tf.INHIBITOR_ACT.value] = 0.0             # self.inhibitor_act = 0.0
        tk[tf.MAX_MAP_UNIT.value] = None             # self.max_map_unit = None
        tk[tf.MAX_MAP.value] = 0.0                   # self.max_map = 0.0
        tk[tf.TD_INPUT.value] = 0.0                  # self.td_input = 0.0
        tk[tf.BU_INPUT.value] = 0.0                  # self.bu_input = 0.0
        tk[tf.LATERAL_INPUT.value] = 0.0             # self.lateral_input = 0.0
        tk[tf.MAP_INPUT.value] = 0.0                 # self.map_input = 0.0
        tk[tf.NET_INPUT.value] = 0.0                 # self.net_input = 0.0
        tk[tf.MY_MAKER_UNIT.value] = None            # self.my_maker_unit = None
        tk[tf.INFERRED.value] = token.inferred_now   # self.inferred = inferred_now
        tk[tf.RETRIEVED.value] = False               # self.retrieved = False
        tk[tf.COPY_FOR_DR.value] = False             # self.copy_for_DR = False
        tk[tf.COPIED_DR_INDEX.value] = None          # self.copied_DR_index = None
        tk[tf.SIM_MADE.value] = token.inferred_now   # self.sim_made = inferred_now
        tk[tf.MY_TYPE] = token.my_type
        tk[tf.IS_DELETED] = False
        # Fill type specific values
        if token != None:
            match token.my_type:
                case 'Group':
                    pass
                case 'P':
                    tk[tf.INHIBITOR_THRESHOLD] = token.inhibitorThreshold
                    tk[tf.MODE] = 0
                    pass
                case 'RB':
                    tk[tf.INHIBITOR_THRESHOLD] = token.inhibitorThreshold
                    tk[tf.TIMES_FIRED] = token.timesFired
                    pass
                case 'PO':
                    tk[tf.INHIBITOR_THRESHOLD] = token.inhibitorThreshold
                    tk[tf.PRED_OR_OBJ] = token.predOrObj
                    tk[tf.SEM_NORM] = token.semNormalisation
                    tk[tf.MAX_SEM_WEIGHT] = token.max_sem_weight
                    pass
                case 'semantic':
                    pass
        
        return tk