import torch
from enum import IntEnum

# Enum to access token values
class tokenFields(IntEnum):
    # INT values:
    ID                  = 0
    TYPE                = 1
    SET                 = 2
    ANALOG              = 3
    MAX_MAP_UNIT        = 4
    MADE_UNIT           = 5
    MAKER_UNIT          = 6
    INHIBITOR_THRESHOLD = 7
    GROUP_LAYER         = 8     # Group
    MODE                = 9     # P unit
    TIMES_FIRED         = 10    # RB unit
    SEM_COUNT           = 11    # PO unit

    # BOOL values:
    INFERRED            = 12
    RETRIEVED           = 13
    COPY_FOR_DR         = 14
    COPIED_DR_INDEX     = 15
    SIM_MADE            = 16
    DELETED             = 17
    PRED                = 18    # PO unit

    # FLOAT values:
    ACT                 = 19
    MAX_ACT             = 20
    INHIBITOR_INPUT     = 21
    INHIBITOR_ACT       = 22
    MAX_MAP             = 23
    TD_INPUT            = 24
    BU_INPUT            = 25
    LATERAL_INPUT       = 26
    MAP_INPUT           = 27
    NET_INPUT           = 28
    MAX_SEM_WEIGHT      = 29    # PO unit

# Enum to access semantic values
class semanticFields(IntEnum):
    # INT values:
    ID                  = 0
    TYPE                = 1
    ONT_STATUS          = 2

    # FLOAT values:
    AMOUNT              = 3
    MYINPUT             = 4
    MAX_SEM_INPUT       = 5
    ACT                 = 6      

class Nodes(object):
    def __init__(self):
        # Set all value, given as arguments in creation
        self.tensor = None
