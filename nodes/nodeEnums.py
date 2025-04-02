# nodeEnums
# Stores enums for types of nodes

from enum import IntEnum

# Enum to access token values
class TF(IntEnum):
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

    # FLOAT values:
    ACT                 = 12
    MAX_ACT             = 13
    INHIBITOR_INPUT     = 14
    INHIBITOR_ACT       = 15
    MAX_MAP             = 16
    TD_INPUT            = 17
    BU_INPUT            = 18
    LATERAL_INPUT       = 19
    MAP_INPUT           = 20
    NET_INPUT           = 21
    MAX_SEM_WEIGHT      = 22    # PO unit

    # BOOL values:
    INFERRED            = 23
    RETRIEVED           = 24
    COPY_FOR_DR         = 25
    COPIED_DR_INDEX     = 26
    SIM_MADE            = 27
    DELETED             = 28
    PRED                = 29    # PO unit


# Enum to access semantic values
class SF(IntEnum):
    # INT values:
    ID                  = 0
    TYPE                = 1
    ONT_STATUS          = 2

    # FLOAT values:
    AMOUNT              = 3
    INPUT               = 4
    MAX_INPUT           = 5
    ACT                 = 6

# Enum to access mapping values
class MappingFields(IntEnum):
    WEIGHT      = 0
    HYPOTHESIS  = 1
    MAX_HYP     = 2
    CONNETIONS  = 3

# Enum to encode my_type field in tokenFields
class Type(IntEnum):
    PO          = 0
    RB          = 1
    P           = 2
    GROUP       = 3
    SEMANTIC    = 4

# Enum to encode my_set field in tokenFields
class Set(IntEnum):
    DRIVER      = 0
    RECIPIENT    = 1
    MEMORY      = 2
    NEW_SET     = 3

# Enum to encode p.mode field in tokenFields
class Mode(IntEnum):
    CHILD       = 0
    NEUTRAL     = 1
    PARENT      = 2

# Enum to encode sem.ont_status in semanticFields
class OntStatus(IntEnum):
    STATE       = 0
    VALUE       = 1
    SDM         = 2

class B(IntEnum):
    TRUE    = 1
    FALSE   = 0
