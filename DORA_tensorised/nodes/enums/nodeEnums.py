# nodes/enums/nodeEnums.py
# Stores enums to access token and semantic features, and to encode feature values.

from enum import IntEnum

class TF(IntEnum):
    """
    Enum to access token values

    Features:
    - ID: ID of token.
    - TYPE: Type of token.
    - SET: Set of token.
    - ANALOG: Analog of token.
    - MAX_MAP_UNIT: Maximum map unit of token.
    - MADE_UNIT: Made unit of token.
    - MAKER_UNIT: Maker unit of token.
    - INHIBITOR_THRESHOLD: Inhibitor threshold of token.
    - GROUP_LAYER: Group layer of token.
    - MODE: Mode of token.
    - TIMES_FIRED: Times fired of token.
    - SEM_COUNT: Semantic count of token.
    - ACT: Act of token.
    - MAX_ACT: Maximum act of token.
    - INHIBITOR_INPUT: Inhibitor input of token.
    - INHIBITOR_ACT: Inhibitor act of token.
    - MAX_MAP: Maximum map of token.
    - TD_INPUT: TD input of token.
    - BU_INPUT: BU input of token.
    - LATERAL_INPUT: Lateral input of token.
    - MAP_INPUT: Map input of token.
    - NET_INPUT: Net input of token.
    - MAX_SEM_WEIGHT: Maximum semantic weight of token.
    - INFERRED: Inferred of token.
    - RETRIEVED: Retrieved of token.
    - COPY_FOR_DR: Copy for DR of token.
    - COPIED_DR_INDEX: Copied DR index of token.
    - SIM_MADE: Sim made of token.
    - DELETED: Deleted of token.
    - PRED: Pred of token.
    """
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


class SF(IntEnum):
    """
    Enum to access semantic values

    Features:
    - ID: ID of semantic.
    - TYPE: Type of semantic.
    - ONT_STATUS: Ontology status of semantic.
    - DELETED: Deleted of semantic.
    - AMOUNT: Amount of semantic.
    - INPUT: Input of semantic.
    - MAX_INPUT: Maximum input of semantic.
    - ACT: Act of semantic.
    """
    # INT values:
    ID                  = 0
    TYPE                = 1
    ONT_STATUS          = 2
    DELETED             = 3

    # FLOAT values:
    AMOUNT              = 4
    INPUT               = 5
    MAX_INPUT           = 6
    ACT                 = 7

class MappingFields(IntEnum):
    """
    Enum to access mapping values

    Fields:
    - WEIGHT: Weight of mapping.
    - HYPOTHESIS: Hypothesis of mapping.
    - MAX_HYP: Maximum hypothesis of mapping.
    - CONNETIONS: Connections of mapping.
    """
    WEIGHT      = 0
    HYPOTHESIS  = 1
    MAX_HYP     = 2
    CONNETIONS  = 3

class Type(IntEnum):
    """
    Enum to encode my_type field in tokenFields

    Types:
    - PO: PO token.
    - RB: RB token.
    - P: P token.
    - GROUP: Group token.
    - SEMANTIC: Semantic token.
    """
    PO          = 0
    RB          = 1
    P           = 2
    GROUP       = 3
    SEMANTIC    = 4

class Set(IntEnum):
    """
    Enum to encode my_set field in tokenFields

    Sets:
    - DRIVER: Driver set.
    - RECIPIENT: Recipient set.
    - MEMORY: Memory set.
    - NEW_SET: New set.
    """
    DRIVER      = 0
    RECIPIENT    = 1
    MEMORY      = 2
    NEW_SET     = 3

class Mode(IntEnum):
    """
    Enum to encode p.mode field in tokenFields

    Modes:
    - CHILD: Child mode.
    - NEUTRAL: Neutral mode.
    - PARENT: Parent mode.
    """
    CHILD       = 0
    NEUTRAL     = 1
    PARENT      = 2

class OntStatus(IntEnum):
    """
    Enum to encode sem.ont_status in semanticFields

    Statuses:
    - STATE: Semantic is a state.
    - VALUE: Semantic is a value.
    - SDM: Semantic is an SDM (greater than/less than/ect.)
    """
    STATE       = 0
    VALUE       = 1
    SDM         = 2

class B(IntEnum):
    TRUE    = 1
    FALSE   = 0

null = -99.99