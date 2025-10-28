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
    """ID of token"""
    TYPE                = 1
    """Type of token"""
    SET                 = 2
    """Set of token"""
    ANALOG              = 3
    """Analog number token belongs to"""
    MAX_MAP_UNIT        = 4
    """Unit token maps most to"""
    MADE_UNIT           = 5
    """Token made by this token"""
    MADE_SET            = 30
    """Set of the made token"""
    MAKER_UNIT          = 6
    """Maker unit of this token"""
    MAKER_SET           = 31
    """Set of the maker token"""
    INHIBITOR_THRESHOLD = 7
    """Inhibitor threshold of token"""
    GROUP_LAYER         = 8     # Group
    """Group layer of token"""
    MODE                = 9     # P unit
    """Mode of P token"""
    TIMES_FIRED         = 10    # RB unit
    """Times the RB token has been fired"""
    SEM_COUNT           = 11    # PO unit
    """Number of semantics connected this PO token"""
    COPIED_DR_INDEX     = 26
    """The index of the token I was copied from in MEMORY"""

    # FLOAT values:
    ACT                 = 12
    """Activation of token"""
    MAX_ACT             = 13
    """Maximum activation of this token"""
    INHIBITOR_INPUT     = 14
    """Inhibitory input to the token"""
    INHIBITOR_ACT       = 15
    """Inhibitory activation of the token"""
    MAX_MAP             = 16
    """Maximum mapping value from/to this token"""
    TD_INPUT            = 17
    """Input from higher tokens"""
    BU_INPUT            = 18
    """Input from lower nodes"""
    LATERAL_INPUT       = 19
    """Input from tokens on the same level"""
    MAP_INPUT           = 20
    """Input from mapping units"""

    NET_INPUT           = 21
    """Total input to the token"""
    MAX_SEM_WEIGHT      = 22    # PO unit
    """Maximum weight of semantic connections to this PO token"""

    # BOOL values:
    INFERRED            = 23
    """If the token was created through inference"""
    RETRIEVED           = 24
    """If the token was retrieved from memory"""
    COPY_FOR_DR         = 25
    """If the token has been copied into the driver or recipient"""
    SIM_MADE            = 27
    """Was I made during a simulation - has same value as inferred, but does not get reset when the new unit leaves the newSet TODO: check if i actually use this?"""
    DELETED             = 28
    """If the token has been deleted"""
    PRED                = 29    # PO unit
    """If this PO token is a predicate (if false, token in an object)"""

def TF_type(feature: TF):
    """
    Get the type of a feature.

    Args:
        feature (TF): The feature to get the type of.

    Returns:
        The type of the feature.
    """
    TF_types = {
        TF.ID: int,
        TF.TYPE: Type,
        TF.SET: Set,
        TF.ANALOG: int,
        TF.MAX_MAP_UNIT: int,
        TF.MADE_UNIT: int,
        TF.MADE_SET: Set,
        TF.MAKER_UNIT: int,
        TF.MAKER_SET: Set,
        TF.INHIBITOR_THRESHOLD: int,
        TF.GROUP_LAYER: int,
        TF.MODE: Mode,
        TF.TIMES_FIRED: int,
        TF.SEM_COUNT: int,
        TF.ACT: float,
        TF.MAX_ACT: float,
        TF.INHIBITOR_INPUT: float,
        TF.INHIBITOR_ACT: float,
        TF.MAX_MAP: float,
        TF.TD_INPUT: float,
        TF.BU_INPUT: float,
        TF.LATERAL_INPUT: float,
        TF.MAP_INPUT: float,
        TF.NET_INPUT: float,
        TF.MAX_SEM_WEIGHT: float,
        TF.INFERRED: bool,
        TF.RETRIEVED: bool,
        TF.COPY_FOR_DR: bool,
        TF.COPIED_DR_INDEX: int,
        TF.SIM_MADE: bool,
        TF.DELETED: bool,
        TF.PRED: bool,
    }
    return TF_types[feature]

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
    - DIMENSION: Encodes dimension of semantic as integer.
    """
    # INT values:
    ID                  = 0
    TYPE                = 1
    """Type of semantic NOTE: can probably be removed as they are all semantics"""
    ONT                 = 2
    """Ontology status of semantic"""
    DIM                 = 8
    """Dimension of semantic"""

    # BOOL values:
    DELETED             = 3

    # FLOAT values:
    AMOUNT              = 4
    INPUT               = 5
    MAX_INPUT           = 6
    ACT                 = 7

def SF_type(feature: SF):
    """
    Get the type of a feature.
    """
    SF_types = {
        SF.ID: int,
        SF.TYPE: Type,
        SF.ONT: OntStatus,
        SF.DIM: int,
        SF.DELETED: bool,
        SF.AMOUNT: float,
        SF.INPUT: float,
        SF.MAX_INPUT: float,
        SF.ACT: float,
    }
    return SF_types[feature]



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
    CONNECTIONS  = 3

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
    HO          = 3

class B(IntEnum):
    TRUE    = 1
    FALSE   = 0

class Routines(IntEnum):
    """
    Enum for routines
    """
    MAP = 0
    RETRIEVE = 1
    PREDICATE = 2
    REL_FORM = 3
    REL_GEN = 4
    SCEMA = 5

class SDM(IntEnum):
    """
    Enum for SDM
    """
    MORE = 0
    LESS = 1
    SAME = 2
    DIFF = 3

class SD(IntEnum):
    """
    Set Dimension Enum [NODES, FEATS]
    """
    NODES = 0
    FEATS = 1


null = -99.0
"""Null value for float tensors: -99.0"""

from torch import float32
tensor_type = float32
"""Tensor type for float tensors: torch.float32"""

MAPPING_SETS = [Set.RECIPIENT]
"""Sets that have mappings"""
