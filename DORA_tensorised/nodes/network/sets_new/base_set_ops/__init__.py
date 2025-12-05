# nodes/network/sets_new/base_set_ops/__init__.py

from .set_tensor import TensorOperations
from .set_update import UpdateOperations
from .set_analog import AnalogOperations
from .set_kludgey import KludgeyOperations
from .set_token import TokenOperations

__all__ = [
    "TensorOperations",
    "UpdateOperations",
    "AnalogOperations",
    "KludgeyOperations",
    "TokenOperations"
]