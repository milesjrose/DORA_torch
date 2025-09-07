import pytest
import torch

from ..builder import NetworkBuilder
from ..enums import *
from ..network.single_nodes import Token
from .sims.sim import symProps
from ..utils import tensor_ops as tOps

from random import shuffle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..network.connections import Mappings
    from ..network import Network
    from ..network.sets import Driver, Recipient
    from ..network.connections import Mappings

@pytest.fixture
def network():
    builder = NetworkBuilder(symProps=symProps)
    net = builder.build_network()
    net.initialise_made_unit()
    return net

def test_get_made_unit_ref(network: 'Network'):
    """
    Test get_made_unit_ref correctly gets the made unit ref.
    """
    driver = network.driver()
    new_set = network.new_set()
    
    # Add a PO token to the driver
    ref_maker = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    idx_maker = network.get_index(ref_maker)
    # Add a new token to the new set
    ref_made = network.add_token(Token(Type.PO, {TF.SET: Set.NEW_SET, TF.ACT: 1.0, TF.PRED: B.TRUE}))
    idx_made = network.get_index(ref_made)
    # Set made features
    network.set_value(ref_maker, TF.MADE_UNIT, idx_made)
    network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
    # Set maker features
    network.set_value(ref_maker, TF.MAKER_UNIT, idx_maker)
    network.set_value(ref_maker, TF.MAKER_SET, ref_maker.set)
    # Get made unit ref
    made_unit_ref = network.get_made_unit_ref(ref_maker)
    # Assert made unit ref is correct
    assert made_unit_ref is not None
    assert made_unit_ref.ID == ref_made.ID
    assert made_unit_ref.set == ref_made.set

def test_get_maker_unit_ref(network: 'Network'):
    """
    Test get_maker_unit_ref correctly gets the maker unit ref.
    """
    driver = network.driver()
    new_set = network.new_set()
    
    # Add a PO token to the driver
    ref_maker = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    idx_maker = network.get_index(ref_maker)
    # Add a new token to the new set
    ref_made = network.add_token(Token(Type.PO, {TF.SET: Set.NEW_SET, TF.ACT: 1.0, TF.PRED: B.TRUE}))
    idx_made = network.get_index(ref_made)
    # Set made features
    network.set_value(ref_maker, TF.MADE_UNIT, idx_made)
    network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
    # Set maker features
    network.set_value(ref_maker, TF.MAKER_UNIT, idx_maker)
    network.set_value(ref_maker, TF.MAKER_SET, ref_maker.set)
    # Get made unit ref
    maker_unit_ref = network.get_maker_unit_ref(ref_maker)
    # Assert made unit ref is correct
    assert maker_unit_ref is not None
    assert maker_unit_ref.ID == ref_maker.ID
    assert maker_unit_ref.set == ref_maker.set