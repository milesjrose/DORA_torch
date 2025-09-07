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
    return builder.build_network()


def test_rel_gen_passes(network: 'Network'):
    """
    Test should pass when at least one mapping exists and all mappings have weight >= 0.7.
    """
    # Setup: Create one valid mapping
    recipient = network.recipient()
    driver = network.driver()
    mappings = network.mappings[Set.RECIPIENT]
    
    # Ensure there's at least one token in each set
    if recipient.nodes.shape[0] == 0: recipient.tensor_ops.add_token(Token(Type.PO))
    if driver.nodes.shape[0] == 0: driver.tensor_ops.add_token(Token(Type.PO))

    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.8

    assert network.routines.rel_gen.requirements() is True


def test_rel_gen_fails_no_mappings(network: 'Network'):
    """
    Test should fail when no mappings exist.
    """
    # Setup: Mappings are empty by default
    assert network.routines.rel_gen.requirements() is False


def test_rel_gen_fails_low_weight(network: 'Network'):
    """
    Test should fail when a mapping has weight < 0.7.
    """
    # Setup: Create one mapping with a low weight
    recipient = network.recipient()
    driver = network.driver()
    mappings = network.mappings[Set.RECIPIENT]
    
    if recipient.nodes.shape[0] == 0: recipient.tensor_ops.add_token(Token(Type.PO))
    if driver.nodes.shape[0] == 0: driver.tensor_ops.add_token(Token(Type.PO))

    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.5

    assert network.routines.rel_gen.requirements() is False
