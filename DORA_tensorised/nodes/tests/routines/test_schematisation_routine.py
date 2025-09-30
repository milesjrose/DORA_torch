import pytest
import torch
import logging

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.single_nodes import Token, Ref_Token
from nodes.tests.sims.sim import symProps
from nodes.utils import tensor_ops as tOps

logger = logging.getLogger(__name__)

from random import shuffle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.network.connections import Mappings
    from nodes.network import Network
    from nodes.network.sets import Driver, Recipient
    from nodes.network.connections import Mappings

@pytest.fixture
def network():
    builder = NetworkBuilder(symProps=symProps)
    net = builder.build_network()
    return net


def test_schema_passes(network: 'Network'):
    """
    Test should pass when all schema requirements are met.
    - All mapped tokens have max_map >= 0.7 -> all tokens connect to valid nodes.
    """
    for s in [network.driver(), network.recipient()]:
        s.nodes[:, TF.MAX_MAP] = 0.8

    assert network.routines.schematisation.requirements() is True


def test_schema_fails_low_max_map(network: 'Network'):
    """
    Test should fail when a token has 0 < max_map < 0.7.
    """
    network.driver().nodes[0, TF.MAX_MAP] = 0.5

    assert network.routines.schematisation.requirements() is False


def test_schema_fails_invalid_child_connection(network: 'Network'):
    """
    Test should fail when a valid token (max_map >= 0.7) is connected to an invalid one (max_map < 0.7).
    """
    driver = network.driver()
    # All tokens are valid by default
    driver.nodes[:, TF.MAX_MAP] = 0.8

    # Make one token invalid
    driver.nodes[1, TF.MAX_MAP] = 0.0

    # Connect a valid token (parent) to the invalid one (child)
    driver.connections[0, 1] = 1.0

    assert network.routines.schematisation.requirements() is False


def test_schema_fails_invalid_parent_connection(network: 'Network'):
    """
    Test should fail when a valid token (max_map >= 0.7) is connected to an invalid one (max_map < 0.7).
    """
    driver = network.driver()
    # All tokens are valid by default
    driver.nodes[:, TF.MAX_MAP] = 0.8

    # Make one token invalid
    driver.nodes[0, TF.MAX_MAP] = 0.0

    # Connect a valid token (child) to the invalid one (parent)
    driver.connections[0, 1] = 1.0

    assert network.routines.schematisation.requirements() is False


def test_schematise_p_infer(network: 'Network'):
    """
    Test shcematise_p correctly infers a new P token in newSet.
    """
    driver = network.driver()
    new_set = network.new_set()

    # Add a P token to the driver
    p_token = network.add_token(Token(Type.P, {TF.SET: Set.DRIVER, TF.ACT: 1.0, TF.MODE: Mode.PARENT}))
    network.set_value(p_token, TF.MAX_MAP, 0.8)
    
    # check most active p is added p
    most_active_p = network.driver().token_op.get_most_active_token(mask=driver.get_mask(Type.P))
    assert most_active_p is not None
    assert most_active_p.ID == p_token.ID
    assert most_active_p.set == p_token.set

    # Add mapping for most active p to recipient (any token will work)
    index = network.get_index(most_active_p)
    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][1, index] = 0.8


    initial_p_count = new_set.get_mask(Type.P).sum()
    
    # Run schematisation for P tokens
    network.routines.schematisation.shcematise_p(Mode.PARENT)
    
    # Assert a new P token was inferred
    assert new_set.get_mask(Type.P).sum() == initial_p_count + 1
    
    # Assert maker/made units are set
    made_unit_ref = network.get_made_unit_ref(p_token)
    assert made_unit_ref is not None
    maker_unit_ref = network.get_maker_unit_ref(made_unit_ref)
    assert maker_unit_ref.ID == p_token.ID

def set_maker_made_units(network: 'Network', ref_maker: Ref_Token, ref_made: Ref_Token):
    """
    Set the maker and made units for a given token.
    """
    network.set_value(ref_maker, TF.MADE_UNIT, network.get_index(ref_made))
    network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
    network.set_value(ref_maker, TF.MAKER_UNIT, network.get_index(ref_maker))
    network.set_value(ref_maker, TF.MAKER_SET, ref_maker.set)

def test_schematise_p_update(network: 'Network'):
    """
    Test shcematise_p correctly updates an existing made P token.
    """
    driver = network.driver()
    new_set = network.new_set()

    # Add a P token to driver and a corresponding made token to newSet
    ref_p = network.add_token(Token(Type.P, {TF.SET: Set.DRIVER, TF.ACT: 0.9, TF.MODE: Mode.PARENT}))
    ref_made = network.add_token(Token(Type.P, {TF.SET: Set.NEW_SET, TF.ACT: 0.0, TF.MODE: Mode.PARENT}))
    set_maker_made_units(network, ref_p, ref_made)
    
    # Add an active RB to newSet to be connected to
    rb_token = network.add_token(Token(Type.RB, {TF.SET: Set.NEW_SET, TF.ACT: 0.6}))

    # Run schematisation for P tokens
    network.routines.schematisation.shcematise_p(Mode.PARENT)
    
    # Assert made_p activation is now 1.0
    made_p_act = network.get_value(ref_made, TF.ACT)
    assert made_p_act == 1.0
    
    # Assert connection was made
    made_p_index = new_set.get_index(ref_made)
    rb_index = new_set.get_index(rb_token)
    assert new_set.connections[made_p_index, rb_index] == B.TRUE


def test_schematise_rb_infer(network: 'Network'):
    """
    Test schematise_rb correctly infers a new RB token in newSet.
    """
    driver = network.driver()
    new_set = network.new_set()

    # Add an RB token to the driver
    rb_token = network.add_token(Token(Type.RB, {TF.SET: Set.DRIVER, TF.ACT: 0.5}))
    # Add mapping for rb to recipient (any token will work)
    index = network.get_index(rb_token)
    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][1, index] = 0.8
    
    initial_rb_count = new_set.get_mask(Type.RB).sum()
    
    # Run schematisation for RB tokens
    network.routines.schematisation.schematise_rb()
    
    # Assert a new RB token was inferred
    assert new_set.get_mask(Type.RB).sum() == initial_rb_count + 1


def test_schematise_po_infer(network: 'Network'):
    """
    Test schematise_po correctly infers a new PO token in newSet.
    """
    driver = network.driver()
    new_set = network.new_set()

    # Add a PO token to the driver
    po_token = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    # Add mapping for po to recipient (any token will work)
    index = network.get_index(po_token)
    network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][1, index] = 0.8
    
    initial_po_count = new_set.get_mask(Type.PO).sum()
    
    # Run schematisation for PO tokens
    network.routines.schematisation.schematise_po()
    
    # Assert a new PO token was inferred
    assert new_set.get_mask(Type.PO).sum() == initial_po_count + 1

def test_infer_token(network: 'Network'):
    """
    Test infer_token correctly infers a new token.
    """
    driver = network.driver()
    new_set = network.new_set()

    # Add a PO token to the driver
    ref_maker = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    network.set_value(ref_maker, TF.MAX_MAP, 0.8)

    # Infer a token
    ref_made = network.routines.schematisation.infer_token(ref_maker)

    # Check made token
    nops = network.node_ops
    assert ref_made is not None
    assert ref_made.set == Set.NEW_SET
    assert nops.get_value(ref_made, TF.TYPE) is not None
    assert nops.get_value(ref_made, TF.TYPE) == Type.PO
    assert nops.get_value(ref_made, TF.ACT) == 1.0
    assert nops.get_value(ref_made, TF.INFERRED) == B.TRUE
    assert nops.get_value(ref_made, TF.SET) == Set.NEW_SET
    assert nops.get_value(ref_made, TF.ANALOG) == null
    assert nops.get_value(ref_made, TF.PRED) == B.TRUE
    # Assert maker unit is set
    assert nops.get_value(ref_made, TF.MAKER_SET) == ref_maker.set
    assert nops.get_value(ref_made, TF.MAKER_UNIT) == network.get_index(ref_maker)

    # Check maker token
    assert nops.get_value(ref_maker, TF.MADE_UNIT) == network.get_index(ref_made)
    assert nops.get_value(ref_maker, TF.MADE_SET) == ref_made.set
