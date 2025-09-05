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

def test_rel_form_passes(network: 'Network'):
    """
    Test should pass when:
    - At least 2 RBs in recipient map to RBs in driver with weight > 0.8
    - Mapped recipient RBs are not connected to any P units
    """
    # ---------------------------[ SETUP ]---------------------------
    driver = network.driver()
    recipient = network.recipient()

    # Ensure at least 2 RBs in driver and recipient
    for s in [driver, recipient]:
        while s.get_mask(Type.RB).sum() < 2:
            s.tensor_ops.add_token(Token(Type.RB))

    d_rb_mask = driver.get_mask(Type.RB)
    r_rb_mask = recipient.get_mask(Type.RB)
    print((r_rb_mask.sum() > 2), (sum(r_rb_mask) > 2))

    from ..utils import nodePrinter
    printer = nodePrinter()
    printer.print_set(network.recipient())
    # Ensure no P units are connected to recipient RBs
    p_mask = recipient.get_mask(Type.P)
    r_rb_indices = torch.where(r_rb_mask)[0]
    p_indices = torch.where(p_mask)[0]
    if p_indices.shape[0] > 0:
        recipient.connections[r_rb_indices[:, None], p_indices] = B.FALSE
        recipient.connections[p_indices[:, None], r_rb_indices] = B.FALSE
    printer = nodePrinter()
    printer.print_set(network.recipient())

    # Map 2 RBs from driver to recipient with high weights
    d_rb_indices = torch.where(d_rb_mask)[0]
    r_rb_indices = torch.where(r_rb_mask)[0]

    mappings: 'Mappings' = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.CONNECTIONS][r_rb_indices[0], d_rb_indices[0]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[0], d_rb_indices[0]] = 0.9
    mappings[MappingFields.CONNECTIONS][r_rb_indices[1], d_rb_indices[1]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[1], d_rb_indices[1]] = 0.9

    # --------------------------[ TEST ]--------------------------
    assert network.routines.rel_form.requirements() is True


def test_rel_form_fails_low_weight(network: 'Network'):
    """
    Test should fail when mapping weights are too low
    """
    # ---------------------------[ SETUP ]---------------------------
    driver = network.driver()
    recipient = network.recipient()

    # Ensure at least 2 RBs in driver and recipient
    for s in [driver, recipient]:
        while s.get_mask(Type.RB).sum() < 2:
            s.tensor_ops.add_token(Token(Type.RB))

    d_rb_mask = driver.get_mask(Type.RB)
    r_rb_mask = recipient.get_mask(Type.RB)

    # Map 2 RBs from driver to recipient with low weights
    d_rb_indices = torch.where(d_rb_mask)[0]
    r_rb_indices = torch.where(r_rb_mask)[0]

    mappings: 'Mappings' = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.CONNECTIONS][r_rb_indices[0], d_rb_indices[0]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[0], d_rb_indices[0]] = 0.5
    mappings[MappingFields.CONNECTIONS][r_rb_indices[1], d_rb_indices[1]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[1], d_rb_indices[1]] = 0.9 # one is still high

    # --------------------------[ TEST ]--------------------------
    assert network.routines.rel_form.requirements() is False


def test_rel_form_fails_connected_to_p(network: 'Network'):
    """
    Test should fail when a mapped recipient RB is connected to a P unit
    """
    # ---------------------------[ SETUP ]---------------------------
    driver = network.driver()
    recipient = network.recipient()

    # Ensure at least 2 RBs in driver and recipient
    for s in [driver, recipient]:
        while s.get_mask(Type.RB).sum() < 2:
            s.tensor_ops.add_token(Token(Type.RB))

    # Ensure at least one P in recipient
    if recipient.get_mask(Type.P).sum() < 1:
        recipient.tensor_ops.add_token(Token(Type.P))

    d_rb_mask = driver.get_mask(Type.RB)
    r_rb_mask = recipient.get_mask(Type.RB)

    # Map 2 RBs from driver to recipient with high weights
    d_rb_indices = torch.where(d_rb_mask)[0]
    r_rb_indices = torch.where(r_rb_mask)[0]

    mappings: 'Mappings' = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.CONNECTIONS][r_rb_indices[0], d_rb_indices[0]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[0], d_rb_indices[0]] = 0.9
    mappings[MappingFields.CONNECTIONS][r_rb_indices[1], d_rb_indices[1]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[1], d_rb_indices[1]] = 0.9

    # Connect one of the mapped RBs to a P unit
    p_index = torch.where(recipient.get_mask(Type.P))[0][0]
    recipient.connections[r_rb_indices[0], p_index] = 1.0

    # --------------------------[ TEST ]--------------------------
    assert network.routines.rel_form.requirements() is False


def test_rel_form_fails_not_enough_rbs(network: 'Network'):
    """
    Test should fail when there is only one RB mapping with high weight
    """
    # ---------------------------[ SETUP ]---------------------------
    driver = network.driver()
    recipient = network.recipient()

    # Ensure at least 1 RB in driver and recipient
    for s in [driver, recipient]:
        if s.get_mask(Type.RB).sum() < 1:
            s.tensor_ops.add_token(Token(Type.RB))

    d_rb_mask = driver.get_mask(Type.RB)
    r_rb_mask = recipient.get_mask(Type.RB)

    # Map 1 RB from driver to recipient with high weights
    d_rb_index = torch.where(d_rb_mask)[0][0]
    r_rb_index = torch.where(r_rb_mask)[0][0]

    mappings: 'Mappings' = network.mappings[Set.RECIPIENT]
    mappings[MappingFields.CONNECTIONS][r_rb_index, d_rb_index] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_index, d_rb_index] = 0.9

    # --------------------------[ TEST ]--------------------------
    assert network.routines.rel_form.requirements() is False

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
