import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.single_nodes import Token
from nodes.tests.sims.sim import symProps
from nodes.utils import tensor_ops as tOps

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

    from nodes.utils import nodePrinter
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

    mappings: 'Mappings' = network.mappings
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

    mappings: 'Mappings' = network.mappings
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

    mappings: 'Mappings' = network.mappings
    mappings[MappingFields.CONNECTIONS][r_rb_indices[0], d_rb_indices[0]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[0], d_rb_indices[0]] = 0.9
    mappings[MappingFields.CONNECTIONS][r_rb_indices[1], d_rb_indices[1]] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_indices[1], d_rb_indices[1]] = 0.9

    # Connect one of the mapped RBs to a P unit
    p_index = torch.where(recipient.get_mask(Type.P))[0][0]
    recipient.token_op.connect_idx(r_rb_indices[0], p_index)

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

    mappings: 'Mappings' = network.mappings
    mappings[MappingFields.CONNECTIONS][r_rb_index, d_rb_index] = 1.0
    mappings[MappingFields.WEIGHT][r_rb_index, d_rb_index] = 0.9

    # --------------------------[ TEST ]--------------------------
    assert network.routines.rel_form.requirements() is False


def test_rel_form_routine_infer_new_p(network: 'Network'):
    """
    Test should infer a new P unit in the recipient when no new P has been made by predication routine.
    """
    # ---------------------------[ SETUP ]---------------------------
    # Make sure no new P has been inferred
    network.routines.predication.made_new_pred = False
    
    recipient = network.recipient()
    initial_p_count = recipient.get_mask(Type.P).sum()

    # --------------------------[ TEST ]--------------------------
    network.routines.rel_form.rel_form_routine()

    # --------------------------[ ASSERT ]--------------------------
    # A new P unit should be inferred
    assert recipient.get_mask(Type.P).sum() == initial_p_count + 1
    # inferred_new_p flag should be True
    assert network.routines.rel_form.inferred_new_p is True
    # inferred_p should hold the new P token
    assert network.routines.rel_form.inferred_p is not None
    inferred_p_ref = network.routines.rel_form.inferred_p
    inferred_p_index = network.recipient().get_index(inferred_p_ref)
    inferred_p_type = network.recipient().nodes[inferred_p_index, TF.TYPE]
    assert inferred_p_type == Type.P


def test_rel_form_routine_connect_inferred_p(network: 'Network'):
    """
    Test should connect an inferred P unit to active RBs in the recipient.
    """
    # ---------------------------[ SETUP ]---------------------------
    # A new P has been inferred by the relation formation routine
    network.routines.rel_form.inferred_new_p = True
    
    recipient = network.recipient()

    # Add an inferred P to the relation formation routine
    inferred_p = Token(Type.P, {TF.SET: Set.RECIPIENT})
    ref_inferred_p = network.add_token(inferred_p)
    network.routines.rel_form.inferred_p = ref_inferred_p

    # Add some RBs to the recipient
    rb1 = network.add_token(Token(Type.RB, {TF.SET: Set.RECIPIENT, TF.ACT: 0.9}))
    rb2 = network.add_token(Token(Type.RB, {TF.SET: Set.RECIPIENT, TF.ACT: 0.7}))
    rb3 = network.add_token(Token(Type.RB, {TF.SET: Set.RECIPIENT, TF.ACT: 0.95}))
    
    initial_p_count = recipient.get_mask(Type.P).sum()

    # --------------------------[ TEST ]--------------------------
    network.routines.rel_form.rel_form_routine()

    # --------------------------[ ASSERT ]--------------------------
    # No new P unit should be inferred
    assert recipient.get_mask(Type.P).sum() == initial_p_count
    
    # The inferred P should be connected to RBs with activation >= 0.8
    connections = recipient.connections
    ref_inferred_p_index = recipient.get_index(ref_inferred_p)
    rb1_index = recipient.get_index(rb1)
    rb2_index = recipient.get_index(rb2)
    rb3_index = recipient.get_index(rb3)
    assert connections[ref_inferred_p_index, rb1_index] == B.TRUE
    assert connections[ref_inferred_p_index, rb2_index] == B.FALSE
    assert connections[ref_inferred_p_index, rb3_index] == B.TRUE
