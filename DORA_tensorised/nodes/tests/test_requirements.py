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

def test_predication_passes(network):
    """
    By default, the test symProps should allow predication to pass
    Test should pass when:
    - All driver POs map to recipient POs with weight above threshold (.8)
    - No mapped recipient POs are connected to RBs
    """
    # ---------------------------[ SETUP ]---------------------------
    # Make mapping connections from driver POs to recipient POs, Mappings empty by default
    driver = network.sets[Set.DRIVER]
    d_po = driver.get_mask(Type.PO)
    print("d_po [" + str(d_po.shape) + "]", d_po)
    print("r_po [" + str(network.recipient().get_mask(Type.PO).shape) + "]", network.recipient().get_mask(Type.PO))
    # Make sure same or more POs in recipient vs driver
    r_po = network.sets[Set.RECIPIENT].get_mask(Type.PO)
    diff = int(sum(d_po)) - int(sum(r_po))
    print("d_pos:", int(sum(d_po)), "r_pos:", int(sum(r_po)), "diff:", diff)
    while diff > 0: # While less pos in recipient, add new po tokens    
        new_po = Token(Type.PO, {TF.PRED: B.FALSE})
        network.sets[Set.RECIPIENT].tensor_ops.add_token(new_po)
        diff -= 1
    r_po = network.sets[Set.RECIPIENT].get_mask(Type.PO)
    no_r_pos = int(sum(r_po))
    no_d_pos = int(sum(d_po))
    assert no_r_pos >= no_d_pos

    # Remove any PO->RB connections
    rb_mask = network.sets[Set.RECIPIENT].get_mask(Type.RB)
    po_mask = network.sets[Set.RECIPIENT].get_mask(Type.PO)
    po_indices = torch.where(po_mask)[0]
    rb_indices = torch.where(rb_mask)[0]
    if po_indices.shape[0] > 0 and rb_indices.shape[0] > 0:
        network.sets[Set.RECIPIENT].connections[po_indices[:, None], rb_indices] = B.FALSE
        network.sets[Set.RECIPIENT].connections[rb_indices[:, None], po_indices] = B.FALSE

    # Now add mapping connections
    d_po_i: 'torch.Tensor' = torch.where(d_po == 1)[0] # Get list of indicies for driver POs
    r_po_i: 'torch.Tensor' = torch.where(r_po == 1)[0] # Get list of indicies for recipient POs

    print("d_po_i", d_po_i)
    print("r_po_i", r_po_i)

    # Make sure the driver has some tokens:
    assert (sum(d_po) > 0)

    # Randomly order the tokens
    d_po_i_list = d_po_i.tolist()
    shuffle(d_po_i_list)
    r_po_i_list = r_po_i.tolist()
    print("d_po_i_list", d_po_i_list)
    print("r_po_i_list", r_po_i_list)

    mappings: 'Mappings' = network.mappings[Set.RECIPIENT]
    # Test
    print("mappings", mappings[MappingFields.CONNECTIONS].shape)
    print("recipient", network.recipient().nodes.shape)
    print("driver", driver.nodes.shape)

    for i, d in enumerate(d_po_i_list):
        print("i", i, "d", d)
        r = r_po_i_list[i]
        network.mappings[Set.RECIPIENT][MappingFields.CONNECTIONS][r, d] = 1.0
        network.mappings[Set.RECIPIENT][MappingFields.WEIGHT][r, d] = 0.85

    # --------------------------[ TEST ]--------------------------
    # Make sure no connections to RBs from recipient POs
    assert check_rb_po_connections(network) is True
    mappings.print(d_mask=d_po)
    assert check_weights(network) is True
    assert network.requirements.predication() is True

def test_predication_fails_low_weight(network):
    # Lower a mapping weight below threshold
    mappings = network.mappings[1]  # Set.RECIPIENT == 1
    # Set all mapping weights to 0.5 for driver POs
    mappings.adj_matrix[:, :, 0] = 0.5  # MappingFields.WEIGHT == 0
    assert network.requirements.predication() is False

def test_predication_fails_connected_to_rb(network):
    # Connect a mapped PO to an RB in the recipient
    recipient = network.sets[1]  # Set.RECIPIENT == 1
    # Find a PO and an RB in recipient
    po_mask = recipient.get_mask(2)  # Type.PO == 2
    rb_mask = recipient.get_mask(1)  # Type.RB == 1
    po_indices = torch.where(po_mask)[0]
    rb_indices = torch.where(rb_mask)[0]
    if len(po_indices) == 0 or len(rb_indices) == 0:
        pytest.skip('No PO or RB in recipient set for this symProps')
    # Connect first PO to first RB
    recipient.connections[po_indices[0], rb_indices[0]] = 1
    assert network.requirements.predication() is False 

def check_rb_po_connections(network: 'Network'):
    recipient: 'Recipient' = network.recipient()
    driver: 'Driver'= network.driver()
    mappings: 'Mappings'= network.mappings[Set.RECIPIENT]

    d_po = driver.get_mask(Type.PO)
    # Get mask of recipient tokens that are mapped to by driver POs
    r_map_mask = (mappings[MappingFields.CONNECTIONS][:, d_po] == 1).any(dim=1)
    # Look for connected RB tokens in the recipient
    r_rbs = recipient.get_mask(Type.RB)
    r_connected = (recipient.connections[r_map_mask][:, r_rbs] == 1)
    print("r_connected", r_connected)
    # If any connected RBs, return false. O.w true.
    return not bool(r_connected.any())

def check_weights(network: 'Network'):
    recipient: 'Recipient' = network.recipient()
    mappings: 'Mappings'= network.mappings[Set.RECIPIENT]
    map_cons: 'torch.Tensor' = mappings[MappingFields.CONNECTIONS]
    map_weights: 'torch.Tensor' = mappings[MappingFields.WEIGHT]
    # Get mask of active mappings
    driver_po_mask = network.driver().get_mask(Type.PO)
    active_maps = map_cons[:, driver_po_mask] == 1
    # Get weights for active mappings
    active_weights = map_weights[:, driver_po_mask][active_maps]

    # Get min weight for active mappings
    min_weight = min(active_weights.tolist())
    return bool(min_weight >= 0.8)


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
    assert network.requirements.rel_form() is True


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
    assert network.requirements.rel_form() is False


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
    assert network.requirements.rel_form() is False


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
    assert network.requirements.rel_form() is False

def test_schema_passes(network: 'Network'):
    """
    Test should pass when all schema requirements are met.
    - All mapped tokens have max_map >= 0.7 -> all tokens connect to valid nodes.
    """
    for s in [network.driver(), network.recipient()]:
        s.nodes[:, TF.MAX_MAP] = 0.8

    assert network.requirements.schema() is True


def test_schema_fails_low_max_map(network: 'Network'):
    """
    Test should fail when a token has 0 < max_map < 0.7.
    """
    network.driver().nodes[0, TF.MAX_MAP] = 0.5

    assert network.requirements.schema() is False


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

    assert network.requirements.schema() is False


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

    assert network.requirements.schema() is False


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

    assert network.requirements.rel_gen() is True


def test_rel_gen_fails_no_mappings(network: 'Network'):
    """
    Test should fail when no mappings exist.
    """
    # Setup: Mappings are empty by default
    assert network.requirements.rel_gen() is False


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

    assert network.requirements.rel_gen() is False
