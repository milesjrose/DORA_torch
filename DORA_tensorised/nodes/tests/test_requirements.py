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
    network.sets[Set.RECIPIENT].connections[po_mask][:, rb_mask] = B.FALSE
    network.sets[Set.RECIPIENT].connections[rb_mask][:, po_mask] = B.FALSE

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
