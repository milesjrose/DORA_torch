import pytest
import torch
from nodes.entropy_net import EntropyNet, Ext, BF

@pytest.fixture
def entropy_net():
    """Returns a new EntropyNet instance for each test."""
    return EntropyNet()

def test_fillin_structure(entropy_net):
    """
    Tests if the fillin method creates tensors of the correct shape and type.
    """
    extent1, extent2 = 5, 10
    entropy_net.fillin(extent1, extent2)

    assert isinstance(entropy_net.nodes, torch.Tensor)
    assert isinstance(entropy_net.connections, torch.Tensor)

    # num_in should be max(extent1, extent2)
    assert entropy_net.num_in == 10
    # nodes tensor should have shape [num_in + 2 outputs, num_features]
    assert entropy_net.nodes.shape == (12, len(BF))
    # connections tensor should have shape [num_outputs, num_inputs]
    assert entropy_net.connections.shape == (2, 10)

def test_fillin_connections(entropy_net):
    """
    Tests if the fillin method sets up the connections correctly.
    """
    extent1, extent2 = 5, 10
    entropy_net.fillin(extent1, extent2)

    small_extent_connections = entropy_net.connections[Ext.SMALL, :]
    large_extent_connections = entropy_net.connections[Ext.LARGE, :]

    # The SMALL extent (5) should have 5 active connections
    assert torch.sum(small_extent_connections) == 5
    assert torch.all(small_extent_connections[:5] == 1.0)
    assert torch.all(small_extent_connections[5:] == 0.0)

    # The LARGE extent (10) should have 10 active connections
    assert torch.sum(large_extent_connections) == 10
    assert torch.all(large_extent_connections == 1.0)

def test_run_entropy_net_large_wins(entropy_net):
    """
    Tests that the output node for the larger extent 'wins' (has higher activation).
    """
    entropy_net.fillin(5, 10)
    entropy_net.run_entropy_net()

    large_extent_act = entropy_net.nodes[Ext.LARGE, BF.ACT]
    small_extent_act = entropy_net.nodes[Ext.SMALL, BF.ACT]

    assert large_extent_act > small_extent_act
    # The winning node's activation should be high (equilibrium is ~0.825)
    assert large_extent_act > 0.8
    # The losing node's activation should be close to 0.0
    assert torch.isclose(small_extent_act, torch.tensor(0.0), atol=1e-3)

def test_run_entropy_net_small_wins_when_larger(entropy_net):
    """
    Tests that the 'SMALL' enum output node still wins if it has the larger extent.
    """
    entropy_net.fillin(10, 5) # extent1 > extent2
    entropy_net.run_entropy_net()

    large_extent_act = entropy_net.nodes[Ext.LARGE, BF.ACT]
    small_extent_act = entropy_net.nodes[Ext.SMALL, BF.ACT]
    
    # The 'LARGE' output node (which corresponds to max(10, 5)) should win.
    assert large_extent_act > small_extent_act
    # The winning node's activation should be high (equilibrium is ~0.825)
    assert large_extent_act > 0.8
    # The losing node's activation should be close to 0.0
    assert torch.isclose(small_extent_act, torch.tensor(0.0), atol=1e-3)

def test_run_entropy_net_equal_extents(entropy_net):
    """
    Tests that for equal extents, the activations are nearly equal.
    """
    entropy_net.fillin(10, 10)
    entropy_net.run_entropy_net()

    large_extent_act = entropy_net.nodes[Ext.LARGE, BF.ACT]
    small_extent_act = entropy_net.nodes[Ext.SMALL, BF.ACT]

    # Activations should be very close to each other
    assert torch.isclose(large_extent_act, small_extent_act, atol=1e-5)
    # Neither should be fully 0 or 1, but some intermediate value
    assert 0.0 < large_extent_act < 1.0
