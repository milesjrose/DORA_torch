# nodes/unit_test/tensor/test_connections.py
# Tests for Connections_Tensor class

import pytest
import torch
from nodes.network.tokens.connections.connections import Connections_Tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor (boolean)."""
    size = 10
    connections = torch.zeros((size, size), dtype=torch.bool)
    return connections


@pytest.fixture
def connections_tensor(mock_connections):
    """Create a Connections_Tensor instance."""
    return Connections_Tensor(mock_connections)


def test_connections_tensor_init(connections_tensor, mock_connections):
    """Test Connections_Tensor initialization."""
    assert torch.equal(connections_tensor.connections, mock_connections)
    assert connections_tensor.connections.dtype == torch.bool


def test_connections_tensor_init_requires_bool():
    """Test that Connections_Tensor requires boolean tensor."""
    float_connections = torch.zeros((5, 5), dtype=torch.float32)
    with pytest.raises(AssertionError, match="Connections tensor must be a boolean tensor"):
        Connections_Tensor(float_connections)


def test_connect(connections_tensor):
    """Test connecting parent to child."""
    # Connect parent 0 to child 1
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    
    # Verify connection
    assert connections_tensor.connections[0, 1] == True
    assert connections_tensor.connections[1, 0] == False  # Not bidirectional
    
    # Connect multiple parents to multiple children
    connections_tensor.connect(torch.tensor([2, 3]), torch.tensor([4, 5]))
    assert connections_tensor.connections[2, 4] == True
    assert connections_tensor.connections[2, 5] == True
    assert connections_tensor.connections[3, 4] == True
    assert connections_tensor.connections[3, 5] == True


def test_connect_with_value(connections_tensor):
    """Test connecting with specific value."""
    # Connect with True
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]), value=True)
    assert connections_tensor.connections[0, 1] == True
    
    # Disconnect with False
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]), value=False)
    assert connections_tensor.connections[0, 1] == False


def test_connect_bi(connections_tensor):
    """Test bidirectional connection."""
    # Connect 0 and 1 bidirectionally
    connections_tensor.connect_bi(torch.tensor([0]), torch.tensor([1]))
    
    # Verify both directions
    assert connections_tensor.connections[0, 1] == True
    assert connections_tensor.connections[1, 0] == True
    
    # Connect multiple nodes bidirectionally
    connections_tensor.connect_bi(torch.tensor([2, 3]), torch.tensor([4, 5]))
    # Should create all combinations
    assert connections_tensor.connections[2, 4] == True
    assert connections_tensor.connections[4, 2] == True
    assert connections_tensor.connections[2, 5] == True
    assert connections_tensor.connections[5, 2] == True
    assert connections_tensor.connections[3, 4] == True
    assert connections_tensor.connections[4, 3] == True
    assert connections_tensor.connections[3, 5] == True
    assert connections_tensor.connections[5, 3] == True


def test_connect_bi_with_value(connections_tensor):
    """Test bidirectional connection with specific value."""
    # Connect bidirectionally
    connections_tensor.connect_bi(torch.tensor([0]), torch.tensor([1]), value=True)
    assert connections_tensor.connections[0, 1] == True
    assert connections_tensor.connections[1, 0] == True
    
    # Disconnect bidirectionally
    connections_tensor.connect_bi(torch.tensor([0]), torch.tensor([1]), value=False)
    assert connections_tensor.connections[0, 1] == False
    assert connections_tensor.connections[1, 0] == False


def test_get_parents(connections_tensor):
    """Test getting parents of a child."""
    # Set up connections: 0->2, 1->2, 3->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    
    # Get parents of child 2
    parents = connections_tensor.get_parents(torch.tensor([2]))
    # Should return [0, 1]
    assert len(parents) == 2
    assert 0 in parents
    assert 1 in parents
    
    # Get parents of child 4
    parents = connections_tensor.get_parents(torch.tensor([4]))
    assert len(parents) == 1
    assert 3 in parents


def test_get_parents_multiple_children(connections_tensor):
    """Test getting parents of multiple children."""
    # Set up: 0->2, 1->2, 0->3, 1->3
    connections_tensor.connect(torch.tensor([0, 1]), torch.tensor([2, 3]))
    
    # Get parents of children [2, 3]
    parents = connections_tensor.get_parents(torch.tensor([2, 3]))
    # Should return all unique parents: [0, 1]
    assert len(parents) == 2
    assert 0 in parents
    assert 1 in parents


def test_get_parents_no_parents(connections_tensor):
    """Test getting parents when child has no parents."""
    # Child 5 has no connections
    parents = connections_tensor.get_parents(torch.tensor([5]))
    assert len(parents) == 0


def test_get_children(connections_tensor):
    """Test getting children of a parent."""
    # Set up connections: 0->2, 0->3, 1->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([4]))
    
    # Get children of parent 0
    children = connections_tensor.get_children(torch.tensor([0]))
    # Should return [2, 3]
    assert len(children) == 2
    assert 2 in children
    assert 3 in children
    
    # Get children of parent 1
    children = connections_tensor.get_children(torch.tensor([1]))
    assert len(children) == 1
    assert 4 in children


def test_get_children_multiple_parents(connections_tensor):
    """Test getting children of multiple parents."""
    # Set up: 0->2, 0->3, 1->2, 1->3
    connections_tensor.connect(torch.tensor([0, 1]), torch.tensor([2, 3]))
    
    # Get children of parents [0, 1]
    children = connections_tensor.get_children(torch.tensor([0, 1]))
    # Should return all unique children: [2, 3]
    assert len(children) == 2
    assert 2 in children
    assert 3 in children


def test_get_children_no_children(connections_tensor):
    """Test getting children when parent has no children."""
    # Parent 5 has no connections
    children = connections_tensor.get_children(torch.tensor([5]))
    assert len(children) == 0


def test_get_all_connected(connections_tensor):
    """Test getting all connected nodes (parents and children)."""
    # Set up: 0->2, 1->2, 2->3
    # So node 2 has parent 0, parent 1, and child 3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get all connected to node 2
    connected = connections_tensor.get_all_connected(torch.tensor([2]))
    # Should return [0, 1, 3] (parents and children)
    assert len(connected) == 3
    assert 0 in connected
    assert 1 in connected
    assert 3 in connected


def test_get_all_connected_no_connections(connections_tensor):
    """Test getting all connected when node has no connections."""
    # Node 5 has no connections
    connected = connections_tensor.get_all_connected(torch.tensor([5]))
    assert len(connected) == 0


def test_get_all_connected_bidirectional(connections_tensor):
    """Test get_all_connected with bidirectional connections."""
    # Set up bidirectional: 0<->1, 1<->2
    connections_tensor.connect_bi(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect_bi(torch.tensor([1]), torch.tensor([2]))
    
    # Get all connected to node 1
    connected = connections_tensor.get_all_connected(torch.tensor([1]))
    # Should return [0, 2] (both directions)
    assert len(connected) == 2
    assert 0 in connected
    assert 2 in connected


def test_expand_to(connections_tensor):
    """Test expanding the connections tensor."""
    original_size = connections_tensor.connections.shape[0]
    
    # Set some connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    
    # Expand to larger size
    new_size = 20
    connections_tensor.expand_to(new_size)
    
    # Verify new size
    assert connections_tensor.connections.shape == (new_size, new_size)
    assert connections_tensor.connections.dtype == torch.bool
    
    # Verify original connections are preserved
    assert connections_tensor.connections[0, 1] == True
    
    # Verify new connections are False
    assert connections_tensor.connections[original_size, original_size] == False


def test_expand_to_smaller_size(connections_tensor):
    """Test that expanding to smaller size still works (but may truncate)."""
    # Set connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([6]))
    
    # Expand to smaller size (should still work, but may lose data)
    new_size = 5
    connections_tensor.expand_to(new_size)
    
    # Verify new size
    assert connections_tensor.connections.shape == (new_size, new_size)
    # Connection at [0, 1] should still be there
    assert connections_tensor.connections[0, 1] == True


def test_complex_connection_graph(connections_tensor):
    """Test a more complex connection graph."""
    # Create a tree structure:
    #     0
    #    / \
    #   1   2
    #  / \   \
    # 3   4   5
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([5]))
    
    # Test getting children of 0
    children = connections_tensor.get_children(torch.tensor([0]))
    assert len(children) == 2
    assert 1 in children
    assert 2 in children
    
    # Test getting children of 1
    children = connections_tensor.get_children(torch.tensor([1]))
    assert len(children) == 2
    assert 3 in children
    assert 4 in children
    
    # Test getting parents of 1
    parents = connections_tensor.get_parents(torch.tensor([1]))
    assert len(parents) == 1
    assert 0 in parents
    
    # Test getting all connected to 1
    connected = connections_tensor.get_all_connected(torch.tensor([1]))
    assert len(connected) == 3  # parent 0, children 3 and 4
    assert 0 in connected
    assert 3 in connected
    assert 4 in connected


def test_get_parents_recursive(connections_tensor):
    """Test getting all parents recursively."""
    # Create a chain: 0->1->2->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get recursive parents of 3 (should get 2, 1, 0)
    parents = connections_tensor.get_parents_recursive(torch.tensor([3]))
    assert len(parents) == 3
    assert 0 in parents
    assert 1 in parents
    assert 2 in parents
    
    # Get recursive parents of 2 (should get 1, 0)
    parents = connections_tensor.get_parents_recursive(torch.tensor([2]))
    assert len(parents) == 2
    assert 0 in parents
    assert 1 in parents


def test_get_parents_recursive_branching(connections_tensor):
    """Test recursive parents with branching structure."""
    # Create: 0->1, 0->2, 1->3, 2->3
    # So 3 has parents 1 and 2, which both have parent 0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get recursive parents of 3 (should get 1, 2, 0)
    parents = connections_tensor.get_parents_recursive(torch.tensor([3]))
    assert len(parents) == 3
    assert 0 in parents
    assert 1 in parents
    assert 2 in parents


def test_get_parents_recursive_handles_cycles(connections_tensor):
    """Test that recursive parents handles cycles correctly."""
    # Create a cycle: 0->1->2->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([0]))
    
    # Should not get stuck in infinite loop
    parents = connections_tensor.get_parents_recursive(torch.tensor([2]))
    # Should get 1 and 0 (but not 2 itself)
    assert len(parents) == 2
    assert 0 in parents
    assert 1 in parents
    assert 2 not in parents


def test_get_parents_recursive_empty(connections_tensor):
    """Test recursive parents with empty input."""
    parents = connections_tensor.get_parents_recursive(torch.tensor([], dtype=torch.long))
    assert len(parents) == 0


def test_get_children_recursive(connections_tensor):
    """Test getting all children recursively."""
    # Create a chain: 0->1->2->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get recursive children of 0 (should get 1, 2, 3)
    children = connections_tensor.get_children_recursive(torch.tensor([0]))
    assert len(children) == 3
    assert 1 in children
    assert 2 in children
    assert 3 in children
    
    # Get recursive children of 1 (should get 2, 3)
    children = connections_tensor.get_children_recursive(torch.tensor([1]))
    assert len(children) == 2
    assert 2 in children
    assert 3 in children


def test_get_children_recursive_branching(connections_tensor):
    """Test recursive children with branching structure."""
    # Create: 0->1, 0->2, 1->3, 2->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get recursive children of 0 (should get 1, 2, 3)
    children = connections_tensor.get_children_recursive(torch.tensor([0]))
    assert len(children) == 3
    assert 1 in children
    assert 2 in children
    assert 3 in children


def test_get_children_recursive_handles_cycles(connections_tensor):
    """Test that recursive children handles cycles correctly."""
    # Create a cycle: 0->1->2->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([0]))
    
    # Should not get stuck in infinite loop
    children = connections_tensor.get_children_recursive(torch.tensor([0]))
    # Should get 1 and 2 (but not 0 itself)
    assert len(children) == 2
    assert 1 in children
    assert 2 in children
    assert 0 not in children


def test_get_children_recursive_empty(connections_tensor):
    """Test recursive children with empty input."""
    children = connections_tensor.get_children_recursive(torch.tensor([], dtype=torch.long))
    assert len(children) == 0


def test_get_all_connected_recursive(connections_tensor):
    """Test getting all connected nodes recursively."""
    # Create: 0->1->2, 3->1
    # So 1 has parent 0 and 3, and child 2
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    
    # Get all recursive connections of 1
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([1]))
    # Should get 0, 2, 3 (parents and children, but not 1 itself)
    assert len(connected) == 3
    assert 0 in connected
    assert 2 in connected
    assert 3 in connected
    assert 1 not in connected


def test_get_all_connected_recursive_chain(connections_tensor):
    """Test recursive connections in a chain."""
    # Create chain: 0->1->2->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get all recursive connections of 1
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([1]))
    # Should get 0 (parent), 2, 3 (descendants)
    assert len(connected) == 3
    assert 0 in connected
    assert 2 in connected
    assert 3 in connected


def test_get_connected_set(connections_tensor):
    """Test getting the connected set (connected component)."""
    # Create two disconnected components:
    # Component 1: 0->1->2
    # Component 2: 3->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    
    # Get connected set starting from 0
    connected = connections_tensor.get_connected_set(torch.tensor([0]))
    # Should get 0, 1, 2 (all in the same component)
    assert len(connected) == 3
    assert 0 in connected
    assert 1 in connected
    assert 2 in connected
    assert 3 not in connected
    assert 4 not in connected
    
    # Get connected set starting from 3
    connected = connections_tensor.get_connected_set(torch.tensor([3]))
    # Should get 3, 4
    assert len(connected) == 2
    assert 3 in connected
    assert 4 in connected


def test_get_connected_set_bidirectional(connections_tensor):
    """Test connected set with bidirectional connections."""
    # Create: 0<->1<->2, 3->4
    connections_tensor.connect_bi(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect_bi(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    
    # Get connected set starting from 0
    connected = connections_tensor.get_connected_set(torch.tensor([0]))
    # Should get 0, 1, 2
    assert len(connected) == 3
    assert 0 in connected
    assert 1 in connected
    assert 2 in connected


def test_get_connected_set_handles_cycles(connections_tensor):
    """Test that connected set handles cycles correctly."""
    # Create a cycle: 0->1->2->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([0]))
    
    # Should not get stuck in infinite loop
    connected = connections_tensor.get_connected_set(torch.tensor([0]))
    # Should get all three nodes
    assert len(connected) == 3
    assert 0 in connected
    assert 1 in connected
    assert 2 in connected


def test_get_connected_set_multiple_starting_points(connections_tensor):
    """Test connected set with multiple starting points."""
    # Create: 0->1->2, 3->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    
    # Get connected set starting from [0, 3]
    connected = connections_tensor.get_connected_set(torch.tensor([0, 3]))
    # Should get all nodes from both components: 0, 1, 2, 3, 4
    assert len(connected) == 5
    assert 0 in connected
    assert 1 in connected
    assert 2 in connected
    assert 3 in connected
    assert 4 in connected


def test_get_connected_set_empty(connections_tensor):
    """Test connected set with empty input."""
    connected = connections_tensor.get_connected_set(torch.tensor([], dtype=torch.long))
    assert len(connected) == 0


def test_get_connected_set_isolated_node(connections_tensor):
    """Test connected set with isolated node."""
    # Node 5 has no connections
    connected = connections_tensor.get_connected_set(torch.tensor([5]))
    # Should only return the node itself
    assert len(connected) == 1
    assert 5 in connected


# Additional tests for get_parents_recursive
def test_get_parents_recursive_multiple_starting_nodes(connections_tensor):
    """Test recursive parents with multiple starting nodes."""
    # Create: 0->1->2, 3->4->2
    # So 2 has parents 1 and 4, which have parents 0 and 3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([4]), torch.tensor([2]))
    
    # Get recursive parents of [1, 4]
    parents = connections_tensor.get_parents_recursive(torch.tensor([1, 4]))
    # Should get 0 and 3 (but not 1 or 4 themselves)
    assert len(parents) == 2
    assert 0 in parents
    assert 3 in parents
    assert 1 not in parents
    assert 4 not in parents


def test_get_parents_recursive_complex_cycle(connections_tensor):
    """Test recursive parents with a more complex cycle structure."""
    # Create: 0->1->2->3->1 (cycle through 1,2,3), 4->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([4]), torch.tensor([0]))
    
    # Get recursive parents of 2
    parents = connections_tensor.get_parents_recursive(torch.tensor([2]))
    # Should get 1, 3, 0, 4 (all ancestors, but not 2 itself)
    assert len(parents) == 4
    assert 0 in parents
    assert 1 in parents
    assert 3 in parents
    assert 4 in parents
    assert 2 not in parents


def test_get_parents_recursive_no_parents(connections_tensor):
    """Test recursive parents when node has no parents."""
    # Node 5 has no connections
    parents = connections_tensor.get_parents_recursive(torch.tensor([5]))
    assert len(parents) == 0


# Additional tests for get_children_recursive
def test_get_children_recursive_multiple_starting_nodes(connections_tensor):
    """Test recursive children with multiple starting nodes."""
    # Create: 0->1->2, 0->3->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    
    # Get recursive children of [1, 3]
    children = connections_tensor.get_children_recursive(torch.tensor([1, 3]))
    # Should get 2 and 4 (but not 1 or 3 themselves)
    assert len(children) == 2
    assert 2 in children
    assert 4 in children
    assert 1 not in children
    assert 3 not in children


def test_get_children_recursive_complex_cycle(connections_tensor):
    """Test recursive children with a more complex cycle structure."""
    # Create: 0->1->2->3->1 (cycle through 1,2,3), 0->4
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([4]))
    
    # Get recursive children of 0
    children = connections_tensor.get_children_recursive(torch.tensor([0]))
    # Should get 1, 2, 3, 4 (all descendants, but not 0 itself)
    assert len(children) == 4
    assert 1 in children
    assert 2 in children
    assert 3 in children
    assert 4 in children
    assert 0 not in children


def test_get_children_recursive_no_children(connections_tensor):
    """Test recursive children when node has no children."""
    # Node 5 has no connections
    children = connections_tensor.get_children_recursive(torch.tensor([5]))
    assert len(children) == 0


# Additional tests for get_all_connected_recursive
def test_get_all_connected_recursive_handles_cycles(connections_tensor):
    """Test recursive connections with cycles."""
    # Create a cycle: 0->1->2->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([0]))
    
    # Get all recursive connections of 0
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([0]))
    # Should get 1 and 2 (but not 0 itself)
    assert len(connected) == 2
    assert 1 in connected
    assert 2 in connected
    assert 0 not in connected


def test_get_all_connected_recursive_multiple_starting_nodes(connections_tensor):
    """Test recursive connections with multiple starting nodes."""
    # Create: 0->1->2, 3->4->2, 2->5
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([4]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([5]))
    
    # Get all recursive connections of [1, 4]
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([1, 4]))
    # Should get 0, 2, 3, 5 (all ancestors and descendants, but not 1 or 4 themselves)
    assert len(connected) == 4
    assert 0 in connected
    assert 2 in connected
    assert 3 in connected
    assert 5 in connected
    assert 1 not in connected
    assert 4 not in connected


def test_get_all_connected_recursive_empty(connections_tensor):
    """Test recursive connections with empty input."""
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([], dtype=torch.long))
    assert len(connected) == 0


def test_get_all_connected_recursive_isolated_node(connections_tensor):
    """Test recursive connections with isolated node."""
    # Node 5 has no connections
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([5]))
    assert len(connected) == 0


def test_get_all_connected_recursive_complex_graph(connections_tensor):
    """Test recursive connections with a complex graph structure."""
    # Create: 0->1->2, 0->3->2, 2->4, 5->1, 6->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([6]), torch.tensor([3]))
    
    # Get all recursive connections of 2
    connected = connections_tensor.get_all_connected_recursive(torch.tensor([2]))
    # Should get all ancestors (0, 1, 3, 5, 6) and descendants (4), but not 2 itself
    assert len(connected) == 6
    assert 0 in connected
    assert 1 in connected
    assert 3 in connected
    assert 4 in connected
    assert 5 in connected
    assert 6 in connected
    assert 2 not in connected


# Additional tests for get_connected_set
def test_get_connected_set_complex_graph(connections_tensor):
    """Test connected set with a complex graph structure."""
    # Create: 0->1->2, 0->3->2, 2->4, 5->1, 6->3
    # All nodes 0-6 are in one connected component
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([6]), torch.tensor([3]))
    
    # Get connected set starting from 2
    connected = connections_tensor.get_connected_set(torch.tensor([2]))
    # Should get all nodes 0-6
    assert len(connected) == 7
    for i in range(7):
        assert i in connected


def test_get_connected_set_diamond_structure(connections_tensor):
    """Test connected set with diamond structure."""
    # Create diamond: 0->1, 0->2, 1->3, 2->3
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([0]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    # Get connected set starting from 0
    connected = connections_tensor.get_connected_set(torch.tensor([0]))
    # Should get all nodes 0-3
    assert len(connected) == 4
    for i in range(4):
        assert i in connected
    
    # Get connected set starting from 3
    connected = connections_tensor.get_connected_set(torch.tensor([3]))
    # Should also get all nodes 0-3
    assert len(connected) == 4
    for i in range(4):
        assert i in connected


def test_get_connected_set_large_cycle(connections_tensor):
    """Test connected set with a larger cycle."""
    # Create cycle: 0->1->2->3->4->0
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([4]), torch.tensor([0]))
    
    # Get connected set starting from any node should get all
    connected = connections_tensor.get_connected_set(torch.tensor([0]))
    assert len(connected) == 5
    for i in range(5):
        assert i in connected
    
    connected = connections_tensor.get_connected_set(torch.tensor([2]))
    assert len(connected) == 5
    for i in range(5):
        assert i in connected
