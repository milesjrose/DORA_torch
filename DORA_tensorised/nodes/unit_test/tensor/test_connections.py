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


# =====================[ get_count tests ]======================

def test_get_count_basic(connections_tensor):
    """Test basic get_count functionality."""
    # Initial count should match the connections tensor size
    count = connections_tensor.get_count()
    assert count == 10  # From mock_connections fixture
    assert isinstance(count, int)


def test_get_count_after_expansion(connections_tensor):
    """Test get_count after expanding the connections tensor."""
    initial_count = connections_tensor.get_count()
    assert initial_count == 10
    
    # Expand to larger size
    connections_tensor.expand_to(15)
    new_count = connections_tensor.get_count()
    assert new_count == 15
    assert new_count > initial_count


def test_get_count_after_shrinking(connections_tensor):
    """Test get_count after shrinking the connections tensor."""
    initial_count = connections_tensor.get_count()
    assert initial_count == 10
    
    # Shrink to smaller size
    connections_tensor.expand_to(5)
    new_count = connections_tensor.get_count()
    assert new_count == 5
    assert new_count < initial_count


def test_get_count_empty_connections():
    """Test get_count with an empty connections tensor."""
    empty_connections = torch.zeros((0, 0), dtype=torch.bool)
    empty_connections_tensor = Connections_Tensor(empty_connections)
    
    count = empty_connections_tensor.get_count()
    assert count == 0
    assert isinstance(count, int)


def test_get_count_after_operations(connections_tensor):
    """Test get_count remains consistent after operations."""
    initial_count = connections_tensor.get_count()
    
    # Perform various operations
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect_bi(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.del_connections(torch.tensor([4]))
    
    # Count should remain the same (operations don't change tensor size)
    count_after_ops = connections_tensor.get_count()
    assert count_after_ops == initial_count


# =====================[ get_view tests ]======================

def test_get_view_basic(connections_tensor):
    """Test basic view creation and access."""
    # Create some connections first
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    indices = torch.tensor([0, 2, 4])
    view = connections_tensor.get_view(indices)
    
    # Verify view properties
    assert view.shape == (3, 10)  # 3 rows (selected indices), 10 columns (all connections)
    assert len(view) == 3
    assert view.dtype == connections_tensor.connections.dtype
    
    # Verify we can read from the view
    assert view[0, 1] == True  # connection[0, 1] should be True
    assert view[1, 3] == True  # connection[2, 3] should be True (view index 1 maps to original index 2)


def test_get_view_updates_original(connections_tensor):
    """Test that modifications to the view update the original tensor."""
    indices = torch.tensor([0, 1, 2])
    view = connections_tensor.get_view(indices)
    
    # Store original values
    original_0_1 = connections_tensor.connections[0, 1].item()
    original_1_2 = connections_tensor.connections[1, 2].item()
    
    # Modify through view
    view[0, 1] = True
    view[1, 2] = True
    
    # Verify original tensor was updated
    assert connections_tensor.connections[0, 1] == True
    assert connections_tensor.connections[1, 2] == True
    
    # Restore original values
    view[0, 1] = original_0_1
    view[1, 2] = original_1_2


def test_get_view_empty_indices(connections_tensor):
    """Test get_view with empty indices."""
    empty_indices = torch.tensor([], dtype=torch.long)
    view = connections_tensor.get_view(empty_indices)
    
    assert len(view) == 0
    assert view.shape == (0, 10)  # 0 rows, 10 columns
    from nodes.network.tokens.tensor_view import TensorView
    assert isinstance(view, TensorView)


def test_get_view_single_index(connections_tensor):
    """Test get_view with a single index."""
    # Create a connection first
    connections_tensor.connect(torch.tensor([5]), torch.tensor([6]))
    
    indices = torch.tensor([5])
    view = connections_tensor.get_view(indices)
    
    assert len(view) == 1
    assert view.shape == (1, 10)
    
    # Verify we can access the single row
    assert view[0, 6] == True  # connection[5, 6] should be True
    
    # Modify through view
    view[0, 7] = True
    assert connections_tensor.connections[5, 7] == True


def test_get_view_all_indices(connections_tensor):
    """Test get_view with all indices."""
    # Create some connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    all_indices = torch.arange(0, connections_tensor.get_count(), dtype=torch.long)
    view = connections_tensor.get_view(all_indices)
    
    assert len(view) == connections_tensor.get_count()
    assert view.shape == (connections_tensor.get_count(), connections_tensor.get_count())
    
    # Verify all connections are accessible
    assert view[0, 1] == True
    assert view[2, 3] == True


def test_get_view_non_contiguous_indices(connections_tensor):
    """Test get_view with non-contiguous indices."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([4]))
    connections_tensor.connect(torch.tensor([7]), torch.tensor([8]))
    
    indices = torch.tensor([0, 3, 7])
    view = connections_tensor.get_view(indices)
    
    assert len(view) == 3
    
    # Verify mapping is correct
    assert view[0, 1] == True  # connection[0, 1]
    assert view[1, 4] == True  # connection[3, 4] (view index 1 maps to original index 3)
    assert view[2, 8] == True  # connection[7, 8] (view index 2 maps to original index 7)


def test_get_view_reordered_indices(connections_tensor):
    """Test get_view with reordered indices."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([6]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    indices = torch.tensor([2, 5, 0])  # Reverse order
    view = connections_tensor.get_view(indices)
    
    # View index 0 should map to original index 2
    assert view[0, 3] == True  # connection[2, 3]
    # View index 1 should map to original index 5
    assert view[1, 6] == True  # connection[5, 6]
    # View index 2 should map to original index 0
    assert view[2, 1] == True  # connection[0, 1]


def test_get_view_slice_indexing(connections_tensor):
    """Test slice indexing on the view."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([1]), torch.tensor([2]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    
    indices = torch.tensor([0, 1, 2, 5, 6])
    view = connections_tensor.get_view(indices)
    
    # Get a slice of the view
    sub_view = view[0:3]  # Should map to original indices 0, 1, 2
    
    # Modify through sub-view
    sub_view[0, 4] = True
    sub_view[1, 5] = True
    sub_view[2, 6] = True
    
    # Verify original tensor was updated
    assert connections_tensor.connections[0, 4] == True
    assert connections_tensor.connections[1, 5] == True
    assert connections_tensor.connections[2, 6] == True


def test_get_view_boolean_mask(connections_tensor):
    """Test boolean mask indexing on the view."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([6]))
    
    indices = torch.tensor([0, 1, 2, 5, 6])
    view = connections_tensor.get_view(indices)
    
    # Create a mask local to the view
    mask = torch.tensor([True, False, True, False, True])
    
    # Get values using mask
    masked_view = view[mask]
    assert len(masked_view) == 3
    
    # Set values using mask
    view[mask, 7] = True
    
    # Verify original tensor was updated (indices 0, 2, 6)
    assert connections_tensor.connections[0, 7] == True
    assert connections_tensor.connections[2, 7] == True
    assert connections_tensor.connections[6, 7] == True


def test_get_view_broadcast_assignment(connections_tensor):
    """Test broadcast assignment through view."""
    indices = torch.tensor([0, 1, 2, 3, 4])
    view = connections_tensor.get_view(indices)
    
    # Broadcast a single value to all connections in view
    view[:, 5] = True
    
    # Verify all were updated
    for idx in indices:
        assert connections_tensor.connections[idx, 5] == True


def test_get_view_column_access(connections_tensor):
    """Test accessing columns through view."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([3]), torch.tensor([1]))
    
    indices = torch.tensor([0, 2, 3])
    view = connections_tensor.get_view(indices)
    
    # Access column 1 (all connections to node 1)
    column_1 = view[:, 1]
    assert len(column_1) == 3
    # All should be True (all connect to node 1)
    assert torch.all(column_1 == True)


def test_get_view_nested_views(connections_tensor):
    """Test nested views (view of a view)."""
    # Create connections
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    connections_tensor.connect(torch.tensor([2]), torch.tensor([3]))
    connections_tensor.connect(torch.tensor([5]), torch.tensor([6]))
    
    indices = torch.tensor([0, 1, 2, 5, 6, 7, 8])
    view1 = connections_tensor.get_view(indices)
    
    # Create a nested view
    # view1 maps: [0,1,2,3,4,5,6] -> original [0,1,2,5,6,7,8]
    # To get original indices [0, 2, 5], we need view1 indices [0, 2, 3]
    # because view1[0]=original[0], view1[2]=original[2], view1[3]=original[5]
    nested_indices = torch.tensor([0, 2, 3])  # Indices into view1 that map to original[0, 2, 5]
    view2 = view1[nested_indices]
    
    # view2[0] should map to view1[0] which maps to original[0]
    assert view2[0, 1] == True  # connection[0, 1]
    # view2[1] should map to view1[2] which maps to original[2]
    assert view2[1, 3] == True  # connection[2, 3]
    # view2[2] should map to view1[3] which maps to original[5]
    assert view2[2, 6] == True  # connection[5, 6]


def test_get_view_clone(connections_tensor):
    """Test cloning a view."""
    # Create a connection
    connections_tensor.connect(torch.tensor([0]), torch.tensor([1]))
    
    indices = torch.tensor([0, 2, 5])
    view = connections_tensor.get_view(indices)
    
    # Clone the view - this creates a copy of the data at the time of cloning
    cloned = view.clone()
    
    # Cloned should be a tensor, not a view
    assert isinstance(cloned, torch.Tensor)
    assert cloned.shape == view.shape
    
    # Verify clone has the same initial values
    assert cloned[0, 1].item() == True
    
    # Modify cloned - should not affect original
    cloned[0, 1] = False
    assert connections_tensor.connections[0, 1] == True  # Original unchanged
    
    # Modify original through view at a different location - clone should remain unchanged
    # This verifies that clone is independent
    view[0, 2] = True  # Modify original at [0, 2]
    assert cloned[0, 2].item() == False  # Clone unchanged (still has original False value)
    
    # Verify modifying clone doesn't affect original
    cloned[0, 3] = True
    assert connections_tensor.connections[0, 3] == False  # Original unchanged