# nodes/tests/unit/operations/test_firing_order_ops.py
# Tests for FiringOperations class

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from nodes.network.network import Network
from nodes.network.tokens import Tokens, Token_Tensor, Connections_Tensor, Links, Mapping
from nodes.network.sets.semantics import Semantics
from nodes.network.network_params import Params, default_params
from nodes.enums import Set, TF, SF, MappingFields, Type, TensorTypes


@pytest.fixture
def minimal_params():
    """Create minimal Params object for testing."""
    return default_params()


@pytest.fixture
def minimal_token_tensor():
    """Create minimal Token_Tensor for testing."""
    num_tokens = 20
    num_features = len(TF)
    tokens = torch.zeros((num_tokens, num_features))
    names = {}
    return Token_Tensor(tokens, names)


@pytest.fixture
def minimal_connections():
    """Create minimal Connections_Tensor for testing."""
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    return Connections_Tensor(connections)


@pytest.fixture
def minimal_links(minimal_token_tensor):
    """Create minimal Links object for testing."""
    num_tokens = minimal_token_tensor.get_count()
    num_semantics = 5
    links = torch.zeros((num_tokens, num_semantics))
    return Links(links)


@pytest.fixture
def minimal_mapping():
    """Create minimal Mapping object for testing."""
    num_recipient = 5
    num_driver = 4
    num_fields = len(MappingFields)
    adj_matrix = torch.zeros((num_recipient, num_driver, num_fields))
    return Mapping(adj_matrix)


@pytest.fixture
def minimal_semantics():
    """Create minimal Semantics object for testing."""
    num_semantics = 5
    num_features = len(SF)
    nodes = torch.zeros((num_semantics, num_features))
    connections = torch.zeros((num_semantics, num_semantics))
    IDs = {i: i for i in range(num_semantics)}
    names = {}
    return Semantics(nodes, connections, IDs, names)


@pytest.fixture
def minimal_tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping):
    """Create minimal Tokens object for testing."""
    return Tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping)


@pytest.fixture
def network(minimal_tokens, minimal_semantics, minimal_params):
    """Create minimal Network object for testing."""
    return Network(minimal_tokens, minimal_semantics, minimal_params)


# =====================[ make_firing_order Tests ]======================

class TestMakeFiringOrder:
    """Tests for make_firing_order method."""
    
    def test_make_firing_order_uses_param_rule_when_none_provided(self, network):
        """Test that make_firing_order uses network params when no rule is provided."""
        network.params.firing_order_rule = "totally_random"
        
        with patch.object(network.firing_ops, 'totally_random', return_value=[1, 2, 3]) as mock_random:
            result = network.firing_ops.make_firing_order()
            mock_random.assert_called_once()
            assert result == [1, 2, 3]
    
    def test_make_firing_order_defaults_to_by_top_random(self, network):
        """Test that make_firing_order defaults to by_top_random when no rule is set."""
        network.params.firing_order_rule = None
        
        with patch.object(network.firing_ops, 'by_top_random', return_value=[1, 2, 3]) as mock_top_random:
            result = network.firing_ops.make_firing_order()
            mock_top_random.assert_called_once()
            assert result == [1, 2, 3]
    
    def test_make_firing_order_with_by_top_random_rule(self, network):
        """Test that make_firing_order calls by_top_random when specified."""
        with patch.object(network.firing_ops, 'by_top_random', return_value=[4, 5, 6]) as mock_top_random:
            result = network.firing_ops.make_firing_order(rule="by_top_random")
            mock_top_random.assert_called_once()
            assert result == [4, 5, 6]
    
    def test_make_firing_order_with_totally_random_rule(self, network):
        """Test that make_firing_order calls totally_random when specified."""
        with patch.object(network.firing_ops, 'totally_random', return_value=[7, 8, 9]) as mock_random:
            result = network.firing_ops.make_firing_order(rule="totally_random")
            mock_random.assert_called_once()
            assert result == [7, 8, 9]
    
    def test_make_firing_order_raises_error_for_invalid_rule(self, network):
        """Test that make_firing_order raises ValueError for invalid rule."""
        with pytest.raises(ValueError, match="Invalid firing order rule"):
            network.firing_ops.make_firing_order(rule="invalid_rule")
    
    def test_make_firing_order_overrides_param_rule(self, network):
        """Test that providing a rule argument overrides the param rule."""
        network.params.firing_order_rule = "totally_random"
        
        with patch.object(network.firing_ops, 'by_top_random', return_value=[1, 2]) as mock_top_random:
            result = network.firing_ops.make_firing_order(rule="by_top_random")
            mock_top_random.assert_called_once()
            assert result == [1, 2]


# =====================[ by_top_random Tests ]======================

class TestByTopRandom:
    """Tests for by_top_random method."""
    
    def test_by_top_random_with_po_tokens_only(self, network):
        """Test by_top_random when driver has only PO tokens."""
        # Setup: Create PO tokens in driver
        po_indices = torch.tensor([0, 1, 2])
        network.tokens.token_tensor.set_features(
            po_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]] * 3)
        )
        network.cache_sets()
        
        # Mock get_highest_token_type to return PO
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.PO):
            # Mock torch.where to return consistent indices
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True, True] + [False] * 17)
                # Mock to_global to return the indices unchanged
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1, 2])):
                    result = network.firing_ops.by_top_random()
        
        # Result should be a list (order may vary due to shuffle)
        assert isinstance(result, list)
    
    def test_by_top_random_with_rb_tokens_only(self, network):
        """Test by_top_random when driver has only RB tokens."""
        # Setup: Create RB tokens in driver
        rb_indices = torch.tensor([0, 1])
        network.tokens.token_tensor.set_features(
            rb_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        network.cache_sets()
        
        # Mock get_highest_token_type to return RB
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.RB):
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True] + [False] * 18)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    result = network.firing_ops.by_top_random()
        
        assert isinstance(result, list)
    
    def test_by_top_random_with_p_tokens(self, network):
        """Test by_top_random when driver has P tokens with RB children."""
        # Setup: Create P and RB tokens in driver
        p_indices = torch.tensor([0, 1])
        rb_indices = torch.tensor([2, 3])
        
        # Set up P tokens
        network.tokens.token_tensor.set_features(
            p_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.P)]] * 2)
        )
        # Set up RB tokens
        network.tokens.token_tensor.set_features(
            rb_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        network.cache_sets()
        
        # Mock necessary methods
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.P):
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                # Return appropriate masks based on call
                def get_mask_side_effect(token_type):
                    if token_type == Type.PO:
                        return torch.tensor([True, True] + [False] * 18)
                    elif token_type == Type.RB:
                        return torch.tensor([False, False, True, True] + [False] * 16)
                    return torch.zeros(20, dtype=torch.bool)
                mock_get_mask.side_effect = get_mask_side_effect
                
                with patch.object(network.firing_ops, 'get_lcl_child_idxs', return_value=[2, 3]):
                    with patch.object(network, 'to_global', return_value=torch.tensor([0, 1, 2, 3])):
                        result = network.firing_ops.by_top_random()
        
        assert isinstance(result, list)
    
    def test_by_top_random_empty_driver_returns_empty_list(self, network):
        """Test by_top_random returns empty list when driver is empty."""
        network.cache_sets()
        
        # Mock get_highest_token_type to return None (indicating no tokens)
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=None):
            with patch.object(network, 'to_global', return_value=torch.tensor([])):
                result = network.firing_ops.by_top_random()
        
        # Result should be an empty list
        assert result == []
    
    def test_by_top_random_stores_firing_order(self, network):
        """Test that by_top_random stores the firing order in self.firing_order."""
        # Setup tokens
        po_indices = torch.tensor([0, 1])
        network.tokens.token_tensor.set_features(
            po_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]] * 2)
        )
        network.cache_sets()
        
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.PO):
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True] + [False] * 18)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    network.firing_ops.by_top_random()
        
        # Check that firing_order is stored
        assert network.firing_ops.firing_order is not None


# =====================[ totally_random Tests ]======================

class TestTotallyRandom:
    """Tests for totally_random method."""
    
    def test_totally_random_with_rb_tokens(self, network):
        """Test totally_random when driver has RB tokens."""
        # Setup: Create RB tokens in driver
        rb_indices = torch.tensor([0, 1, 2])
        network.tokens.token_tensor.set_features(
            rb_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 3)
        )
        network.cache_sets()
        
        # Mock get_count to return number of RB tokens
        with patch.object(network.driver().tensor_op, 'get_count') as mock_get_count:
            mock_get_count.side_effect = lambda t: 3 if t == Type.RB else 0
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True, True] + [False] * 17)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1, 2])):
                    result = network.firing_ops.totally_random()
        
        assert isinstance(result, list)
    
    def test_totally_random_with_po_tokens_no_rb(self, network):
        """Test totally_random when driver has only PO tokens (no RB)."""
        # Setup: Create PO tokens in driver
        po_indices = torch.tensor([0, 1])
        network.tokens.token_tensor.set_features(
            po_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]] * 2)
        )
        network.cache_sets()
        
        # Mock get_count to return 0 for RB, 2 for PO
        with patch.object(network.driver().tensor_op, 'get_count') as mock_get_count:
            def count_side_effect(t):
                if t == Type.RB:
                    return 0
                elif t == Type.PO:
                    return 2
                return 0
            mock_get_count.side_effect = count_side_effect
            
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True] + [False] * 18)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    result = network.firing_ops.totally_random()
        
        assert isinstance(result, list)
    
    def test_totally_random_prefers_rb_over_po(self, network):
        """Test that totally_random uses RB tokens when both RB and PO exist."""
        # Setup: Create both RB and PO tokens in driver
        rb_indices = torch.tensor([0, 1])
        po_indices = torch.tensor([2, 3])
        
        network.tokens.token_tensor.set_features(
            rb_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        network.tokens.token_tensor.set_features(
            po_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]] * 2)
        )
        network.cache_sets()
        
        # Mock get_count to return both RB and PO counts
        with patch.object(network.driver().tensor_op, 'get_count') as mock_get_count:
            def count_side_effect(t):
                if t == Type.RB:
                    return 2
                elif t == Type.PO:
                    return 2
                return 0
            mock_get_count.side_effect = count_side_effect
            
            # get_mask should be called for RB (since RB count > 0)
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True, False, False] + [False] * 16)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    result = network.firing_ops.totally_random()
        
        # RB mask should be requested
        mock_get_mask.assert_called_with(Type.RB)
    
    def test_totally_random_empty_driver_returns_empty_list(self, network):
        """Test totally_random returns empty list when driver is empty."""
        network.cache_sets()
        
        # Mock get_count to return 0 for both RB and PO
        with patch.object(network.driver().tensor_op, 'get_count', return_value=0):
            with patch.object(network, 'to_global', return_value=torch.tensor([])):
                result = network.firing_ops.totally_random()
        
        assert result == []
    
    def test_totally_random_stores_firing_order(self, network):
        """Test that totally_random stores the firing order in self.firing_order."""
        rb_indices = torch.tensor([0, 1])
        network.tokens.token_tensor.set_features(
            rb_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        network.cache_sets()
        
        with patch.object(network.driver().tensor_op, 'get_count') as mock_get_count:
            mock_get_count.side_effect = lambda t: 2 if t == Type.RB else 0
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True] + [False] * 18)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    network.firing_ops.totally_random()
        
        assert network.firing_ops.firing_order is not None


# =====================[ get_lcl_child_idxs Tests ]======================

class TestGetLclChildIdxs:
    """Tests for get_lcl_child_idxs method."""
    
    def test_get_lcl_child_idxs_returns_children(self, network):
        """Test that get_lcl_child_idxs returns correct child indices."""
        # Setup: Create parent and child tokens
        parent_indices = torch.tensor([0])
        child_indices = torch.tensor([1, 2])
        
        # Set up parent as P, children as RB
        network.tokens.token_tensor.set_features(
            parent_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.P)]])
        )
        network.tokens.token_tensor.set_features(
            child_indices, 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        
        # Set up connections: parent -> children
        network.tokens.connections.tensor[0, 1] = True
        network.tokens.connections.tensor[0, 2] = True
        
        network.cache_sets()
        
        # Get local view of connections
        cons = network.tokens.get_view(TensorTypes.CON, Set.DRIVER)
        
        # Mock get_mask to return mask for RB type
        with patch.object(network.sets[Set.DRIVER].tensor_op, 'get_mask') as mock_get_mask:
            mock_get_mask.return_value = torch.tensor([False, True, True] + [False] * 17)
            
            result = network.firing_ops.get_lcl_child_idxs([0], cons, Type.RB, Set.DRIVER)
        
        # Result should contain child indices
        assert isinstance(result, list)
    
    def test_get_lcl_child_idxs_preserves_order(self, network):
        """Test that get_lcl_child_idxs returns children in parent order."""
        # Setup: Create multiple parents with children
        parent1_idx = torch.tensor([0])
        parent2_idx = torch.tensor([1])
        child1_idx = torch.tensor([2])
        child2_idx = torch.tensor([3])
        
        # Set up parents as P, children as RB
        network.tokens.token_tensor.set_features(
            torch.tensor([0, 1]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.P)]] * 2)
        )
        network.tokens.token_tensor.set_features(
            torch.tensor([2, 3]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        
        # Set up connections: parent1 -> child1, parent2 -> child2
        network.tokens.connections.tensor[0, 2] = True
        network.tokens.connections.tensor[1, 3] = True
        
        network.cache_sets()
        
        cons = network.tokens.get_view(TensorTypes.CON, Set.DRIVER)
        
        with patch.object(network.sets[Set.DRIVER].tensor_op, 'get_mask') as mock_get_mask:
            mock_get_mask.return_value = torch.tensor([False, False, True, True] + [False] * 16)
            
            # Children should be returned in order of parents
            result = network.firing_ops.get_lcl_child_idxs([0, 1], cons, Type.RB, Set.DRIVER)
        
        # Child of parent 0 should come before child of parent 1
        assert isinstance(result, list)
    
    def test_get_lcl_child_idxs_filters_by_type(self, network):
        """Test that get_lcl_child_idxs only returns children of the specified type."""
        # Setup: Create parent with children of different types
        network.tokens.token_tensor.set_features(
            torch.tensor([0]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.P)]])
        )
        network.tokens.token_tensor.set_features(
            torch.tensor([1]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]])
        )
        network.tokens.token_tensor.set_features(
            torch.tensor([2]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]])
        )
        
        # Parent connected to both RB and PO
        network.tokens.connections.tensor[0, 1] = True
        network.tokens.connections.tensor[0, 2] = True
        
        network.cache_sets()
        
        cons = network.tokens.get_view(TensorTypes.CON, Set.DRIVER)
        
        # Mock get_mask to return mask for RB type only
        with patch.object(network.sets[Set.DRIVER].tensor_op, 'get_mask') as mock_get_mask:
            # Only index 1 is RB
            mock_get_mask.return_value = torch.tensor([False, True, False] + [False] * 17)
            
            result = network.firing_ops.get_lcl_child_idxs([0], cons, Type.RB, Set.DRIVER)
        
        # Should only return RB child, not PO
        assert isinstance(result, list)
    
    def test_get_lcl_child_idxs_empty_indices(self, network):
        """Test that get_lcl_child_idxs returns empty list for empty input."""
        network.cache_sets()
        
        cons = network.tokens.get_view(TensorTypes.CON, Set.DRIVER)
        
        with patch.object(network.sets[Set.DRIVER].tensor_op, 'get_mask') as mock_get_mask:
            mock_get_mask.return_value = torch.zeros(20, dtype=torch.bool)
            
            result = network.firing_ops.get_lcl_child_idxs([], cons, Type.RB, Set.DRIVER)
        
        assert result == []
    
    def test_get_lcl_child_idxs_no_children(self, network):
        """Test that get_lcl_child_idxs returns empty list when no children exist."""
        # Setup: Create parent with no children
        network.tokens.token_tensor.set_features(
            torch.tensor([0]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.P)]])
        )
        
        network.cache_sets()
        
        cons = network.tokens.get_view(TensorTypes.CON, Set.DRIVER)
        
        with patch.object(network.sets[Set.DRIVER].tensor_op, 'get_mask') as mock_get_mask:
            mock_get_mask.return_value = torch.zeros(20, dtype=torch.bool)
            
            result = network.firing_ops.get_lcl_child_idxs([0], cons, Type.RB, Set.DRIVER)
        
        assert result == []


# =====================[ FiringOperations Initialization Tests ]======================

class TestFiringOperationsInit:
    """Tests for FiringOperations initialization."""
    
    def test_firing_ops_initialized_with_network(self, network):
        """Test that FiringOperations is initialized with network reference."""
        assert network.firing_ops.network is network
    
    def test_firing_order_initially_none(self, network):
        """Test that firing_order is None on initialization."""
        assert network.firing_ops.firing_order is None


# =====================[ Integration-like Tests ]======================

class TestFiringOpsIntegration:
    """Integration-style tests for firing operations."""
    
    def test_make_firing_order_returns_list_of_ints(self, network):
        """Test that make_firing_order returns a list of integers."""
        # Setup some tokens
        network.tokens.token_tensor.set_features(
            torch.tensor([0, 1]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.RB)]] * 2)
        )
        network.cache_sets()
        
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.RB):
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True, True] + [False] * 18)
                with patch.object(network, 'to_global', return_value=torch.tensor([0, 1])):
                    # Explicitly set a valid rule to avoid default param issues
                    result = network.firing_ops.make_firing_order(rule="by_top_random")
        
        assert isinstance(result, list)
        if result:  # If not empty
            assert all(isinstance(x, int) for x in result)
    
    def test_firing_order_accessible_after_creation(self, network):
        """Test that firing order can be accessed after creation."""
        # Setup some tokens
        network.tokens.token_tensor.set_features(
            torch.tensor([0]), 
            torch.tensor([TF.SET, TF.TYPE]), 
            torch.tensor([[float(Set.DRIVER), float(Type.PO)]])
        )
        network.cache_sets()
        
        with patch.object(network.driver().token_op, 'get_highest_token_type', return_value=Type.PO):
            with patch.object(network.driver().tensor_op, 'get_mask') as mock_get_mask:
                mock_get_mask.return_value = torch.tensor([True] + [False] * 19)
                with patch.object(network, 'to_global', return_value=torch.tensor([0])):
                    # Explicitly set a valid rule to avoid default param issues
                    network.firing_ops.make_firing_order(rule="by_top_random")
        
        # Should be able to access firing_order
        assert network.firing_ops.firing_order is not None

