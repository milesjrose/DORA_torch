# nodes/unit_test/sets/test_base_set.py
# Tests for Base_Set class

import pytest
import torch
from nodes.network.sets_new.base_set import Base_Set
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, B, null, tensor_type


@pytest.fixture
def mock_tensor():
    """
    Create a mock tensor with multiple tokens across different sets.
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for active tokens (0-14)
    tensor[0:15, TF.DELETED] = B.FALSE
    # Set DELETED to True for deleted tokens (15-19)
    tensor[15:20, TF.DELETED] = B.TRUE
    
    # Create tokens in different sets
    # Set 0 (DRIVER): tokens 0-4
    tensor[0:5, TF.SET] = Set.DRIVER
    tensor[0:5, TF.ANALOG] = 0
    tensor[0:5, TF.ACT] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    tensor[0:5, TF.ID] = torch.arange(0, 5)
    
    # Set 1 (RECIPIENT): tokens 5-9
    tensor[5:10, TF.SET] = Set.RECIPIENT
    tensor[5:10, TF.ANALOG] = 1
    tensor[5:10, TF.ACT] = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0])
    tensor[5:10, TF.ID] = torch.arange(5, 10)
    
    # Set 2 (MEMORY): tokens 10-14
    tensor[10:15, TF.SET] = Set.MEMORY
    tensor[10:15, TF.ANALOG] = 2
    tensor[10:15, TF.ACT] = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5])
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    
    return tensor


@pytest.fixture
def mock_connections():
    """Create a mock connections tensor."""
    num_tokens = 20
    return torch.zeros((num_tokens, num_tokens), dtype=tensor_type)


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(15)}


@pytest.fixture
def token_tensor(mock_tensor, mock_connections, mock_names):
    """Create a Token_Tensor instance with mock data."""
    return Token_Tensor(mock_tensor, mock_connections, mock_names)


@pytest.fixture
def mock_params():
    """Create a mock Params object."""
    from nodes.network.default_parameters import parameters
    return Params(parameters)


# =====================[ __init__ tests ]======================

def test_base_set_init_driver(token_tensor, mock_params):
    """Test Base_Set initialization with DRIVER set."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    
    assert base_set.glbl is token_tensor
    assert base_set.tk_set == Set.DRIVER
    assert base_set.lcl is not None


def test_base_set_init_recipient(token_tensor, mock_params):
    """Test Base_Set initialization with RECIPIENT set."""
    base_set = Base_Set(token_tensor, Set.RECIPIENT, mock_params)
    
    assert base_set.glbl is token_tensor
    assert base_set.tk_set == Set.RECIPIENT
    assert base_set.lcl is not None


def test_base_set_init_memory(token_tensor, mock_params):
    """Test Base_Set initialization with MEMORY set."""
    base_set = Base_Set(token_tensor, Set.MEMORY, mock_params)
    
    assert base_set.glbl is token_tensor
    assert base_set.tk_set == Set.MEMORY
    assert base_set.lcl is not None


# =====================[ get_tensor tests ]======================

def test_get_tensor_driver(token_tensor, mock_params):
    """Test get_tensor returns the view for DRIVER set."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    tensor = base_set.get_tensor()
    
    # Should return a TensorView
    assert tensor is not None
    # Should contain 5 DRIVER tokens (indices 0-4)
    assert len(tensor) == 5


def test_get_tensor_recipient(token_tensor, mock_params):
    """Test get_tensor returns the view for RECIPIENT set."""
    base_set = Base_Set(token_tensor, Set.RECIPIENT, mock_params)
    tensor = base_set.get_tensor()
    
    # Should return a TensorView
    assert tensor is not None
    # Should contain 5 RECIPIENT tokens (indices 5-9)
    assert len(tensor) == 5


def test_get_tensor_memory(token_tensor, mock_params):
    """Test get_tensor returns the view for MEMORY set."""
    base_set = Base_Set(token_tensor, Set.MEMORY, mock_params)
    tensor = base_set.get_tensor()
    
    # Should return a TensorView
    assert tensor is not None
    # Should contain 5 MEMORY tokens (indices 10-14)
    assert len(tensor) == 5


def test_get_tensor_returns_same_view(token_tensor, mock_params):
    """Test that get_tensor returns the same view object."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    tensor1 = base_set.get_tensor()
    tensor2 = base_set.get_tensor()
    
    # Should return the same object (not create a new one)
    assert tensor1 is tensor2


# =====================[ get_token_set tests ]======================

def test_get_token_set_driver(token_tensor, mock_params):
    """Test get_token_set returns DRIVER."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    assert base_set.get_token_set() == Set.DRIVER


def test_get_token_set_recipient(token_tensor, mock_params):
    """Test get_token_set returns RECIPIENT."""
    base_set = Base_Set(token_tensor, Set.RECIPIENT, mock_params)
    assert base_set.get_token_set() == Set.RECIPIENT


def test_get_token_set_memory(token_tensor, mock_params):
    """Test get_token_set returns MEMORY."""
    base_set = Base_Set(token_tensor, Set.MEMORY, mock_params)
    assert base_set.get_token_set() == Set.MEMORY


# =====================[ update_view tests ]======================

def test_update_view_basic(token_tensor, mock_params):
    """Test basic update_view functionality."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    original_tensor = base_set.get_tensor()
    
    # Update the view
    updated_tensor = base_set.update_view()
    
    # Should return a TensorView
    assert updated_tensor is not None
    # Should update the internal tensor reference
    assert base_set.lcl is updated_tensor


def test_update_view_after_token_addition(token_tensor, mock_params):
    """Test update_view after adding tokens to the set."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    original_count = len(base_set.get_tensor())
    
    # Add a new DRIVER token
    new_tokens = torch.zeros((1, len(TF)), dtype=tensor_type)
    new_tokens[0, TF.DELETED] = B.FALSE
    new_tokens[0, TF.SET] = Set.DRIVER
    new_tokens[0, TF.ACT] = 0.6
    new_names = ["new_driver_token"]
    token_tensor.add_tokens(new_tokens, new_names)
    
    # Update the view
    base_set.update_view()
    
    # Should now see the new token
    updated_tensor = base_set.get_tensor()
    assert len(updated_tensor) == original_count + 1


def test_update_view_after_token_deletion(token_tensor, mock_params):
    """Test update_view after deleting tokens from the set."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    original_count = len(base_set.get_tensor())
    
    # Delete a DRIVER token (token 0)
    token_tensor.del_tokens(torch.tensor([0]))
    
    # Update the view
    base_set.update_view()
    
    # Should now see one fewer token
    updated_tensor = base_set.get_tensor()
    assert len(updated_tensor) == original_count - 1


def test_update_view_preserves_modifications(token_tensor, mock_params):
    """Test that modifications through the view are preserved after update_view."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    view = base_set.get_tensor()
    
    # Modify through the view
    view[0, TF.ACT] = 0.99
    
    # Update the view
    base_set.update_view()
    
    # The modification should still be in the original tensor
    # (view[0] corresponds to original token 0)
    assert token_tensor.tensor[0, TF.ACT].item() == pytest.approx(0.99, abs=1e-6)


def test_update_view_returns_tensor_view(token_tensor, mock_params):
    """Test that update_view returns a TensorView."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    updated_tensor = base_set.update_view()
    
    # Should return a TensorView (or tensor-like object)
    assert updated_tensor is not None
    assert hasattr(updated_tensor, '__getitem__')  # Should support indexing


def test_update_view_multiple_calls(token_tensor, mock_params):
    """Test calling update_view multiple times."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    
    # Call update_view multiple times
    view1 = base_set.update_view()
    view2 = base_set.update_view()
    view3 = base_set.update_view()
    
    # All should return valid views
    assert view1 is not None
    assert view2 is not None
    assert view3 is not None
    # All should have the same content (same indices)
    assert len(view1) == len(view2) == len(view3)
    assert len(view1) == 5
    # The internal tensor reference should be updated
    assert base_set.lcl is view3


def test_update_view_different_sets(token_tensor, mock_params):
    """Test update_view works for different sets."""
    driver_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    recipient_set = Base_Set(token_tensor, Set.RECIPIENT, mock_params)
    memory_set = Base_Set(token_tensor, Set.MEMORY, mock_params)
    
    # Update all views
    driver_view = driver_set.update_view()
    recipient_view = recipient_set.update_view()
    memory_view = memory_set.update_view()
    
    # All should have correct counts
    assert len(driver_view) == 5
    assert len(recipient_view) == 5
    assert len(memory_view) == 5


def test_update_view_after_set_change(token_tensor, mock_params):
    """Test update_view after changing a token's set."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    original_count = len(base_set.get_tensor())
    assert original_count == 5  # Should have 5 DRIVER tokens initially
    
    # Move token 0 from DRIVER to RECIPIENT
    # Note: move_tokens should invalidate cache for both old and new sets
    token_tensor.move_tokens(torch.tensor([0]), Set.RECIPIENT)
    
    # Update the view to refresh it
    base_set.update_view()
    
    # Should now see one fewer DRIVER token (4 instead of 5)
    updated_tensor = base_set.get_tensor()
    assert len(updated_tensor) == 4
    assert len(updated_tensor) == original_count - 1


# =====================[ Integration tests ]======================

def test_base_set_full_workflow(token_tensor, mock_params):
    """Test a full workflow with Base_Set."""
    # Create base set
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    
    # Get initial tensor
    tensor1 = base_set.get_tensor()
    assert len(tensor1) == 5
    
    # Verify token_set
    assert base_set.get_token_set() == Set.DRIVER
    
    # Add a token and update view
    new_tokens = torch.zeros((1, len(TF)), dtype=tensor_type)
    new_tokens[0, TF.DELETED] = B.FALSE
    new_tokens[0, TF.SET] = Set.DRIVER
    new_names = ["new_token"]
    token_tensor.add_tokens(new_tokens, new_names)
    
    base_set.update_view()
    tensor2 = base_set.get_tensor()
    assert len(tensor2) == 6
    
    # Get tensor again (should be same object)
    tensor3 = base_set.get_tensor()
    assert tensor3 is tensor2


def test_base_set_modifications_propagate(token_tensor, mock_params):
    """Test that modifications through Base_Set view propagate to original tensor."""
    base_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    view = base_set.get_tensor()
    
    # Modify through view
    original_act = token_tensor.tensor[0, TF.ACT].item()
    view[0, TF.ACT] = 0.88
    
    # Should propagate to original
    assert token_tensor.tensor[0, TF.ACT].item() == pytest.approx(0.88, abs=1e-6)
    assert view[0, TF.ACT].item() == pytest.approx(0.88, abs=1e-6)


def test_base_set_multiple_instances(token_tensor, mock_params):
    """Test creating multiple Base_Set instances for different sets."""
    driver_set = Base_Set(token_tensor, Set.DRIVER, mock_params)
    recipient_set = Base_Set(token_tensor, Set.RECIPIENT, mock_params)
    memory_set = Base_Set(token_tensor, Set.MEMORY, mock_params)
    
    # All should work independently
    assert len(driver_set.get_tensor()) == 5
    assert len(recipient_set.get_tensor()) == 5
    assert len(memory_set.get_tensor()) == 5
    
    assert driver_set.get_token_set() == Set.DRIVER
    assert recipient_set.get_token_set() == Set.RECIPIENT
    assert memory_set.get_token_set() == Set.MEMORY

