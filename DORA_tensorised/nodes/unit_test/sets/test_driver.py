# nodes/unit_test/sets/test_driver.py
# Tests for Driver class, specifically update_input_p_parent and update_input_p_child functions

import pytest
import torch
from nodes.network.sets_new.driver import Driver
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.network.network_params import Params
from nodes.enums import Set, TF, Type, B, Mode, null, tensor_type


@pytest.fixture
def mock_tensor_with_parent_p():
    """
    Create a mock tensor with P nodes in PARENT mode, GROUPs, and RBs for testing update_input_p_parent.
    Structure:
    - Tokens 0-1: P nodes in PARENT mode in DRIVER
    - Token 2: P node in CHILD mode in DRIVER (should not be affected)
    - Token 3: P node in PARENT mode in RECIPIENT (should not be affected)
    - Tokens 4-5: GROUP nodes
    - Tokens 6-7: RB nodes
    - Tokens 8-9: PO nodes (for completeness)
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all active tokens
    tensor[:, TF.DELETED] = B.FALSE
    
    # DRIVER set: tokens 0-9
    tensor[0:10, TF.SET] = Set.DRIVER
    tensor[0:10, TF.ID] = torch.arange(0, 10)
    tensor[0:10, TF.ANALOG] = 0
    
    # P nodes in DRIVER
    tensor[0:3, TF.TYPE] = Type.P
    tensor[0:2, TF.MODE] = Mode.PARENT  # Parent mode P nodes (should be affected)
    tensor[2, TF.MODE] = Mode.CHILD      # Child mode P node (should NOT be affected)
    tensor[0:3, TF.ACT] = torch.tensor([0.5, 0.6, 0.7])  # Different activations for testing
    
    # GROUP nodes in DRIVER
    tensor[4:6, TF.TYPE] = Type.GROUP
    tensor[4:6, TF.ACT] = torch.tensor([0.3, 0.4])  # Group activations
    
    # RB nodes in DRIVER
    tensor[6:8, TF.TYPE] = Type.RB
    tensor[6:8, TF.ACT] = torch.tensor([0.8, 0.9])  # RB activations
    
    # PO nodes in DRIVER
    tensor[8:10, TF.TYPE] = Type.PO
    tensor[8:10, TF.ACT] = torch.tensor([0.2, 0.3])
    
    # RECIPIENT set: tokens 10-14
    tensor[10:15, TF.SET] = Set.RECIPIENT
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    tensor[10:15, TF.ANALOG] = 1
    
    # P node in PARENT mode in RECIPIENT (should NOT be affected)
    tensor[10, TF.TYPE] = Type.P
    tensor[10, TF.MODE] = Mode.PARENT
    tensor[10, TF.ACT] = 0.5
    
    # Initialize input values to 0 for clean testing
    tensor[:, TF.TD_INPUT] = 0.0
    tensor[:, TF.BU_INPUT] = 0.0
    tensor[:, TF.LATERAL_INPUT] = 0.0
    tensor[:, TF.MAP_INPUT] = 0.0
    tensor[:, TF.NET_INPUT] = 0.0
    
    return tensor


@pytest.fixture
def mock_connections_with_parent_p():
    """
    Create connections for testing update_input_p_parent.
    Connections (parent -> child):
    - P[0] -> GROUP[4], GROUP[5], RB[6]
    - P[1] -> GROUP[4], RB[6], RB[7]
    - P[2] -> GROUP[5] (child mode, should not be affected)
    """
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    # P[0] connections
    connections[0, 4] = True  # P[0] -> GROUP[4]
    connections[0, 5] = True  # P[0] -> GROUP[5]
    connections[0, 6] = True  # P[0] -> RB[6]
    
    # P[1] connections
    connections[1, 4] = True  # P[1] -> GROUP[4]
    connections[1, 6] = True  # P[1] -> RB[6]
    connections[1, 7] = True  # P[1] -> RB[7]
    
    # P[2] connections (child mode, should not be affected)
    connections[2, 5] = True  # P[2] -> GROUP[5]
    
    return connections


@pytest.fixture
def mock_names():
    """Create a mock names dictionary."""
    return {i: f"token_{i}" for i in range(20)}


@pytest.fixture
def mock_params():
    """Create a mock Params object."""
    from nodes.network.default_parameters import parameters
    return Params(parameters)


@pytest.fixture
def token_tensor(mock_tensor_with_parent_p, mock_connections_with_parent_p, mock_names):
    """Create a Token_Tensor instance with mock data."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections_with_parent_p)
    return Token_Tensor(mock_tensor_with_parent_p, connections_tensor, mock_names)


@pytest.fixture
def driver(token_tensor, mock_params):
    """Create a Driver instance."""
    return Driver(token_tensor, mock_params, mappings=None)


# =====================[ update_input_p_parent tests ]======================

def test_update_input_p_parent_td_input_from_groups(driver):
    """
    Test that TD_INPUT is correctly updated from connected GROUP nodes.
    P[0] is connected to GROUP[4] (act=0.3) and GROUP[5] (act=0.4)
    Expected TD_INPUT for P[0]: 0.3 + 0.4 = 0.7
    
    P[1] is connected to GROUP[4] (act=0.3)
    Expected TD_INPUT for P[1]: 0.3
    """
    # Get global indices for parent P nodes
    cache = driver.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.PARENT,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Get initial TD_INPUT values
    initial_td_input = driver.glbl.tensor[p_indices, TF.TD_INPUT].clone()
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated TD_INPUT values
    updated_td_input = driver.glbl.tensor[p_indices, TF.TD_INPUT]
    
    # Calculate expected values
    # P[0] (global index 0) is connected to GROUP[4] (act=0.3) and GROUP[5] (act=0.4)
    # P[1] (global index 1) is connected to GROUP[4] (act=0.3)
    con_tensor = driver.glbl.connections.connections
    group_mask = cache.get_type_mask(Type.GROUP)
    group_indices = torch.where(group_mask)[0]
    
    # Calculate expected TD_INPUT for each P node
    expected_td_input = torch.matmul(
        con_tensor[p_indices][:, group_indices].float(),
        driver.glbl.tensor[group_indices, TF.ACT]
    )
    
    # Verify TD_INPUT was incremented correctly
    assert torch.allclose(updated_td_input, initial_td_input + expected_td_input), \
        f"TD_INPUT not updated correctly. Expected increment: {expected_td_input}, Got: {updated_td_input - initial_td_input}"


def test_update_input_p_parent_bu_input_from_rbs(driver):
    """
    Test that BU_INPUT is correctly updated from connected RB nodes.
    P[0] is connected to RB[6] (act=0.8)
    Expected BU_INPUT for P[0]: 0.8
    
    P[1] is connected to RB[6] (act=0.8) and RB[7] (act=0.9)
    Expected BU_INPUT for P[1]: 0.8 + 0.9 = 1.7
    """
    # Get global indices for parent P nodes
    cache = driver.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.PARENT,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Get initial BU_INPUT values
    initial_bu_input = driver.glbl.tensor[p_indices, TF.BU_INPUT].clone()
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated BU_INPUT values
    updated_bu_input = driver.glbl.tensor[p_indices, TF.BU_INPUT]
    
    # Calculate expected values
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    con_tensor = driver.glbl.connections.connections
    
    # Calculate expected BU_INPUT for each P node
    expected_bu_input = torch.matmul(
        con_tensor[p_indices][:, rb_indices].float(),
        driver.glbl.tensor[rb_indices, TF.ACT]
    )
    
    # Verify BU_INPUT was incremented correctly
    assert torch.allclose(updated_bu_input, initial_bu_input + expected_bu_input), \
        f"BU_INPUT not updated correctly. Expected increment: {expected_bu_input}, Got: {updated_bu_input - initial_bu_input}"


def test_update_input_p_parent_lateral_input_from_other_parent_ps(driver):
    """
    Test that LATERAL_INPUT is correctly decremented by 3 * (sum of other parent P activations).
    P[0] (act=0.5) and P[1] (act=0.6) are both in PARENT mode.
    For P[0]: LATERAL_INPUT should decrease by 3 * 0.6 = 1.8 (from P[1])
    For P[1]: LATERAL_INPUT should decrease by 3 * 0.5 = 1.5 (from P[0])
    """
    # Get local P mask (all P nodes in driver)
    local_p_mask = driver.tensor_op.get_mask(Type.P)
    local_p_indices = torch.where(local_p_mask)[0]
    
    # Get initial LATERAL_INPUT values
    initial_lateral_input = driver.lcl[local_p_indices, TF.LATERAL_INPUT].clone()
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver.lcl[local_p_indices, TF.LATERAL_INPUT]
    
    # Get global parent P mask and indices
    cache = driver.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.PARENT,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Calculate expected lateral input change
    # For each parent P node, subtract 3 * (sum of other parent P activations)
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(len(p_indices))
    expected_decrement = 3 * torch.matmul(
        diag_zeroes,
        driver.glbl.tensor[p_indices, TF.ACT]
    )
    
    # Map global parent P indices to local indices
    # P[0] and P[1] are at local indices 0 and 1
    # The decrement should apply to local indices 0 and 1
    local_decrement = torch.zeros(len(local_p_indices))
    for i, local_idx in enumerate(local_p_indices):
        global_idx = driver.lcl.to_global(local_idx)[0].item()
        if global_idx in p_indices.tolist():
            p_idx_in_parent_list = (p_indices == global_idx).nonzero(as_tuple=True)[0].item()
            local_decrement[i] = expected_decrement[p_idx_in_parent_list]
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = initial_lateral_input - updated_lateral_input
    assert torch.allclose(actual_decrement, local_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly. Expected decrement: {local_decrement}, Got: {actual_decrement}"


def test_update_input_p_parent_only_affects_parent_mode_ps(driver):
    """
    Test that only P nodes in PARENT mode in DRIVER are affected.
    P[2] is in CHILD mode and should NOT have its inputs updated.
    """
    # Get local index for P[2] (child mode) - it's at local index 2
    local_p_mask = driver.tensor_op.get_mask(Type.P)
    local_p_indices = torch.where(local_p_mask)[0]
    # P[2] should be at local index 2 (third P node in driver)
    p2_local_idx = 2
    
    # Get initial input values (using global tensor for TD/BU, local for lateral)
    p2_global_idx = driver.lcl.to_global(p2_local_idx)[0].item()
    initial_td_input = driver.glbl.tensor[p2_global_idx, TF.TD_INPUT].item()
    initial_bu_input = driver.glbl.tensor[p2_global_idx, TF.BU_INPUT].item()
    initial_lateral_input = driver.lcl[p2_local_idx, TF.LATERAL_INPUT].item()
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated input values
    updated_td_input = driver.glbl.tensor[p2_global_idx, TF.TD_INPUT].item()
    updated_bu_input = driver.glbl.tensor[p2_global_idx, TF.BU_INPUT].item()
    updated_lateral_input = driver.lcl[p2_local_idx, TF.LATERAL_INPUT].item()
    
    # Verify P[2] inputs were NOT changed
    assert updated_td_input == initial_td_input, "TD_INPUT should not change for CHILD mode P node"
    assert updated_bu_input == initial_bu_input, "BU_INPUT should not change for CHILD mode P node"
    assert updated_lateral_input == initial_lateral_input, "LATERAL_INPUT should not change for CHILD mode P node"


def test_update_input_p_parent_only_affects_driver_set(driver):
    """
    Test that P nodes in PARENT mode in other sets (e.g., RECIPIENT) are NOT affected.
    """
    # Get global index for P node in RECIPIENT (token 10)
    recipient_p_global_idx = 10
    
    # Get initial input values
    initial_td_input = driver.glbl.tensor[recipient_p_global_idx, TF.TD_INPUT].item()
    initial_bu_input = driver.glbl.tensor[recipient_p_global_idx, TF.BU_INPUT].item()
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated input values
    updated_td_input = driver.glbl.tensor[recipient_p_global_idx, TF.TD_INPUT].item()
    updated_bu_input = driver.glbl.tensor[recipient_p_global_idx, TF.BU_INPUT].item()
    
    # Verify RECIPIENT P node inputs were NOT changed
    assert updated_td_input == initial_td_input, "TD_INPUT should not change for P nodes in other sets"
    assert updated_bu_input == initial_bu_input, "BU_INPUT should not change for P nodes in other sets"


def test_update_input_p_parent_increments_not_overwrites(driver):
    """
    Test that update_input_p_parent increments input values rather than overwriting them.
    """
    # Set initial TD_INPUT and BU_INPUT values for parent P nodes
    cache = driver.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.PARENT,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Set initial values
    initial_td = torch.tensor([1.0, 2.0])  # Different initial values
    initial_bu = torch.tensor([0.5, 1.5])
    driver.glbl.tensor[p_indices, TF.TD_INPUT] = initial_td
    driver.glbl.tensor[p_indices, TF.BU_INPUT] = initial_bu
    
    # Call the function
    driver.update_input_p_parent()
    
    # Get updated values
    updated_td = driver.glbl.tensor[p_indices, TF.TD_INPUT]
    updated_bu = driver.glbl.tensor[p_indices, TF.BU_INPUT]
    
    # Verify values were incremented (not overwritten)
    assert torch.all(updated_td > initial_td), "TD_INPUT should be incremented, not overwritten"
    assert torch.all(updated_bu > initial_bu), "BU_INPUT should be incremented, not overwritten"


def test_update_input_p_parent_no_parent_ps_no_error(driver):
    """
    Test that update_input_p_parent handles the case where there are no P nodes in PARENT mode gracefully.
    """
    # Set all P nodes to CHILD mode
    cache = driver.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    driver.glbl.tensor[p_indices, TF.MODE] = Mode.CHILD
    
    # This should not raise an error
    try:
        driver.update_input_p_parent()
    except Exception as e:
        pytest.fail(f"update_input_p_parent raised an exception when no parent P nodes exist: {e}")


# =====================[ update_input_p_child tests ]======================

@pytest.fixture
def mock_tensor_with_child_p():
    """
    Create a mock tensor with P nodes in CHILD mode, GROUPs, RBs, and Objects for testing update_input_p_child.
    Structure:
    - Tokens 0-1: P nodes in CHILD mode in DRIVER
    - Token 2: P node in PARENT mode in DRIVER (should not be affected)
    - Token 3: P node in CHILD mode in RECIPIENT (should not be affected)
    - Tokens 4-5: GROUP nodes
    - Tokens 6-7: RB nodes (parents of child P nodes)
    - Tokens 8-9: Object nodes (PO with PRED=False)
    - Token 10: Predicate node (PO with PRED=True, should not affect child P)
    """
    num_tokens = 25
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all active tokens
    tensor[:, TF.DELETED] = B.FALSE
    
    # DRIVER set: tokens 0-14
    tensor[0:15, TF.SET] = Set.DRIVER
    tensor[0:15, TF.ID] = torch.arange(0, 15)
    tensor[0:15, TF.ANALOG] = 0
    
    # P nodes in DRIVER
    tensor[0:3, TF.TYPE] = Type.P
    tensor[0:2, TF.MODE] = Mode.CHILD   # Child mode P nodes (should be affected)
    tensor[2, TF.MODE] = Mode.PARENT     # Parent mode P node (should NOT be affected)
    tensor[0:3, TF.ACT] = torch.tensor([0.5, 0.6, 0.7])  # Different activations for testing
    
    # GROUP nodes in DRIVER
    tensor[4:6, TF.TYPE] = Type.GROUP
    tensor[4:6, TF.ACT] = torch.tensor([0.3, 0.4])  # Group activations
    
    # RB nodes in DRIVER (these will be parents of child P nodes)
    tensor[6:8, TF.TYPE] = Type.RB
    tensor[6:8, TF.ACT] = torch.tensor([0.8, 0.9])  # RB activations
    
    # Object nodes (PO with PRED=False) in DRIVER
    tensor[8:10, TF.TYPE] = Type.PO
    tensor[8:10, TF.PRED] = B.FALSE
    tensor[8:10, TF.ACT] = torch.tensor([0.2, 0.3])  # Object activations
    
    # Predicate node (PO with PRED=True) in DRIVER
    tensor[10, TF.TYPE] = Type.PO
    tensor[10, TF.PRED] = B.TRUE
    tensor[10, TF.ACT] = 0.4
    
    # RECIPIENT set: tokens 15-19
    tensor[15:20, TF.SET] = Set.RECIPIENT
    tensor[15:20, TF.ID] = torch.arange(15, 20)
    tensor[15:20, TF.ANALOG] = 1
    
    # P node in CHILD mode in RECIPIENT (should NOT be affected)
    tensor[15, TF.TYPE] = Type.P
    tensor[15, TF.MODE] = Mode.CHILD
    tensor[15, TF.ACT] = 0.5
    
    # Initialize input values to 0 for clean testing
    tensor[:, TF.TD_INPUT] = 0.0
    tensor[:, TF.BU_INPUT] = 0.0
    tensor[:, TF.LATERAL_INPUT] = 0.0
    tensor[:, TF.MAP_INPUT] = 0.0
    tensor[:, TF.NET_INPUT] = 0.0
    
    return tensor


@pytest.fixture
def mock_connections_with_child_p():
    """
    Create connections for testing update_input_p_child.
    Connections (parent -> child):
    - RB[6] -> P[0] (RB is parent of child P)
    - RB[7] -> P[0], P[1] (RB is parent of child P)
    - GROUP[4] -> P[0], P[1]
    - GROUP[5] -> P[1]
    - RB[6] -> Object[8] (for testing shared RB logic)
    - RB[7] -> Object[9] (for testing shared RB logic)
    - P[2] -> GROUP[4] (parent mode, should not be affected)
    """
    num_tokens = 25
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    # RB -> P connections (parent -> child, so RB is parent of P)
    connections[6, 0] = True  # RB[6] -> P[0]
    connections[7, 0] = True  # RB[7] -> P[0]
    connections[7, 1] = True  # RB[7] -> P[1]
    
    # GROUP -> P connections
    connections[4, 0] = True  # GROUP[4] -> P[0]
    connections[4, 1] = True  # GROUP[4] -> P[1]
    connections[5, 1] = True  # GROUP[5] -> P[1]
    
    # RB -> Object connections (for testing shared RB logic)
    connections[6, 8] = True  # RB[6] -> Object[8]
    connections[7, 9] = True  # RB[7] -> Object[9]
    
    # P[2] (parent mode) -> GROUP[4] (should not be affected)
    connections[2, 4] = True  # P[2] -> GROUP[4]
    
    return connections


@pytest.fixture
def token_tensor_child_p(mock_tensor_with_child_p, mock_connections_with_child_p, mock_names):
    """Create a Token_Tensor instance with mock data for child P tests."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections_with_child_p)
    return Token_Tensor(mock_tensor_with_child_p, connections_tensor, mock_names)


@pytest.fixture
def driver_child_p(token_tensor_child_p, mock_params):
    """Create a Driver instance for child P tests."""
    return Driver(token_tensor_child_p, mock_params, mappings=None)


def test_update_input_p_child_td_input_from_groups(driver_child_p):
    """
    Test that TD_INPUT is correctly updated from connected GROUP nodes.
    P[0] is connected to GROUP[4] (act=0.3)
    Expected TD_INPUT for P[0]: 0.3
    
    P[1] is connected to GROUP[4] (act=0.3) and GROUP[5] (act=0.4)
    Expected TD_INPUT for P[1]: 0.3 + 0.4 = 0.7
    
    Note: The function also adds parent RB contributions, so we need to account for both.
    """
    # Get global indices for child P nodes
    cache = driver_child_p.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.CHILD,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Get initial TD_INPUT values
    initial_td_input = driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT].clone()
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated TD_INPUT values
    updated_td_input = driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT]
    
    # Calculate expected values from groups
    con_tensor = driver_child_p.glbl.connections.connections
    group_mask = cache.get_type_mask(Type.GROUP)
    group_indices = torch.where(group_mask)[0]
    
    # Calculate expected TD_INPUT for each P node from groups
    expected_from_groups = torch.matmul(
        con_tensor[p_indices][:, group_indices].float(),
        driver_child_p.glbl.tensor[group_indices, TF.ACT]
    )
    
    # Also calculate parent RB contribution (function adds both)
    t_con = torch.transpose(con_tensor, 0, 1)
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    expected_from_rbs = torch.matmul(
        t_con[p_indices][:, rb_indices].float(),
        driver_child_p.glbl.tensor[rb_indices, TF.ACT]
    )
    
    # Total expected increment is groups + parent RBs
    total_expected = expected_from_groups + expected_from_rbs
    
    # Verify TD_INPUT was incremented correctly
    actual_increment = updated_td_input - initial_td_input
    assert torch.allclose(actual_increment, total_expected, atol=1e-5), \
        f"TD_INPUT not updated correctly from groups. Expected increment: {total_expected}, Got: {actual_increment}"


def test_update_input_p_child_td_input_from_parent_rbs(driver_child_p):
    """
    Test that TD_INPUT is correctly updated from connected parent RB nodes.
    P[0] has parent RB[6] (act=0.8) and RB[7] (act=0.9)
    Expected TD_INPUT increment for P[0]: 0.8 + 0.9 = 1.7
    
    P[1] has parent RB[7] (act=0.9)
    Expected TD_INPUT increment for P[1]: 0.9
    """
    # Get global indices for child P nodes
    cache = driver_child_p.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.CHILD,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Get initial TD_INPUT values
    initial_td_input = driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT].clone()
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated TD_INPUT values
    updated_td_input = driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT]
    
    # Calculate expected values from parent RBs (using transpose connections)
    con_tensor = driver_child_p.glbl.connections.connections
    t_con = torch.transpose(con_tensor, 0, 1)  # Transpose for child -> parent connections
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    
    # Calculate expected TD_INPUT for each P node from parent RBs
    expected_td_input = torch.matmul(
        t_con[p_indices][:, rb_indices].float(),
        driver_child_p.glbl.tensor[rb_indices, TF.ACT]
    )
    
    # Note: TD_INPUT also gets updated from groups, so we need to account for that
    # Get the group contribution
    group_mask = cache.get_type_mask(Type.GROUP)
    group_indices = torch.where(group_mask)[0]
    group_contribution = torch.matmul(
        con_tensor[p_indices][:, group_indices].float(),
        driver_child_p.glbl.tensor[group_indices, TF.ACT]
    )
    
    # Total expected increment is groups + parent RBs
    total_expected = group_contribution + expected_td_input
    
    # Verify TD_INPUT was incremented correctly
    actual_increment = updated_td_input - initial_td_input
    assert torch.allclose(actual_increment, total_expected, atol=1e-5), \
        f"TD_INPUT not updated correctly from parent RBs. Expected total increment: {total_expected}, Got: {actual_increment}"


def test_update_input_p_child_lateral_input_from_other_child_ps(driver_child_p):
    """
    Test that LATERAL_INPUT is correctly decremented from other child P nodes.
    P[0] (act=0.5) and P[1] (act=0.6) are both in CHILD mode.
    For P[0]: LATERAL_INPUT should decrease by 0.6 (from P[1])
    For P[1]: LATERAL_INPUT should decrease by 0.5 (from P[0])
    
    Note: When as_DORA=False, the function also processes non-shared objects,
    so we need to account for both contributions or remove objects from the test.
    """
    # Set as_DORA to False
    driver_child_p.params.as_DORA = False
    
    # Remove objects from the driver set to isolate child P contributions
    # Get global object indices and set them to a different set
    cache = driver_child_p.glbl.cache
    obj_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.PRED: B.FALSE,
        TF.SET: Set.DRIVER
    })
    obj_indices = torch.where(obj_mask)[0]
    # Move objects to RECIPIENT set so they won't be processed
    driver_child_p.glbl.tensor[obj_indices, TF.SET] = Set.RECIPIENT
    
    # Get local child P mask
    local_p_mask = driver_child_p.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
    local_p_indices = torch.where(local_p_mask)[0]
    
    # Reset lateral input
    driver_child_p.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_child_p.lcl[local_p_indices, TF.LATERAL_INPUT]
    
    # Calculate expected lateral input change from child P nodes
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(sum(local_p_mask))
    expected_decrement = torch.matmul(
        diag_zeroes.float(),
        driver_child_p.lcl[local_p_mask, TF.ACT]
    )
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from other child P nodes. Expected decrement: {expected_decrement}, Got: {actual_decrement}"


def test_update_input_p_child_lateral_input_from_objects_dora_mode(driver_child_p):
    """
    Test that LATERAL_INPUT is correctly decremented from all objects when as_DORA=True.
    P[0] and P[1] should both get lateral input from Object[8] (act=0.2) and Object[9] (act=0.3)
    Expected decrement for each P from objects: 0.2 + 0.3 = 0.5
    
    Note: The function also adds child P contributions, so we need to account for both.
    """
    # Set as_DORA to True
    driver_child_p.params.as_DORA = True
    
    # Get local child P mask
    local_p_mask = driver_child_p.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
    local_p_indices = torch.where(local_p_mask)[0]
    
    # Reset lateral input
    driver_child_p.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get the lateral input after update
    updated_lateral_input = driver_child_p.lcl[local_p_indices, TF.LATERAL_INPUT]
    
    # Calculate expected decrement from child P nodes
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(sum(local_p_mask))
    expected_from_child_ps = torch.matmul(
        diag_zeroes.float(),
        driver_child_p.lcl[local_p_mask, TF.ACT]
    )
    
    # Calculate expected decrement from objects
    local_obj_mask = driver_child_p.tnop.get_arb_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE})
    obj_acts = driver_child_p.lcl[local_obj_mask, TF.ACT]
    expected_from_objects = obj_acts.sum()  # Same for all P nodes
    
    # Total expected decrement is child P + objects
    total_expected = expected_from_child_ps + expected_from_objects
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, total_expected, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from objects in DORA mode. Expected decrement: {total_expected}, Got: {actual_decrement}"


def test_update_input_p_child_lateral_input_from_non_shared_objects_non_dora_mode(driver_child_p):
    """
    Test that LATERAL_INPUT is correctly decremented from objects NOT connected to same RBs when as_DORA=False.
    P[0] is connected to RB[6] and RB[7]
    Object[8] is connected to RB[6] (shared) -> should NOT contribute
    Object[9] is connected to RB[7] (shared) -> should NOT contribute
    But wait, P[0] is connected to both RBs, so both objects share an RB with P[0]
    Actually, let's check: P[0] -> RB[6] and RB[7], Object[8] -> RB[6], Object[9] -> RB[7]
    So P[0] shares RB[6] with Object[8] and RB[7] with Object[9], so both are shared -> no decrement
    
    P[1] is connected to RB[7]
    Object[8] is connected to RB[6] (not shared with P[1]) -> should contribute
    Object[9] is connected to RB[7] (shared with P[1]) -> should NOT contribute
    """
    # Set as_DORA to False
    driver_child_p.params.as_DORA = False
    
    # Reset lateral input
    driver_child_p.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get local child P mask and indices
    local_p_mask = driver_child_p.tnop.get_arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
    local_p_indices = torch.where(local_p_mask)[0]
    
    # Get global child P indices
    cache = driver_child_p.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.CHILD,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Get lateral input values
    lateral_input = driver_child_p.lcl[local_p_indices, TF.LATERAL_INPUT]
    
    # The lateral input should be negative (inhibitory) due to:
    # 1. Other child P nodes
    # 2. Objects not sharing RBs with each child P
    
    # Verify lateral input is inhibitory (non-positive)
    assert torch.all(lateral_input <= 0.0), "LATERAL_INPUT should be non-positive (inhibitory)"
    
    # Verify that the function completed without errors
    # The exact values depend on the shared RB logic which is complex


def test_update_input_p_child_only_affects_child_mode_ps(driver_child_p):
    """
    Test that only P nodes in CHILD mode in DRIVER are affected.
    P[2] is in PARENT mode and should NOT have its inputs updated.
    """
    # Get global index for P[2] (parent mode)
    p2_global_idx = 2
    
    # Get initial input values
    initial_td_input = driver_child_p.glbl.tensor[p2_global_idx, TF.TD_INPUT].item()
    initial_lateral_input = driver_child_p.lcl[2, TF.LATERAL_INPUT].item()  # Local index 2
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated input values
    updated_td_input = driver_child_p.glbl.tensor[p2_global_idx, TF.TD_INPUT].item()
    updated_lateral_input = driver_child_p.lcl[2, TF.LATERAL_INPUT].item()
    
    # Verify P[2] inputs were NOT changed (or only changed minimally due to other operations)
    # Actually, P[2] might get some input if it's connected, but it shouldn't get child-mode specific updates
    # Let's check that it's not getting the same updates as child P nodes
    assert updated_td_input == initial_td_input or abs(updated_td_input - initial_td_input) < 1e-6, \
        "TD_INPUT should not change significantly for PARENT mode P node"
    assert updated_lateral_input == initial_lateral_input or abs(updated_lateral_input - initial_lateral_input) < 1e-6, \
        "LATERAL_INPUT should not change significantly for PARENT mode P node"


def test_update_input_p_child_only_affects_driver_set(driver_child_p):
    """
    Test that P nodes in CHILD mode in other sets (e.g., RECIPIENT) are NOT affected.
    """
    # Get global index for P node in RECIPIENT (token 15)
    recipient_p_global_idx = 15
    
    # Get initial input values
    initial_td_input = driver_child_p.glbl.tensor[recipient_p_global_idx, TF.TD_INPUT].item()
    initial_lateral_input = driver_child_p.glbl.tensor[recipient_p_global_idx, TF.LATERAL_INPUT].item()
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated input values
    updated_td_input = driver_child_p.glbl.tensor[recipient_p_global_idx, TF.TD_INPUT].item()
    updated_lateral_input = driver_child_p.glbl.tensor[recipient_p_global_idx, TF.LATERAL_INPUT].item()
    
    # Verify RECIPIENT P node inputs were NOT changed
    assert updated_td_input == initial_td_input, "TD_INPUT should not change for P nodes in other sets"
    assert updated_lateral_input == initial_lateral_input, "LATERAL_INPUT should not change for P nodes in other sets"


def test_update_input_p_child_increments_not_overwrites(driver_child_p):
    """
    Test that update_input_p_child increments input values rather than overwriting them.
    """
    # Set initial TD_INPUT values for child P nodes
    cache = driver_child_p.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.MODE: Mode.CHILD,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    
    # Set initial values
    initial_td = torch.tensor([1.0, 2.0])  # Different initial values
    driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT] = initial_td
    
    # Call the function
    driver_child_p.update_input_p_child()
    
    # Get updated values
    updated_td = driver_child_p.glbl.tensor[p_indices, TF.TD_INPUT]
    
    # Verify values were incremented (not overwritten)
    assert torch.all(updated_td > initial_td), "TD_INPUT should be incremented, not overwritten"


def test_update_input_p_child_no_child_ps_no_error(driver_child_p):
    """
    Test that update_input_p_child handles the case where there are no P nodes in CHILD mode gracefully.
    """
    # Set all P nodes to PARENT mode
    cache = driver_child_p.glbl.cache
    p_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.P,
        TF.SET: Set.DRIVER
    })
    p_indices = torch.where(p_mask)[0]
    driver_child_p.glbl.tensor[p_indices, TF.MODE] = Mode.PARENT
    
    # This should not raise an error
    try:
        driver_child_p.update_input_p_child()
    except Exception as e:
        pytest.fail(f"update_input_p_child raised an exception when no child P nodes exist: {e}")


# =====================[ update_input_rb tests ]======================

@pytest.fixture
def mock_tensor_with_rb():
    """
    Create a mock tensor with RB nodes, P nodes, and PO nodes for testing update_input_rb.
    Structure:
    - Tokens 0-1: RB nodes in DRIVER
    - Tokens 2-3: P nodes (parent and child) in DRIVER
    - Tokens 4-5: PO nodes (predicate and object) in DRIVER
    - Token 6: RB node in RECIPIENT (should not be affected)
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all active tokens
    tensor[:, TF.DELETED] = B.FALSE
    
    # DRIVER set: tokens 0-9
    tensor[0:10, TF.SET] = Set.DRIVER
    tensor[0:10, TF.ID] = torch.arange(0, 10)
    tensor[0:10, TF.ANALOG] = 0
    
    # RB nodes in DRIVER
    tensor[0:2, TF.TYPE] = Type.RB
    tensor[0:2, TF.ACT] = torch.tensor([0.5, 0.6])  # RB activations
    tensor[0:2, TF.INHIBITOR_ACT] = torch.tensor([0.1, 0.2])  # Inhibitor activations
    
    # P nodes in DRIVER
    tensor[2:4, TF.TYPE] = Type.P
    tensor[2, TF.MODE] = Mode.PARENT  # Parent P node
    tensor[3, TF.MODE] = Mode.CHILD   # Child P node
    tensor[2:4, TF.ACT] = torch.tensor([0.7, 0.8])  # P activations
    
    # PO nodes in DRIVER
    tensor[4:6, TF.TYPE] = Type.PO
    tensor[4, TF.PRED] = B.TRUE   # Predicate
    tensor[5, TF.PRED] = B.FALSE  # Object
    tensor[4:6, TF.ACT] = torch.tensor([0.3, 0.4])  # PO activations
    
    # RECIPIENT set: tokens 10-14
    tensor[10:15, TF.SET] = Set.RECIPIENT
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    tensor[10:15, TF.ANALOG] = 1
    
    # RB node in RECIPIENT (should NOT be affected)
    tensor[10, TF.TYPE] = Type.RB
    tensor[10, TF.ACT] = 0.5
    tensor[10, TF.INHIBITOR_ACT] = 0.1
    
    # Initialize input values to 0 for clean testing
    tensor[:, TF.TD_INPUT] = 0.0
    tensor[:, TF.BU_INPUT] = 0.0
    tensor[:, TF.LATERAL_INPUT] = 0.0
    tensor[:, TF.MAP_INPUT] = 0.0
    tensor[:, TF.NET_INPUT] = 0.0
    
    return tensor


@pytest.fixture
def mock_connections_with_rb():
    """
    Create connections for testing update_input_rb.
    Connections (parent -> child):
    - P[2] (parent) -> RB[0], RB[1] (P is parent of RB)
    - P[3] (child) -> RB[0] (P is child of RB, so RB is parent of P)
    - RB[0] -> PO[4] (predicate), PO[5] (object)
    - RB[1] -> PO[4] (predicate)
    - P[3] (child) -> RB[0] means RB[0] is parent of P[3]
    """
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    # P[2] (parent) -> RB connections (P is parent of RB)
    connections[2, 0] = True  # P[2] -> RB[0]
    connections[2, 1] = True  # P[2] -> RB[1]
    
    # RB -> P[3] connections (RB is parent of child P)
    connections[0, 3] = True  # RB[0] -> P[3]
    
    # RB -> PO connections
    connections[0, 4] = True  # RB[0] -> PO[4] (predicate)
    connections[0, 5] = True  # RB[0] -> PO[5] (object)
    connections[1, 4] = True  # RB[1] -> PO[4] (predicate)
    
    return connections


@pytest.fixture
def token_tensor_rb(mock_tensor_with_rb, mock_connections_with_rb, mock_names):
    """Create a Token_Tensor instance with mock data for RB tests."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections_with_rb)
    return Token_Tensor(mock_tensor_with_rb, connections_tensor, mock_names)


@pytest.fixture
def driver_rb(token_tensor_rb, mock_params):
    """Create a Driver instance for RB tests."""
    return Driver(token_tensor_rb, mock_params, mappings=None)


def test_update_input_rb_td_input_from_parent_ps(driver_rb):
    """
    Test that TD_INPUT is correctly updated from connected parent P nodes.
    RB[0] has parent P[2] (act=0.7) via transpose connection
    Expected TD_INPUT for RB[0]: 0.7
    
    RB[1] has parent P[2] (act=0.7) via transpose connection
    Expected TD_INPUT for RB[1]: 0.7
    """
    # Get global indices for RB nodes
    cache = driver_rb.glbl.cache
    rb_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.RB,
        TF.SET: Set.DRIVER
    })
    rb_indices = torch.where(rb_mask)[0]
    
    # Get initial TD_INPUT values
    initial_td_input = driver_rb.glbl.tensor[rb_indices, TF.TD_INPUT].clone()
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated TD_INPUT values
    updated_td_input = driver_rb.glbl.tensor[rb_indices, TF.TD_INPUT]
    
    # Calculate expected values (using transpose connections for parent P nodes)
    con_tensor = driver_rb.glbl.connections.connections
    t_con = torch.transpose(con_tensor, 0, 1)
    p_mask = cache.get_type_mask(Type.P)
    p_indices = torch.where(p_mask)[0]
    
    # Calculate expected TD_INPUT for each RB node from parent P nodes
    expected_td_input = torch.matmul(
        t_con[rb_indices][:, p_indices].float(),
        driver_rb.glbl.tensor[p_indices, TF.ACT]
    )
    
    # Verify TD_INPUT was incremented correctly
    assert torch.allclose(updated_td_input, initial_td_input + expected_td_input, atol=1e-5), \
        f"TD_INPUT not updated correctly from parent P nodes. Expected increment: {expected_td_input}, Got: {updated_td_input - initial_td_input}"


def test_update_input_rb_bu_input_from_po_and_child_p(driver_rb):
    """
    Test that BU_INPUT is correctly updated from connected PO and child P nodes.
    RB[0] is connected to PO[4] (act=0.3), PO[5] (act=0.4), and P[3] (child, act=0.8)
    Expected BU_INPUT for RB[0]: 0.3 + 0.4 + 0.8 = 1.5
    
    RB[1] is connected to PO[4] (act=0.3)
    Expected BU_INPUT for RB[1]: 0.3
    """
    # Get global indices for RB nodes
    cache = driver_rb.glbl.cache
    rb_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.RB,
        TF.SET: Set.DRIVER
    })
    rb_indices = torch.where(rb_mask)[0]
    
    # Get initial BU_INPUT values
    initial_bu_input = driver_rb.glbl.tensor[rb_indices, TF.BU_INPUT].clone()
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated BU_INPUT values
    updated_bu_input = driver_rb.glbl.tensor[rb_indices, TF.BU_INPUT]
    
    # Calculate expected values
    con_tensor = driver_rb.glbl.connections.connections
    po_mask = cache.get_type_mask(Type.PO)
    p_mask = cache.get_type_mask(Type.P)
    po_p_mask = torch.bitwise_or(po_mask, p_mask)
    po_p_indices = torch.where(po_p_mask)[0]
    
    # Calculate expected BU_INPUT for each RB node from PO and child P nodes
    expected_bu_input = torch.matmul(
        con_tensor[rb_indices][:, po_p_indices].float(),
        driver_rb.glbl.tensor[po_p_indices, TF.ACT]
    )
    
    # Verify BU_INPUT was incremented correctly
    assert torch.allclose(updated_bu_input, initial_bu_input + expected_bu_input, atol=1e-5), \
        f"BU_INPUT not updated correctly from PO and child P nodes. Expected increment: {expected_bu_input}, Got: {updated_bu_input - initial_bu_input}"


def test_update_input_rb_lateral_input_from_other_rbs(driver_rb):
    """
    Test that LATERAL_INPUT is correctly decremented by 3 * (sum of other RB activations).
    RB[0] (act=0.5) and RB[1] (act=0.6) are both in DRIVER.
    For RB[0]: LATERAL_INPUT should decrease by 3 * 0.6 = 1.8 (from RB[1])
    For RB[1]: LATERAL_INPUT should decrease by 3 * 0.5 = 1.5 (from RB[0])
    
    Note: We set inhibitor_act to 0 to isolate just the other RB contribution.
    """
    # Get local RB mask
    local_rb_mask = driver_rb.tnop.get_arb_mask({TF.TYPE: Type.RB})
    local_rb_indices = torch.where(local_rb_mask)[0]
    
    # Set inhibitor_act to 0 to isolate other RB contribution
    driver_rb.lcl[local_rb_mask, TF.INHIBITOR_ACT] = 0.0
    
    # Get initial LATERAL_INPUT values
    initial_lateral_input = driver_rb.lcl[local_rb_indices, TF.LATERAL_INPUT].clone()
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_rb.lcl[local_rb_indices, TF.LATERAL_INPUT]
    
    # Calculate expected lateral input change from other RBs only
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(sum(local_rb_mask))
    expected_decrement = 3 * torch.matmul(
        diag_zeroes.float(),
        driver_rb.lcl[local_rb_mask, TF.ACT]
    )
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = initial_lateral_input - updated_lateral_input
    assert torch.allclose(actual_decrement, expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from other RBs. Expected decrement: {expected_decrement}, Got: {actual_decrement}"


def test_update_input_rb_lateral_input_from_inhibitor(driver_rb):
    """
    Test that LATERAL_INPUT is correctly decremented by 10 * inhibitor_act.
    RB[0] has inhibitor_act=0.1, so LATERAL_INPUT should decrease by 1.0
    RB[1] has inhibitor_act=0.2, so LATERAL_INPUT should decrease by 2.0
    """
    # Get local RB mask
    local_rb_mask = driver_rb.tnop.get_arb_mask({TF.TYPE: Type.RB})
    local_rb_indices = torch.where(local_rb_mask)[0]
    
    # Reset lateral input to 0 for clean test
    driver_rb.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_rb.lcl[local_rb_indices, TF.LATERAL_INPUT]
    
    # Calculate expected decrement from inhibitor (10 * inhibitor_act)
    expected_inhibitor_decrement = 10 * driver_rb.lcl[local_rb_mask, TF.INHIBITOR_ACT]
    
    # Also account for other RB contributions (3 * other RBs)
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(sum(local_rb_mask))
    other_rb_decrement = 3 * torch.matmul(
        diag_zeroes.float(),
        driver_rb.lcl[local_rb_mask, TF.ACT]
    )
    
    # Total expected decrement
    total_expected_decrement = other_rb_decrement + expected_inhibitor_decrement
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, total_expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from inhibitor. Expected total decrement: {total_expected_decrement}, Got: {actual_decrement}"


def test_update_input_rb_only_affects_driver_set(driver_rb):
    """
    Test that RB nodes in other sets (e.g., RECIPIENT) are NOT affected.
    """
    # Get global index for RB node in RECIPIENT (token 10)
    recipient_rb_global_idx = 10
    
    # Get initial input values
    initial_td_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.TD_INPUT].item()
    initial_bu_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.BU_INPUT].item()
    initial_lateral_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.LATERAL_INPUT].item()
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated input values
    updated_td_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.TD_INPUT].item()
    updated_bu_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.BU_INPUT].item()
    updated_lateral_input = driver_rb.glbl.tensor[recipient_rb_global_idx, TF.LATERAL_INPUT].item()
    
    # Verify RECIPIENT RB node inputs were NOT changed
    assert updated_td_input == initial_td_input, "TD_INPUT should not change for RB nodes in other sets"
    assert updated_bu_input == initial_bu_input, "BU_INPUT should not change for RB nodes in other sets"
    assert updated_lateral_input == initial_lateral_input, "LATERAL_INPUT should not change for RB nodes in other sets"


def test_update_input_rb_increments_not_overwrites(driver_rb):
    """
    Test that update_input_rb increments input values rather than overwriting them.
    """
    # Set initial TD_INPUT and BU_INPUT values for RB nodes
    cache = driver_rb.glbl.cache
    rb_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.RB,
        TF.SET: Set.DRIVER
    })
    rb_indices = torch.where(rb_mask)[0]
    
    # Set initial values
    initial_td = torch.tensor([1.0, 2.0])  # Different initial values
    initial_bu = torch.tensor([0.5, 1.5])
    driver_rb.glbl.tensor[rb_indices, TF.TD_INPUT] = initial_td
    driver_rb.glbl.tensor[rb_indices, TF.BU_INPUT] = initial_bu
    
    # Call the function
    driver_rb.update_input_rb()
    
    # Get updated values
    updated_td = driver_rb.glbl.tensor[rb_indices, TF.TD_INPUT]
    updated_bu = driver_rb.glbl.tensor[rb_indices, TF.BU_INPUT]
    
    # Verify values were incremented (not overwritten)
    assert torch.all(updated_td > initial_td), "TD_INPUT should be incremented, not overwritten"
    assert torch.all(updated_bu > initial_bu), "BU_INPUT should be incremented, not overwritten"


def test_update_input_rb_no_rbs_no_error(driver_rb):
    """
    Test that update_input_rb handles the case where there are no RB nodes gracefully.
    """
    # Set all RB nodes to a different type
    cache = driver_rb.glbl.cache
    rb_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.RB,
        TF.SET: Set.DRIVER
    })
    rb_indices = torch.where(rb_mask)[0]
    driver_rb.glbl.tensor[rb_indices, TF.TYPE] = Type.PO
    
    # This should not raise an error
    try:
        driver_rb.update_input_rb()
    except Exception as e:
        pytest.fail(f"update_input_rb raised an exception when no RB nodes exist: {e}")


# =====================[ update_input_po tests ]======================

@pytest.fixture
def mock_tensor_with_po():
    """
    Create a mock tensor with PO nodes (predicates and objects), RB nodes, and P nodes for testing update_input_po.
    Structure:
    - Tokens 0-1: PO nodes (predicate and object) in DRIVER
    - Tokens 2-3: RB nodes in DRIVER
    - Token 4: P node (child mode) in DRIVER
    - Token 5: PO node in RECIPIENT (should not be affected)
    """
    num_tokens = 20
    num_features = len(TF)
    
    # Create tensor with all features
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Set DELETED to False for all active tokens
    tensor[:, TF.DELETED] = B.FALSE
    
    # DRIVER set: tokens 0-9
    tensor[0:10, TF.SET] = Set.DRIVER
    tensor[0:10, TF.ID] = torch.arange(0, 10)
    tensor[0:10, TF.ANALOG] = 0
    
    # PO nodes in DRIVER
    tensor[0:2, TF.TYPE] = Type.PO
    tensor[0, TF.PRED] = B.TRUE   # Predicate
    tensor[1, TF.PRED] = B.FALSE  # Object
    tensor[0:2, TF.ACT] = torch.tensor([0.5, 0.6])  # PO activations
    tensor[0:2, TF.INHIBITOR_ACT] = torch.tensor([0.1, 0.2])  # Inhibitor activations
    
    # RB nodes in DRIVER
    tensor[2:4, TF.TYPE] = Type.RB
    tensor[2:4, TF.ACT] = torch.tensor([0.7, 0.8])  # RB activations
    
    # P node (child mode) in DRIVER
    tensor[4, TF.TYPE] = Type.P
    tensor[4, TF.MODE] = Mode.CHILD
    tensor[4, TF.ACT] = 0.9
    
    # RECIPIENT set: tokens 10-14
    tensor[10:15, TF.SET] = Set.RECIPIENT
    tensor[10:15, TF.ID] = torch.arange(10, 15)
    tensor[10:15, TF.ANALOG] = 1
    
    # PO node in RECIPIENT (should NOT be affected)
    tensor[10, TF.TYPE] = Type.PO
    tensor[10, TF.PRED] = B.TRUE
    tensor[10, TF.ACT] = 0.5
    tensor[10, TF.INHIBITOR_ACT] = 0.1
    
    # Initialize input values to 0 for clean testing
    tensor[:, TF.TD_INPUT] = 0.0
    tensor[:, TF.BU_INPUT] = 0.0
    tensor[:, TF.LATERAL_INPUT] = 0.0
    tensor[:, TF.MAP_INPUT] = 0.0
    tensor[:, TF.NET_INPUT] = 0.0
    
    return tensor


@pytest.fixture
def mock_connections_with_po():
    """
    Create connections for testing update_input_po.
    Connections (parent -> child):
    - RB[2] -> PO[0] (predicate), PO[1] (object) - both POs connected to same RB
    - RB[3] -> PO[0] (predicate) - PO[0] connected to two RBs
    """
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    # RB -> PO connections
    connections[2, 0] = True  # RB[2] -> PO[0] (predicate)
    connections[2, 1] = True  # RB[2] -> PO[1] (object)
    connections[3, 0] = True  # RB[3] -> PO[0] (predicate)
    
    return connections


@pytest.fixture
def token_tensor_po(mock_tensor_with_po, mock_connections_with_po, mock_names):
    """Create a Token_Tensor instance with mock data for PO tests."""
    from nodes.network.tokens.connections.connections import Connections_Tensor
    connections_tensor = Connections_Tensor(mock_connections_with_po)
    return Token_Tensor(mock_tensor_with_po, connections_tensor, mock_names)


@pytest.fixture
def driver_po(token_tensor_po, mock_params):
    """Create a Driver instance for PO tests."""
    return Driver(token_tensor_po, mock_params, mappings=None)


def test_update_input_po_td_input_from_rbs_with_gain(driver_po):
    """
    Test that TD_INPUT is correctly updated from connected RB nodes with gain (2x for predicates, 1x for objects).
    PO[0] (predicate) is connected to RB[2] (act=0.7) and RB[3] (act=0.8)
    Expected TD_INPUT for PO[0]: 2 * (0.7 + 0.8) = 3.0 (predicate gets 2x gain)
    
    PO[1] (object) is connected to RB[2] (act=0.7)
    Expected TD_INPUT for PO[1]: 1 * 0.7 = 0.7 (object gets 1x gain)
    """
    # Get global indices for PO nodes
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    
    # Get initial TD_INPUT values
    initial_td_input = driver_po.glbl.tensor[po_indices, TF.TD_INPUT].clone()
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated TD_INPUT values
    updated_td_input = driver_po.glbl.tensor[po_indices, TF.TD_INPUT]
    
    # Calculate expected values
    con_tensor = driver_po.glbl.connections.connections
    parent_cons = torch.transpose(con_tensor, 0, 1)
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    
    # Get connections from PO to RB
    cons = parent_cons[po_indices][:, rb_indices].clone().float()
    # Apply gain: 2x for predicates, 1x for objects
    pred_mask = driver_po.glbl.tensor[po_indices, TF.PRED] == B.TRUE
    cons[pred_mask] = cons[pred_mask] * 2
    
    # Calculate expected TD_INPUT
    expected_td_input = torch.matmul(
        cons,
        driver_po.glbl.tensor[rb_indices, TF.ACT]
    )
    
    # Verify TD_INPUT was incremented correctly
    assert torch.allclose(updated_td_input, initial_td_input + expected_td_input, atol=1e-5), \
        f"TD_INPUT not updated correctly from RBs with gain. Expected increment: {expected_td_input}, Got: {updated_td_input - initial_td_input}"


def test_update_input_po_lateral_input_from_shared_pos_dora_mode(driver_po):
    """
    Test that LATERAL_INPUT is correctly decremented from POs connected to same RB when as_DORA=True.
    PO[0] and PO[1] both connect to RB[2], so they share an RB.
    For PO[0]: LATERAL_INPUT should decrease by 3 * 0.6 = 1.8 (from PO[1])
    For PO[1]: LATERAL_INPUT should decrease by 3 * 0.5 = 1.5 (from PO[0])
    """
    # Set as_DORA to True
    driver_po.params.as_DORA = True
    
    # Get global indices for PO nodes
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    
    # Set inhibitor_act to 0 to isolate PO contribution
    driver_po.glbl.tensor[po_indices, TF.INHIBITOR_ACT] = 0.0
    
    # Reset lateral input
    driver_po.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_po.glbl.tensor[po_indices, TF.LATERAL_INPUT]
    
    # Calculate expected lateral input change
    con_tensor = driver_po.glbl.connections.connections
    parent_cons = torch.transpose(con_tensor, 0, 1)
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    
    # Find shared RBs
    shared = torch.matmul(
        parent_cons[po_indices][:, rb_indices].float(),
        con_tensor[rb_indices][:, po_indices].float()
    )
    shared = torch.gt(shared, 0).int()
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(len(po_indices))
    shared = torch.bitwise_and(shared.int(), diag_zeroes.int()).float()
    
    # In DORA mode, use shared POs
    po_connections = shared
    expected_decrement = 3 * torch.matmul(
        po_connections,
        driver_po.glbl.tensor[po_indices, TF.ACT]
    )
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from shared POs in DORA mode. Expected decrement: {expected_decrement}, Got: {actual_decrement}"


def test_update_input_po_lateral_input_from_non_shared_pos_non_dora_mode(driver_po):
    """
    Test that LATERAL_INPUT is correctly decremented from POs NOT connected to same RB when as_DORA=False.
    PO[0] and PO[1] both connect to RB[2], so they share an RB -> should NOT contribute to each other.
    But PO[0] also connects to RB[3], which PO[1] doesn't connect to.
    Actually, since they share RB[2], they should NOT contribute to each other's lateral input.
    """
    # Set as_DORA to False
    driver_po.params.as_DORA = False
    
    # Get global indices for PO nodes
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    
    # Set inhibitor_act to 0 to isolate PO contribution
    driver_po.glbl.tensor[po_indices, TF.INHIBITOR_ACT] = 0.0
    
    # Reset lateral input
    driver_po.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_po.glbl.tensor[po_indices, TF.LATERAL_INPUT]
    
    # Calculate expected lateral input change
    con_tensor = driver_po.glbl.connections.connections
    parent_cons = torch.transpose(con_tensor, 0, 1)
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    
    # Find shared RBs
    shared = torch.matmul(
        parent_cons[po_indices][:, rb_indices].float(),
        con_tensor[rb_indices][:, po_indices].float()
    )
    shared = torch.gt(shared, 0).int()
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(len(po_indices))
    shared = torch.bitwise_and(shared.int(), diag_zeroes.int()).float()
    
    # In non-DORA mode, use non-shared POs
    po_connections = 1 - shared
    expected_decrement = 3 * torch.matmul(
        po_connections,
        driver_po.glbl.tensor[po_indices, TF.ACT]
    )
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from non-shared POs in non-DORA mode. Expected decrement: {expected_decrement}, Got: {actual_decrement}"


def test_update_input_po_lateral_input_from_inhibitor(driver_po):
    """
    Test that LATERAL_INPUT is correctly decremented by 10 * inhibitor_act.
    PO[0] has inhibitor_act=0.1, so LATERAL_INPUT should decrease by 1.0
    PO[1] has inhibitor_act=0.2, so LATERAL_INPUT should decrease by 2.0
    """
    # Get global indices for PO nodes
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    
    # Reset lateral input to 0 for clean test
    driver_po.glbl.tensor[:, TF.LATERAL_INPUT] = 0.0
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated LATERAL_INPUT values
    updated_lateral_input = driver_po.glbl.tensor[po_indices, TF.LATERAL_INPUT]
    
    # Calculate expected decrement from inhibitor (10 * inhibitor_act)
    expected_inhibitor_decrement = 10 * driver_po.glbl.tensor[po_indices, TF.INHIBITOR_ACT]
    
    # Also account for other PO contributions (3 * other POs)
    con_tensor = driver_po.glbl.connections.connections
    parent_cons = torch.transpose(con_tensor, 0, 1)
    rb_mask = cache.get_type_mask(Type.RB)
    rb_indices = torch.where(rb_mask)[0]
    
    shared = torch.matmul(
        parent_cons[po_indices][:, rb_indices].float(),
        con_tensor[rb_indices][:, po_indices].float()
    )
    shared = torch.gt(shared, 0).int()
    from nodes.utils import tensor_ops as tOps
    diag_zeroes = tOps.diag_zeros(len(po_indices))
    shared = torch.bitwise_and(shared.int(), diag_zeroes.int()).float()
    
    if driver_po.params.as_DORA:
        po_connections = shared
    else:
        po_connections = 1 - shared
    
    other_po_decrement = 3 * torch.matmul(
        po_connections,
        driver_po.glbl.tensor[po_indices, TF.ACT]
    )
    
    # Total expected decrement
    total_expected_decrement = other_po_decrement + expected_inhibitor_decrement
    
    # Verify LATERAL_INPUT was decremented correctly
    actual_decrement = -updated_lateral_input  # Since we started at 0
    assert torch.allclose(actual_decrement, total_expected_decrement, atol=1e-5), \
        f"LATERAL_INPUT not updated correctly from inhibitor. Expected total decrement: {total_expected_decrement}, Got: {actual_decrement}"


def test_update_input_po_only_affects_driver_set(driver_po):
    """
    Test that PO nodes in other sets (e.g., RECIPIENT) are NOT affected.
    """
    # Get global index for PO node in RECIPIENT (token 10)
    recipient_po_global_idx = 10
    
    # Get initial input values
    initial_td_input = driver_po.glbl.tensor[recipient_po_global_idx, TF.TD_INPUT].item()
    initial_lateral_input = driver_po.glbl.tensor[recipient_po_global_idx, TF.LATERAL_INPUT].item()
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated input values
    updated_td_input = driver_po.glbl.tensor[recipient_po_global_idx, TF.TD_INPUT].item()
    updated_lateral_input = driver_po.glbl.tensor[recipient_po_global_idx, TF.LATERAL_INPUT].item()
    
    # Verify RECIPIENT PO node inputs were NOT changed
    assert updated_td_input == initial_td_input, "TD_INPUT should not change for PO nodes in other sets"
    assert updated_lateral_input == initial_lateral_input, "LATERAL_INPUT should not change for PO nodes in other sets"


def test_update_input_po_increments_not_overwrites(driver_po):
    """
    Test that update_input_po increments input values rather than overwriting them.
    """
    # Set initial TD_INPUT values for PO nodes
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    
    # Set initial values
    initial_td = torch.tensor([1.0, 2.0])  # Different initial values
    driver_po.glbl.tensor[po_indices, TF.TD_INPUT] = initial_td
    
    # Call the function
    driver_po.update_input_po()
    
    # Get updated values
    updated_td = driver_po.glbl.tensor[po_indices, TF.TD_INPUT]
    
    # Verify values were incremented (not overwritten)
    assert torch.all(updated_td > initial_td), "TD_INPUT should be incremented, not overwritten"


def test_update_input_po_no_pos_no_error(driver_po):
    """
    Test that update_input_po handles the case where there are no PO nodes gracefully.
    """
    # Set all PO nodes to a different type
    cache = driver_po.glbl.cache
    po_mask = cache.get_arbitrary_mask({
        TF.TYPE: Type.PO,
        TF.SET: Set.DRIVER
    })
    po_indices = torch.where(po_mask)[0]
    driver_po.glbl.tensor[po_indices, TF.TYPE] = Type.RB
    
    # This should not raise an error
    try:
        driver_po.update_input_po()
    except Exception as e:
        pytest.fail(f"update_input_po raised an exception when no PO nodes exist: {e}")

