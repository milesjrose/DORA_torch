# nodes/tests/test_node_ops.py
# Tests for node operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.single_nodes import Token
from nodes.network.network import Network

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_get_most_active_token(network):
    """Test getting the most active token."""
    # work with DRIVER set
    driver_set = network.sets[Set.DRIVER]
    
    # get all non-deleted tokens in the driver set
    driver_mask = driver_set.tensor_op.get_all_nodes_mask()
    driver_indices = torch.where(driver_mask)[0]
    
    # ensure we have tokens to work with
    assert len(driver_indices) > 0, "No tokens in DRIVER set to test."
    
    # set all activations to 0
    driver_set.nodes[driver_indices, TF.ACT] = 0.0
    
    # pick a token to be the most active
    target_index = driver_indices[0]
    target_id = driver_set.nodes[target_index, TF.ID].item()
    
    # set its activation to a high value
    high_activation_value = 0.9
    driver_set.nodes[target_index, TF.ACT] = high_activation_value
    
    masks = {Set.DRIVER: driver_mask}
    
    # Test with id=False
    most_active_tokens = network.node_ops.get_most_active_token(masks, id=False)
    
    assert Set.DRIVER in most_active_tokens
    
    retrieved_token_ref = most_active_tokens[Set.DRIVER]
    
    assert retrieved_token_ref.ID == target_id, "The retrieved token ID does not match the most active token's ID."

    # Test with id=True
    most_active_tokens_ids = network.node_ops.get_most_active_token(masks, id=True)
    assert isinstance(most_active_tokens_ids, list)
    retrieved_id = most_active_tokens_ids[0]

    assert retrieved_id == target_id, "The retrieved ID should be the ID of the most active token."

def test_get_made_unit_ref(network: 'Network'):
    """
    Test get_made_unit_ref correctly gets the made unit ref.
    """
    driver = network.driver()
    new_set = network.new_set()
    
    # Add a PO token to the driver
    ref_maker = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    idx_maker = network.get_index(ref_maker)
    # Add a new token to the new set
    ref_made = network.add_token(Token(Type.PO, {TF.SET: Set.NEW_SET, TF.ACT: 1.0, TF.PRED: B.TRUE}))
    idx_made = network.get_index(ref_made)
    # Set made features
    network.set_value(ref_maker, TF.MADE_UNIT, idx_made)
    network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
    # Set maker features
    network.set_value(ref_maker, TF.MAKER_UNIT, idx_maker)
    network.set_value(ref_maker, TF.MAKER_SET, ref_maker.set)
    # Get made unit ref
    made_unit_ref = network.get_made_unit_ref(ref_maker)
    # Assert made unit ref is correct
    assert made_unit_ref is not None
    assert made_unit_ref.ID == ref_made.ID
    assert made_unit_ref.set == ref_made.set

def test_get_maker_unit_ref(network: 'Network'):
    """
    Test get_maker_unit_ref correctly gets the maker unit ref.
    """
    driver = network.driver()
    new_set = network.new_set()
    
    # Add a PO token to the driver
    ref_maker = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    idx_maker = network.get_index(ref_maker)
    # Add a new token to the new set
    ref_made = network.add_token(Token(Type.PO, {TF.SET: Set.NEW_SET, TF.ACT: 1.0, TF.PRED: B.TRUE}))
    idx_made = network.get_index(ref_made)
    # Set made features
    network.set_value(ref_maker, TF.MADE_UNIT, idx_made)
    network.set_value(ref_maker, TF.MADE_SET, ref_made.set)
    # Set maker features
    network.set_value(ref_maker, TF.MAKER_UNIT, idx_maker)
    network.set_value(ref_maker, TF.MAKER_SET, ref_maker.set)
    # Get made unit ref
    maker_unit_ref = network.get_maker_unit_ref(ref_maker)
    # Assert made unit ref is correct
    assert maker_unit_ref is not None
    assert maker_unit_ref.ID == ref_maker.ID
    assert maker_unit_ref.set == ref_maker.set

def test_kludgey_comparitor(network: 'Network'):
    """
    Test the kludgey comparator functionality.
    Tests comparison of two PO tokens based on their highest weight linked semantics.
    """
    from nodes.network.single_nodes import Semantic
    from nodes.enums import SF, OntStatus
    
    # Create two PO tokens in the driver set
    po1 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    po2 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    
    # Create semantics with same dimension but different amounts
    sem1 = Semantic("size1", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 5.0, SF.ONT: OntStatus.VALUE})
    sem2 = Semantic("size2", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 3.0, SF.ONT: OntStatus.VALUE})
    
    # Add semantics to the network
    ref_sem1 = network.semantics.add_semantic(sem1)
    ref_sem2 = network.semantics.add_semantic(sem2)
    
    # Set dimensions for the semantics (same dimension for comparison)
    network.semantics.set_dim(ref_sem1, "size")
    network.semantics.set_dim(ref_sem2, "size")
    
    # Set up links between PO tokens and semantics
    # Make sem1 the highest weight link for po1, sem2 for po2
    po1_idx = network.get_index(po1)
    po2_idx = network.get_index(po2)
    sem1_idx = network.get_index(ref_sem1)
    sem2_idx = network.get_index(ref_sem2)
    
    # Set high weights for the intended connections
    network.links.sets[Set.DRIVER][po1_idx, sem1_idx] = 1.0
    network.links.sets[Set.DRIVER][po2_idx, sem2_idx] = 1.0
    # Set lower weights for other connections
    network.links.sets[Set.DRIVER][po1_idx, sem2_idx] = 0.1
    network.links.sets[Set.DRIVER][po2_idx, sem1_idx] = 0.1
    
    # Test the kludgey comparator
    network.node_ops.kludgey_comparitor(po1.set, po1_idx, po2_idx)
    
    # Check that comparative semantics were created
    for sdm in SDM:
        assert network.semantics.sdms[sdm] is not None, f"{sdm.NAME} should be created"
    
    # Check that po1 (higher amount) is connected to "more" and po2 (lower amount) to "less"
    more_idx = network.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.get_index(network.semantics.sdms[SDM.LESS])
    
    # Verify connections
    assert network.links.sets[Set.DRIVER][po1_idx, more_idx] == 1.0, "po1 should be connected to 'more'"
    assert network.links.sets[Set.DRIVER][po2_idx, less_idx] == 1.0, "po2 should be connected to 'less'"

def test_kludgey_comparitor_reverse_order(network: 'Network'):
    """
    Test kludgey comparator with reversed order (po1 < po2).
    """
    from nodes.network.single_nodes import Semantic
    from nodes.enums import SF, OntStatus
    
    # Create two PO tokens in the driver set
    po1 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    po2 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    
    # Create semantics with same dimension but different amounts (po1 < po2)
    sem1 = Semantic("size1", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 2.0, SF.ONT: OntStatus.VALUE})
    sem2 = Semantic("size2", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 8.0, SF.ONT: OntStatus.VALUE})
    
    # Add semantics to the network
    ref_sem1 = network.semantics.add_semantic(sem1)
    ref_sem2 = network.semantics.add_semantic(sem2)
    
    # Set dimensions for the semantics (same dimension for comparison)
    network.semantics.set_dim(ref_sem1, "size")
    network.semantics.set_dim(ref_sem2, "size")
    
    # Set up links between PO tokens and semantics
    po1_idx = network.get_index(po1)
    po2_idx = network.get_index(po2)
    sem1_idx = network.get_index(ref_sem1)
    sem2_idx = network.get_index(ref_sem2)
    
    # Set high weights for the intended connections
    network.links.sets[Set.DRIVER][po1_idx, sem1_idx] = 1.0
    network.links.sets[Set.DRIVER][po2_idx, sem2_idx] = 1.0
    # Set lower weights for other connections
    network.links.sets[Set.DRIVER][po1_idx, sem2_idx] = 0.1
    network.links.sets[Set.DRIVER][po2_idx, sem1_idx] = 0.1
    
    # Test the kludgey comparator
    network.node_ops.kludgey_comparitor(po1.set, po1_idx, po2_idx)
    
    # Check that po1 (lower amount) is connected to "less" and po2 (higher amount) to "more"
    more_idx = network.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.get_index(network.semantics.sdms[SDM.LESS])
    
    # Verify connections
    assert network.links.sets[Set.DRIVER][po1_idx, less_idx] == 1.0, "po1 should be connected to 'less'"
    assert network.links.sets[Set.DRIVER][po2_idx, more_idx] == 1.0, "po2 should be connected to 'more'"

def test_kludgey_comparitor_equal_values(network: 'Network'):
    """
    Test kludgey comparator with equal semantic values.
    """
    from nodes.network.single_nodes import Semantic
    from nodes.enums import SF, OntStatus
    
    # Create two PO tokens in the driver set
    po1 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    po2 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    
    # Create semantics with same dimension and same amounts
    sem1 = Semantic("size1", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 5.0, SF.ONT: OntStatus.VALUE})
    sem2 = Semantic("size2", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 5.0, SF.ONT: OntStatus.VALUE})
    
    # Add semantics to the network
    ref_sem1 = network.semantics.add_semantic(sem1)
    ref_sem2 = network.semantics.add_semantic(sem2)
    
    # Set dimensions for the semantics (same dimension for comparison)
    network.semantics.set_dim(ref_sem1, "size")
    network.semantics.set_dim(ref_sem2, "size")
    
    # Set up links between PO tokens and semantics
    po1_idx = network.get_index(po1)
    po2_idx = network.get_index(po2)
    sem1_idx = network.get_index(ref_sem1)
    sem2_idx = network.get_index(ref_sem2)
    
    # Set high weights for the intended connections
    network.links.sets[Set.DRIVER][po1_idx, sem1_idx] = 1.0
    network.links.sets[Set.DRIVER][po2_idx, sem2_idx] = 1.0
    # Set lower weights for other connections
    network.links.sets[Set.DRIVER][po1_idx, sem2_idx] = 0.1
    network.links.sets[Set.DRIVER][po2_idx, sem1_idx] = 0.1
    
    # Test the kludgey comparator
    network.node_ops.kludgey_comparitor(po1.set, po1_idx, po2_idx)
    
    # Check that both PO tokens are connected to "same"
    same_idx = network.get_index(network.semantics.sdms[SDM.SAME])
    
    # Verify connections
    assert network.links.sets[Set.DRIVER][po1_idx, same_idx] == 1.0, "po1 should be connected to 'same'"
    assert network.links.sets[Set.DRIVER][po2_idx, same_idx] == 1.0, "po2 should be connected to 'same'"

def test_kludgey_comparitor_different_dimensions(network: 'Network'):
    """
    Test kludgey comparator with semantics of different dimensions (should not connect anything).
    """
    from nodes.network.single_nodes import Semantic
    from nodes.enums import SF, OntStatus
    
    # Create two PO tokens in the driver set
    po1 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    po2 = network.add_token(Token(Type.PO, {TF.SET: Set.DRIVER, TF.ACT: 0.5, TF.PRED: B.TRUE}))
    
    # Create semantics with different dimensions
    sem1 = Semantic("size1", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 5.0, SF.ONT: OntStatus.VALUE})
    sem2 = Semantic("color1", {SF.TYPE: Type.SEMANTIC, SF.AMOUNT: 3.0, SF.ONT: OntStatus.VALUE})
    
    # Add semantics to the network
    ref_sem1 = network.semantics.add_semantic(sem1)
    ref_sem2 = network.semantics.add_semantic(sem2)
    
    # Set different dimensions for the semantics
    network.semantics.set_dim(ref_sem1, "size")
    network.semantics.set_dim(ref_sem2, "color")
    
    # Set up links between PO tokens and semantics
    po1_idx = network.get_index(po1)
    po2_idx = network.get_index(po2)
    sem1_idx = network.get_index(ref_sem1)
    sem2_idx = network.get_index(ref_sem2)
    
    # Set high weights for the intended connections
    network.links.sets[Set.DRIVER][po1_idx, sem1_idx] = 1.0
    network.links.sets[Set.DRIVER][po2_idx, sem2_idx] = 1.0
    # Set lower weights for other connections
    network.links.sets[Set.DRIVER][po1_idx, sem2_idx] = 0.1
    network.links.sets[Set.DRIVER][po2_idx, sem1_idx] = 0.1
    
    # Test the kludgey comparator
    network.node_ops.kludgey_comparitor(po1.set, po1_idx, po2_idx)
    
    # Check that no comparative connections were made
    # The comparative semantics will be created, but no connections should be made to them
    # since the dimensions are different
    
    # Get indices of the comparative semantics
    more_idx = network.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.get_index(network.semantics.sdms[SDM.LESS])
    same_idx = network.get_index(network.semantics.sdms[SDM.SAME])
    
    # Verify that no connections were made to comparative semantics
    assert network.links.sets[Set.DRIVER][po1_idx, more_idx] == 0.0, "po1 should not be connected to 'more'"
    assert network.links.sets[Set.DRIVER][po1_idx, less_idx] == 0.0, "po1 should not be connected to 'less'"
    assert network.links.sets[Set.DRIVER][po1_idx, same_idx] == 0.0, "po1 should not be connected to 'same'"
    assert network.links.sets[Set.DRIVER][po2_idx, more_idx] == 0.0, "po2 should not be connected to 'more'"
    assert network.links.sets[Set.DRIVER][po2_idx, less_idx] == 0.0, "po2 should not be connected to 'less'"
    assert network.links.sets[Set.DRIVER][po2_idx, same_idx] == 0.0, "po2 should not be connected to 'same'"