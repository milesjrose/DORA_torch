# DORA_tensorised/nodes/tests/test_predication_routine.py
# Tests for the predication routine.

import pytest
import torch
from random import shuffle

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.single_nodes import Token
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    net = builder.build_network()
    net.routines.predication.debug = True  # Enable debug prints
    return net

def setup_predication_environment(network, mapping_weight=0.85):
    """Helper to synthetically create mappings for predication tests."""
    driver = network.driver()
    recipient = network.recipient()
    mappings = network.mappings

    # Ensure recipient has enough POs for 1-to-1 mapping
    d_po_mask = driver.get_mask(Type.PO)
    r_po_mask = recipient.get_mask(Type.PO)
    diff = d_po_mask.sum().item() - r_po_mask.sum().item()
    if diff > 0:
        for _ in range(diff):
            new_po = Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT)
            recipient.add_token(new_po)
    
    d_po_mask = driver.get_mask(Type.PO)
    r_po_mask = recipient.get_mask(Type.PO)
    
    d_indices = torch.where(d_po_mask)[0].tolist()
    r_indices = torch.where(r_po_mask)[0].tolist()
    shuffle(d_indices)
    
    if not d_indices:
        pytest.skip("No driver POs found in symProps to set up for predication test.")

    # Create 1-to-1 mappings
    mapped_pairs = []
    for i, d_idx in enumerate(d_indices):
        r_idx = r_indices[i]
        mappings[MappingFields.CONNECTIONS][r_idx, d_idx] = 1.0
        mappings[MappingFields.WEIGHT][r_idx, d_idx] = mapping_weight
        d_ref = driver.token_op.get_reference(index=d_idx)
        r_ref = recipient.token_op.get_reference(index=r_idx)
        mapped_pairs.append((d_ref, r_ref))
        
    # Calculate the max_map fields required by some checks
    network.mapping_ops.get_max_maps()
        
    return mapped_pairs

def test_requirements_pass(network):
    """Test that a synthetically valid network state passes predication requirements."""
    setup_predication_environment(network, mapping_weight=0.85)
    assert network.routines.predication.requirements() is True, "Network should pass predication requirements"

def test_requirements_fail_low_weight(network):
    """Test requirements failure when a mapping weight is too low."""
    setup_predication_environment(network, mapping_weight=0.5)
    assert network.routines.predication.requirements() is False, "Requirements should fail with low mapping weight"

def test_requirements_fail_rb_connection(network):
    """Test requirements failure when a mapped recipient PO is connected to an RB."""
    mapped_pairs = setup_predication_environment(network)
    _ , r_po = mapped_pairs[0]
    
    new_rb = Token(Type.RB, {TF.PRED: B.FALSE}, set=Set.RECIPIENT)
    rb_ref = network.recipient().add_token(new_rb)
    
    r_po_index = network.recipient().token_op.get_index(r_po)
    rb_ref_index = network.recipient().token_op.get_index(rb_ref)

    network.recipient().token_op.connect_idx(r_po_index, rb_ref_index)

    assert network.routines.predication.requirements() is False, "Requirements should fail with existing RB connection"

def test_check_po_requirements_pass(network):
    """Test that a PO correctly passes the predication checks."""
    mapped_pairs = setup_predication_environment(network)
    d_po, r_po = mapped_pairs[0]
    
    d_po_index = network.driver().token_op.get_index(d_po)
    r_po_index = network.recipient().token_op.get_index(r_po)

    # Setup conditions to pass on that pair
    network.recipient().nodes[r_po_index, TF.ACT] = 0.7
    network.driver().nodes[d_po_index, TF.ACT] = 0.7
    network.recipient().nodes[r_po_index, TF.PRED] = B.FALSE

    assert network.routines.predication.check_po_requirements(r_po) is True, "PO should pass requirements"

def test_predication_routine_no_new_pred_success(network):
    """Test the successful creation of a new predicate."""
    mapped_pairs = setup_predication_environment(network)
    d_po, r_po = mapped_pairs[0]

    d_po_index = network.driver().token_op.get_index(d_po)
    r_po_index = network.recipient().token_op.get_index(r_po)

    # Make r_po the most active PO in the recipient set
    r_po_mask = network.recipient().get_mask(Type.PO)
    network.recipient().nodes[r_po_mask, TF.ACT] = 0.1
    network.recipient().nodes[r_po_index, TF.ACT] = 0.9
    network.driver().nodes[d_po_index, TF.ACT] = 0.7
    network.recipient().nodes[r_po_index, TF.PRED] = B.FALSE

    initial_new_set_count = network.new_set().get_count()

    network.routines.predication.predication_routine()

    assert network.routines.predication.made_new_pred is True
    assert network.routines.predication.inferred_pred is not None
    # A new PO copy, a new Predicate, and a new RB are created
    assert network.new_set().get_count() == initial_new_set_count + 3

    inferred_pred_ref = network.routines.predication.inferred_pred
    assert inferred_pred_ref.set == Set.NEW_SET
    inferred_pred_token = network.new_set().token_op.get_single_token(inferred_pred_ref)
    assert inferred_pred_token.tensor[TF.TYPE] == Type.PO
    assert inferred_pred_token.tensor[TF.PRED] == B.TRUE
    assert inferred_pred_token.tensor[TF.INFERRED] == B.TRUE

    original_po_tensor = network.recipient().token_op.get_single_token(r_po).tensor
    assert int(original_po_tensor[TF.MADE_UNIT]) != null

def test_predication_routine_made_new_pred_learning(network):
    """Test that link weights are updated correctly when a new predicate exists."""
    network.routines.predication.made_new_pred = True
    # In a real scenario, the new pred would be in the NEW_SET
    new_pred_token = Token(Type.PO, {TF.SET: Set.NEW_SET, TF.PRED: B.TRUE})
    inferred_pred_ref = network.new_set().add_token(new_pred_token)
    network.routines.predication.inferred_pred = inferred_pred_ref
    inferred_pred_index = network.new_set().token_op.get_index(inferred_pred_ref)

    sem_indices = [0, 1]
    sem_acts = [0.8, 0.9]
    for i, act in zip(sem_indices, sem_acts):
        network.semantics.nodes[i, SF.ACT] = act
    active_sem_mask = network.semantics.nodes[:, SF.ACT] > 0
    
    links = network.links.sets[Set.NEW_SET]
    initial_weights = links[inferred_pred_index, active_sem_mask].clone()

    network.routines.predication.predication_routine()

    gamma = network.params.gamma
    expected_change = 1 * (torch.tensor(sem_acts) - initial_weights) * gamma
    final_weights = links[inferred_pred_index, active_sem_mask]
    
    assert torch.allclose(final_weights, initial_weights + expected_change), "Link weights did not update as expected"