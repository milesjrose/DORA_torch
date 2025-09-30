import pytest
import torch

from nodes.network import Network
from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token, Semantic
from nodes.enums import *

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_ent_magnitude_more_less_same(network: Network):
    """Test the ent_magnitude_more_less_same function."""
    entropy_ops = network.entropy_ops
    
    # Test case 1: extent1 > extent2
    more, less, same_flag, _ = entropy_ops.ent_magnitude_more_less_same(10, 5)
    assert more == 10
    assert less == 5
    assert not same_flag

    # Test case 2: extent2 > extent1
    more, less, same_flag, _ = entropy_ops.ent_magnitude_more_less_same(5, 10)
    assert more == 10
    assert less == 5
    assert not same_flag

    # Test case 3: extent1 == extent2
    more, less, same_flag, _ = entropy_ops.ent_magnitude_more_less_same(7, 7)
    assert more is None
    assert less is None
    assert same_flag

def test_ent_overall_same_diff(network: Network):
    """Test the ent_overall_same_diff function."""
    entropy_ops = network.entropy_ops
    
    # Activate some semantics
    network.semantics.nodes[0, SF.ACT] = 0.9
    network.semantics.nodes[1, SF.ACT] = 0.8
    network.semantics.nodes[2, SF.ACT] = 0.1  # This will be masked out
    
    # Expected values
    expected_sum_diff = (1.0 - 0.9) + (1.0 - 0.8)
    expected_sum_act = 0.9 + 0.8
    expected_ratio = expected_sum_diff / expected_sum_act
    
    ratio = entropy_ops.ent_overall_same_diff()
    assert torch.isclose(torch.tensor(ratio), torch.tensor(expected_ratio))

def test_attach_mag_semantics_not_same(network: Network):
    """Test attach_mag_semantics with same_flag=False."""
    # Add two PO tokens
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    po1_idx = network.get_index(po1_ref)
    po2_idx = network.get_index(po2_ref)

    # Mock sem_links - create a mask of the correct size
    num_semantics = network.semantics.get_count()
    po1_mask = torch.zeros(num_semantics, dtype=torch.bool)
    po1_mask[0] = True
    po1_mask[2] = True
    po2_mask = torch.zeros(num_semantics, dtype=torch.bool)
    po2_mask[1] = True
    po2_mask[2] = True
    sem_links = {
        po1_ref: po1_mask,
        po2_ref: po2_mask,
    }
    idxs = {po1_ref: po1_idx, po2_ref: po2_idx}

    # Test case: Not same
    network.entropy_ops.attach_mag_semantics(False, po1_ref, po2_ref, sem_links, idxs)

    # Check SDM connections
    more_idx = network.semantics.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.semantics.get_index(network.semantics.sdms[SDM.LESS])

    assert network.links[Set.DRIVER][po1_idx, more_idx].item() == 1.0
    assert network.links[Set.DRIVER][po2_idx, less_idx].item() == 1.0

def test_attach_mag_semantics_same(network: Network):
    """Test attach_mag_semantics with same_flag=True."""
    # Add two PO tokens
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    
    po1_idx = network.get_index(po1_ref)
    po2_idx = network.get_index(po2_ref)

    # Mock sem_links - create a mask of the correct size
    num_semantics = network.semantics.get_count()
    po1_mask = torch.zeros(num_semantics, dtype=torch.bool)
    po1_mask[0] = True
    po1_mask[2] = True
    po2_mask = torch.zeros(num_semantics, dtype=torch.bool)
    po2_mask[1] = True
    po2_mask[2] = True
    sem_links = {
        po1_ref: po1_mask,
        po2_ref: po2_mask,
    }
    idxs = {po1_ref: po1_idx, po2_ref: po2_idx}

    # Test case: Same
    network.entropy_ops.attach_mag_semantics(True, po1_ref, po2_ref, sem_links, idxs)
    
    # Check SDM connections
    same_idx = network.semantics.get_index(network.semantics.sdms[SDM.SAME])

    assert network.links[Set.DRIVER][po1_idx, same_idx].item() == 1.0
    assert network.links[Set.DRIVER][po2_idx, same_idx].item() == 1.0

def test_en_based_mag_checks(network: Network):
    """Test the en_based_mag_checks function."""
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    po1_idx = network.get_index(po1_ref)
    po2_idx = network.get_index(po2_ref)
    new_dim1 = network.semantics.add_dim("dim1")
    new_dim2 = network.semantics.add_dim("dim2")

    # Add some semantics with dimensions
    sem0_ref = network.semantics.add_semantic(Semantic(f"sem0", {SF.DIM: new_dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 10.0}))
    sem1_ref = network.semantics.add_semantic(Semantic(f"sem1", {SF.DIM: new_dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 20.0}))
    sem2_ref = network.semantics.add_semantic(Semantic(f"sem2", {SF.DIM: new_dim2, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 5.0}))
    sem3_ref = network.semantics.add_semantic(Semantic(f"sem3", {SF.DIM: new_dim2, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 1.0}))
    sem0_idx = network.get_index(sem0_ref)
    sem1_idx = network.get_index(sem1_ref)
    sem2_idx = network.get_index(sem2_ref)
    sem3_idx = network.get_index(sem3_ref)

    # Create links
    network.links.update_link(po1_ref.set, po1_idx, sem0_idx, 0.95)
    network.links.update_link(po2_ref.set, po2_idx, sem1_idx, 0.94)
    network.links.update_link(po1_ref.set, po1_idx, sem2_idx, 0.92)
    network.links.update_link(po2_ref.set, po2_idx, sem3_idx, 0.93)

    # Create links to SDM semantics
    more_idx = network.semantics.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.semantics.get_index(network.semantics.sdms[SDM.LESS])
    same_idx = network.semantics.get_index(network.semantics.sdms[SDM.SAME])
    diff_idx = network.semantics.get_index(network.semantics.sdms[SDM.DIFF])

    network.links.update_link(po1_ref.set, po1_idx, more_idx, 0.5)
    network.links.update_link(po2_ref.set, po2_idx, less_idx, 0.93)
    
    high_dim, num_sdm_above, num_sdm_below = network.entropy_ops.en_based_mag_checks(po1_ref, po2_ref)

    assert high_dim == [new_dim1]
    assert num_sdm_above == 1
    assert num_sdm_below == 1

def test_update_mag_semantics(network: Network):
    """Test the update_mag_semantics function."""
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    po1_idx = network.get_index(po1_ref)
    po2_idx = network.get_index(po2_ref)
    
    sem_links = {
        po1_ref: torch.zeros(network.semantics.get_count(), dtype=torch.bool),
        po2_ref: torch.zeros(network.semantics.get_count(), dtype=torch.bool),
    }
    sem_links[po1_ref][0] = True
    sem_links[po2_ref][1] = True
    
    idxs = {po1_ref: po1_idx, po2_ref: po2_idx}

    # Test case: same_flag = True
    network.entropy_ops.update_mag_semantics(True, po1_ref, po2_ref, sem_links, idxs)
    same_idx = network.semantics.get_index(network.semantics.sdms[SDM.SAME])
    assert network.links[Set.DRIVER][po1_idx, same_idx].item() == 1.0
    assert network.links[Set.DRIVER][po2_idx, same_idx].item() == 1.0
    assert torch.allclose(network.links[Set.DRIVER][po1_idx, sem_links[po1_ref]], torch.tensor(1.0))

    # Reset links
    network.links[Set.DRIVER][po1_idx, :] = 0.0
    network.links[Set.DRIVER][po2_idx, :] = 0.0

    # Test case: same_flag = False
    network.entropy_ops.update_mag_semantics(False, po1_ref, po2_ref, sem_links, idxs)
    more_idx = network.semantics.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.semantics.get_index(network.semantics.sdms[SDM.LESS])
    assert network.links[Set.DRIVER][po1_idx, more_idx].item() == 1.0
    assert network.links[Set.DRIVER][po2_idx, less_idx].item() == 1.0
    assert torch.allclose(network.links[Set.DRIVER][po1_idx, sem_links[po1_ref]], torch.tensor(1.0))
    assert torch.allclose(network.links[Set.DRIVER][po2_idx, sem_links[po2_ref]], torch.tensor(1.0))

def test_basic_en_based_mag_comparison(network: Network):
    """Test the basic_en_based_mag_comparison function."""
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    po1_idx = network.get_index(po1_ref)
    po2_idx = network.get_index(po2_ref)

    # Add semantics with a shared dimension
    new_dim1 = network.semantics.add_dim("dim1")
    sem_dim1_val10_ref = network.semantics.add_semantic(Semantic("dim1_val10", {SF.DIM: new_dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 10.0}))
    sem_dim1_val20_ref = network.semantics.add_semantic(Semantic("dim1_val20", {SF.DIM: new_dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 20.0}))
    sem_dim1_val10_idx = network.get_index(sem_dim1_val10_ref)
    sem_dim1_val20_idx = network.get_index(sem_dim1_val20_ref)

    # Link POs to semantics
    network.links.update_link(po1_ref.set, po1_idx, sem_dim1_val10_idx, 1.0)
    network.links.update_link(po2_ref.set, po2_idx, sem_dim1_val20_idx, 1.0)

    # Check that no SDM links are present
    for sdm in network.semantics.get_sdm_indices():
        assert network.links[Set.DRIVER][po1_idx, sdm].item() == 0.0
        assert network.links[Set.DRIVER][po2_idx, sdm].item() == 0.0

    # Run comparison
    network.entropy_ops.basic_en_based_mag_comparison(po1_ref, po2_ref, [new_dim1], 0)

    # po2 > po1, so po2 is MORE, po1 is LESS
    more_idx = network.semantics.get_index(network.semantics.sdms[SDM.MORE])
    less_idx = network.semantics.get_index(network.semantics.sdms[SDM.LESS])
    
    assert network.links[Set.DRIVER][po2_idx, more_idx].item() == 1.0
    assert network.links[Set.DRIVER][po1_idx, less_idx].item() == 1.0
    
    # Check that original link weights are halved
    assert network.links[Set.DRIVER][po1_idx, sem_dim1_val10_idx].item() == 0.5
    assert network.links[Set.DRIVER][po2_idx, sem_dim1_val20_idx].item() == 0.5

def test_find_links_to_abs_dim(network: Network):
    """Test the find_links_to_abs_dim helper function."""
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    po1_idx = network.get_index(po1_ref)
    new_dim1 = network.semantics.add_dim("dim1")
    new_dim2 = network.semantics.add_dim("dim2")

    # Add semantics
    sem1_ref = network.semantics.add_semantic(Semantic("sem1", {SF.DIM: new_dim1, SF.ONT: OntStatus.VALUE}))
    sem2_ref = network.semantics.add_semantic(Semantic("sem2", {SF.DIM: new_dim1, SF.ONT: OntStatus.STATE}))
    sem3_ref = network.semantics.add_semantic(Semantic("sem3", {SF.DIM: new_dim2, SF.ONT: OntStatus.VALUE}))
    sem1_idx = network.get_index(sem1_ref)
    sem2_idx = network.get_index(sem2_ref)
    sem3_idx = network.get_index(sem3_ref)
    
    # Link PO to semantics
    network.links.update_link(po1_ref.set, po1_idx, sem1_idx, 1.0)
    network.links.update_link(po1_ref.set, po1_idx, sem2_idx, 1.0)
    network.links.update_link(po1_ref.set, po1_idx, sem3_idx, 1.0)
    
    # Find links to dim 1, ont_status VALUE
    result_mask = network.entropy_ops.find_links_to_abs_dim(po1_ref.set, po1_idx, new_dim1, OntStatus.VALUE)
    
    # Create expected mask
    expected_mask = torch.zeros(network.semantics.get_count(), dtype=torch.bool)
    expected_mask[sem1_idx] = True
    
    assert torch.equal(result_mask, expected_mask)

def test_basic_en_based_mag_refinement(network: Network):
    """Test the basic_en_based_mag_refinement function."""
    # This function is complex and has dependencies on the network structure.
    # A full test would require setting up a complex state.
    # This test will be a placeholder to be expanded upon.
    # For now, we'll just call it to ensure no trivial errors.
    
    po1_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.TRUE}, set=Set.DRIVER))
    po2_ref = network.driver().add_token(Token(Type.PO, {TF.PRED: B.TRUE}, set=Set.DRIVER))

    # To make this test meaningful, we would need to:
    # 1. Create a more complex network state with objects, predicates, and RBs.
    # 2. Link them in a way that basic_en_based_mag_refinement can operate.
    # 3. This includes setting up intersecting dimensions and pre-existing magnitude semantics.
    
    # For now, just a smoke test.
    try:
        network.entropy_ops.basic_en_based_mag_refinement(po1_ref, po2_ref)
    except Exception as e:
        pytest.fail(f"basic_en_based_mag_refinement raised an exception: {e}")
