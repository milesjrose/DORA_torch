# nodes/tests/unit/operations/test_entropy_ops.py
# Unit tests for EntropyOperations class

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from nodes.network.network import Network
from nodes.network.tokens import Tokens, Token_Tensor, Connections_Tensor, Links, Mapping
from nodes.network.sets import Semantics
from nodes.network.network_params import default_params
from nodes.network.single_nodes import Token, Semantic
from nodes.network.operations.entropy_ops import EntropyOperations, en_based_mag_checks_results
from nodes.enums import Set, TF, SF, MappingFields, Type, null, OntStatus, B, SDM


# =====================[ Fixtures ]======================

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
    for i in range(num_tokens):
        tokens[i, TF.ID] = i
        tokens[i, TF.SET] = Set.DRIVER.value
    names = {i: f"token_{i}" for i in range(num_tokens)}
    return Token_Tensor(tokens, names)


@pytest.fixture
def minimal_connections():
    """Create minimal Connections_Tensor for testing."""
    num_tokens = 20
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    return Connections_Tensor(connections)


@pytest.fixture
def minimal_semantics():
    """Create minimal Semantics object for testing with SDM support."""
    num_semantics = 20
    num_features = len(SF)
    nodes = torch.zeros((num_semantics, num_features))
    nodes[:, SF.DELETED] = B.TRUE  # Mark all as deleted initially
    connections = torch.zeros((num_semantics, num_semantics))
    IDs = {}
    names = {}
    semantics = Semantics(nodes, connections, IDs, names)
    return semantics


@pytest.fixture
def minimal_links(minimal_token_tensor, minimal_semantics):
    """Create minimal Links object for testing."""
    num_tokens = minimal_token_tensor.get_count()
    num_semantics = minimal_semantics.get_count()
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
def minimal_tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping):
    """Create minimal Tokens object for testing."""
    return Tokens(minimal_token_tensor, minimal_connections, minimal_links, minimal_mapping)


@pytest.fixture
def network(minimal_tokens, minimal_semantics, minimal_params):
    """Create minimal Network object for testing."""
    net = Network(minimal_tokens, minimal_semantics, minimal_params)
    # Initialize SDM semantics
    net.semantics.init_sdm()
    return net


# =====================[ Initialization Tests ]======================

class TestEntropyOpsInitialization:
    """Tests for EntropyOperations initialization."""

    def test_entropy_ops_initialization(self, network):
        """Test that EntropyOperations is properly initialized."""
        assert network.entropy_ops is not None
        assert network.entropy_ops.network is network

    def test_entropy_ops_is_instance(self, network):
        """Test that entropy_ops is an instance of EntropyOperations."""
        assert isinstance(network.entropy_ops, EntropyOperations)


# =====================[ ent_magnitude_more_less_same Tests ]======================

class TestEntMagnitudeMoreLessSame:
    """Tests for the ent_magnitude_more_less_same function."""

    def test_extent1_greater_than_extent2(self, network):
        """Test when extent1 > extent2."""
        more, less, same_flag, iterations = network.entropy_ops.ent_magnitude_more_less_same(10.0, 5.0)
        assert more == 10.0
        assert less == 5.0
        assert same_flag is False
        assert isinstance(iterations, int)

    def test_extent2_greater_than_extent1(self, network):
        """Test when extent2 > extent1."""
        more, less, same_flag, iterations = network.entropy_ops.ent_magnitude_more_less_same(5.0, 10.0)
        assert more == 10.0
        assert less == 5.0
        assert same_flag is False

    def test_extents_are_equal(self, network):
        """Test when extent1 == extent2."""
        more, less, same_flag, iterations = network.entropy_ops.ent_magnitude_more_less_same(7.0, 7.0)
        assert more is None
        assert less is None
        assert same_flag is True

    def test_with_decimal_precision_zero(self, network):
        """Test with mag_decimal_precision=0."""
        more, less, same_flag, _ = network.entropy_ops.ent_magnitude_more_less_same(10.5, 10.4, 0)
        # With precision 0, both round to 11, so should be same
        assert same_flag is True

    def test_with_decimal_precision_one(self, network):
        """Test with mag_decimal_precision=1."""
        more, less, same_flag, _ = network.entropy_ops.ent_magnitude_more_less_same(10.5, 10.4, 1)
        # With precision 1, they should be different
        assert same_flag is False
        assert more == 10.5
        assert less == 10.4

    def test_zero_extents(self, network):
        """Test with zero extents."""
        more, less, same_flag, _ = network.entropy_ops.ent_magnitude_more_less_same(0.0, 0.0)
        # Both are 0, so should be same
        assert same_flag is True

    def test_large_difference(self, network):
        """Test with a large difference between extents."""
        more, less, same_flag, _ = network.entropy_ops.ent_magnitude_more_less_same(100.0, 1.0)
        assert more == 100.0
        assert less == 1.0
        assert same_flag is False


# =====================[ ent_overall_same_diff Tests ]======================

class TestEntOverallSameDiff:
    """Tests for the ent_overall_same_diff function."""

    def test_with_active_semantics(self, network):
        """Test with some active semantics."""
        # Set up some semantic activations
        network.semantics.nodes[0, SF.ACT] = 0.9
        network.semantics.nodes[0, SF.DELETED] = B.FALSE
        network.semantics.nodes[1, SF.ACT] = 0.8
        network.semantics.nodes[1, SF.DELETED] = B.FALSE
        
        ratio = network.entropy_ops.ent_overall_same_diff()
        
        # Expected: (1.0 - 0.9) + (1.0 - 0.8) / (0.9 + 0.8)
        expected_ratio = (0.1 + 0.2) / (0.9 + 0.8)
        assert abs(ratio - expected_ratio) < 0.001

    def test_with_no_active_semantics(self, network):
        """Test with no active semantics (all below threshold)."""
        # Set all activations below threshold (0.1)
        network.semantics.nodes[:, SF.ACT] = 0.05
        
        ratio = network.entropy_ops.ent_overall_same_diff()
        
        # Should return 0.0 when no active semantics
        assert ratio == 0.0

    def test_with_perfect_activation(self, network):
        """Test with perfect activation (act = 1.0)."""
        network.semantics.nodes[0, SF.ACT] = 1.0
        network.semantics.nodes[0, SF.DELETED] = B.FALSE
        
        ratio = network.entropy_ops.ent_overall_same_diff()
        
        # (1.0 - 1.0) / 1.0 = 0.0
        assert ratio == 0.0

    def test_with_threshold_activation(self, network):
        """Test with activation exactly at threshold."""
        network.semantics.nodes[0, SF.ACT] = 0.1
        network.semantics.nodes[0, SF.DELETED] = B.FALSE
        
        ratio = network.entropy_ops.ent_overall_same_diff()
        
        # 0.1 is not > 0.1, so should not be counted
        assert ratio == 0.0


# =====================[ find_links_to_abs_dim Tests ]======================

class TestFindLinksToAbsDim:
    """Tests for the find_links_to_abs_dim helper function."""

    def test_finds_matching_links(self, network):
        """Test finding links that match dimension and ontological status."""
        po_idx = 0
        
        # Add a dimension
        dim1 = network.semantics.add_dim("height")
        
        # Add semantics with the dimension
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE})
        )
        sem1_idx = network.semantics.get_index(sem1)
        
        # Create link from PO to semantic
        network.links[po_idx, sem1_idx] = 1.0
        
        result = network.entropy_ops.find_links_to_abs_dim(po_idx, dim1, OntStatus.VALUE)
        
        assert result[sem1_idx].item() is True
        assert result.sum().item() == 1

    def test_filters_by_ont_status(self, network):
        """Test that filtering by ontological status works."""
        po_idx = 0
        dim1 = network.semantics.add_dim("height")
        
        # Add semantic with VALUE ont_status
        sem_value = network.semantics.add_semantic(
            Semantic("sem_value", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE})
        )
        # Add semantic with STATE ont_status
        sem_state = network.semantics.add_semantic(
            Semantic("sem_state", {SF.DIM: dim1, SF.ONT: OntStatus.STATE})
        )
        
        sem_value_idx = network.semantics.get_index(sem_value)
        sem_state_idx = network.semantics.get_index(sem_state)
        
        # Link PO to both
        network.links[po_idx, sem_value_idx] = 1.0
        network.links[po_idx, sem_state_idx] = 1.0
        
        # Search for VALUE only
        result = network.entropy_ops.find_links_to_abs_dim(po_idx, dim1, OntStatus.VALUE)
        
        assert result[sem_value_idx].item() is True
        assert result[sem_state_idx].item() is False

    def test_filters_by_dimension(self, network):
        """Test that filtering by dimension works."""
        po_idx = 0
        dim1 = network.semantics.add_dim("height")
        dim2 = network.semantics.add_dim("width")
        
        # Add semantics with different dimensions
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE})
        )
        sem2 = network.semantics.add_semantic(
            Semantic("sem2", {SF.DIM: dim2, SF.ONT: OntStatus.VALUE})
        )
        
        sem1_idx = network.semantics.get_index(sem1)
        sem2_idx = network.semantics.get_index(sem2)
        
        # Link PO to both
        network.links[po_idx, sem1_idx] = 1.0
        network.links[po_idx, sem2_idx] = 1.0
        
        # Search for dim1 only
        result = network.entropy_ops.find_links_to_abs_dim(po_idx, dim1, OntStatus.VALUE)
        
        assert result[sem1_idx].item() is True
        assert result[sem2_idx].item() is False

    def test_no_matching_links(self, network):
        """Test when no links match the criteria."""
        po_idx = 0
        dim1 = network.semantics.add_dim("height")
        
        # No links created
        result = network.entropy_ops.find_links_to_abs_dim(po_idx, dim1, OntStatus.VALUE)
        
        assert result.sum().item() == 0


# =====================[ en_based_mag_checks Tests ]======================

class TestEnBasedMagChecks:
    """Tests for the en_based_mag_checks function."""

    def test_no_intersecting_dimensions(self, network):
        """Test when POs have no intersecting dimensions."""
        po1_idx = 0
        po2_idx = 1
        
        # Set up POs
        network.token_tensor.tensor[po1_idx, TF.TYPE] = Type.PO.value
        network.token_tensor.tensor[po2_idx, TF.TYPE] = Type.PO.value
        
        # Add different dimensions
        dim1 = network.semantics.add_dim("height")
        dim2 = network.semantics.add_dim("width")
        
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE})
        )
        sem2 = network.semantics.add_semantic(
            Semantic("sem2", {SF.DIM: dim2, SF.ONT: OntStatus.VALUE})
        )
        
        # Link each PO to different dimensions
        network.links[po1_idx, network.semantics.get_index(sem1)] = 0.95
        network.links[po2_idx, network.semantics.get_index(sem2)] = 0.95
        
        high_dim, num_sdm_above, num_sdm_below = network.entropy_ops.en_based_mag_checks(po1_idx, po2_idx)
        
        assert high_dim == []
        assert num_sdm_above == 0
        assert num_sdm_below == 0

    def test_single_intersecting_dimension(self, network):
        """Test when POs share one dimension."""
        po1_idx = 0
        po2_idx = 1
        
        # Add shared dimension
        dim1 = network.semantics.add_dim("height")
        
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 10.0})
        )
        sem2 = network.semantics.add_semantic(
            Semantic("sem2", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE, SF.AMOUNT: 20.0})
        )
        
        # Link both POs to the shared dimension
        network.links[po1_idx, network.semantics.get_index(sem1)] = 0.95
        network.links[po2_idx, network.semantics.get_index(sem2)] = 0.95
        
        high_dim, num_sdm_above, num_sdm_below = network.entropy_ops.en_based_mag_checks(po1_idx, po2_idx)
        
        assert high_dim == [dim1]
        assert num_sdm_above == 0
        assert num_sdm_below == 0

    def test_sdm_connections_above_threshold(self, network):
        """Test detection of SDM connections above threshold."""
        po1_idx = 0
        po2_idx = 1
        
        # Get SDM indices
        sdm_indices = network.semantics.get_sdm_indices()
        more_idx = sdm_indices[0].item()  # MORE is first
        
        # Create link above threshold to SDM
        network.links[po1_idx, more_idx] = 0.95
        
        high_dim, num_sdm_above, num_sdm_below = network.entropy_ops.en_based_mag_checks(po1_idx, po2_idx)
        
        assert num_sdm_above == 1

    def test_sdm_connections_below_threshold(self, network):
        """Test detection of SDM connections below threshold."""
        po1_idx = 0
        po2_idx = 1
        
        # Get SDM indices
        sdm_indices = network.semantics.get_sdm_indices()
        more_idx = sdm_indices[0].item()
        
        # Create link below threshold to SDM
        network.links[po1_idx, more_idx] = 0.5
        
        high_dim, num_sdm_above, num_sdm_below = network.entropy_ops.en_based_mag_checks(po1_idx, po2_idx)
        
        assert num_sdm_below == 1


# =====================[ attach_mag_semantics Tests ]======================

class TestAttachMagSemantics:
    """Tests for the attach_mag_semantics function."""

    def test_attach_more_less_semantics(self, network):
        """Test attaching MORE/LESS semantics when same_flag is False."""
        po1_idx = 0
        po2_idx = 1
        
        # Create sem_links dict
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        
        network.entropy_ops.attach_mag_semantics(False, po1_idx, po2_idx, sem_links)
        
        # Check that MORE is connected to po1 and LESS to po2
        more_ref = network.semantics.sdms[SDM.MORE]
        less_ref = network.semantics.sdms[SDM.LESS]
        more_idx = network.semantics.get_index(more_ref)
        less_idx = network.semantics.get_index(less_ref)
        
        assert network.links[po1_idx, more_idx].item() == 1.0
        assert network.links[po2_idx, less_idx].item() == 1.0

    def test_attach_same_semantics(self, network):
        """Test attaching SAME semantics when same_flag is True."""
        po1_idx = 0
        po2_idx = 1
        
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        
        network.entropy_ops.attach_mag_semantics(True, po1_idx, po2_idx, sem_links)
        
        # Check that SAME is connected to both
        same_ref = network.semantics.sdms[SDM.SAME]
        same_idx = network.semantics.get_index(same_ref)
        
        assert network.links[po1_idx, same_idx].item() == 1.0
        assert network.links[po2_idx, same_idx].item() == 1.0

    def test_halves_sem_link_weights(self, network):
        """Test that semantic link weights are halved."""
        po1_idx = 0
        po2_idx = 1
        
        # Add a semantic and link it
        dim1 = network.semantics.add_dim("height")
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.VALUE})
        )
        sem1_idx = network.semantics.get_index(sem1)
        
        # Set initial weight
        network.links[po1_idx, sem1_idx] = 1.0
        
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        sem_links[po1_idx][sem1_idx] = True
        
        network.entropy_ops.attach_mag_semantics(False, po1_idx, po2_idx, sem_links)
        
        # Check weight was halved
        assert network.links[po1_idx, sem1_idx].item() == 0.5


# =====================[ update_mag_semantics Tests ]======================

class TestUpdateMagSemantics:
    """Tests for the update_mag_semantics function."""

    def test_update_with_same_flag_true(self, network):
        """Test updating semantics when same_flag is True."""
        po1_idx = 0
        po2_idx = 1
        
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        sem_links[po1_idx][0] = True
        sem_links[po2_idx][1] = True
        
        network.entropy_ops.update_mag_semantics(True, po1_idx, po2_idx, sem_links)
        
        same_ref = network.semantics.sdms[SDM.SAME]
        same_idx = network.semantics.get_index(same_ref)
        
        assert network.links[po1_idx, same_idx].item() == 1.0
        assert network.links[po2_idx, same_idx].item() == 1.0

    def test_update_with_same_flag_false(self, network):
        """Test updating semantics when same_flag is False."""
        po1_idx = 0
        po2_idx = 1
        
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        
        network.entropy_ops.update_mag_semantics(False, po1_idx, po2_idx, sem_links)
        
        more_ref = network.semantics.sdms[SDM.MORE]
        less_ref = network.semantics.sdms[SDM.LESS]
        more_idx = network.semantics.get_index(more_ref)
        less_idx = network.semantics.get_index(less_ref)
        
        assert network.links[po1_idx, more_idx].item() == 1.0
        assert network.links[po2_idx, less_idx].item() == 1.0

    def test_sem_links_weights_restored_to_one(self, network):
        """Test that sem_links weights are restored to 1.0."""
        po1_idx = 0
        po2_idx = 1
        
        # Add a semantic
        dim1 = network.semantics.add_dim("height")
        sem1 = network.semantics.add_semantic(
            Semantic("sem1", {SF.DIM: dim1, SF.ONT: OntStatus.STATE})
        )
        sem1_idx = network.semantics.get_index(sem1)
        
        # Set initial weight
        network.links[po1_idx, sem1_idx] = 0.5
        
        num_sems = network.semantics.nodes.shape[0]
        sem_links = {
            po1_idx: torch.zeros(num_sems, dtype=torch.bool),
            po2_idx: torch.zeros(num_sems, dtype=torch.bool),
        }
        sem_links[po1_idx][sem1_idx] = True
        
        network.entropy_ops.update_mag_semantics(True, po1_idx, po2_idx, sem_links)
        
        # Check weight was restored to 1.0
        assert network.links[po1_idx, sem1_idx].item() == 1.0


# =====================[ check_and_run_ent_ops_within Tests ]======================

class TestCheckAndRunEntOpsWithin:
    """Tests for the check_and_run_ent_ops_within function."""

    def test_pred_calls_basic_comparison(self, network):
        """Test that predicates with intersecting dims call basic_en_based_mag_comparison."""
        po1_idx = 0
        po2_idx = 1
        
        # Set up as predicates
        network.token_tensor.tensor[po1_idx, TF.PRED] = B.TRUE
        network.token_tensor.tensor[po2_idx, TF.PRED] = B.TRUE
        
        with patch.object(network.entropy_ops, 'basic_en_based_mag_comparison', return_value="comparison") as mock_comp:
            result = network.entropy_ops.check_and_run_ent_ops_within(
                po1_idx, po2_idx, [1], 0, 1, False, False, False, 1
            )
            mock_comp.assert_called_once()
            assert result == "comparison"

    def test_pred_calls_refinement(self, network):
        """Test that predicates with SDM connections call basic_en_based_mag_refinement."""
        po1_idx = 0
        po2_idx = 1
        
        network.token_tensor.tensor[po1_idx, TF.PRED] = B.TRUE
        network.token_tensor.tensor[po2_idx, TF.PRED] = B.TRUE
        
        with patch.object(network.entropy_ops, 'basic_en_based_mag_refinement', return_value="refinement") as mock_ref:
            result = network.entropy_ops.check_and_run_ent_ops_within(
                po1_idx, po2_idx, [], 1, 0, True, False, False, 1
            )
            mock_ref.assert_called_once()
            assert result == "refinement"

    def test_object_calls_basic_comparison(self, network):
        """Test that objects with intersecting dims call basic_en_based_mag_comparison."""
        po1_idx = 0
        po2_idx = 1
        
        # Set up as objects (not predicates)
        network.token_tensor.tensor[po1_idx, TF.PRED] = B.FALSE
        network.token_tensor.tensor[po2_idx, TF.PRED] = B.FALSE
        
        with patch.object(network.entropy_ops, 'basic_en_based_mag_comparison', return_value="comparison") as mock_comp:
            result = network.entropy_ops.check_and_run_ent_ops_within(
                po1_idx, po2_idx, [1], 0, 0, False, False, False, 1
            )
            mock_comp.assert_called_once()
            assert result == "comparison"

    def test_no_conditions_met_returns_none(self, network):
        """Test that None is returned when no conditions are met."""
        po1_idx = 0
        po2_idx = 1
        
        network.token_tensor.tensor[po1_idx, TF.PRED] = B.TRUE
        
        result = network.entropy_ops.check_and_run_ent_ops_within(
            po1_idx, po2_idx, [], 0, 0, False, False, False, 1
        )
        assert result is None


# =====================[ en_based_mag_checks_results Tests ]======================

class TestEnBasedMagChecksResults:
    """Tests for the en_based_mag_checks_results helper class."""

    def test_initialization(self):
        """Test that the results object initializes correctly."""
        result = en_based_mag_checks_results(
            po1=0, po2=1, high_dim=5, num_sdm_above=2, num_sdm_below=1
        )
        
        assert result.po1 == 0
        assert result.po2 == 1
        assert result.high_dim == 5
        assert result.num_sdm_above == 2
        assert result.num_sdm_below == 1

    def test_stores_all_values(self):
        """Test that all values are stored correctly."""
        result = en_based_mag_checks_results(
            po1=10, po2=20, high_dim=3, num_sdm_above=0, num_sdm_below=0
        )
        
        assert result.po1 == 10
        assert result.po2 == 20
        assert result.high_dim == 3
        assert result.num_sdm_above == 0
        assert result.num_sdm_below == 0
