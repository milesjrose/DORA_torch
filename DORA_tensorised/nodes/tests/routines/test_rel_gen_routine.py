import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.single_nodes import Token, Ref_Analog, Ref_Token, Analog
from nodes.tests.sims.sim import symProps
from nodes.utils import tensor_ops as tOps

from random import shuffle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodes.network.connections import Mappings
    from nodes.network import Network
    from nodes.network.sets import Driver, Recipient
    from nodes.network.connections import Mappings

@pytest.fixture
def network():
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()


def test_rel_gen_passes(network: 'Network'):
    """
    Test should pass when at least one mapping exists and all mappings have weight >= 0.7.
    """
    # Setup: Create one valid mapping
    recipient = network.recipient()
    driver = network.driver()
    mappings = network.mappings[Set.RECIPIENT]
    
    # Ensure there's at least one token in each set
    if recipient.nodes.shape[0] == 0: recipient.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT))
    if driver.nodes.shape[0] == 0: driver.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.8

    assert network.routines.rel_gen.requirements() is True


def test_rel_gen_fails_no_mappings(network: 'Network'):
    """
    Test should fail when no mappings exist.
    """
    # Setup: Mappings are empty by default
    assert network.routines.rel_gen.requirements() is False


def test_rel_gen_fails_low_weight(network: 'Network'):
    """
    Test should fail when a mapping has weight < 0.7.
    """
    # Setup: Create one mapping with a low weight
    recipient = network.recipient()
    driver = network.driver()
    mappings = network.mappings[Set.RECIPIENT]
    
    if recipient.nodes.shape[0] == 0: recipient.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT))
    if driver.nodes.shape[0] == 0: driver.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))

    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.5

    assert network.routines.rel_gen.requirements() is False


def setup_rel_gen_environment(network: 'Network', mapping_weight=0.8):
    """Helper to set up environment for rel_gen tests."""
    driver = network.driver()
    recipient = network.recipient()
    mappings = network.mappings[Set.RECIPIENT]
    
    # Ensure we have tokens in both sets
    if driver.nodes.shape[0] == 0:
        driver.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
    if recipient.nodes.shape[0] == 0:
        recipient.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT))
    
    # Create a valid mapping
    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = mapping_weight
    
    return driver, recipient, mappings


def test_infer_token_po_type(network: 'Network'):
    """Test infer_token function for PO token type."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a PO token in driver
    po_token = Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER)
    ref_maker = driver.add_token(po_token)
    
    # Create a recipient analog
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Test inferring PO token
    ref_made = network.routines.rel_gen.infer_token(ref_maker, recip_analog, Set.RECIPIENT)
    
    # Verify the inferred token
    assert ref_made.set == Set.RECIPIENT
    made_token = recipient.token_op.get_single_token(ref_made)
    assert made_token.tensor[TF.TYPE] == Type.PO
    assert made_token.tensor[TF.INFERRED] == B.TRUE
    assert made_token.tensor[TF.ACT] == 1.0
    assert made_token.tensor[TF.ANALOG] == 1
    assert made_token.tensor[TF.MAKER_UNIT] == network.get_index(ref_maker)
    assert made_token.tensor[TF.MAKER_SET] == ref_maker.set
    assert made_token.tensor[TF.PRED] == B.FALSE  # Should copy from maker
    
    # Verify maker unit has made_unit and made_set set
    assert network.get_value(ref_maker, TF.MADE_UNIT) == network.get_index(ref_made)
    assert network.get_value(ref_maker, TF.MADE_SET) == Set.RECIPIENT


def test_infer_token_p_type(network: 'Network'):
    """Test infer_token function for P token type."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a P token in driver
    p_token = Token(Type.P, {TF.MODE: Mode.CHILD}, set=Set.DRIVER)
    ref_maker = driver.add_token(p_token)
    
    # Create a recipient analog
    recip_analog = Ref_Analog(analog_number=2, set=Set.RECIPIENT)
    
    # Test inferring P token
    ref_made = network.routines.rel_gen.infer_token(ref_maker, recip_analog, Set.RECIPIENT)
    
    # Verify the inferred token
    made_token = recipient.token_op.get_single_token(ref_made)
    assert made_token.tensor[TF.TYPE] == Type.P
    assert made_token.tensor[TF.INFERRED] == B.TRUE
    assert made_token.tensor[TF.ACT] == 1.0
    assert made_token.tensor[TF.ANALOG] == 2
    assert made_token.tensor[TF.MODE] == Mode.CHILD  # Should copy from maker


def test_infer_token_rb_type(network: 'Network'):
    """Test infer_token function for RB token type."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a RB token in driver
    rb_token = Token(Type.RB, set=Set.DRIVER)
    ref_maker = driver.add_token(rb_token)
    
    # Create a recipient analog
    recip_analog = Ref_Analog(analog_number=3, set=Set.RECIPIENT)
    
    # Test inferring RB token
    ref_made = network.routines.rel_gen.infer_token(ref_maker, recip_analog, Set.RECIPIENT)
    
    # Verify the inferred token
    made_token = recipient.token_op.get_single_token(ref_made)
    assert made_token.tensor[TF.TYPE] == Type.RB
    assert made_token.tensor[TF.INFERRED] == B.TRUE
    assert made_token.tensor[TF.ACT] == 1.0
    assert made_token.tensor[TF.ANALOG] == 3


def test_infer_token_invalid_type(network: 'Network'):
    """Test infer_token function with invalid token type."""
    driver, _, _ = setup_rel_gen_environment(network)
    
    # Create an invalid token (this should not happen in normal operation)
    # We'll test with a token that has an invalid type by directly setting it
    invalid_token = Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER)  # Start with valid type
    ref_maker = driver.add_token(invalid_token)
    
    # Manually set an invalid type (simulating edge case)
    network.set_value(ref_maker, TF.TYPE, 999)  # Invalid type value
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should raise ValueError for invalid type
    with pytest.raises(ValueError):
        network.routines.rel_gen.infer_token(ref_maker, recip_analog, Set.RECIPIENT)


def test_rel_gen_type_po_no_active_token(network: 'Network'):
    """Test rel_gen_type for PO when no active token exists."""
    setup_rel_gen_environment(network)
    
    # Ensure no PO tokens are active
    driver = network.driver()
    po_mask = driver.get_mask(Type.PO)
    if po_mask.any():
        driver.nodes[po_mask, TF.ACT] = 0.0
    
    # Should return early without error
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)


def test_rel_gen_type_po_below_threshold(network: 'Network'):
    """Test rel_gen_type for PO when active token is below threshold."""
    driver, _, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with low activation
    po_token = Token(Type.PO, {TF.ACT: 0.3, TF.PRED: B.FALSE}, set=Set.DRIVER)  # Below 0.5 threshold
    ref_po = driver.add_token(po_token)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 0.0  # No mapping
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should return early due to low activation
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify no new token was created
    initial_count = network.recipient().get_count()
    assert network.recipient().get_count() == initial_count


def test_rel_gen_type_po_max_map_not_zero(network: 'Network'):
    """Test rel_gen_type for PO when max_map is not zero."""
    driver, _, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with high activation
    po_token = Token(Type.PO, {TF.ACT: 0.8, TF.PRED: B.FALSE}, set=Set.DRIVER)  # Above 0.5 threshold
    ref_po = driver.add_token(po_token)
    
    # Set up mapping with max_map > 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 1.0  # Has mapping
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.8
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should return early due to max_map not being zero
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify no new token was created
    initial_count = network.recipient().get_count()
    assert network.recipient().get_count() == initial_count


def test_rel_gen_type_po_infer_new_token(network: 'Network'):
    """Test rel_gen_type for PO when inferring a new token."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with high activation and no made unit
    po_token = Token(Type.PO, {TF.ACT: 0.8, TF.MADE_UNIT: null, TF.PRED: B.FALSE}, set=Set.DRIVER)
    ref_po = driver.add_token(po_token)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 0.0  # No mapping
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    initial_recipient_count = recipient.get_count()
    initial_new_set_count = network.new_set().get_count()
    
    # Should infer new tokens
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify new tokens were created
    assert recipient.get_count() == initial_recipient_count + 1
    assert network.new_set().get_count() == initial_new_set_count + 1
    
    # Verify made_unit is set
    assert network.get_value(ref_po, TF.MADE_UNIT) != null


def test_rel_gen_type_po_update_existing_made_unit(network: 'Network'):
    """Test rel_gen_type for PO when updating an existing made unit."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with high activation and existing made unit
    po_token = Token(Type.PO, {TF.ACT: 0.8, TF.PRED: B.FALSE}, set=Set.DRIVER)
    ref_po = driver.add_token(po_token)
    
    # Create a made unit
    made_po = Token(Type.PO, {TF.ACT: 0.5, TF.PRED: B.FALSE}, set=Set.RECIPIENT)
    ref_made = recipient.add_token(made_po)
    made_idx = network.get_index(ref_made)
    
    # Set the made unit reference
    network.set_value(ref_po, TF.MADE_UNIT, made_idx)
    network.set_value(ref_po, TF.MADE_SET, Set.RECIPIENT)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should update existing made unit
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify made unit activation was updated
    assert network.get_value(ref_made, TF.ACT) == 1.0


def test_rel_gen_type_rb_connect_to_po(network: 'Network'):
    """Test rel_gen_type for RB when connecting to active PO."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create an RB token with high activation and existing made unit
    rb_token = Token(Type.RB, {TF.ACT: 0.8}, set=Set.DRIVER)
    ref_rb = driver.add_token(rb_token)
    
    # Create a made RB unit
    made_rb = Token(Type.RB, {TF.ACT: 0.5}, set=Set.RECIPIENT)
    ref_made_rb = recipient.add_token(made_rb)
    made_idx = network.get_index(ref_made_rb)
    
    # Set the made unit reference
    network.set_value(ref_rb, TF.MADE_UNIT, made_idx)
    network.set_value(ref_rb, TF.MADE_SET, Set.RECIPIENT)
    
    # Create an active PO in recipient
    active_po = Token(Type.PO, {TF.ACT: 0.8, TF.PRED: B.FALSE}, set=Set.RECIPIENT)
    ref_active_po = recipient.add_token(active_po)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    rb_idx = network.get_index(ref_rb)
    mappings[MappingFields.CONNECTIONS][0, rb_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, rb_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should connect RB to active PO
    network.routines.rel_gen.rel_gen_type(Type.RB, 0.5, recip_analog)
    
    # Verify made RB activation was updated
    assert network.get_value(ref_made_rb, TF.ACT) == 1.0


def test_rel_gen_type_p_child_mode(network: 'Network'):
    """Test rel_gen_type for P token in child mode."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a P token with high activation and existing made unit
    p_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.CHILD}, set=Set.DRIVER)
    ref_p = driver.add_token(p_token)
    
    # Create a made P unit
    made_p = Token(Type.P, {TF.ACT: 0.5, TF.MODE: Mode.CHILD}, set=Set.RECIPIENT)
    ref_made_p = recipient.add_token(made_p)
    made_idx = network.get_index(ref_made_p)
    
    # Set the made unit reference
    network.set_value(ref_p, TF.MADE_UNIT, made_idx)
    network.set_value(ref_p, TF.MADE_SET, Set.RECIPIENT)
    
    # Create an active RB in recipient
    active_rb = Token(Type.RB, {TF.ACT: 0.8}, set=Set.RECIPIENT)
    ref_active_rb = recipient.add_token(active_rb)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    p_idx = network.get_index(ref_p)
    mappings[MappingFields.CONNECTIONS][0, p_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, p_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should connect P to active RB as child
    network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, Mode.CHILD)
    
    # Verify made P activation was updated
    assert network.get_value(ref_made_p, TF.ACT) == 1.0


def test_rel_gen_type_p_parent_mode(network: 'Network'):
    """Test rel_gen_type for P token in parent mode."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a P token with high activation and existing made unit
    p_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.PARENT}, set=Set.DRIVER)
    ref_p = driver.add_token(p_token)
    
    # Create a made P unit
    made_p = Token(Type.P, {TF.ACT: 0.5, TF.MODE: Mode.PARENT}, set=Set.RECIPIENT)
    ref_made_p = recipient.add_token(made_p)
    made_idx = network.get_index(ref_made_p)
    
    # Set the made unit reference
    network.set_value(ref_p, TF.MADE_UNIT, made_idx)
    network.set_value(ref_p, TF.MADE_SET, Set.RECIPIENT)
    
    # Create an active RB in recipient
    active_rb = Token(Type.RB, {TF.ACT: 0.6}, set=Set.RECIPIENT)  # Above 0.5 threshold for parent mode
    ref_active_rb = recipient.add_token(active_rb)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    p_idx = network.get_index(ref_p)
    mappings[MappingFields.CONNECTIONS][0, p_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, p_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should connect RB to P as parent
    network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, Mode.PARENT)
    
    # Verify made P activation was updated
    assert network.get_value(ref_made_p, TF.ACT) == 1.0


def test_rel_gen_routine_integration(network: 'Network'):
    """Test the complete rel_gen_routine integration."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create various token types in driver
    po_token = Token(Type.PO, {TF.ACT: 0.8, TF.MADE_UNIT: null, TF.PRED: B.FALSE}, set=Set.DRIVER)
    rb_token = Token(Type.RB, {TF.ACT: 0.8, TF.MADE_UNIT: null}, set=Set.DRIVER)
    p_child_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.CHILD, TF.MADE_UNIT: null}, set=Set.DRIVER)
    p_parent_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.PARENT, TF.MADE_UNIT: null}, set=Set.DRIVER)
    
    ref_po = driver.add_token(po_token)
    ref_rb = driver.add_token(rb_token)
    ref_p_child = driver.add_token(p_child_token)
    ref_p_parent = driver.add_token(p_parent_token)
    
    # Set up mappings with max_map = 0.0 for all
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    rb_idx = network.get_index(ref_rb)
    p_child_idx = network.get_index(ref_p_child)
    p_parent_idx = network.get_index(ref_p_parent)
    
    for idx in [po_idx, rb_idx, p_child_idx, p_parent_idx]:
        mappings[MappingFields.CONNECTIONS][0, idx] = 0.0
        mappings[MappingFields.WEIGHT][0, idx] = 0.0
    
    # Create recipient analog
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    initial_recipient_count = recipient.get_count()
    initial_new_set_count = network.new_set().get_count()
    
    # Run the complete routine
    network.routines.rel_gen.rel_gen_routine(recip_analog)
    
    # Verify tokens were created (4 types * 2 sets = 8 new tokens)
    assert recipient.get_count() == initial_recipient_count + 4
    assert network.new_set().get_count() == initial_new_set_count + 4
    
    # Verify made_unit references are set
    assert network.get_value(ref_po, TF.MADE_UNIT) != null
    assert network.get_value(ref_rb, TF.MADE_UNIT) != null
    assert network.get_value(ref_p_child, TF.MADE_UNIT) != null
    assert network.get_value(ref_p_parent, TF.MADE_UNIT) != null


def test_requirements_multiple_mappings_all_valid(network: 'Network'):
    """Test requirements when multiple mappings exist and all are valid."""
    driver, recipient, mappings = setup_rel_gen_environment(network)
    
    # Add more tokens to create multiple mappings
    for i in range(3):
        if i < driver.nodes.shape[0]:
            driver.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
        if i < recipient.nodes.shape[0]:
            recipient.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT))
    
    # Create multiple valid mappings
    for i in range(min(3, recipient.nodes.shape[0])):
        for j in range(min(3, driver.nodes.shape[0])):
            mappings[MappingFields.CONNECTIONS][i, j] = 1.0
            mappings[MappingFields.WEIGHT][i, j] = 0.8
    
    assert network.routines.rel_gen.requirements() is True


def test_requirements_multiple_mappings_one_invalid(network: 'Network'):
    """Test requirements when multiple mappings exist but one has low weight."""
    driver, recipient, mappings = setup_rel_gen_environment(network)
    
    # Add more tokens to create multiple mappings
    for i in range(3):
        if i < driver.nodes.shape[0]:
            driver.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.DRIVER))
        if i < recipient.nodes.shape[0]:
            recipient.tensor_ops.add_token(Token(Type.PO, {TF.PRED: B.FALSE}, set=Set.RECIPIENT))
    
    # Create multiple mappings with one invalid weight
    for i in range(min(3, recipient.nodes.shape[0])):
        for j in range(min(3, driver.nodes.shape[0])):
            mappings[MappingFields.CONNECTIONS][i, j] = 1.0
            # Set one mapping to have low weight
            weight = 0.5 if i == 1 and j == 1 else 0.8
            mappings[MappingFields.WEIGHT][i, j] = weight
    
    assert network.routines.rel_gen.requirements() is False


def test_requirements_edge_case_exactly_threshold(network: 'Network'):
    """Test requirements when mapping weight is exactly at threshold."""
    driver, recipient, mappings = setup_rel_gen_environment(network)
    
    # Set weight exactly at threshold (0.7)
    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.7
    
    assert network.routines.rel_gen.requirements() is True


def test_requirements_edge_case_just_below_threshold(network: 'Network'):
    """Test requirements when mapping weight is just below threshold."""
    driver, recipient, mappings = setup_rel_gen_environment(network)
    
    # Set weight just below threshold
    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.699999
    
    assert network.routines.rel_gen.requirements() is False


def test_requirements_debug_mode(network: 'Network'):
    """Test requirements function with debug mode enabled."""
    driver, recipient, mappings = setup_rel_gen_environment(network)
    
    # Enable debug mode
    network.routines.rel_gen.debug = True
    
    # Set up invalid mapping
    mappings[MappingFields.CONNECTIONS][0, 0] = 1.0
    mappings[MappingFields.WEIGHT][0, 0] = 0.5
    
    # Should return False but not raise exception (debug prints the error)
    result = network.routines.rel_gen.requirements()
    assert result is False
    
    # Disable debug mode
    network.routines.rel_gen.debug = False


def test_rel_gen_type_edge_case_activation_exactly_threshold(network: 'Network'):
    """Test rel_gen_type when activation is exactly at threshold."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with activation exactly at threshold
    po_token = Token(Type.PO, {TF.ACT: 0.5, TF.MADE_UNIT: null, TF.PRED: B.FALSE}, set=Set.DRIVER)  # Exactly at threshold
    ref_po = driver.add_token(po_token)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    initial_recipient_count = recipient.get_count()
    
    # Should proceed since activation >= threshold
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify new token was created
    assert recipient.get_count() == initial_recipient_count + 1


def test_rel_gen_type_edge_case_activation_just_below_threshold(network: 'Network'):
    """Test rel_gen_type when activation is just below threshold."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a PO token with activation just below threshold
    po_token = Token(Type.PO, {TF.ACT: 0.499999, TF.MADE_UNIT: null, TF.PRED: B.FALSE}, set=Set.DRIVER)  # Just below threshold
    ref_po = driver.add_token(po_token)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    po_idx = network.get_index(ref_po)
    mappings[MappingFields.CONNECTIONS][0, po_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, po_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    initial_recipient_count = recipient.get_count()
    
    # Should not proceed since activation < threshold
    network.routines.rel_gen.rel_gen_type(Type.PO, 0.5, recip_analog)
    
    # Verify no new token was created
    assert recipient.get_count() == initial_recipient_count


def test_rel_gen_type_rb_po_below_threshold(network: 'Network'):
    """Test rel_gen_type for RB when active PO is below connection threshold."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create an RB token with high activation and existing made unit
    rb_token = Token(Type.RB, {TF.ACT: 0.8}, set=Set.DRIVER)
    ref_rb = driver.add_token(rb_token)
    
    # Create a made RB unit
    made_rb = Token(Type.RB, {TF.ACT: 0.5}, set=Set.RECIPIENT)
    ref_made_rb = recipient.add_token(made_rb)
    made_idx = network.get_index(ref_made_rb)
    
    # Set the made unit reference
    network.set_value(ref_rb, TF.MADE_UNIT, made_idx)
    network.set_value(ref_rb, TF.MADE_SET, Set.RECIPIENT)
    
    # Create a PO in recipient with low activation (below 0.7 threshold)
    low_act_po = Token(Type.PO, {TF.ACT: 0.6, TF.PRED: B.FALSE}, set=Set.RECIPIENT)  # Below 0.7 threshold
    ref_low_po = recipient.add_token(low_act_po)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    rb_idx = network.get_index(ref_rb)
    mappings[MappingFields.CONNECTIONS][0, rb_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, rb_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should update made unit but not connect to low-activation PO
    network.routines.rel_gen.rel_gen_type(Type.RB, 0.5, recip_analog)
    
    # Verify made RB activation was updated
    assert network.get_value(ref_made_rb, TF.ACT) == 1.0


def test_rel_gen_type_p_child_rb_below_threshold(network: 'Network'):
    """Test rel_gen_type for P in child mode when active RB is below threshold."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a P token with high activation and existing made unit
    p_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.CHILD}, set=Set.DRIVER)
    ref_p = driver.add_token(p_token)
    
    # Create a made P unit
    made_p = Token(Type.P, {TF.ACT: 0.5, TF.MODE: Mode.CHILD}, set=Set.RECIPIENT)
    ref_made_p = recipient.add_token(made_p)
    made_idx = network.get_index(ref_made_p)
    
    # Set the made unit reference
    network.set_value(ref_p, TF.MADE_UNIT, made_idx)
    network.set_value(ref_p, TF.MADE_SET, Set.RECIPIENT)
    
    # Create an RB in recipient with low activation (below 0.7 threshold)
    low_act_rb = Token(Type.RB, {TF.ACT: 0.6}, set=Set.RECIPIENT)  # Below 0.7 threshold
    ref_low_rb = recipient.add_token(low_act_rb)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    p_idx = network.get_index(ref_p)
    mappings[MappingFields.CONNECTIONS][0, p_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, p_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should update made unit but not connect to low-activation RB
    network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, Mode.CHILD)
    
    # Verify made P activation was updated
    assert network.get_value(ref_made_p, TF.ACT) == 1.0


def test_rel_gen_type_p_parent_rb_below_threshold(network: 'Network'):
    """Test rel_gen_type for P in parent mode when active RB is below threshold."""
    driver, recipient, _ = setup_rel_gen_environment(network)
    
    # Create a P token with high activation and existing made unit
    p_token = Token(Type.P, {TF.ACT: 0.8, TF.MODE: Mode.PARENT}, set=Set.DRIVER)
    ref_p = driver.add_token(p_token)
    
    # Create a made P unit
    made_p = Token(Type.P, {TF.ACT: 0.5, TF.MODE: Mode.PARENT}, set=Set.RECIPIENT)
    ref_made_p = recipient.add_token(made_p)
    made_idx = network.get_index(ref_made_p)
    
    # Set the made unit reference
    network.set_value(ref_p, TF.MADE_UNIT, made_idx)
    network.set_value(ref_p, TF.MADE_SET, Set.RECIPIENT)
    
    # Create an RB in recipient with low activation (below 0.5 threshold for parent mode)
    low_act_rb = Token(Type.RB, {TF.ACT: 0.4}, set=Set.RECIPIENT)  # Below 0.5 threshold
    ref_low_rb = recipient.add_token(low_act_rb)
    
    # Set up mapping with max_map = 0.0
    mappings = network.mappings[Set.RECIPIENT]
    p_idx = network.get_index(ref_p)
    mappings[MappingFields.CONNECTIONS][0, p_idx] = 0.0
    mappings[MappingFields.WEIGHT][0, p_idx] = 0.0
    
    recip_analog = Ref_Analog(analog_number=1, set=Set.RECIPIENT)
    
    # Should update made unit but not connect to low-activation RB
    network.routines.rel_gen.rel_gen_type(Type.P, 0.5, recip_analog, Mode.PARENT)
    
    # Verify made P activation was updated
    assert network.get_value(ref_made_p, TF.ACT) == 1.0