# nodes/tests/test_links.py
# Tests for links..

import pytest
import torch
import logging
logger = logging.getLogger(__name__)

from nodes.network import Network
from nodes.builder import NetworkBuilder
from nodes.network.single_nodes import Token
from nodes.enums import *
from nodes.network.single_nodes import Ref_Analog, Analog, Ref_Token

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

@pytest.fixture
def network():
    """Create a Nodes object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_calibrate_weights(network: Network):
    """Test calibrating most active links to driver"""
    # Set link weights for driver token
    sets = network.links.sets
    po_mask = network.driver().get_mask(Type.PO)
    network.links.sets[Set.DRIVER][po_mask, :] = 0.0
    network.links.sets[Set.DRIVER][po_mask, 1] = 0.9
    logger.debug(f"po_links:{sets[Set.DRIVER][po_mask,:]}")
    #Check weight
    failed = network.links[Set.DRIVER][po_mask, 1] != 0.9
    assert failed.nonzero().shape[0] == 0
    # calibrate weights
    network.links.calibrate_weights()
    logger.debug(f"po_links:{sets[Set.DRIVER][po_mask,:]}")
    # Check new weights
    failed = network.links[Set.DRIVER][po_mask, 1] != 1.0
    assert failed.nonzero().shape[0] == 0

def test_get_max_linked_sem(network: Network):
    """Test getting the semantic with the highest link weight to a token."""
    # Get a token from the driver set
    driver_set = network.sets[Set.DRIVER]
    po_mask = driver_set.get_mask(Type.PO)
    po_indices = torch.where(po_mask)[0]
    
    if len(po_indices) > 0:
        # Get the first PO token
        token_index = po_indices[0].item()
        token_id = driver_set.nodes[token_index, TF.ID].item()
        ref_token = Ref_Token(Set.DRIVER, token_id)
        
        # Clear all link weights for this token first
        network.links.sets[Set.DRIVER][token_index, :] = 0.0
        
        # Set up some link weights for this token
        # Set a high weight for semantic at index 1
        network.links.sets[Set.DRIVER][token_index, 1] = 0.9
        # Set lower weights for other semantics
        network.links.sets[Set.DRIVER][token_index, 0] = 0.3
        if network.links.sets[Set.DRIVER].size(1) > 2:
            network.links.sets[Set.DRIVER][token_index, 2] = 0.5
        
        # Get the semantic with maximum link weight
        max_linked_sem = network.links.get_max_linked_sem(ref_token)
        
        # Should return a Ref_Semantic object
        assert hasattr(max_linked_sem, 'ID')
        assert hasattr(max_linked_sem, 'name')
        
        # The semantic with highest weight should be at index 1
        # Get the actual ID of the semantic at index 1
        expected_semantic_id = network.semantics.nodes[1, SF.ID].item()
        assert max_linked_sem.ID == expected_semantic_id
        
        # Test with a different token setup
        if len(po_indices) > 1:
            token_index2 = po_indices[1].item()
            token_id2 = driver_set.nodes[token_index2, TF.ID].item()
            ref_token2 = Ref_Token(Set.DRIVER, token_id2)
            
            # Clear all link weights for this token first
            network.links.sets[Set.DRIVER][token_index2, :] = 0.0
            
            # Set different weights for this token
            network.links.sets[Set.DRIVER][token_index2, 0] = 0.8
            network.links.sets[Set.DRIVER][token_index2, 1] = 0.2
            
            max_linked_sem2 = network.links.get_max_linked_sem(ref_token2)
            
            # Should return the semantic at index 0 for this token
            expected_semantic_id2 = network.semantics.nodes[0, SF.ID].item()
            assert max_linked_sem2.ID == expected_semantic_id2