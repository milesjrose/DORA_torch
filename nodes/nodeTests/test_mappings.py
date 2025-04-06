import pytest
import torch
from nodes.nodeMemObjects import Mappings
from nodes.nodeEnums import MappingFields

def test_mappings_initialization():
    # Create test tensors
    size = 3
    connections = torch.ones(size, size)
    weights = torch.ones(size, size)
    hypotheses = torch.ones(size, size)
    max_hyps = torch.ones(size, size)
    driver = None
    
    # Test initialization
    mappings = Mappings(driver, connections, weights, hypotheses, max_hyps)
    
    # Test tensor shape
    assert mappings.adj_matrix.shape == (size, size, len(MappingFields))
    
    # Test accessor methods
    assert torch.all(mappings.connections() == connections)
    assert torch.all(mappings.weights() == weights)
    assert torch.all(mappings.hypotheses() == hypotheses)
    assert torch.all(mappings.max_hyps() == max_hyps)

def test_mappings_invalid_input():
    # Test with mismatched tensor shapes
    size1, size2 = 3, 4
    connections = torch.ones(size1, size2)
    weights = torch.ones(size1, size2)
    hypotheses = torch.ones(size2, size1)
    max_hyps = torch.ones(size1, size2)
    driver = None
    
    # Should raise error for invalid shapes
    with pytest.raises(ValueError):
        Mappings(driver, connections, weights, hypotheses, max_hyps)

def test_mappings_update():
    # Create test tensors
    size1 , size2 = 3, 4
    connections = torch.zeros(size1, size2)
    weights = torch.zeros(size1, size2)
    hypotheses = torch.zeros(size1, size2)
    max_hyps = torch.zeros(size1, size2)
    driver = None
    
    mappings = Mappings(driver, connections, weights, hypotheses, max_hyps)
    
    # Test update methods (once implemented)
    # mappings.updateHypotheses(new_hypotheses)
    # mappings.add_mappings(new_mappings) 