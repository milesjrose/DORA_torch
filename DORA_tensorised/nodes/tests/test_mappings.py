# nodes/tests/test_mappings.py
# Tests the mappings class.

import pytest
import torch

from ..enums import MappingFields, Set
from ..network.connections import Mappings
from ..builder import NetworkBuilder
from ..network import Network


# Import the symProps from sim.py
from .sims.sim import symProps


@pytest.fixture
def network():
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_mappings_initialization(network: Network):
    # Create test tensors
    size = 3
    fields = {}

    driver = network.sets[Set.DRIVER]
    size = driver.nodes.shape[0]
    for field in MappingFields:
        fields[field] = torch.ones(size, size)
    
    # Test initialization
    mappings = Mappings(driver, fields)
    
    # Test tensor shape
    assert mappings.adj_matrix.shape == (size, size, len(MappingFields))

    assert mappings.adj_matrix.dtype == torch.float32
    
    # Test accessor methods
    for field in MappingFields:
        assert torch.all(mappings[field] == fields[field])

    # Test size method
    assert mappings.size(0) == size
    assert mappings.size(1) == size
    assert mappings.size(2) == len(MappingFields)

def test_mappings_invalid_input(network: Network):
    # Test with mismatched tensor shapes
    driver = network.sets[Set.DRIVER]
    size = driver.nodes.shape[0]

    # Should raise error for invalid shapes
    fields = {}
    for i, field in enumerate(MappingFields):
        fields[field] = torch.ones(size, size + i)
    with pytest.raises(ValueError):
        Mappings(driver, fields)

    # Should raise error for driver nodes mismatch
    fields = {}
    for field in MappingFields:
        fields[field] = torch.ones(size, size + 1)
    with pytest.raises(ValueError):
        Mappings(driver, fields)
    