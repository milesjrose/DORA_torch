# nodes/tests/test_semantics.py
# Tests for semantics operations.

import pytest
import torch

from nodes.builder import NetworkBuilder
from nodes.enums import *
from nodes.network.sets.semantics import Semantics
from nodes.network.single_nodes import Semantic, Ref_Semantic
from nodes.network.connections import Links

# Import the symProps from sim.py
from nodes.tests.sims.sim import symProps

from nodes.network import Network


@pytest.fixture
def network():
    """Create a Network object using the sim.py data."""
    builder = NetworkBuilder(symProps=symProps)
    return builder.build_network()

def test_init_sdm(network:Network):
    """Test that the sdm semantics are initialised correctly."""
    sems:Semantics = network.semantics
    sems.init_sdm()
    for sdm in SDM:
        assert sems.sdms[sdm] is not None
        assert sems.sdms[sdm].name == sdm.name
        assert sems.sdm_dims[sdm] is not None
        
        sdm_sem_ref = sems.sdms[sdm]
        
        assert sems.get(sdm_sem_ref, SF.TYPE) == Type.SEMANTIC
        assert sems.get(sdm_sem_ref, SF.DIM) == sems.sdm_dims[sdm]
        assert sems.get(sdm_sem_ref, SF.ONT) == OntStatus.SDM

def test_get_sdm_indices(network:Network):
    """Test that the sdm indices are retrieved correctly."""
    sems:Semantics = network.semantics
    
    indices = sems.get_sdm_indices()
    assert indices.shape == (3, 1)
    
    sdm_indices = [
        network.get_index(sems.sdms[SDM.MORE]),
        network.get_index(sems.sdms[SDM.LESS]),
        network.get_index(sems.sdms[SDM.SAME]),
    ]
    
    for index in sdm_indices:
        assert index in indices

    indices = sems.get_sdm_indices(include_diff=True)
    assert indices.shape == (4, 1)
    
    sdm_indices.append(network.get_index(sems.sdms[SDM.DIFF]))
    
    for index in sdm_indices:
        assert index in indices

