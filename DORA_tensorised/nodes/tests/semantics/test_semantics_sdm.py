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


def test_add_dim(network:Network):
    """Test that a dimension can be added to the semantics."""
    sems:Semantics = network.semantics
    initial_dim_count = len(sems.dimensions)
    new_dim_key = sems.add_dim("test_dim")
    assert len(sems.dimensions) == initial_dim_count + 1
    assert sems.dimensions[new_dim_key] == "test_dim"

def test_get_dim_and_name(network:Network):
    """Test that the dimension of a semantic can be retrieved."""
    sems:Semantics = network.semantics
    sems.init_sdm()
    
    more_sem_ref = sems.sdms[SDM.MORE]
    dim_key = sems.get_dim(more_sem_ref)
    assert dim_key == sems.sdm_dims[SDM.MORE]
    
    dim_name = sems.get_dim_name(dim_key)
    assert dim_name == SDM.MORE.name

def test_set_dim_and_name(network:Network):
    """Test that the dimension of a semantic can be set."""
    sems:Semantics = network.semantics
    sem_ref = Ref_Semantic(ID=1, name="sem1")

    sems.set_dim(sem_ref, "new_dim")
    dim_key = sems.get_dim(sem_ref)
    dim_name = sems.get_dim_name(dim_key)
    assert dim_name == "new_dim"
    
    sems.set_dim_name(dim_key, "updated_dim")
    dim_name = sems.get_dim_name(dim_key)
    assert dim_name == "updated_dim"

