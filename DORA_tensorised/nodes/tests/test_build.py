# nodes/tests/test_build.py
# Tests the build function.

import pytest

from nodes.builder import NetworkBuilder
from nodes.enums import Set

# Import the symProps from sim.py
from .sims.sim import symProps

def test_nodes_from_file():
    test_file = './DORA_tensorised/nodes/tests/sims/sim.py'

    try:
        # Build nodes from the file
        builder = NetworkBuilder(file_path=test_file)
        nodes = builder.build_network()

        # Verify that driver and recipient exist and have the correct properties
        assert nodes[Set.DRIVER] is not None, "Driver should exist"
        assert nodes[Set.RECIPIENT] is not None, "Recipient should exist"

    finally:
        pass

def test_nodes_from_props():
    try:
        # Build nodes from the file
        builder = NetworkBuilder(symProps=symProps)
        nodes = builder.build_network()

        # Verify that driver and recipient exist and have the correct properties
        assert nodes[Set.DRIVER] is not None, "Driver should exist"
        assert nodes[Set.RECIPIENT] is not None, "Recipient should exist"

    finally:
        pass