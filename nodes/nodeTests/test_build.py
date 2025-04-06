import pytest
import os
import json
from nodes.nodeBuilder import nodeBuilder
from nodes.nodeEnums import Set

# Import the symProps from sim.py
from .sim import symProps

def test_nodes_from_file():
    test_file = './nodes/nodeTests/sim.py'

    try:
        # Build nodes from the file
        builder = nodeBuilder(file_path=test_file)
        nodes = builder.build_nodes()

        # Verify that driver and recipient exist and have the correct properties
        assert nodes.driver is not None, "Driver should exist"
        assert nodes.recipient is not None, "Recipient should exist"

    finally:
        pass

def test_nodes_from_props():
    try:
        # Build nodes from the file
        builder = nodeBuilder(symProps=symProps)
        nodes = builder.build_nodes()

        # Verify that driver and recipient exist and have the correct properties
        assert nodes.driver is not None, "Driver should exist"
        assert nodes.recipient is not None, "Recipient should exist"

    finally:
        pass