import pytest
import os
import json
from nodes.nodeBuilder import nodeBuilder
from nodes.nodeEnums import Set

def test_nodes_from_file():
    test_file = 'testsim15.py'

    try:
        # Build nodes from the file
        builder = nodeBuilder(file_path=test_file)
        nodes = builder.build_nodes()

        # Verify that driver and recipient exist and have the correct properties
        assert nodes.driver is not None, "Driver should exist"
        assert nodes.recipient is not None, "Recipient should exist"

    finally:
        pass
