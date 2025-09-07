# nodes/tests/test_firing_ops.py
# Tests for FiringOperations class

import pytest
import nodes.enums as enums

def test_feature_type():
    for feature in enums.TF:
        assert isinstance(enums.feature_type(feature), type)