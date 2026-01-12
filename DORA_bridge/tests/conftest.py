# conftest.py for DORA_bridge tests

import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def testsim_path():
    """Path to the testsim.py simulation file."""
    return str(project_root / 'nodes' / 'tests' / 'test_sims' / 'testsim.py')


@pytest.fixture
def testsym_path():
    """Path to the testsym.py simulation file."""
    return str(project_root / 'nodes' / 'tests' / 'test_sims' / 'testsym.py')


@pytest.fixture
def simple_props():
    """Simple symProps for testing without file loading."""
    return [
        {
            "name": "prop1",
            "RBs": [
                {
                    "pred_name": "pred1",
                    "pred_sem": ["sem1", "sem2"],
                    "higher_order": False,
                    "object_name": "obj1",
                    "object_sem": ["sem3", "sem4"],
                    "P": "non_exist",
                }
            ],
            "set": "driver",
            "analog": 0,
        },
        {
            "name": "prop2",
            "RBs": [
                {
                    "pred_name": "pred2",
                    "pred_sem": ["sem1", "sem2"],
                    "higher_order": False,
                    "object_name": "obj2",
                    "object_sem": ["sem5", "sem6"],
                    "P": "non_exist",
                }
            ],
            "set": "recipient",
            "analog": 1,
        },
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test output files."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir

