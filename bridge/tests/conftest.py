# conftest.py for DORA_bridge tests
# Adds DORA_torch to Python path so 'nodes' module can be imported

import sys
from pathlib import Path
import logging
# Add project root to path - MUST be done before any other imports
# conftest.py is in DORA_torch/DORA_bridge/tests/
# So we need to go up 3 levels: tests -> DORA_bridge -> DORA_torch
project_root = Path(__file__).resolve().parent.parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import pytest


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

# =====================[ Logging Configuration ]====================
# Configure logging levels for tests
# You can modify these levels to control verbosity during test runs

def configure_test_logging():
    """
    Configure logging for tests.
    Set logging levels for specific modules/classes here.
    """
    # Set root logger level (affects all loggers unless overridden)
    logging.getLogger().setLevel(logging.DEBUG)  # Only show WARNING and above by default
    info_loggers = ["set", "tns", "tkn", "con", "SP_DEBUG"]
    for logger_name in info_loggers:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Set specific logger levels for classes/modules
    # Examples:
    # logging.getLogger('nodes.network.routines.rel_form').setLevel(logging.DEBUG)
    # logging.getLogger('nodes.network.operations.node_ops').setLevel(logging.INFO)
    # logging.getLogger('nodes.network.operations.analog_ops').setLevel(logging.WARNING)
    
    # You can also set levels for entire packages:
    # logging.getLogger('nodes.network.routines').setLevel(logging.DEBUG)
    # logging.getLogger('nodes.network.operations').setLevel(logging.INFO)


# Configure logging when pytest loads this conftest
configure_test_logging()


@pytest.fixture(scope="function")
def reset_logging():
    """
    Fixture to reset logging levels before each test.
    Useful if tests modify logging levels and you want clean state.
    """
    # Store original levels
    original_levels = {}
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if hasattr(logger, 'level'):
            original_levels[logger_name] = logger.level
    
    yield
    
    # Restore original levels
    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)


