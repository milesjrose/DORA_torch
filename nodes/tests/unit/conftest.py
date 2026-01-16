# conftest.py for unit tests
# Adds DORA_tensorised to Python path so 'nodes' module can be imported

import sys
import logging
import pytest
from pathlib import Path

# Get the DORA_tensorised directory (parent of nodes)
# conftest.py is in DORA_tensorised/nodes/tests/unit/
# So we need to go up 4 levels: unit -> tests -> nodes -> DORA_tensorised
dora_tensorised_dir = Path(__file__).parent.parent.parent.parent

# Add to Python path if not already there
if str(dora_tensorised_dir) not in sys.path:
    sys.path.insert(0, str(dora_tensorised_dir))


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

