# conftest.py
import pytest
import logging
from nodes.utils.log_utils import ShortPathFormatter # Your updated class

def pytest_configure(config):
    logging_plugin = config.pluginmanager.getplugin("logging-plugin")
    handler = logging_plugin.report_handler
    
    # Add %(log_color)s to the beginning of the format string
    formatter = ShortPathFormatter(
        fmt='%(log_color)s%(levelname)-8s%(reset)s [%(short_name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler.setFormatter(formatter)