# conftest.py
import pytest
import logging
from nodes.utils.log_utils import ShortPathFormatter

suppress_warnings = True

def suppress_warnings_filter(record):
    """A custom filter to block any log messages with the WARNING level."""
    return record.levelno != logging.WARNING

def pytest_configure(config):
    logging_plugin = config.pluginmanager.getplugin("logging-plugin")
    if logging_plugin is not None and hasattr(logging_plugin, 'report_handler'):
        handler = logging_plugin.report_handler
        
        # Add %(log_color)s to the beginning of the format string
        formatter = ShortPathFormatter(
            fmt='%(log_color)s%(levelname)-8s%(reset)s [%(short_name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)

        # --- NEW PART: Add the custom filter to the handler ---
        if suppress_warnings:
            handler.addFilter(suppress_warnings_filter)
