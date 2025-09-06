# log_utils.py
import logging
from colorlog import ColoredFormatter # <-- Import this

# Inherit from ColoredFormatter instead of logging.Formatter
class ShortPathFormatter(ColoredFormatter):
    """
    A custom logging formatter that shortens the logger name and adds color.
    """
    def format(self, record):
        # This part is the same as before: create the short name
        full_path = record.name
        parts = full_path.split('.')[-2:]
        record.short_name = '.'.join(parts)
        
        # Now let the ColoredFormatter do its magic to add colors
        return super().format(record)