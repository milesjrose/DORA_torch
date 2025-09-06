# log_utils.py
import logging
from colorlog import ColoredFormatter

class ShortPathFormatter(ColoredFormatter):
    """
    A custom logging formatter that shortens the logger name and adds color.
    """
    def format(self, record):
        full_path = record.name
        parts = full_path.split('.')[-2:]
        record.short_name = '.'.join(parts)

        return super().format(record)