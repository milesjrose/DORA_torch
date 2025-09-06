# logging_config.py

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'short': {
            # This tells Python to use your custom class
            'class': 'nodes.utils.log_utils.ShortPathFormatter',
            # Note: We now use '%(short_name)s' which we created in our custom class
            'format': '%(levelname)-8s %(asctime)s [%(short_name)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'standard': {
            'format': '%(levelname)-8s %(asctime)s [%(name)s] %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'short', # Use our new 'short' formatter
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}